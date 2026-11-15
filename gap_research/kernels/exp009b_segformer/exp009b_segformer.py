"""EXP-009b (Wave 3 GPU Kernel): 5-Fold Pure Hierarchical Transformer SegFormer-B3 Stage-2 Segmenter + Topology Loss.

Research Gap Addressed:
  Prior Stage-2 segmenters are CNN U-Net/Unet++ (ResNeXt50, EfficientNet-B4), Hiera SAM2, or ConvNeXt-Small.
  SegFormer (`mit_b3`, hierarchical Mix-Transformer encoder + lightweight All-MLP decoder) captures long-range
  global continuity across broken faint filament segments without convolutional receptive-field bias.
  1. Uses strictly competition-only `CropDataset(..., trunc=0.4, pad_hi=0.6)` on `enhance.planes`.
  2. Loss: 0.45 BCEWithLogits + 0.35 Tversky(0.3, 0.7) + 0.20 Soft-clDice continuity.
  3. 5-fold strictly OOF inference + 5-model Test inference across ms, s, main, F3, P6, and P4a boxes -> /tmp/segformer.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import numpy as np

FULL = 2048


def _find_comp_root() -> Path:
    for p in Path("/kaggle/input").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"):
        if "magfilo-extra" not in str(p).lower():
            return p.parents[1]
    raise FileNotFoundError("MAGFiLO_1.0_Kaggle_2026 not found under /kaggle/input")


def _find_input_dir(slug: str) -> Path:
    cands = [
        Path(f"/kaggle/input/datasets/ajikajadeh/{slug}"),
        Path(f"/kaggle/input/notebooks/ajikajadeh/{slug}"),
        Path(f"/kaggle/input/{slug}"),
    ]
    for p in cands:
        if p.exists():
            return p
    for p in Path("/kaggle/input").glob(f"**/{slug}"):
        if p.is_dir():
            return p
    raise FileNotFoundError(f"Could not locate {slug} under /kaggle/input")


def _bind_filseg(repo_dir: Path, data_root: Path, work_dir: Path):
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    os.environ["MAGFILO_ROOT"] = str(data_root)
    os.environ["FILSEG_WORK"] = str(work_dir)
    from filseg import data, enhance, postprocess, segmenter

    data.ROOT = data_root
    data.TRAIN_JSON = data_root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    data.TRAIN_IMAGES = data_root / "train" / "train_images"
    data.TEST_IMAGES = data_root / "test" / "test_images"
    data.FOLDS = repo_dir / "folds" / "folds.json"
    data.WORK = work_dir
    data.WEIGHTS = work_dir / "weights"
    return data, enhance, postprocess, segmenter


def _soft_erode(img):
    import torch.nn.functional as F

    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return p1.minimum(p2)


def _soft_dilate(img):
    import torch.nn.functional as F

    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def _soft_skel(img, iters: int = 5):
    import torch.nn.functional as F

    img1 = _soft_dilate(_soft_erode(img))
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = _soft_erode(img)
        img1 = _soft_dilate(_soft_erode(img))
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def topo_loss(logit, target):
    import torch.nn.functional as F

    x = logit.float()
    t = target.float()
    bce = F.binary_cross_entropy_with_logits(x, t)
    p = x.sigmoid()
    tp = (p * t).sum()
    fp = (p * (1.0 - t)).sum()
    fn = ((1.0 - p) * t).sum()
    tversky = 1.0 - (tp + 1e-6) / (tp + 0.3 * fp + 0.7 * fn + 1e-6)

    sp = _soft_skel(p, 5)
    st = _soft_skel(t, 5)
    tprec = ((sp * t).sum() + 1e-5) / (sp.sum() + 1e-5)
    tsens = ((st * p).sum() + 1e-5) / (st.sum() + 1e-5)
    cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens + 1e-5)
    return 0.45 * bce + 0.35 * tversky + 0.20 * cl_dice


def build_segformer(pretrained: bool = True):
    import segmentation_models_pytorch as smp

    return smp.Segformer(
        encoder_name="mit_b3",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=1,
        activation=None,
    )


def worker_train_fold(repo_dir: Path, data_root: Path, work_dir: Path, fold: int, epochs: int) -> None:
    import torch
    from torch.utils.data import DataLoader

    data, _, _, segmenter = _bind_filseg(repo_dir, data_root, work_dir)
    tr, va = data.split(fold)
    ds_tr = segmenter.CropDataset(tr, size=512, pad=0.2, jitter=0.12, train=True, seed=100 + fold, trunc=0.4, pad_hi=0.6)
    ds_va = segmenter.CropDataset(va, size=512, pad=0.2, jitter=0.0, train=False, seed=fold, trunc=0.4, pad_hi=0.6)
    dl_tr = DataLoader(ds_tr, batch_size=8, shuffle=True, num_workers=2, drop_last=True, pin_memory=True, worker_init_fn=segmenter._reseed)
    dl_va = DataLoader(ds_va, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_segformer(pretrained=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    out_w = work_dir / "weights" / f"segformer_b3_f{fold}.pth"
    out_w.parent.mkdir(parents=True, exist_ok=True)
    best_dice = -1.0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for x, t in dl_tr:
            x, t = x.to(dev, non_blocking=True), t.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                logit = model(x)
            loss = topo_loss(logit, t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))
        sched.step()

        model.eval()
        inter = psum = tsum = 0.0
        with torch.no_grad():
            for x, t in dl_va:
                with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                    p = (model(x.to(dev)).float().sigmoid() > 0.5).float()
                t = t.to(dev)
                inter += float((p * t).sum())
                psum += float(p.sum())
                tsum += float(t.sum())
        dice = 2.0 * inter / max(psum + tsum, 1.0)
        print(f"[SegFormer-B3 f{fold}] ep={ep}/{epochs} loss={np.mean(losses):.4f} val_dice={dice:.5f} ({(time.time()-t0)/60:.1f}m)", flush=True)
        if dice > best_dice:
            best_dice = dice
            torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, out_w)


def _load_ckpt(ckpt: Path, dev):
    import torch

    m = build_segformer(pretrained=False)
    sd = torch.load(ckpt, map_location="cpu")
    m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()}, strict=True)
    return m.to(dev).eval()


def worker_infer(repo_dir: Path, data_root: Path, work_dir: Path, index_json: Path, out_dir: Path, part: int) -> None:
    data, enhance, postprocess, segmenter = _bind_filseg(repo_dir, data_root, work_dir)
    folds_map = data.folds()
    idx = json.loads(index_json.read_text())
    dev = "cuda:0"
    models = {k: _load_ckpt(work_dir / "weights" / f"segformer_b3_f{k}.pth", dev) for k in range(5)}
    all_models = [models[k] for k in range(5)]

    tasks = []
    for sname, sp_map in idx.items():
        for sp, stem_map in sp_map.items():
            for st, npz_p in sorted(stem_map.items()):
                tasks.append((sname, sp, st, Path(npz_p)))
    my_tasks = tasks[part::2]
    t0 = time.time()
    for i, (sname, sp, st, npz_p) in enumerate(my_tasks):
        z = np.load(npz_p)
        conf, boxes = z["conf"], z["box"]
        ws = [tuple(int(v) for v in b) for b in boxes]
        if len(ws):
            segs = [models[folds_map[st]]] if sp in ("oof", "val") else all_models
            planes = enhance.planes(st)
            probs = segmenter.predict(segs, planes, ws, batch=24, tta=True, device=dev)
        else:
            probs = []
        postprocess.save_cache(out_dir / sname / sp / f"{st}.npz", conf, ws, probs)
        if (i + 1) % 300 == 0:
            print(f"[SegFormer infer p{part}] {i+1}/{len(my_tasks)} ({(time.time()-t0)/60:.1f}m)", flush=True)


def _pack(src_dir: Path, dst_tgz: Path) -> None:
    tmp = Path("/tmp") / f".tmp_{dst_tgz.name}"
    with tarfile.open(tmp, "w:gz") as tf:
        tf.add(src_dir, arcname=src_dir.name)
    dst_tgz.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp), str(dst_tgz))
    print(f"Packed {dst_tgz.name}: {dst_tgz.stat().st_size/1e6:.2f} MB", flush=True)


def run_driver(epochs: int) -> None:
    t0 = time.time()
    subprocess.run(
        "pip install -q segmentation-models-pytorch==0.5.0 pycocotools scikit-image 2>&1 | tail -2",
        shell=True,
        check=True,
    )

    import torch

    n_gpu = torch.cuda.device_count()
    data_root = _find_comp_root()
    repo_dir = next(p for p in Path("/kaggle/input").rglob("run.py") if (p.parent / "filseg").is_dir()).parent
    infer_ms = _find_input_dir("filseg-infer-ms")
    infer_s = _find_input_dir("filseg-infer-s")
    infer_main = _find_input_dir("filseg-infer-main")
    g19_dir = _find_input_dir("filseg-twostage-g19")
    final_dir = _find_input_dir("filseg-final-pipeline")

    work_dir = Path("/tmp/work")
    planes_tmp = Path("/tmp/planes")
    weights_dir = work_dir / "weights"
    out_dir = Path("/tmp/segformer")
    for d in (work_dir, planes_tmp, weights_dir, out_dir):
        d.mkdir(parents=True, exist_ok=True)
    planes_link = work_dir / "planes"
    if not planes_link.exists():
        planes_link.symlink_to(planes_tmp)

    base_cache = Path("/tmp/base_caches")
    for sname, inp in (("ms", infer_ms), ("s", infer_s), ("main", infer_main)):
        tgz = inp / f"cache_{sname}.tgz"
        if tgz.exists():
            with tarfile.open(tgz, "r:gz") as tf:
                tf.extractall(base_cache)
        elif (inp / "work" / "cache" / sname).exists():
            shutil.copytree(inp / "work" / "cache" / sname, base_cache / "cache" / sname, dirs_exist_ok=True)

    idx = {}
    for sname in ("ms", "s", "main"):
        idx[sname] = {}
        for sp in ("oof", "test"):
            sub = base_cache / "cache" / sname / sp
            idx[sname][sp] = {f.stem: str(f) for f in sorted(sub.rglob("*.npz"))} if sub.exists() else {}
    for sname, parent, rel in (("F3", g19_dir, "tsF3/cache/S1E"), ("P6", g19_dir, "tsP6/cache/S1E"), ("P4a", final_dir, "final/tsP4a/cache/S1E")):
        sub_root = parent / rel
        idx[sname] = {}
        for sp in ("val", "test"):
            sub = sub_root / sp
            idx[sname][sp] = {f.stem: str(f) for f in sorted(sub.rglob("*.npz"))} if sub.exists() else {}
    index_json = Path("/tmp/stream_index.json")
    index_json.write_text(json.dumps(idx))

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import segmentation_models_pytorch as smp; "
            "smp.Segformer('mit_b3', encoder_weights='imagenet', in_channels=3, classes=1); "
            "print('MiT-B3 ready')",
        ],
        check=True,
    )

    py = sys.executable
    self_path = str(Path(__file__).resolve())
    common_env = dict(
        os.environ,
        MAGFILO_ROOT=str(data_root),
        FILSEG_WORK=str(work_dir),
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(repo_dir),
    )

    subprocess.run([py, str(repo_dir / "run.py"), "planes", "--procs", "2"], env=common_env, check=True)

    fold_groups = [(0, (0, 2, 4)), (min(1, max(0, n_gpu - 1)), (1, 3))] if n_gpu >= 2 else [(0, (0, 1, 2, 3, 4))]
    errs: list[BaseException] = []

    def _run_folds(gpu: int, fks: tuple[int, ...]):
        try:
            for fk in fks:
                rc = subprocess.call(
                    [
                        py, self_path, "--worker", "train_fold", "--fold", str(fk), "--epochs", str(epochs),
                        "--repo", str(repo_dir), "--data-root", str(data_root), "--work-dir", str(work_dir),
                    ],
                    env=dict(common_env, CUDA_VISIBLE_DEVICES=str(gpu)),
                )
                assert rc == 0, f"SegFormer fold {fk} failed"
        except BaseException as exc:
            errs.append(exc)

    ts = [threading.Thread(target=_run_folds, args=(g, fks)) for g, fks in fold_groups]
    for th in ts:
        th.start()
    for th in ts:
        th.join()
    assert not errs, errs

    p_errs: list[BaseException] = []

    def _run_inf(part: int):
        try:
            gpu = part if n_gpu >= 2 else 0
            rc = subprocess.call(
                [
                    py, self_path, "--worker", "infer", "--part", str(part),
                    "--repo", str(repo_dir), "--data-root", str(data_root), "--work-dir", str(work_dir),
                    "--index", str(index_json), "--out-dir", str(out_dir),
                ],
                env=dict(common_env, CUDA_VISIBLE_DEVICES=str(gpu)),
            )
            assert rc == 0, f"SegFormer infer p{part} failed"
        except BaseException as exc:
            p_errs.append(exc)

    tis = [threading.Thread(target=_run_inf, args=(0,)), threading.Thread(target=_run_inf, args=(1,))]
    for th in tis:
        th.start()
    for th in tis:
        th.join()
    assert not p_errs, p_errs

    w_dir = Path("/tmp/segformer_weights")
    w_dir.mkdir(parents=True, exist_ok=True)
    for k in range(5):
        shutil.copy2(weights_dir / f"segformer_b3_f{k}.pth", w_dir / f"segformer_b3_f{k}.pth")

    _pack(out_dir, Path("/kaggle/working/exp009b_segformer.tar.gz"))
    _pack(w_dir, Path("/kaggle/working/exp009b_weights.tar.gz"))
    summary = {"elapsed_min": round((time.time() - t0) / 60, 2), "epochs": epochs, "encoder": "mit_b3"}
    Path("/kaggle/working/exp009b_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"EXP-009b COMPLETE in {summary['elapsed_min']}m", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["train_fold", "infer"])
    ap.add_argument("--repo", type=Path)
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--work-dir", type=Path)
    ap.add_argument("--index", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    if args.worker is None:
        run_driver(args.epochs)
    elif args.worker == "train_fold":
        worker_train_fold(args.repo, args.data_root, args.work_dir, args.fold, args.epochs)
    elif args.worker == "infer":
        worker_infer(args.repo, args.data_root, args.work_dir, args.index, args.out_dir, args.part)


if __name__ == "__main__":
    main()
