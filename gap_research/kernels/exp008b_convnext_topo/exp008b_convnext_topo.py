"""EXP-008b (Wave 2 GPU Kernel #4): 5-Fold ConvNeXt-Small UNet++ w/ Soft-clDice Topology & Tversky Loss.

True third Stage-2 segmentation family & loss family:
  - Architecture: smp.UnetPlusPlus('tu-convnext_small.fb_in22k', in_channels=3 [enhance.planes])
  - Loss: 0.40 * BCE + 0.35 * Tversky(alpha=0.40, beta=0.60) + 0.25 * Soft-clDice(iter=6)
    explicitly penalizing breaks / pinched tips on thin elongated solar filaments (PQ IoU > 0.50 cliff).
  - Crop augmentation: CropDataset(..., trunc=0.4, pad_hi=0.6), AdamW + cosine decay, 8 epochs/fold on 2 x T4.
  - Infers strictly OOF + Test on ms (all 707 + 180) and Fold-0 + Test on F3, P6, P4a, s, main.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np


def _find_comp_root() -> Path:
    for p in Path("/kaggle/input").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"):
        if "magfilo-extra" not in str(p).lower():
            return p.parents[1]
    raise FileNotFoundError("MAGFiLO_1.0_Kaggle_2026 not found under /kaggle/input")


def _find_input_dir(slug: str) -> Path:
    for c in (
        Path(f"/kaggle/input/datasets/ajikajadeh/{slug}"),
        Path(f"/kaggle/input/notebooks/ajikajadeh/{slug}"),
        Path(f"/kaggle/input/{slug}"),
    ):
        if c.exists():
            return c
    for p in Path("/kaggle/input").glob(f"**/{slug}"):
        if p.is_dir():
            return p
    raise FileNotFoundError(slug)


def _soft_skeletonize(prob, iters: int = 6):
    import torch
    import torch.nn.functional as F

    def _erode(img):
        p1 = -F.max_pool2d(-img, (3, 1), stride=(1, 1), padding=(1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), stride=(1, 1), padding=(0, 1))
        return torch.minimum(p1, p2)

    def _dilate(img):
        return F.max_pool2d(img, (3, 3), stride=(1, 1), padding=(1, 1))

    img = prob
    img1 = _dilate(_erode(img))
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = _erode(img)
        img1 = _dilate(_erode(img))
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def topo_loss_fn(logit, target):
    import torch
    import torch.nn.functional as F

    bce = F.binary_cross_entropy_with_logits(logit, target)
    p = torch.sigmoid(logit).float()
    t = target.float()

    # Recall-slightly-favored Tversky (alpha=0.40 FP, beta=0.60 FN) prevents thin filament shrinkage
    tp = (p * t).sum(dim=(1, 2, 3))
    fp = (p * (1.0 - t)).sum(dim=(1, 2, 3))
    fn = ((1.0 - p) * t).sum(dim=(1, 2, 3))
    tversky = 1.0 - ((tp + 1.0) / (tp + 0.40 * fp + 0.60 * fn + 1.0)).mean()

    # Soft-clDice centerline continuity loss (Shit et al., CVPR 2021) + Skeleton Recall Loss (Kirchhoff et al., ECCV 2024)
    skel_p = _soft_skeletonize(p, iters=6)
    skel_t = _soft_skeletonize(t, iters=6)
    tprec = ((skel_p * t).sum(dim=(1, 2, 3)) + 1.0) / (skel_p.sum(dim=(1, 2, 3)) + 1.0)
    tsens = ((skel_t * p).sum(dim=(1, 2, 3)) + 1.0) / (skel_t.sum(dim=(1, 2, 3)) + 1.0)
    cldice = 1.0 - (2.0 * tprec * tsens / (tprec + tsens + 1e-6)).mean()

    return 0.35 * bce + 0.35 * tversky + 0.15 * cldice + 0.15 * (1.0 - tsens.mean())


def _build_convnext_model(pretrained: bool = True):
    import segmentation_models_pytorch as smp

    # UnetPlusPlus fails on 0-channel skip level 1 of ConvNeXt in smp 0.5.0; Unet handles ConvNeXt skips cleanly.
    return smp.Unet(
        encoder_name="tu-convnext_small.fb_in22k",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=1,
    )


def worker_train_fold(args) -> None:
    import torch
    from torch.utils.data import DataLoader

    t0 = time.time()
    repo_dir = Path(args.repo)
    sys.path.insert(0, str(repo_dir))
    os.environ["MAGFILO_ROOT"] = args.data_root
    os.environ["FILSEG_WORK"] = args.work_dir
    from filseg import data, segmenter

    fold = args.fold
    torch.manual_seed(fold)
    np.random.seed(fold)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tr, va = data.split(fold)
    ds_tr = segmenter.CropDataset(tr, size=512, pad=0.2, jitter=0.12, train=True, seed=fold, trunc=0.4, pad_hi=0.6)
    ds_va = segmenter.CropDataset(va, size=512, pad=0.2, jitter=0.0, train=False, seed=fold, trunc=0.4, pad_hi=0.6)

    dl_tr = DataLoader(ds_tr, batch_size=8, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    model = _build_convnext_model(pretrained=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-4)
    total_steps = args.epochs * len(dl_tr)
    warm_steps = max(1, len(dl_tr))

    def _lr_fn(step: int) -> float:
        if step < warm_steps:
            return (step + 1) / warm_steps
        prog = (step - warm_steps) / max(1, total_steps - warm_steps)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_fn)
    scaler = torch.amp.GradScaler("cuda")
    best_dice = -1.0
    out_w = Path(args.work_dir) / "weights" / f"seg_cnxtopo_f{fold}.pth"
    out_w.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        if ep > 1 and (time.time() - t0) / 60.0 >= args.max_fold_minutes:
            print(f"[ConvNeXtTopo f{fold}] time budget {args.max_fold_minutes:.1f}m reached before ep {ep}; keeping best_dice={best_dice:.4f}", flush=True)
            break
        model.train()
        t_losses = []
        ep_t0 = time.time()
        for x, y in dl_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = topo_loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            t_losses.append(float(loss.item()))

        model.eval()
        inter, union = 0.0, 0.0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    pr = (model(x).float().sigmoid() > 0.5).float()
                inter += float((pr * y).sum().item())
                union += float((pr + y).sum().item())
        dice = (2.0 * inter + 1e-6) / (union + 1e-6)
        print(
            f"[ConvNeXtTopo f{fold}] ep={ep}/{args.epochs} loss={np.mean(t_losses):.4f} "
            f"val_dice={dice:.4f} ({(time.time()-ep_t0)/60:.1f}m)",
            flush=True,
        )
        if dice > best_dice:
            best_dice = dice
            torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, out_w)
    print(f"[ConvNeXtTopo f{fold}] DONE best_dice={best_dice:.4f} total={(time.time()-t0)/60:.1f}m", flush=True)


def worker_infer(args) -> None:
    import torch

    t0 = time.time()
    repo_dir = Path(args.repo)
    sys.path.insert(0, str(repo_dir))
    os.environ["MAGFILO_ROOT"] = args.data_root
    os.environ["FILSEG_WORK"] = args.work_dir
    from filseg import data, enhance, postprocess, segmenter

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w_dir = Path(args.work_dir) / "weights"
    models = {}
    for k in range(data.N_FOLDS):
        m = _build_convnext_model(pretrained=False)
        sd = torch.load(w_dir / f"seg_cnxtopo_f{k}.pth", map_location="cpu")
        m.load_state_dict({kk: vv.float() if vv.is_floating_point() else vv for kk, vv in sd.items()})
        models[k] = m.to(dev).eval()

    idx = json.loads(Path(args.index_json).read_text())
    stem_to_fold = {s: k for k in range(data.N_FOLDS) for s in data.split(k)[1]}

    tasks = []
    for sname in args.streams.split(","):
        if not sname:
            continue
        for sp in ("oof", "val", "test"):
            for stem in sorted(idx.get(sname, {}).get(sp, {})):
                fk = stem_to_fold[stem] if sp in ("oof", "val") else None
                tasks.append((sname, sp, stem, fk))

    if args.num_shards > 1:
        tasks = [t for t in tasks if int(hashlib.md5(f"{t[1]}_{t[2]}".encode()).hexdigest()[:8], 16) % args.num_shards == args.shard_id]

    by_photo = {}
    for sname, sp, stem, fk in tasks:
        by_photo.setdefault((sp, stem, fk), []).append(sname)

    out_dir = Path(args.out_dir)
    n_done = 0
    for (sp, stem, fk), pstreams in sorted(by_photo.items()):
        if all((out_dir / sname / sp / f"{stem}.npz").exists() for sname in pstreams):
            continue
        planes = enhance.planes(stem, cache=True)
        segs = [models[fk]] if fk is not None else [models[k] for k in range(data.N_FOLDS)]
        for sname in pstreams:
            dst = out_dir / sname / sp / f"{stem}.npz"
            if dst.exists():
                continue
            with np.load(idx[sname][sp][stem]) as d:
                conf = np.asarray(d["conf"], np.float32)
                boxes = np.asarray(d["box"], np.int32).reshape(-1, 4)
            ws = [tuple(int(v) for v in b) for b in boxes]
            if not ws:
                postprocess.save_cache(dst, conf, ws, [])
            else:
                probs = segmenter.predict(segs, planes, ws, batch=24, tta=True, device=dev)
                postprocess.save_cache(dst, conf, ws, probs)
        n_done += 1
        if n_done % 50 == 0 or n_done == len(by_photo):
            print(f"[cnxtopo_infer shard {args.shard_id}] {n_done}/{len(by_photo)} ({(time.time()-t0)/60:.1f}m)", flush=True)


def orchestrate() -> None:
    t0 = time.time()
    print(
        subprocess.run(
            "pip install -q ultralytics==8.4.160 segmentation-models-pytorch==0.5.0 pycocotools scikit-image 2>&1 | tail -2",
            shell=True,
            capture_output=True,
            text=True,
        ).stdout,
        flush=True,
    )
    subprocess.run("nvidia-smi -L; nproc", shell=True)

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
    out_dir = Path("/tmp/out_cnxtopo")
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

    # Pre-warm timm ConvNeXt-Small ImageNet-22k weights cleanly
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import segmentation_models_pytorch as smp, torch; "
            "m = smp.Unet('tu-convnext_small.fb_in22k', encoder_weights='imagenet', in_channels=3, classes=1); "
            "m(torch.zeros(2, 3, 512, 512)); "
            "print('ConvNeXt-Small UNet ImageNet-22k weights & forward shape OK')",
        ],
        check=True,
    )

    py = sys.executable
    self_path = str(Path(__file__).resolve())
    common_env = dict(os.environ, MAGFILO_ROOT=str(data_root), FILSEG_WORK=str(work_dir), PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(repo_dir))

    rc_planes = subprocess.call([py, str(repo_dir / "run.py"), "planes", "--procs", "4"], env=common_env)
    assert rc_planes == 0, "planes failed"

    err_box: list[BaseException] = []

    def _train(gpu: int, folds: tuple[int, ...]):
        try:
            for fk in folds:
                rc = subprocess.call(
                    [
                        py, self_path, "--worker", "train_fold", "--fold", str(fk), "--epochs", "9",
                        "--repo", str(repo_dir), "--data-root", str(data_root), "--work-dir", str(work_dir),
                    ],
                    env=dict(common_env, CUDA_VISIBLE_DEVICES=str(gpu)),
                )
                assert rc == 0, f"ConvNeXt fold {fk} failed"
        except BaseException as exc:
            err_box.append(exc)

    th0 = threading.Thread(target=_train, args=(0, (0, 2, 4)))
    th1 = threading.Thread(target=_train, args=(1, (1, 3)))
    th0.start(); th1.start()
    th0.join(); th1.join()
    if err_box:
        raise err_box[0]

    ps = [
        subprocess.Popen(
            [
                py, self_path, "--worker", "infer",
                "--repo", str(repo_dir), "--data-root", str(data_root), "--work-dir", str(work_dir),
                "--index-json", str(index_json), "--out-dir", str(out_dir),
                "--streams", "ms,F3,P6,P4a,s,main", "--shard-id", str(g), "--num-shards", "2",
            ],
            env=dict(common_env, CUDA_VISIBLE_DEVICES=str(g)),
        )
        for g in (0, 1)
    ]
    assert all(p.wait() == 0 for p in ps), "cnxtopo infer failed"

    w_bundle = Path("/tmp/cnxtopo_weights")
    w_bundle.mkdir(parents=True, exist_ok=True)
    for k in range(5):
        shutil.copy2(weights_dir / f"seg_cnxtopo_f{k}.pth", w_bundle / f"seg_cnxtopo_f{k}.pth")

    summary = {"minutes": round((time.time() - t0) / 60.0, 2), "archives": {}}
    for sdir, tpath in (
        (out_dir, "/kaggle/working/exp008b_cnxtopo.tar.gz"),
        (w_bundle, "/kaggle/working/exp008b_weights.tar.gz"),
    ):
        with tarfile.open(tpath, "w:gz") as tf:
            tf.add(sdir, arcname=sdir.name)
        summary["archives"][Path(tpath).name] = round(Path(tpath).stat().st_size / 1e6, 2)
    Path("/kaggle/working/exp008b_summary.json").write_text(json.dumps(summary, indent=2))
    print("EXP008B SUMMARY:", json.dumps(summary), flush=True)
    shutil.rmtree(work_dir, ignore_errors=True)
    shutil.rmtree(planes_tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default="orchestrate", choices=["orchestrate", "train_fold", "infer"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=9)
    ap.add_argument("--max-fold-minutes", type=float, default=65.0)
    ap.add_argument("--repo", default="")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--work-dir", default="")
    ap.add_argument("--index-json", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--streams", default="ms,F3,P6,P4a,s,main")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    if args.worker == "orchestrate":
        orchestrate()
    elif args.worker == "train_fold":
        worker_train_fold(args)
    elif args.worker == "infer":
        worker_infer(args)


if __name__ == "__main__":
    main()
