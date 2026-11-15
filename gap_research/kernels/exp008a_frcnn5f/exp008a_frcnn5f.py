"""EXP-008a (Wave 2 GPU Kernel #3): 5-Fold Cross-Fitted Faster R-CNN R50-FPN v2 + Multi-Segmenter.

Eliminates F3's Fold-0-only coverage gap (missing 31 overlapping-date Test images + Folds 1..4 OOF):
  1. Trains clean 5-fold Faster R-CNN ResNet50-FPN v2 (imgsz=1408, competition-only annotator records,
     strictly OOF across folds 0..4, dihedral augmentation, AdamW cosine, fp16 AMP, best val AP50).
  2. Runs 3-view flip TTA (identity, hflip, vflip) + WBF inference on 707 OOF + 180 Test images.
  3. Segments F_5F boxes with seg_e4 (+ grow=1), S1E, and SAM2.1 Hiera-B+ -> compact .tar.gz bundles.
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

FULL = 2048
IMGSZ = 1408


def _find_comp_root() -> Path:
    for p in Path("/kaggle/input").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"):
        if "magfilo-extra" not in str(p).lower():
            return p.parents[1]
    raise FileNotFoundError("MAGFiLO_1.0_Kaggle_2026 not found under /kaggle/input")


def _build_frcnn(imgsz: int = IMGSZ, pretrained: bool = True):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    m = fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT" if pretrained else None,
        weights_backbone=None,
        min_size=imgsz,
        max_size=imgsz,
        box_detections_per_img=100,
    )
    m.roi_heads.box_predictor = FastRCNNPredictor(1024, 2)
    return m


class _FrcnnRecordsDataset:
    def __init__(self, img_dir: Path, stems: list[str], recs: dict, anns: dict, train: bool, seed: int = 0):
        self.samples: list[tuple[Path, np.ndarray]] = []
        self.train = train
        self.rng = np.random.default_rng(seed)
        for s in stems:
            rids = recs[s] if train else recs[s][:1]
            for rid in rids:
                bs = []
                for a in anns.get(rid, []):
                    x, y, w, h = a["bbox"]
                    if w >= 2 and h >= 2:
                        bs.append([x, y, x + w, y + h])
                arr = np.asarray(bs, np.float32).reshape(-1, 4)
                if train and len(arr) == 0:
                    continue
                self.samples.append((img_dir / f"{s}.jpeg", arr))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import torch

        p, boxes = self.samples[idx]
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise FileNotFoundError(str(p))
        boxes = boxes.copy()
        if self.train:
            k = int(self.rng.integers(4))
            if k:
                g = np.rot90(g, k)
                for _ in range(k):
                    x1, y1, x2, y2 = boxes[:, 0].copy(), boxes[:, 1].copy(), boxes[:, 2].copy(), boxes[:, 3].copy()
                    # CCW rotation by 90 deg: (x, y) -> (y, FULL - x)
                    boxes = np.stack([y1, FULL - x2, y2, FULL - x1], axis=1)
            if self.rng.random() < 0.5:
                g = g[:, ::-1]
                boxes[:, [0, 2]] = FULL - boxes[:, [2, 0]]
            # Mild contrast/brightness jitter matching filament H-alpha robustness
            alpha = float(self.rng.uniform(0.92, 1.08))
            beta = float(self.rng.uniform(-6.0, 6.0))
            g = np.clip(g.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        g = np.ascontiguousarray(g)
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, FULL)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, FULL)
        keep = (boxes[:, 2] - boxes[:, 0] >= 2) & (boxes[:, 3] - boxes[:, 1] >= 2)
        boxes = boxes[keep]
        img_t = torch.from_numpy(g).float().div(255.0).unsqueeze(0).expand(3, -1, -1).contiguous()
        target = {
            "boxes": torch.from_numpy(boxes.astype(np.float32)),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
        }
        return img_t, target


def _eval_ap50(model, ds, dev) -> float:
    import torch
    from torchvision.ops import box_iou

    model.eval()
    scores, hits, npos = [], [], 0
    with torch.no_grad():
        for i in range(len(ds)):
            img_t, tgt = ds[i]
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                out = model([img_t.to(dev)])[0]
            pb = out["boxes"].detach().cpu()
            ps = out["scores"].detach().cpu().numpy()
            gt = tgt["boxes"]
            npos += len(gt)
            used = set()
            if len(pb) and len(gt):
                ious = box_iou(pb, gt).numpy()
            else:
                ious = np.zeros((len(pb), len(gt)))
            order = np.argsort(-ps)
            for pi in order:
                scores.append(float(ps[pi]))
                if len(gt) == 0:
                    hits.append(0)
                    continue
                gi = int(np.argmax(ious[pi]))
                if ious[pi, gi] >= 0.5 and gi not in used:
                    used.add(gi)
                    hits.append(1)
                else:
                    hits.append(0)
    if npos == 0 or not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.cumsum(np.asarray(hits)[order])
    fp = np.cumsum(1 - np.asarray(hits)[order])
    rec = tp / npos
    prec = tp / np.maximum(tp + fp, 1)
    # 101-point COCO AP50
    ap = 0.0
    for r in np.linspace(0.0, 1.0, 101):
        m = prec[rec >= r]
        ap += float(m.max()) if len(m) else 0.0
    return ap / 101.0


def worker_train_fold(args) -> None:
    import torch
    from torch.utils.data import DataLoader

    t0 = time.time()
    repo_dir = Path(args.repo)
    data_root = Path(args.data_root)
    work_dir = Path(args.work_dir)
    fold = args.fold

    sys.path.insert(0, str(repo_dir))
    os.environ["MAGFILO_ROOT"] = str(data_root)
    from filseg import data

    torch.manual_seed(fold)
    np.random.seed(fold)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    recs, anns = data.coco_index()
    tr, va = data.split(fold)
    ds_tr = _FrcnnRecordsDataset(data.TRAIN_IMAGES, tr, recs, anns, train=True, seed=fold)
    ds_va = _FrcnnRecordsDataset(data.TRAIN_IMAGES, va, recs, anns, train=False, seed=fold)

    bs = 4
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=2, collate_fn=lambda b: tuple(zip(*b)), pin_memory=True)

    model = _build_frcnn(IMGSZ, pretrained=True).to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1.5e-4, weight_decay=1e-4)
    total_steps = args.epochs * len(dl_tr)
    warm_steps = max(1, len(dl_tr))

    def _lr_fn(step: int) -> float:
        if step < warm_steps:
            return (step + 1) / warm_steps
        prog = (step - warm_steps) / max(1, total_steps - warm_steps)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_fn)
    scaler = torch.amp.GradScaler("cuda")

    best_ap = -1.0
    out_w = work_dir / "weights" / f"frcnn_f{fold}.pth"
    out_w.parent.mkdir(parents=True, exist_ok=True)
    bad = 0

    for ep in range(1, args.epochs + 1):
        if ep > 1 and (time.time() - t0) / 60.0 >= args.max_fold_minutes:
            print(f"[FRCNN f{fold}] time budget {args.max_fold_minutes:.1f}m reached before ep {ep}; keeping best_ap50={best_ap:.4f}", flush=True)
            break
        model.train()
        losses = []
        ep_t0 = time.time()
        for imgs, tgts in dl_tr:
            imgs = [im.to(dev, non_blocking=True) for im in imgs]
            tgts = [{k: v.to(dev, non_blocking=True) for k, v in t.items()} for t in tgts]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                ld = model(imgs, tgts)
                loss = sum(ld.values())
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss.item()))

        ap50 = _eval_ap50(model, ds_va, dev)
        mean_l = float(np.mean(losses)) if losses else 0.0
        print(f"[FRCNN f{fold}] ep={ep}/{args.epochs} loss={mean_l:.4f} val_ap50={ap50:.4f} ({(time.time()-ep_t0)/60:.1f}m)", flush=True)
        if ap50 > best_ap:
            best_ap = ap50
            bad = 0
            torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, out_w)
        else:
            bad += 1
            if bad >= 4 and ep >= 8:
                print(f"[FRCNN f{fold}] early stop at ep {ep} best_ap50={best_ap:.4f}", flush=True)
                break
    print(f"[FRCNN f{fold}] DONE best_ap50={best_ap:.4f} total={(time.time()-t0)/60:.1f}m", flush=True)


def worker_infer_boxes(args) -> None:
    import torch

    t0 = time.time()
    repo_dir = Path(args.repo)
    data_root = Path(args.data_root)
    work_dir = Path(args.work_dir)
    sys.path.insert(0, str(repo_dir))
    os.environ["MAGFILO_ROOT"] = str(data_root)
    from filseg import data, detector

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = {}
    for k in range(data.N_FOLDS):
        m = _build_frcnn(IMGSZ, pretrained=False)
        sd = torch.load(work_dir / "weights" / f"frcnn_f{k}.pth", map_location="cpu")
        m.load_state_dict({kk: vv.float() if vv.is_floating_point() else vv for kk, vv in sd.items()})
        models[k] = m.to(dev).eval()

    tasks = []
    for k in range(data.N_FOLDS):
        for s in data.split(k)[1]:
            tasks.append(("oof", s, k, data.TRAIN_IMAGES / f"{s}.jpeg"))
    for s in data.test_stems():
        tasks.append(("test", s, None, data.TEST_IMAGES / f"{s}.jpeg"))

    if args.num_shards > 1:
        tasks = [t for t in tasks if int(hashlib.md5(f"{t[0]}_{t[1]}".encode()).hexdigest()[:8], 16) % args.num_shards == args.shard_id]

    boxes_dir = Path(args.boxes_dir)
    boxes_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for i, (sp, stem, fk, img_p) in enumerate(tasks, 1):
            dst = boxes_dir / sp / f"{stem}.npz"
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            g = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
            views = []
            ms = [models[fk]] if fk is not None else [models[k] for k in range(data.N_FOLDS)]
            for m in ms:
                for flip_mode in (0, 1, 2):  # 0=id, 1=hflip, 2=vflip
                    gi = g if flip_mode == 0 else np.ascontiguousarray(np.flip(g, 1 if flip_mode == 1 else 0))
                    t = torch.from_numpy(gi).to(dev).float().div(255.0).unsqueeze(0).expand(3, -1, -1)
                    with torch.autocast("cuda", dtype=torch.float16):
                        out = m([t])[0]
                    bs = out["boxes"].cpu().numpy()
                    cs = out["scores"].cpu().numpy()
                    keep = cs >= 0.05
                    bs, cs = bs[keep], cs[keep]
                    if flip_mode == 1 and len(bs):
                        bs[:, [0, 2]] = FULL - bs[:, [2, 0]]
                    elif flip_mode == 2 and len(bs):
                        bs[:, [1, 3]] = FULL - bs[:, [3, 1]]
                    views.append(np.c_[bs, cs] if len(bs) else np.zeros((0, 5), np.float32))
            fused = sorted(detector.wbf(views), key=lambda x: -x[1])[:60]
            b_arr = np.array([f[0] for f in fused], np.float32).reshape(-1, 4)
            c_arr = np.array([f[1] for f in fused], np.float32)
            keep = c_arr >= 0.05
            np.savez_compressed(dst, box=b_arr[keep], conf=c_arr[keep])
            if i % 60 == 0 or i == len(tasks):
                print(f"[frcnn_infer shard {args.shard_id}] {i}/{len(tasks)} ({(time.time()-t0)/60:.1f}m)", flush=True)


def worker_seg_stage2(args) -> None:
    import torch

    t0 = time.time()
    repo_dir = Path(args.repo)
    data_root = Path(args.data_root)
    work_dir = Path(args.work_dir)
    boxes_dir = Path(args.boxes_dir)
    sys.path.insert(0, str(repo_dir))
    os.environ["MAGFILO_ROOT"] = str(data_root)
    os.environ["FILSEG_WORK"] = str(work_dir)
    from filseg import data, enhance, postprocess, segmenter

    stem_to_fold = {s: k for k in range(data.N_FOLDS) for s in data.split(k)[1]}
    stem_paths = {p.stem: p for p in data.TRAIN_IMAGES.glob("*.jpeg")}
    stem_paths.update({p.stem: p for p in data.TEST_IMAGES.glob("*.jpeg")})

    items = []
    for sp in ("oof", "test"):
        for f in sorted((boxes_dir / sp).glob("*.npz")):
            fk = stem_to_fold[f.stem] if sp == "oof" else None
            items.append((sp, f.stem, fk, f))
    if args.num_shards > 1:
        items = [t for t in items if int(hashlib.md5(f"{t[0]}_{t[1]}".encode()).hexdigest()[:8], 16) % args.num_shards == args.shard_id]

    out_dir = Path(args.out_dir)
    if args.seg_family == "e4":
        dev = "cuda"
        models = {k: segmenter.load(data.WEIGHTS / f"seg_e4_f{k}.pth", dev) for k in range(data.N_FOLDS)}
        out_o, out_g = out_dir / "frcnn_e4", out_dir / "frcnn_e4_g1"
        for i, (sp, stem, fk, bf) in enumerate(items, 1):
            po, pg = out_o / sp / f"{stem}.npz", out_g / sp / f"{stem}.npz"
            if po.exists() and pg.exists():
                continue
            with np.load(bf) as bd:
                boxes, conf = bd["box"], bd["conf"]
            ws = [segmenter.crop_box(float(x1), float(y1), float(x2 - x1), float(y2 - y1), 0.2) for x1, y1, x2, y2 in boxes]
            if not ws:
                postprocess.save_cache(po, conf, ws, [])
                postprocess.save_cache(pg, conf, ws, [])
                continue
            planes = enhance.planes(stem, cache=True)
            segs = [models[fk]] if fk is not None else [models[k] for k in range(data.N_FOLDS)]
            probs = segmenter.predict(segs, planes, ws, batch=32, tta=True, device=dev)
            postprocess.save_cache(po, conf, ws, probs)
            ws_g, ch = segmenter.grow_windows(ws, probs, pad=0.2)
            probs_g = list(probs)
            if ch:
                np_g = segmenter.predict(segs, planes, [ws_g[c] for c in ch], batch=32, tta=True, device=dev)
                for ci, pa in zip(ch, np_g):
                    probs_g[ci] = pa
            postprocess.save_cache(pg, conf, ws_g, probs_g)
            if i % 80 == 0 or i == len(items):
                print(f"[frcnn seg e4 shard {args.shard_id}] {i}/{len(items)} ({(time.time()-t0)/60:.1f}m)", flush=True)
        return

    tpu_dir = Path(args.tpu_code)
    for k in list(sys.modules):
        if k == "filseg" or k.startswith("filseg."):
            del sys.modules[k]
    if str(repo_dir) in sys.path:
        sys.path.remove(str(repo_dir))
    sys.path.insert(0, str(tpu_dir / "tpu"))
    sys.path.insert(0, str(tpu_dir / "src"))
    import sam_crop
    import twostage as ts

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.seg_family == "sam2":
        model = sam_crop.SamCrop("sam2", 512).to(dev)
        model.load_state_dict(torch.load(args.seg_weights, map_location="cpu"))
        model.eval()
        pre, bs = "raw", 12
    else:
        cfg_p = Path(args.seg_weights).with_name("S1E_config.json")
        cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {"pre": "bth7+flat+ridge", "arch": "unetpp"}
        pre, bs = cfg["pre"], 32
        model = ts.seg_model(pre, cfg["arch"], cfg.get("encoder", "efficientnet-b4"))
        model.load_state_dict(torch.load(args.seg_weights, map_location="cpu"))
        model.to(dev).eval()

    items.sort(key=lambda t: (0 if (t[2] == 0 or t[0] == "test") else 1, t[0], t[1]))
    out_seg = out_dir / f"frcnn_{args.seg_family}"
    for i, (sp, stem, _, bf) in enumerate(items, 1):
        dst = out_seg / sp / f"{stem}.npz"
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with np.load(bf) as bd:
            boxes, conf = bd["box"], bd["conf"]
        ws = np.array([ts.crop_box(float(x1), float(y1), float(x2 - x1), float(y2 - y1), 0.2) for x1, y1, x2, y2 in boxes], np.int32).reshape(-1, 4)
        img = cv2.imread(str(stem_paths[stem]), cv2.IMREAD_GRAYSCALE)
        planes = ts.channels(img, pre)
        crops = [(ts.crop_planes(planes, b, 512) - 0.5) / 0.5 for b in ws]
        probs_u8 = []
        j, cur_bs = 0, bs
        while j < len(crops):
            chunk = crops[j : j + cur_bs]
            try:
                x = torch.from_numpy(np.stack(chunk).astype(np.float32)).to(dev)
                acc = 0
                with torch.no_grad():
                    for k in range(4):
                        xx = x.flip(3) if k in (1, 3) else x
                        xx = xx.flip(2) if k in (2, 3) else xx
                        with torch.autocast("cuda", dtype=torch.float16):
                            y = model(xx).float().sigmoid()
                        y = y.flip(2) if k in (2, 3) else y
                        acc = acc + (y.flip(3) if k in (1, 3) else y)
                for p, b in zip((acc / 4.0).cpu().numpy()[:, 0], ws[j : j + len(chunk)]):
                    probs_u8.append((cv2.resize(p, (max(1, int(b[2] - b[0])), max(1, int(b[3] - b[1])))) * 255.0 + 0.5).astype(np.uint8))
                j += len(chunk)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and cur_bs > 2:
                    torch.cuda.empty_cache()
                    cur_bs = max(2, cur_bs // 2)
                else:
                    raise
        flat = np.concatenate([p.ravel() for p in probs_u8]) if probs_u8 else np.zeros(0, np.uint8)
        np.savez_compressed(dst, conf=conf.astype(np.float32), box=ws, p=flat)
        if i % 80 == 0 or i == len(items):
            print(f"[frcnn seg {args.seg_family}] {i}/{len(items)} ({(time.time()-t0)/60:.1f}m)", flush=True)


def _tar_dir(src_dir: Path, tar_path: Path) -> float:
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(src_dir, arcname=src_dir.name)
    return tar_path.stat().st_size / 1e6


def orchestrate() -> None:
    t0 = time.time()
    print(
        subprocess.run(
            "pip install -q ultralytics==8.4.160 segmentation-models-pytorch==0.5.0 pycocotools scikit-image 'transformers==5.14.1' 2>&1 | tail -2",
            shell=True,
            capture_output=True,
            text=True,
        ).stdout,
        flush=True,
    )
    subprocess.run("nvidia-smi -L; nproc", shell=True)

    data_root = _find_comp_root()
    repo_dir = next(p for p in Path("/kaggle/input").rglob("run.py") if (p.parent / "filseg").is_dir()).parent
    tpu_dir = next(p for p in Path("/kaggle/input").rglob("sam_crop.py") if p.parent.name == "tpu").parents[1]
    a100_weights = next(Path("/kaggle/input").rglob("seg_e4_f0.pth")).parent
    s1e_weights = next(Path("/kaggle/input").rglob("S1E_best.pth"))
    sam2_weights = next(p for p in Path("/kaggle/input").rglob("best.pth") if p.parent.name == "SAM2")

    work_dir = Path("/tmp/work")
    planes_tmp = Path("/tmp/planes")
    boxes_dir = Path("/tmp/frcnn_boxes")
    out_root = Path("/tmp/out")
    weights_dir = work_dir / "weights"
    for d in (work_dir, planes_tmp, boxes_dir, out_root, weights_dir):
        d.mkdir(parents=True, exist_ok=True)
    planes_link = work_dir / "planes"
    if not planes_link.exists():
        planes_link.symlink_to(planes_tmp)

    for k in range(5):
        shutil.copy2(a100_weights / f"seg_e4_f{k}.pth", weights_dir / f"seg_e4_f{k}.pth")

    # Pre-warm torchvision FRCNN + HF SAM2 weights once before parallel workers start
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2; "
            "from transformers import Sam2Model; "
            "fasterrcnn_resnet50_fpn_v2(weights='DEFAULT', weights_backbone=None); "
            "Sam2Model.from_pretrained('facebook/sam2.1-hiera-base-plus'); print('Pretrained ready')",
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

    p_planes = subprocess.Popen(
        [py, str(repo_dir / "run.py"), "planes", "--procs", "2"],
        env=common_env,
        stdout=open("/tmp/log_planes.txt", "w"),
        stderr=subprocess.STDOUT,
    )

    err_box: list[BaseException] = []

    def _train_folds(gpu: int, folds: tuple[int, ...]):
        try:
            for fk in folds:
                rc = subprocess.call(
                    [
                        py, self_path, "--worker", "train_fold", "--fold", str(fk), "--epochs", "15",
                        "--repo", str(repo_dir), "--data-root", str(data_root), "--work-dir", str(work_dir),
                    ],
                    env=dict(common_env, CUDA_VISIBLE_DEVICES=str(gpu)),
                )
                assert rc == 0, f"FRCNN fold {fk} failed"
        except BaseException as exc:
            err_box.append(exc)

    th0 = threading.Thread(target=_train_folds, args=(0, (0, 2, 4)))
    th1 = threading.Thread(target=_train_folds, args=(1, (1, 3)))
    th0.start(); th1.start()
    th0.join(); th1.join()
    assert p_planes.wait() == 0, "planes failed"
    if err_box:
        raise err_box[0]

    print(f"=== 5-fold FRCNN trained in {(time.time()-t0)/60:.1f}m -> 2-GPU Box Infer ===", flush=True)
    ps_box = [
        subprocess.Popen(
            [
                py, self_path, "--worker", "infer_boxes", "--repo", str(repo_dir), "--data-root", str(data_root),
                "--work-dir", str(work_dir), "--boxes-dir", str(boxes_dir), "--shard-id", str(g), "--num-shards", "2",
            ],
            env=dict(common_env, CUDA_VISIBLE_DEVICES=str(g)),
        )
        for g in (0, 1)
    ]
    assert all(p.wait() == 0 for p in ps_box), "FRCNN box infer failed"

    ps_e4 = [
        subprocess.Popen(
            [
                py, self_path, "--worker", "seg_stage2", "--seg-family", "e4",
                "--repo", str(repo_dir), "--data-root", str(data_root), "--work-dir", str(work_dir),
                "--boxes-dir", str(boxes_dir), "--out-dir", str(out_root), "--shard-id", str(g), "--num-shards", "2",
            ],
            env=dict(common_env, CUDA_VISIBLE_DEVICES=str(g)),
        )
        for g in (0, 1)
    ]
    assert all(p.wait() == 0 for p in ps_e4), "FRCNN seg_e4 failed"

    p_sam2 = subprocess.Popen(
        [
            py, self_path, "--worker", "seg_stage2", "--seg-family", "sam2",
            "--repo", str(repo_dir), "--data-root", str(data_root), "--tpu-code", str(tpu_dir),
            "--work-dir", str(work_dir), "--boxes-dir", str(boxes_dir), "--out-dir", str(out_root),
            "--seg-weights", str(sam2_weights),
        ],
        env=dict(common_env, CUDA_VISIBLE_DEVICES="0", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"),
    )
    p_s1e = subprocess.Popen(
        [
            py, self_path, "--worker", "seg_stage2", "--seg-family", "s1e",
            "--repo", str(repo_dir), "--data-root", str(data_root), "--tpu-code", str(tpu_dir),
            "--work-dir", str(work_dir), "--boxes-dir", str(boxes_dir), "--out-dir", str(out_root),
            "--seg-weights", str(s1e_weights),
        ],
        env=dict(common_env, CUDA_VISIBLE_DEVICES="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"),
    )
    assert p_sam2.wait() == 0 and p_s1e.wait() == 0, "FRCNN SAM2 or S1E failed"

    fw_dir = Path("/tmp/frcnn_weights")
    fw_dir.mkdir(parents=True, exist_ok=True)
    for k in range(5):
        shutil.copy2(weights_dir / f"frcnn_f{k}.pth", fw_dir / f"frcnn_f{k}.pth")

    summary = {"minutes": round((time.time() - t0) / 60.0, 2), "archives": {}, "counts": {}}
    for name, sdir, tpath in (
        ("frcnn_boxes", boxes_dir, "/kaggle/working/exp008a_frcnn_boxes.tar.gz"),
        ("frcnn_e4", out_root / "frcnn_e4", "/kaggle/working/exp008a_frcnn_e4.tar.gz"),
        ("frcnn_e4_g1", out_root / "frcnn_e4_g1", "/kaggle/working/exp008a_frcnn_e4_g1.tar.gz"),
        ("frcnn_s1e", out_root / "frcnn_s1e", "/kaggle/working/exp008a_frcnn_s1e.tar.gz"),
        ("frcnn_sam2", out_root / "frcnn_sam2", "/kaggle/working/exp008a_frcnn_sam2.tar.gz"),
        ("frcnn_weights", fw_dir, "/kaggle/working/exp008a_frcnn_weights.tar.gz"),
    ):
        if sdir.exists() and any(sdir.rglob("*")):
            sz = _tar_dir(sdir, Path(tpath))
            summary["archives"][Path(tpath).name] = round(sz, 2)
            summary["counts"][name] = sum(1 for _ in sdir.rglob("*"))

    Path("/kaggle/working/exp008a_summary.json").write_text(json.dumps(summary, indent=2))
    print("EXP008A SUMMARY:", json.dumps(summary), flush=True)
    shutil.rmtree(work_dir, ignore_errors=True)
    shutil.rmtree(planes_tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default="orchestrate", choices=["orchestrate", "train_fold", "infer_boxes", "seg_stage2"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--max-fold-minutes", type=float, default=50.0)
    ap.add_argument("--repo", default="")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--tpu-code", default="")
    ap.add_argument("--work-dir", default="")
    ap.add_argument("--boxes-dir", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--seg-weights", default="")
    ap.add_argument("--seg-family", default="e4")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    if args.worker == "orchestrate":
        orchestrate()
    elif args.worker == "train_fold":
        worker_train_fold(args)
    elif args.worker == "infer_boxes":
        worker_infer_boxes(args)
    elif args.worker == "seg_stage2":
        worker_seg_stage2(args)


if __name__ == "__main__":
    main()
