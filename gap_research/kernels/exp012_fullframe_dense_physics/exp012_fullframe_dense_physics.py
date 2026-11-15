"""EXP-012 (Wave-3 GPU Kernel): Full-Frame Physics-Normalized Dense Ensemble (5-Fold OOF + Test).

Research Gap Addressed:
  ALL existing & queued models (ms, s, main, SAHI, F3, P6, P4a, FRCNN, EnhYOLO, SegFormer/ConvNeXt crops)
  rely on bounding-box proposals from Stage-1 detectors, inheriting Stage-1 FN misses (32% on faint <580px
  filaments) and box-edge truncation/merge ambiguities. Cross-paradigm evidence proves Full-Frame Dense
  segmentation has ~26% orthogonal disagreement with YOLO pipelines and yields huge gains on fusion.

Key Engineering Design:
  1. Subpixel Ray-Cast Solar Limb Fit (cx, cy, R) + 2D Robust CLV Radial Profile Removal (kills limb darkening
     without attenuating wide filaments) + Black Top-Hat bth7 + Local Flat101 + Normalized Radius rho (4 ch).
  2. Full-frame 1536x1536 (3/4 native 2048) training using soft annotator-mean (y_fg) and instance-separated
     centerline core (y_spine) weighted per-image by R_i annotator count (matches official PQ numerator/denom).
  3. Zero clDice (preserves dense probability calibration!). Loss = BCE + SoftDice on fg + BCE + SoftDice on spine.
  4. Two orthogonal 5-fold backbones (strictly OOF via folds.json, 4-view dihedral TTA):
       - Model A (`cnx_upp` key): smp.Unet("tu-convnext_small.fb_in22k", in_channels=4, classes=2)  (avoids UnetPlusPlus 0-ch skip bug)
       - Model B: smp.Unet("mit_b3", in_channels=4, classes=2)  (Mix-Transformer + U-Net dense skip)
  5. Exports compact 1024x1024 uint8 dense probability maps (p_fg, p_spine) + Spine-Watershed component
     proposals in standard .npz (conf, box, p) format for direct local fusion & R_i-Hurdle Antichain DP.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

FULL = 2048
TRAIN_SZ = 1536
OUT_SZ = 1024


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
    raise FileNotFoundError(f"Could not locate {slug} under /kaggle/input")


def _bind_filseg(repo_dir: Path, data_root: Path, work_dir: Path):
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    os.environ["MAGFILO_ROOT"] = str(data_root)
    os.environ["FILSEG_WORK"] = str(work_dir)
    from filseg import data, metric
    from filseg import postprocess

    data.ROOT = data_root
    data.TRAIN_JSON = data_root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    data.TRAIN_IMAGES = data_root / "train" / "train_images"
    data.TEST_IMAGES = data_root / "test" / "test_images"
    data.FOLDS = repo_dir / "folds" / "folds.json"
    data.WORK = work_dir
    data.WEIGHTS = work_dir / "weights"
    return data, metric, postprocess


def fit_solar_disk_raycast(img: np.ndarray) -> tuple[float, float, float]:
    """Subpixel ray-cast steepest-limb circle fit robust to off-center GONG frames."""
    h, w = img.shape
    p99 = float(np.percentile(img, 99))
    mask = img > max(18.0, 0.25 * p99)
    ys, xs = np.nonzero(mask)
    if len(xs) < 1000:
        return w / 2.0, h / 2.0, min(w, h) * 0.43
    cx0, cy0 = float(xs.mean()), float(ys.mean())
    r0 = float(np.sqrt(len(xs) / np.pi))

    thetas = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    rs = np.linspace(max(10.0, 0.85 * r0), min(w * 0.65, 1.15 * r0), 96)
    cos_t, sin_t = np.cos(thetas)[:, None], np.sin(thetas)[:, None]
    gx = np.clip(np.round(cx0 + cos_t * rs[None, :]).astype(np.int32), 0, w - 1)
    gy = np.clip(np.round(cy0 + sin_t * rs[None, :]).astype(np.int32), 0, h - 1)
    prof = img[gy, gx].astype(np.float32)
    grad = prof[:, :-2] - prof[:, 2:]  # positive when transitioning bright disk -> dark sky
    best_idx = np.argmax(grad, axis=1) + 1
    r_limb = rs[best_idx]

    med_r = float(np.median(r_limb))
    mad_r = 1.4826 * float(np.median(np.abs(r_limb - med_r))) + 1e-3
    inliers = np.abs(r_limb - med_r) <= max(6.0, 2.5 * mad_r)
    if int(inliers.sum()) < 120:
        return cx0, cy0, r0

    px = cx0 + r_limb[inliers] * np.cos(thetas[inliers])
    py = cy0 + r_limb[inliers] * np.sin(thetas[inliers])
    A = np.c_[2.0 * px, 2.0 * py, np.ones(len(px), dtype=np.float64)]
    b = px * px + py * py
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = float(sol[0]), float(sol[1])
        r = float(np.sqrt(max(1.0, sol[2] + cx * cx + cy * cy)))
        if abs(cx - cx0) < 40 and abs(cy - cy0) < 40 and abs(r - r0) < 40:
            return cx, cy, r
    except Exception:
        pass
    return cx0, cy0, r0


def physics_planes_1536(img: np.ndarray) -> np.ndarray:
    """2048x2048 uint8 gray -> (4, 1536, 1536) uint8 [flat_clv, bth7, clahe_clv, rho]."""
    cv2.setNumThreads(0)
    cx, cy, R = fit_solar_disk_raycast(img)
    yy, xx = np.mgrid[:FULL, :FULL].astype(np.float32)
    rho = np.hypot(xx - cx, yy - cy) / max(R, 1.0)
    disk = rho < 0.992

    x = img.astype(np.float32)
    # 2D Robust CLV radial profile (64 radial annuli over disk)
    n_bins = 64
    bin_edges = np.linspace(0.0, 0.995, n_bins + 1)
    # Subsample 4x for ultra-fast quantile median per bin
    rho_s = rho[::4, ::4].ravel()
    x_s = x[::4, ::4].ravel()
    b_idx = np.clip(np.searchsorted(bin_edges, rho_s, side="right") - 1, 0, n_bins - 1)
    med_prof = np.zeros(n_bins, dtype=np.float32)
    global_med = float(np.median(x_s[rho_s < 0.98])) if (rho_s < 0.98).any() else float(np.median(x_s))
    for k in range(n_bins):
        sel = x_s[(b_idx == k) & (rho_s < 0.995)]
        med_prof[k] = float(np.median(sel)) if len(sel) >= 20 else global_med
    # Mild 3-tap smoothing respecting center-to-limb profile
    med_sm = np.convolve(np.pad(med_prof, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    clv_map = np.interp(np.clip(rho, 0.0, 0.995), bin_centers, med_sm).astype(np.float32)
    flat_clv = np.where(disk, x - clv_map, 0.0)

    # Remove solar CLV limb darkening BEFORE CLAHE(2.0, (8,8)) so limb filaments preserve full SNR
    ratio = np.where(disk, x / np.maximum(clv_map, 22.0), 0.0)
    clv_norm_u8 = np.where(disk, np.clip((ratio - 0.45) / (1.35 - 0.45) * 255.0, 0.0, 255.0), 0.0).astype(np.uint8)
    clahe_clv_u8 = np.where(disk, cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(clv_norm_u8), 0).astype(np.uint8)

    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bth7 = -cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, k7).astype(np.float32)

    chs = []
    for arr in (flat_clv, bth7):
        v = arr[disk]
        med = float(np.median(v)) if len(v) else 0.0
        mad = 1.4826 * float(np.median(np.abs(v - med))) + 1e-6 if len(v) else 1.0
        z = np.where(disk, np.clip((arr - med) / mad, -6.0, 6.0), 0.0)
        u8 = np.clip((z + 6.0) * (255.0 / 12.0), 0.0, 255.0).astype(np.uint8)
        chs.append(cv2.resize(u8, (TRAIN_SZ, TRAIN_SZ), interpolation=cv2.INTER_AREA))

    chs.append(cv2.resize(clahe_clv_u8, (TRAIN_SZ, TRAIN_SZ), interpolation=cv2.INTER_AREA))
    rho_u8 = np.clip((rho / 1.05) * 255.0, 0.0, 255.0).astype(np.uint8)
    chs.append(cv2.resize(rho_u8, (TRAIN_SZ, TRAIN_SZ), interpolation=cv2.INTER_AREA))
    return np.stack(chs, axis=0)


def _prepare_one_frame(task: tuple) -> str:
    stem, img_path, x_out, y_out, rec_anns_list = task
    if not x_out.exists():
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(img_path)
        x_arr = physics_planes_1536(img)
        x_out.parent.mkdir(parents=True, exist_ok=True)
        tmp = x_out.with_suffix(".tmp.npy")
        np.save(tmp, x_arr)
        tmp.replace(x_out)

    if rec_anns_list is not None and not y_out.exists():
        from pycocotools import mask as mu

        n_rec = max(1, len(rec_anns_list))
        fg_sum = np.zeros((TRAIN_SZ, TRAIN_SZ), dtype=np.float32)
        sp_sum = np.zeros((TRAIN_SZ, TRAIN_SZ), dtype=np.float32)
        for ann_list in rec_anns_list:
            rec_fg = np.zeros((TRAIN_SZ, TRAIN_SZ), dtype=np.uint8)
            rec_sp = np.zeros((TRAIN_SZ, TRAIN_SZ), dtype=np.uint8)
            for a in ann_list:
                seg = a.get("segmentation")
                if not seg:
                    continue
                if isinstance(seg, list):
                    polys = [p for p in seg if isinstance(p, (list, tuple)) and len(p) >= 6]
                    if not polys:
                        continue
                    rle = mu.merge(mu.frPyObjects(polys, FULL, FULL))
                elif isinstance(seg, dict):
                    rle = mu.frPyObjects(seg, FULL, FULL) if isinstance(seg.get("counts"), list) else seg
                else:
                    continue
                m = mu.decode(rle).astype(np.float32)
                ms = (cv2.resize(m, (TRAIN_SZ, TRAIN_SZ), interpolation=cv2.INTER_AREA) >= 0.5).astype(np.uint8)
                if not ms.any():
                    continue
                rec_fg |= ms
                dt = cv2.distanceTransform(ms, cv2.DIST_L2, 3)
                dmax = float(dt.max())
                # Keep core ridge >= 0.45*max inside each instance so touching filaments stay separated
                core = (dt >= max(0.8, 0.45 * dmax)).astype(np.uint8)
                if not core.any():
                    core = ms
                rec_sp |= core
            fg_sum += rec_fg.astype(np.float32)
            sp_sum += rec_sp.astype(np.float32)
        y_fg = np.clip(np.round((fg_sum / n_rec) * 255.0), 0, 255).astype(np.uint8)
        y_sp = np.clip(np.round((sp_sum / n_rec) * 255.0), 0, 255).astype(np.uint8)
        y_out.parent.mkdir(parents=True, exist_ok=True)
        tmp = y_out.with_suffix(".tmp.npy")
        np.save(tmp, np.stack([y_fg, y_sp], axis=0))
        tmp.replace(y_out)
    return stem


def worker_prepare(args) -> None:
    data, _, _ = _bind_filseg(args.repo, args.data_root, args.work_dir)
    recs, anns = data.coco_index()
    cache = args.cache_dir
    tasks = []
    for s in sorted(data.folds()):
        rec_anns = [anns.get(rid, []) for rid in recs[s]]
        tasks.append((s, data.TRAIN_IMAGES / f"{s}.jpeg", cache / "x" / f"{s}.npy", cache / "y" / f"{s}.npy", rec_anns))
    for s in data.test_stems():
        tasks.append((s, data.TEST_IMAGES / f"{s}.jpeg", cache / "x" / f"{s}.npy", Path("/dev/null"), None))

    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.procs) as ex:
        list(ex.map(_prepare_one_frame, tasks, chunksize=8))
    print(f"[EXP-012 prepare] Prepared {len(tasks)} frames @ {TRAIN_SZ}px in {(time.time()-t0)/60:.1f}m", flush=True)


def _patch_grouped_convs(m):
    """Bypass cuDNN 9.27 Graph API bug on depthwise convs (groups > 1) and speed up T4 grouped convs."""
    import torch
    import torch.nn as nn

    for mod in m.modules():
        if isinstance(mod, nn.Conv2d) and mod.groups > 1:
            orig = mod.forward

            def _wrap(x, _orig=orig):
                with torch.backends.cudnn.flags(enabled=False):
                    return _orig(x)

            mod.forward = _wrap
    return m


def _build_dense_model(arch: str, pretrained: bool = True):
    import segmentation_models_pytorch as smp

    if arch == "cnx_upp":
        m = smp.Unet(
            encoder_name="tu-convnext_small.fb_in22k",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=4,
            classes=2,
        )
        return _patch_grouped_convs(m)
    if arch == "mit_b3":
        m = smp.Unet(
            encoder_name="mit_b3",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=4,
            classes=2,
        )
        return _patch_grouped_convs(m)
    if arch == "res50":
        m = smp.UnetPlusPlus(
            encoder_name="resnet50",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=4,
            classes=2,
        )
        return _patch_grouped_convs(m)
    raise ValueError(arch)


class _FullFrameDataset:
    def __init__(self, stems: list[str], cache_dir: Path, weights_by_stem: dict[str, float], train: bool, seed: int = 0):
        self.stems = stems
        self.cache_dir = cache_dir
        self.weights = weights_by_stem
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        import torch

        s = self.stems[idx]
        x = np.load(self.cache_dir / "x" / f"{s}.npy").astype(np.float32) / 255.0
        y = np.load(self.cache_dir / "y" / f"{s}.npy").astype(np.float32) / 255.0
        if self.train:
            k = int(self.rng.integers(4))
            if k:
                x = np.rot90(x, k, axes=(1, 2))
                y = np.rot90(y, k, axes=(1, 2))
            if self.rng.random() < 0.5:
                x = x[:, :, ::-1]
                y = y[:, :, ::-1]
            # Mild brightness/gain jitter only on physics contrast channels 0..2, never corrupt ch3 (rho)
            gain = float(self.rng.uniform(0.94, 1.06))
            bias = float(self.rng.uniform(-0.03, 0.03))
            x[:3] = np.clip((x[:3] - 0.5) * gain + 0.5 + bias, 0.0, 1.0)
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        return torch.from_numpy(x), torch.from_numpy(y), float(self.weights[s])


def _dense_loss(logits, targets, sample_w):
    import torch
    import torch.nn.functional as F

    # logits, targets: (B, 2, H, W), sample_w: (B,) normalized by R_i
    sw = sample_w.view(-1, 1, 1)
    sw = sw / (sw.mean() + 1e-6)

    l_fg, l_sp = logits[:, 0].float(), logits[:, 1].float()
    t_fg, t_sp = targets[:, 0].float(), targets[:, 1].float()

    bce_fg = (F.binary_cross_entropy_with_logits(l_fg, t_fg, reduction="none") * sw[:, :, :]).mean()
    bce_sp = (F.binary_cross_entropy_with_logits(l_sp, t_sp, reduction="none") * sw[:, :, :]).mean()

    p_fg = l_fg.sigmoid()
    p_sp = l_sp.sigmoid()
    # Soft Dice weighted per image by R_i against soft consensus target (no clDice!)
    inter_fg = (p_fg * t_fg * sw).sum(dim=(1, 2))
    den_fg = ((p_fg + t_fg) * sw).sum(dim=(1, 2)) + 1.0
    dice_fg = 1.0 - (2.0 * inter_fg + 1.0).sum() / den_fg.sum()

    inter_sp = (p_sp * t_sp * sw).sum(dim=(1, 2))
    den_sp = ((p_sp + t_sp) * sw).sum(dim=(1, 2)) + 1.0
    dice_sp = 1.0 - (2.0 * inter_sp + 1.0).sum() / den_sp.sum()

    return (0.65 * bce_fg + 0.35 * dice_fg) + 0.30 * (0.50 * bce_sp + 0.50 * dice_sp)


def worker_train(args) -> None:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    t0 = time.time()
    torch.backends.cudnn.benchmark = False
    data, _, _ = _bind_filseg(args.repo, args.data_root, args.work_dir)
    recs, _ = data.coco_index()
    weights_by_stem = {s: float(len(rids)) for s, rids in recs.items()}

    tr, va = data.split(args.fold)
    ds_tr = _FullFrameDataset(tr, args.cache_dir, weights_by_stem, train=True, seed=args.fold * 10 + (0 if args.arch == "cnx_upp" else 1))
    ds_va = _FullFrameDataset(va, args.cache_dir, weights_by_stem, train=False, seed=0)

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 16.0
    if vram_gb >= 38:
        b_candidates, train_sz = (8, 4, 2), TRAIN_SZ
    elif args.arch == "mit_b3":
        b_candidates, train_sz = (1,), 1024
    else:
        b_candidates, train_sz = (2, 1), 1280
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and (torch.cuda.get_device_capability(0)[0] >= 8) and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_scaler = amp_dtype == torch.float16

    for b_try in b_candidates:
        grad_accum = 2 if (vram_gb < 38 and b_try == 1) else 1
        try:
            dl_tr = DataLoader(ds_tr, batch_size=b_try, shuffle=True, num_workers=4, drop_last=(b_try > 1), pin_memory=True)
            dl_va = DataLoader(ds_va, batch_size=b_try, shuffle=False, num_workers=2, pin_memory=True)
            model = _build_dense_model(args.arch, pretrained=True).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
            opt_steps_per_ep = max(1, math.ceil(len(dl_tr) / grad_accum))
            total_steps = args.epochs * opt_steps_per_ep
            warm_steps = max(1, 2 * opt_steps_per_ep)

            def _lr_fn(step: int) -> float:
                if step < warm_steps:
                    return (step + 1) / warm_steps
                prog = (step - warm_steps) / max(1, total_steps - warm_steps)
                return 0.03 + 0.97 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

            sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_fn)
            scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

            out_w = args.work_dir / "weights" / f"dense_{args.arch}_f{args.fold}.pth"
            done_p = out_w.with_suffix(".done.json")
            out_w.parent.mkdir(parents=True, exist_ok=True)
            if out_w.exists() and done_p.exists():
                print(f"[{args.arch} f{args.fold}] already completed -> skipping", flush=True)
                return

            best_dice = -1.0
            best_ep = 0
            no_imp = 0
            patience = max(1, int(args.patience))
            min_delta = 1e-4

            for ep in range(1, args.epochs + 1):
                model.train()
                losses = []
                ep_t0 = time.time()
                opt.zero_grad(set_to_none=True)
                n_batches = len(dl_tr)
                for step_i, (x, y, sw) in enumerate(dl_tr, 1):
                    x = x.to(dev, non_blocking=True)
                    y = y.to(dev, non_blocking=True)
                    if x.shape[-1] != train_sz:
                        x = F.interpolate(x, size=(train_sz, train_sz), mode="bilinear", align_corners=False)
                        y = F.interpolate(y, size=(train_sz, train_sz), mode="bilinear", align_corners=False)
                    sw = sw.to(dev, non_blocking=True).float()
                    with torch.autocast("cuda", dtype=amp_dtype):
                        loss = _dense_loss(model(x), y, sw)
                    loss_scaled = loss / grad_accum
                    if use_scaler:
                        scaler.scale(loss_scaled).backward()
                    else:
                        loss_scaled.backward()
                    if (step_i % grad_accum == 0) or (step_i == n_batches):
                        if use_scaler:
                            scaler.unscale_(opt)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                            scaler.step(opt)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                            opt.step()
                        opt.zero_grad(set_to_none=True)
                        sched.step()
                    losses.append(float(loss.item()))

                model.eval()
                inter, union = 0.0, 0.0
                with torch.no_grad():
                    for x, y, sw in dl_va:
                        x = x.to(dev, non_blocking=True)
                        y = y.to(dev, non_blocking=True)
                        if x.shape[-1] != train_sz:
                            x = F.interpolate(x, size=(train_sz, train_sz), mode="bilinear", align_corners=False)
                            y = F.interpolate(y, size=(train_sz, train_sz), mode="bilinear", align_corners=False)
                        y_bin = (y[:, 0] > 0.30).float()
                        sw_d = sw.to(dev, non_blocking=True).float().view(-1, 1, 1)
                        with torch.autocast("cuda", dtype=amp_dtype):
                            pr = (model(x)[:, 0].float().sigmoid() > 0.42).float()
                        inter += float((pr * y_bin * sw_d).sum().item())
                        union += float(((pr + y_bin) * sw_d).sum().item())
                dice = (2.0 * inter + 1e-6) / (union + 1e-6)
                if dice > best_dice + min_delta:
                    best_dice = dice
                    best_ep = ep
                    no_imp = 0
                    torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, out_w)
                else:
                    if dice > best_dice:
                        best_dice = dice
                        best_ep = ep
                        torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, out_w)
                    no_imp += 1
                print(
                    f"[{args.arch} f{args.fold} sz={train_sz} bs={b_try}x{grad_accum}] ep={ep}/{args.epochs} loss={np.mean(losses):.4f} "
                    f"w_dice={dice:.4f} best={best_dice:.4f} no_imp={no_imp}/{patience} ({(time.time()-ep_t0):.1f}s)",
                    flush=True,
                )
                if no_imp >= patience:
                    print(f"[{args.arch} f{args.fold}] early-stop at ep={ep} (best_ep={best_ep} best_w_dice={best_dice:.4f})", flush=True)
                    break

            done_p.write_text(json.dumps({"arch": args.arch, "fold": args.fold, "train_sz": train_sz, "best_dice": best_dice, "best_ep": best_ep, "epochs_run": ep}))
            print(f"[{args.arch} f{args.fold}] DONE best_w_dice={best_dice:.4f} in {(time.time()-t0)/60:.1f}m", flush=True)
            return
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and b_try > 1:
                torch.cuda.empty_cache()
                print(f"[{args.arch} f{args.fold}] OOM bs={b_try}, retrying smaller bs", flush=True)
                continue
            raise


def _predict_4tta(model, x_np: np.ndarray, dev, amp_dtype) -> np.ndarray:
    import torch

    acc = None
    for k in range(4):
        if k == 0:
            v = x_np
        elif k == 1:
            v = np.ascontiguousarray(x_np[:, :, ::-1])
        elif k == 2:
            v = np.ascontiguousarray(x_np[:, ::-1, :])
        else:
            v = np.ascontiguousarray(np.rot90(x_np, 1, axes=(1, 2)))
        xt = torch.from_numpy(v[None]).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype):
            p = model(xt).float().sigmoid()[0].cpu().numpy()
        if k == 1:
            p = p[:, :, ::-1]
        elif k == 2:
            p = p[:, ::-1, :]
        elif k == 3:
            p = np.rot90(p, 3, axes=(1, 2))
        acc = p if acc is None else (acc + p)
    return acc * 0.25


def _extract_watershed_proposals(p_fg_1024: np.ndarray, p_sp_1024: np.ndarray, seed_thr: float = 0.45, fg_thr: float = 0.34, min_area_2048: int = 80):
    """Split dense foreground components cleanly at spine peaks using watershed, emit 2048-res (conf, box, crop_prob_u8)."""
    from skimage.segmentation import watershed

    p_fg = cv2.resize(p_fg_1024, (FULL, FULL), interpolation=cv2.INTER_LINEAR)
    p_sp = cv2.resize(p_sp_1024, (FULL, FULL), interpolation=cv2.INTER_LINEAR)
    fg_mask = p_fg > fg_thr
    if not fg_mask.any():
        return [], [], []

    n_cc, cc_lab = cv2.connectedComponents(fg_mask.astype(np.uint8), connectivity=8)
    confs, boxes, probs = [], [], []
    for cid in range(1, n_cc):
        comp = cc_lab == cid
        area = int(comp.sum())
        if area < min_area_2048:
            continue
        ys, xs = np.nonzero(comp)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        sub_fg = p_fg[y1:y2, x1:x2]
        sub_sp = p_sp[y1:y2, x1:x2]
        sub_mask = comp[y1:y2, x1:x2]

        seeds = sub_mask & (sub_sp > seed_thr)
        n_s, s_lab = cv2.connectedComponents(seeds.astype(np.uint8), connectivity=8)
        # Filter tiny spurious spine speckles < 15px before watershed split
        valid_sids = [sid for sid in range(1, n_s) if int((s_lab == sid).sum()) >= 15]
        if len(valid_sids) >= 2:
            clean_markers = np.zeros_like(s_lab)
            for new_id, sid in enumerate(valid_sids, 1):
                clean_markers[s_lab == sid] = new_id
            ws = watershed(-sub_fg, clean_markers, mask=sub_mask)
            parts = [ws == nid for nid in range(1, len(valid_sids) + 1)]
        else:
            parts = [sub_mask]

        for pm in parts:
            pa = int(pm.sum())
            if pa < min_area_2048:
                continue
            pys, pxs = np.nonzero(pm)
            bx1, by1 = x1 + int(pxs.min()), y1 + int(pys.min())
            bx2, by2 = x1 + int(pxs.max()) + 1, y1 + int(pys.max()) + 1
            crop_p = np.where(pm[int(pys.min()):int(pys.max())+1, int(pxs.min()):int(pxs.max())+1], p_fg[by1:by2, bx1:bx2], 0.0)
            # Confidence calibrated to mean top-50% dense probability * size factor
            vals = crop_p[pm[int(pys.min()):int(pys.max())+1, int(pxs.min()):int(pxs.max())+1]]
            c = float(0.65 * vals.mean() + 0.35 * np.percentile(vals, 85))
            confs.append(c)
            boxes.append((bx1, by1, bx2, by2))
            probs.append(np.clip(crop_p * 255.0 + 0.5, 0, 255).astype(np.uint8))
    return confs, boxes, probs


def worker_infer(args) -> None:
    import torch

    t0 = time.time()
    torch.backends.cudnn.benchmark = False
    data, metric, postprocess = _bind_filseg(args.repo, args.data_root, args.work_dir)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 16.0
    use_bf16 = torch.cuda.is_available() and (torch.cuda.get_device_capability(0)[0] >= 8) and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    def _infer_sz(arch_name: str) -> int:
        if vram_gb >= 38:
            return TRAIN_SZ
        return 1024 if arch_name == "mit_b3" else 1280

    def _pred_to_1024(model, x_np: np.ndarray, arch_name: str) -> np.ndarray:
        sz = _infer_sz(arch_name)
        x_in = x_np if sz == x_np.shape[-1] else np.stack([cv2.resize(c, (sz, sz), interpolation=cv2.INTER_AREA) for c in x_np], axis=0)
        p = _predict_4tta(model, x_in, dev, amp_dtype)
        if p.shape[-1] == OUT_SZ:
            return p
        interp = cv2.INTER_AREA if p.shape[-1] >= OUT_SZ else cv2.INTER_LINEAR
        return np.stack([cv2.resize(p[0], (OUT_SZ, OUT_SZ), interpolation=interp), cv2.resize(p[1], (OUT_SZ, OUT_SZ), interpolation=interp)], axis=0)

    w_dir = args.work_dir / "weights"
    models: dict[tuple[str, int], torch.nn.Module] = {}
    for arch in args.arches:
        for k in range(5):
            wp = w_dir / f"dense_{arch}_f{k}.pth"
            if not wp.exists():
                raise FileNotFoundError(wp)
            m = _build_dense_model(arch, pretrained=False)
            sd = torch.load(wp, map_location="cpu")
            m.load_state_dict({kk: vv.float() if vv.is_floating_point() else vv for kk, vv in sd.items()})
            models[(arch, k)] = m.to(dev).eval()

    folds = data.folds()
    all_oof = sorted(folds)
    test_stems = data.test_stems()

    out_probs = args.out_dir / "dense_probs"
    out_cc = args.out_dir / "cc_proposals"
    (out_probs / "oof").mkdir(parents=True, exist_ok=True)
    (out_probs / "test").mkdir(parents=True, exist_ok=True)

    recs, anns = data.coco_index()
    gt_records = {s: [[data.ann_rle(a) for a in anns.get(rid, [])] for rid in recs[s]] for s in all_oof}

    rows_solo = []
    for sp, stems in (("oof", all_oof), ("test", test_stems)):
        for i, s in enumerate(stems):
            x = np.load(args.cache_dir / "x" / f"{s}.npy").astype(np.float32) / 255.0
            if sp == "oof":
                fk = folds[s]
                preds = [_pred_to_1024(models[(arch, fk)], x, arch) for arch in args.arches]
            else:
                preds = [_pred_to_1024(models[(arch, k)], x, arch) for arch in args.arches for k in range(5)]
            p1024 = np.mean(preds, axis=0)  # (2, 1024, 1024)
            p_u8 = np.clip(p1024 * 255.0 + 0.5, 0, 255).astype(np.uint8)
            np.savez_compressed(out_probs / sp / f"{s}.npz", p=p_u8)

            confs, boxes, cprobs = _extract_watershed_proposals(p1024[0], p1024[1])
            postprocess.save_cache(out_cc / sp / f"{s}.npz", confs, boxes, cprobs)

            if sp == "oof":
                inst = sorted(zip(confs, boxes, cprobs), key=lambda z: -z[0])
                painted = postprocess.paint(inst, thr=0.45, conf=0.40, drop=0.80, min_area=100)
                rows_solo.append(metric.counts(postprocess.to_rles(painted), gt_records[s]))
            if (i + 1) % 100 == 0 or i + 1 == len(stems):
                print(f"[EXP-012 infer {sp}] {i+1}/{len(stems)} ({(time.time()-t0)/60:.1f}m)", flush=True)

    solo_sum = metric.summarize(rows_solo)
    print(f"\n[EXP-012 SOLO 707 OOF] PQ={solo_sum['pq']:.5f} SQ={solo_sum['sq']:.5f} RQ={solo_sum['rq']:.5f} TP={solo_sum['tp']} FP={solo_sum['fp']} FN={solo_sum['fn']}", flush=True)
    (args.out_dir / "exp012_solo_oof.json").write_text(json.dumps(solo_sum, indent=2))


def _pack(src_dir: Path, dst_tgz: Path) -> None:
    tmp = dst_tgz.with_suffix(".tmp.gz")
    with tarfile.open(tmp, "w:gz", compresslevel=1) as tf:
        tf.add(src_dir, arcname=src_dir.name)
    tmp.replace(dst_tgz)
    print(f"Packed {dst_tgz.name}: {dst_tgz.stat().st_size / 1e6:.1f} MB", flush=True)


def run_driver(arches_req: list[str] | None = None, epochs: int = 36, patience: int = 5) -> None:
    import torch

    t0 = time.time()
    data_root = _find_comp_root()
    repo_dir = _find_input_dir("filseg-a100-code")
    work_dir = Path("/tmp/exp012_work")
    cache_dir = Path("/tmp/exp012_cache")
    out_dir = Path("/tmp/exp012_out")
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    arches = list(arches_req) if arches_req else ["cnx_upp", "mit_b3"]
    if "mit_b3" in arches:
        try:
            _build_dense_model("mit_b3", pretrained=True)
        except Exception as exc:
            print(f"[warn] mit_b3 unavailable ({exc}), falling back to res50", flush=True)
            arches = ["res50" if a == "mit_b3" else a for a in arches]

    py = sys.executable
    self_path = str(Path(__file__).resolve())
    common = ["--repo", str(repo_dir), "--data-root", str(data_root), "--work-dir", str(work_dir), "--cache-dir", str(cache_dir), "--out-dir", str(out_dir)]

    rc = subprocess.call([py, self_path, "--worker", "prepare", "--procs", str(min(8, os.cpu_count() or 4))] + common)
    assert rc == 0, "EXP-012 prepare failed"

    for arch in arches:
        for fold in range(5):
            rc = subprocess.call([py, self_path, "--worker", "train", "--arch", arch, "--fold", str(fold), "--epochs", str(epochs), "--patience", str(patience)] + common)
            assert rc == 0, f"EXP-012 train {arch} fold {fold} failed"

    rc = subprocess.call([py, self_path, "--worker", "infer", "--arches", *arches] + common)
    assert rc == 0, "EXP-012 infer failed"

    Path("/kaggle/working").mkdir(parents=True, exist_ok=True)
    _pack(out_dir / "dense_probs", Path("/kaggle/working/exp012_dense_probs.tar.gz"))
    _pack(out_dir / "cc_proposals", Path("/kaggle/working/exp012_cc_proposals.tar.gz"))
    if (out_dir / "exp012_solo_oof.json").exists():
        shutil.copy2(out_dir / "exp012_solo_oof.json", Path("/kaggle/working/exp012_summary.json"))

    print(f"EXP-012 COMPLETE in {(time.time() - t0)/60:.1f}m", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["prepare", "train", "infer"])
    ap.add_argument("--repo", type=Path)
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--work-dir", type=Path)
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--arch", default="cnx_upp")
    ap.add_argument("--arches", nargs="+", default=["cnx_upp", "mit_b3"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=36)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()

    if args.worker is None:
        run_driver(args.arches, args.epochs, args.patience)
    elif args.worker == "prepare":
        worker_prepare(args)
    elif args.worker == "train":
        worker_train(args)
    elif args.worker == "infer":
        worker_infer(args)


if __name__ == "__main__":
    main()
