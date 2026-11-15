"""Strictly leak-free SolarRestormer DL Image Quality Enhancer + Train-Only Filament Copy-Paste.

Guarantees zero validation/test leakage:
1. Self-supervised Restormer (MDTA + GDFN) trains ONLY on sharp frames inside `folds != fold`
   with synthetic PSF seeing blur + limb haze + sensor noise -> restores crisp dark filaments
   on out-of-fold validation/test frames without ever fitting on `va` or test pixels/labels.
2. Object-aware Copy-Paste augments ONLY `tr` (`folds != fold`), pasting small (<950 px) real
   filaments onto clean quiet-Sun disk regions of `tr` photos; `va` stays 100% original.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FULL = 2048


def disk_geometry(im_u8: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    h, w = im_u8.shape
    q99 = float(np.percentile(im_u8, 99))
    ys, xs = np.nonzero(im_u8 > 0.25 * q99)
    if len(xs) > 100:
        cx, cy = float(xs.mean()), float(ys.mean())
        rad = float(0.99 * np.sqrt(len(xs) / np.pi))
    else:
        cx, cy, rad = 0.5 * w, 0.5 * h, 0.44 * w
    yy, xx = np.ogrid[:h, :w]
    disk = ((xx - cx) ** 2 + (yy - cy) ** 2) < (rad * rad)
    return cx, cy, rad, disk


def physics_3ch_from_gray(im_u8: np.ndarray) -> np.ndarray:
    """Single-image deterministic 3-ch physical normalized uint8 BGR [raw_clahe, flat_z, bth7_z]."""
    _, _, _, disk = disk_geometry(im_u8)
    xf = im_u8.astype(np.float32)

    sm = cv2.resize(xf, (256, 256), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.GaussianBlur(sm, (0, 0), 6.25), (im_u8.shape[1], im_u8.shape[0]), interpolation=cv2.INTER_LINEAR)
    flat = xf - bg

    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bth = -cv2.morphologyEx(im_u8, cv2.MORPH_BLACKHAT, k7).astype(np.float32)

    def _to_u8(arr: np.ndarray, lo: float = -6.0, hi: float = 5.0) -> np.ndarray:
        vals = arr[disk][::16] if disk.any() else arr.ravel()[::16]
        med = float(np.median(vals))
        mad = 1.4826 * float(np.median(np.abs(vals - med))) + 1e-6
        z = np.where(disk, (arr - med) / mad, 0.0)
        return np.clip((z - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)

    cx, cy, rad, _ = disk_geometry(im_u8)
    ch0 = radial_clv_clahe(im_u8, cx, cy, rad, disk)
    ch1 = _to_u8(flat, -6.0, 5.0)
    ch2 = _to_u8(bth, -6.0, 5.0)
    return np.stack([ch0, ch1, ch2], axis=-1)


def radial_clv_clahe(im_u8: np.ndarray, cx: float, cy: float, rad: float, disk: np.ndarray) -> np.ndarray:
    """Remove solar Center-to-Limb Variation (CLV) BEFORE applying CLAHE(2.0, (8,8))."""
    h, w = im_u8.shape[:2]
    scale = 4 if min(h, w) >= 1024 else 1
    sh, sw = max(64, h // scale), max(64, w // scale)
    im_s = cv2.resize(im_u8, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
    disk_s = cv2.resize(disk.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    rn_s = np.sqrt((xx - (cx / scale)) ** 2 + (yy - (cy / scale)) ** 2) / max(1.0, float(rad) / scale)

    n_bins = 64
    edges = np.linspace(0.0, 1.02, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    prof = np.full(n_bins, np.nan, dtype=np.float32)
    for bi in range(n_bins):
        sel = disk_s & (rn_s >= edges[bi]) & (rn_s < edges[bi + 1])
        vals = im_s[sel]
        if vals.size >= 16:
            prof[bi] = float(np.percentile(vals, 65.0))
    valid = np.isfinite(prof)
    if valid.any():
        curve = np.interp(centers, centers[valid], prof[valid]).astype(np.float32)
        if curve.size >= 5:
            ker = np.ones(5, dtype=np.float32) / 5.0
            curve = np.convolve(np.pad(curve, 2, mode="edge"), ker, mode="valid")
        clv_s = np.interp(np.clip(rn_s, 0.0, 1.02), centers, curve).astype(np.float32)
        clv_map = cv2.resize(clv_s, (w, h), interpolation=cv2.INTER_LINEAR)
        ratio = np.where(disk, im_u8.astype(np.float32) / np.maximum(clv_map, 22.0), 0.0)
        clv_u8 = np.where(disk, np.clip((ratio - 0.45) / (1.35 - 0.45) * 255.0, 0.0, 255.0), 0.0).astype(np.uint8)
    else:
        clv_u8 = np.where(disk, im_u8, 0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(clv_u8)
    return np.where(disk, clahe, 0).astype(np.uint8)


def filter_legal_gong_paths(gong_paths: list, test_stems: list[str], va_stems: list[str], min_gap_sec: int = 3600) -> list:
    """Strictly filter unlabeled GONG frames: exclude +-1h of test timestamps, all test dates, and current fold va dates."""
    from datetime import datetime

    def _parse_ts(stem: str):
        for i in range(0, max(1, len(stem) - 13)):
            sub = stem[i : i + 14]
            if sub.isdigit():
                try:
                    return datetime.strptime(sub, "%Y%m%d%H%M%S")
                except ValueError:
                    pass
        return None

    test_ts = [t for t in (_parse_ts(str(s)) for s in test_stems) if t is not None]
    test_dates = {str(s)[:8] for s in test_stems if len(str(s)) >= 8 and str(s)[:8].isdigit()}
    va_dates = {str(s)[:8] for s in va_stems if len(str(s)) >= 8 and str(s)[:8].isdigit()}
    forbidden_dates = test_dates | va_dates

    kept = []
    for p in gong_paths:
        stem = getattr(p, "stem", str(p))
        ts = _parse_ts(stem)
        if ts is None:
            continue
        if ts.strftime("%Y%m%d") in forbidden_dates:
            continue
        if any(abs((ts - tt).total_seconds()) <= min_gap_sec for tt in test_ts):
            continue
        kept.append(p)
    return sorted(kept)


# ---------------------------------------------------------------------------
# Compact Restormer (MDTA Transposed Channel Attention + GDFN)
# ---------------------------------------------------------------------------

class MDTA(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.temp = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=False)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv_in = self.qkv(x)
        with torch.backends.cudnn.flags(enabled=False):
            q, k, v = self.qkv_dw(qkv_in).chunk(3, dim=1)
        dh = c // self.num_heads
        q = F.normalize(q.reshape(b, self.num_heads, dh, h * w), dim=-1)
        k = F.normalize(k.reshape(b, self.num_heads, dh, h * w), dim=-1)
        v = v.reshape(b, self.num_heads, dh, h * w)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.temp, dim=-1)
        out = (attn @ v).reshape(b, c, h, w)
        return self.proj(out)


class GDFN(nn.Module):
    def __init__(self, dim: int, exp: float = 2.0):
        super().__init__()
        hid = int(dim * exp)
        self.in_p = nn.Conv2d(dim, hid * 2, kernel_size=1, bias=False)
        self.dw = nn.Conv2d(hid * 2, hid * 2, kernel_size=3, padding=1, groups=hid * 2, bias=False)
        self.out_p = nn.Conv2d(hid, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = self.in_p(x)
        with torch.backends.cudnn.flags(enabled=False):
            x1, x2 = self.dw(inp).chunk(2, dim=1)
        return self.out_p(F.gelu(x1) * x2)


class RestormerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.n1 = nn.GroupNorm(1, dim)
        self.attn = MDTA(dim, heads)
        self.n2 = nn.GroupNorm(1, dim)
        self.ffn = GDFN(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        x = x + self.ffn(self.n2(x))
        return x


class SolarRestormerNet(nn.Module):
    """Lightweight U-shaped Restormer predicting clean residual restoration + local contrast boost."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.in_c = nn.Conv2d(1, dim, 3, padding=1)
        self.enc1 = nn.Sequential(RestormerBlock(dim, 4), RestormerBlock(dim, 4))
        self.down = nn.Conv2d(dim, dim * 2, 2, stride=2)
        self.mid = nn.Sequential(RestormerBlock(dim * 2, 4), RestormerBlock(dim * 2, 4))
        self.up = nn.ConvTranspose2d(dim * 2, dim, 2, stride=2)
        self.dec1 = nn.Sequential(RestormerBlock(dim * 2, 4), nn.Conv2d(dim * 2, dim, 1))
        self.out_c = nn.Conv2d(dim, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0 = self.in_c(x)
        e1 = self.enc1(f0)
        m = self.mid(self.down(e1))
        d1 = self.dec1(torch.cat([self.up(m), e1], dim=1))
        return torch.clamp(x + self.out_c(d1), 0.0, 1.0)


def _sharpness_score(im_u8: np.ndarray) -> float:
    sm = cv2.resize(im_u8, (512, 512), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(sm, cv2.CV_32F, ksize=3)
    return float(lap.var())


def _degrade_solar_patch(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Synthetic realistic H-alpha telescope degradation (PSF seeing blur + haze + noise)."""
    sig = float(rng.uniform(0.7, 2.0))
    deg = cv2.GaussianBlur(patch.astype(np.float32), (0, 0), sig)
    if rng.random() < 0.7:
        haze = float(rng.uniform(0.82, 0.98))
        deg = deg * haze + float(rng.uniform(-4.0, 8.0))
    noise = rng.normal(0.0, float(rng.uniform(1.5, 6.0)), size=deg.shape).astype(np.float32)
    return np.clip(deg + noise, 0.0, 255.0).astype(np.float32) / 255.0


def train_fold_restormer(
    tr_paths: list[Path],
    out_w: Path,
    device: str = "cuda",
    epochs: int = 20,
    patience: int = 5,
    gong_paths: list[Path] | None = None,
) -> SolarRestormerNet:
    """Train (or load cached) self-supervised SolarRestormer strictly on sharp `tr` (+ legal GONG SSL) images with patience=5."""
    if out_w.exists():
        model = SolarRestormerNet(dim=32).to(device)
        model.load_state_dict(torch.load(out_w, map_location=device))
        model.eval()
        return model

    # Combine fold tr images + legal external GONG SSL images (already filtered to exclude test +-1h/dates & va dates)
    pool = list(tr_paths)
    if gong_paths:
        rng_g = np.random.default_rng(17)
        pick = min(len(gong_paths), 120)
        idx = rng_g.choice(len(gong_paths), size=pick, replace=False)
        pool.extend(gong_paths[int(j)] for j in idx)

    scored = sorted(((_sharpness_score(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)), p) for p in pool), reverse=True)
    sharp_paths = [p for _, p in scored[: max(48, int(0.45 * len(scored)))]]
    imgs = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in sharp_paths if p.exists()]
    imgs = [im for im in imgs if im is not None and min(im.shape[:2]) >= 768]

    # Hold out 10% of internal sharp pool strictly inside train/legal-SSL to govern patience=5 early stopping
    n_hold = max(4, min(16, len(imgs) // 10))
    hold_imgs = imgs[:n_hold]
    fit_imgs = imgs[n_hold:] if len(imgs) > n_hold + 8 else imgs

    model = SolarRestormerNet(dim=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=2e-5)
    rng = np.random.default_rng(42)
    val_rng = np.random.default_rng(999)

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-2, -1)

    best_loss = float("inf")
    best_sd = None
    no_imp = 0
    torch.backends.cudnn.benchmark = False

    is_cuda = (device == "cuda") and torch.cuda.is_available()
    use_bf16 = is_cuda and (torch.cuda.get_device_capability(0)[0] >= 8) and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(is_cuda and not use_bf16))

    patch_sz = 384
    bs_rest = 4
    grad_accum = 2
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(fit_imgs))
        steps = list(range(0, len(order), bs_rest))
        for si, i in enumerate(steps):
            if si % grad_accum == 0:
                opt.zero_grad(set_to_none=True)
            xb_l, yb_l = [], []
            for k in order[i : i + bs_rest]:
                im = fit_imgs[int(k)]
                y0 = int(rng.integers(128, im.shape[0] - patch_sz - 128))
                x0 = int(rng.integers(128, im.shape[1] - patch_sz - 128))
                crop = im[y0 : y0 + patch_sz, x0 : x0 + patch_sz]
                xb_l.append(_degrade_solar_patch(crop, rng)[None])
                yb_l.append((crop.astype(np.float32) / 255.0)[None])
            xb = torch.from_numpy(np.stack(xb_l)).to(device)
            yb = torch.from_numpy(np.stack(yb_l)).to(device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=is_cuda):
                pred = model(xb)
            pred_f = pred.float()
            l1 = F.smooth_l1_loss(pred_f, yb, beta=0.01)
            gx_p, gy_p = F.conv2d(pred_f, sobel_x, padding=1), F.conv2d(pred_f, sobel_y, padding=1)
            gx_y, gy_y = F.conv2d(yb, sobel_x, padding=1), F.conv2d(yb, sobel_y, padding=1)
            ledge = F.l1_loss(gx_p, gx_y) + F.l1_loss(gy_p, gy_y)
            loss = (l1 + 0.35 * ledge) / float(grad_accum)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if (si + 1) % grad_accum == 0 or (si + 1) == len(steps):
                    scaler.step(opt)
                    scaler.update()
            else:
                loss.backward()
                if (si + 1) % grad_accum == 0 or (si + 1) == len(steps):
                    opt.step()
        sched.step()

        model.eval()
        v_losses = []
        with torch.no_grad():
            for im in hold_imgs:
                y0 = (im.shape[0] - patch_sz) // 2
                x0 = (im.shape[1] - patch_sz) // 2
                crop = im[y0 : y0 + patch_sz, x0 : x0 + patch_sz]
                xb = torch.from_numpy(_degrade_solar_patch(crop, val_rng)[None, None]).to(device)
                yb = torch.from_numpy((crop.astype(np.float32) / 255.0)[None, None]).to(device)
                pred = model(xb)
                v_losses.append(float(F.smooth_l1_loss(pred, yb, beta=0.01).item()))
        v_mean = float(np.mean(v_losses)) if v_losses else 0.0
        if v_mean + 1e-5 < best_loss:
            best_loss = v_mean
            best_sd = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"[Restormer] early-stop ep={ep} best_hold_l1={best_loss:.5f} (patience={patience})", flush=True)
                break

    if best_sd is not None:
        model.load_state_dict({kk: vv.to(device) for kk, vv in best_sd.items()})
    out_w.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_w)
    model.eval()
    return model


@torch.no_grad()
def enhance_full_image_with_restormer(model: SolarRestormerNet, im_u8: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Restore full 2048x2048 solar image cleanly using 4x4 overlapping 576x576 tiles -> 3-ch physical BGR."""
    h, w = im_u8.shape
    xf = im_u8.astype(np.float32) / 255.0
    acc = np.zeros((h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)
    tile = 576
    ys = sorted(set([int(round(k * (h - tile) / 3.0)) for k in range(4)])) if h > tile else [0]
    xs = sorted(set([int(round(k * (w - tile) / 3.0)) for k in range(4)])) if w > tile else [0]
    is_cuda = ("cuda" in str(device)) and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if (is_cuda and torch.cuda.get_device_capability(0)[0] >= 8 and torch.cuda.is_bf16_supported()) else torch.float16
    for y0 in ys:
        for x0 in xs:
            inp = torch.from_numpy(xf[y0 : y0 + tile, x0 : x0 + tile])[None, None].to(device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=is_cuda):
                out = model(inp)[0, 0].float().cpu().numpy()
            acc[y0 : y0 + tile, x0 : x0 + tile] += out
            cnt[y0 : y0 + tile, x0 : x0 + tile] += 1.0
    restored = np.clip((acc / np.maximum(cnt, 1.0)) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return physics_3ch_from_gray(restored)


def decode_ann_mask(a: dict, full: int = FULL) -> np.ndarray:
    """Decode COCO polygon or RLE (compressed/uncompressed) segmentation to uint8 mask."""
    from pycocotools import mask as mu

    seg = a.get("segmentation")
    if not seg:
        return np.zeros((full, full), dtype=np.uint8)
    if isinstance(seg, list):
        polys = [p for p in seg if isinstance(p, (list, tuple)) and len(p) >= 6]
        if not polys:
            return np.zeros((full, full), dtype=np.uint8)
        rle = mu.merge(mu.frPyObjects(polys, full, full))
    elif isinstance(seg, dict):
        rle = mu.frPyObjects(seg, full, full) if isinstance(seg.get("counts"), list) else seg
    else:
        return np.zeros((full, full), dtype=np.uint8)
    return mu.decode(rle).astype(np.uint8)


def build_tr_copypaste_bank(tr_stems: list[str], img_dir: Path, recs: dict, anns: dict, max_bank: int = 420, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Collect clean small/medium (<1400 px) filament patches STRICTLY from `tr_stems` across all annotators."""
    rng = np.random.default_rng(seed)
    bank = []
    for s in rng.permutation(tr_stems):
        im = cv2.imread(str(img_dir / f"{s}.png" if (img_dir / f"{s}.png").exists() else img_dir / f"{s}.jpeg"))
        if im is None:
            continue
        for rid in recs.get(s, []):
            for a in anns.get(rid, []):
                x, y, w, h = [int(round(v)) for v in a["bbox"]]
                if w < 10 or h < 10 or max(w, h) > 140 or min(w, h) < 6:
                    continue
                m = decode_ann_mask(a, FULL)
                area = int(m.sum())
                if area < 85 or area > 1400:
                    continue
                x1, y1 = max(0, x - 3), max(0, y - 3)
                x2, y2 = min(FULL, x + w + 3), min(FULL, y + h + 3)
                bank.append((im[y1:y2, x1:x2].copy(), m[y1:y2, x1:x2].copy()))
                if len(bank) >= max_bank:
                    return bank
    return bank


def _rotate_patch_and_mask(patch: np.ndarray, pmask: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate small filament crop cleanly inside tight padded canvas without clipping."""
    ph, pw = patch.shape[:2]
    side = int(math.ceil(math.hypot(pw, ph))) + 4
    cp = np.zeros((side, side, patch.shape[2]), dtype=patch.dtype) if patch.ndim == 3 else np.zeros((side, side), dtype=patch.dtype)
    cm = np.zeros((side, side), dtype=pmask.dtype)
    ox, oy = (side - pw) // 2, (side - ph) // 2
    cp[oy : oy + ph, ox : ox + pw] = patch
    cm[oy : oy + ph, ox : ox + pw] = pmask
    M = cv2.getRotationMatrix2D((side * 0.5, side * 0.5), angle_deg, 1.0)
    rp = cv2.warpAffine(cp, M, (side, side), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    rm = (cv2.warpAffine(cm, M, (side, side), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.5).astype(np.uint8)
    ys, xs = np.nonzero(rm)
    if not len(xs):
        return patch, pmask
    x1, y1 = max(0, int(xs.min()) - 2), max(0, int(ys.min()) - 2)
    x2, y2 = min(side, int(xs.max()) + 3), min(side, int(ys.max()) + 3)
    return rp[y1:y2, x1:x2], rm[y1:y2, x1:x2]


def apply_copypaste_sample(
    bgr: np.ndarray,
    boxes_xywh: list[list[float]],
    bank: list[tuple[np.ndarray, np.ndarray]],
    rng: np.random.Generator,
    n_paste: int = 3,
    return_polys: bool = False,
):
    """Paste `n_paste` rotated training-only filaments onto clean quiet-Sun regions of `tr` photo."""
    if not bank:
        return (bgr, boxes_xywh, []) if return_polys else (bgr, boxes_xywh)
    out = bgr.copy()
    new_boxes = list(boxes_xywh)
    pasted_polys = []
    h_img, w_img = out.shape[:2]
    gray = out[:, :, 0]
    cx, cy, rad, _ = disk_geometry(gray)

    occ = np.zeros((h_img, w_img), dtype=bool)
    for bx, by, bw, bh in new_boxes:
        occ[max(0, int(by)) : min(h_img, int(by + bh)), max(0, int(bx)) : min(w_img, int(bx + bw))] = True

    for _ in range(n_paste):
        patch, pmask = bank[int(rng.integers(len(bank)))]
        if rng.random() < 0.5:
            patch = patch[:, ::-1]
            pmask = pmask[:, ::-1]
        if rng.random() < 0.75:
            patch, pmask = _rotate_patch_and_mask(patch, pmask, float(rng.uniform(-180.0, 180.0)))
        ph, pw = patch.shape[:2]
        for _try in range(14):
            ang = float(rng.uniform(0, 2 * math.pi))
            rr = float(rng.uniform(0.08, 0.78)) * rad
            px = int(cx + rr * math.cos(ang) - pw * 0.5)
            py = int(cy + rr * math.sin(ang) - ph * 0.5)
            if px < 4 or py < 4 or px + pw >= w_img - 4 or py + ph >= h_img - 4:
                continue
            if occ[py : py + ph, px : px + pw].any():
                continue
            alpha = cv2.GaussianBlur(pmask.astype(np.float32), (5, 5), 1.0)[..., None]
            alpha = np.clip(alpha, 0.0, 1.0)
            roi = out[py : py + ph, px : px + pw].astype(np.float32)
            # Match local solar continuum ch0 while preserving relative absorption contrast
            ch0_diff = float(np.median(roi[:, :, 0]) - np.median(patch[:, :, 0][pmask == 0] if (pmask == 0).any() else patch[:, :, 0]))
            p_adj = patch.astype(np.float32)
            p_adj[:, :, 0] = np.clip(p_adj[:, :, 0] + 0.5 * ch0_diff, 0.0, 255.0)
            out[py : py + ph, px : px + pw] = np.clip(alpha * p_adj + (1.0 - alpha) * roi, 0, 255).astype(np.uint8)
            occ[py : py + ph, px : px + pw] = True
            ys, xs = np.nonzero(pmask > 0)
            if len(xs):
                new_boxes.append([float(px + xs.min()), float(py + ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)])
                if return_polys:
                    cnts, _ = cv2.findContours(pmask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in cnts:
                        if len(cnt) >= 3:
                            pts = cnt.reshape(-1, 2).astype(np.float32)
                            pts[:, 0] += px
                            pts[:, 1] += py
                            pasted_polys.append(pts.reshape(-1).tolist())
                            break
            break
    if return_polys:
        return out, new_boxes, pasted_polys
    return out, new_boxes
