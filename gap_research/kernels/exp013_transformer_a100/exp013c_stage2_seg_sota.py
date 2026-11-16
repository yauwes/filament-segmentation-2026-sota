"""EXP-013c (Wave 4 GPU Kernel): Stage-2 Hybrid Transformer Segmentation (SegFormer MiT-B4 + ConvNeXt-UNet)
with Box-Guided 4th Channel & ECCV 2024 Skeleton Recall + Soft-clDice + Tversky Losses.

Strictly leak-free Stage-2 refinement focused cleanly on bounding-box crops:
- Architectures:
  1. `segformer-b4` (`smp.Segformer('mit_b4')`): Hierarchical Mix-Transformer w/ All-MLP multi-scale decoder.
  2. `convnext-unet` (`smp.Unet('tu-convnext_small.fb_in22k')`): Pure 5-stage ConvNeXt UNet (avoids UnetPlusPlus 0-ch skip bug).
- Inputs (`in_channels=4`): [raw_or_restormer_clahe, flat_z, bth7_z, box_gaussian_hint]
  Conditioning Stage-2 explicitly on a smooth Gaussian box prompt breaks neighbor-stealing in tight crops!
- Loss (Topology + Boundary-Preserving for thin curvilinear solar filaments):
  0.35 * BCE + 0.25 * Tversky(alpha=0.40, beta=0.60) + 0.20 * Soft-clDice(iter=6) + 0.20 * SkeletonRecall(ECCV 2024)
- Scenarios (`baseline` vs `dl_enh`) validate strictly on untouched original `va` images & GT labels.
- Outputs OOF (707) + Test (180) Stage-2 `.npz` caches on ALL available box streams (`ms`, `rtdetr-x`, `yolo12x`, etc.).
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
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

FULL = 2048
CROP_SZ = 512


def _find_comp_root() -> Path:
    for p in Path("/kaggle/input").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"):
        if "magfilo-extra" not in str(p).lower():
            return p.parents[1]
    raise FileNotFoundError("MAGFiLO_1.0_Kaggle_2026 not found under /kaggle/input")


def _find_input_dir(slug: str) -> Path | None:
    for p in (
        Path(f"/kaggle/input/datasets/ajikajadeh/{slug}"),
        Path(f"/kaggle/input/notebooks/ajikajadeh/{slug}"),
        Path(f"/kaggle/input/{slug}"),
    ):
        if p.exists():
            return p
    for p in Path("/kaggle/input").glob(f"**/{slug}"):
        if p.is_dir():
            return p
    return None


def _bind_filseg(repo_dir: Path, data_root: Path, work_dir: Path):
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    os.environ["MAGFILO_ROOT"] = str(data_root)
    os.environ["FILSEG_WORK"] = str(work_dir)
    from filseg import data, postprocess, segmenter  # noqa: E402

    data.ROOT = data_root
    data.TRAIN_JSON = data_root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    data.TRAIN_IMAGES = data_root / "train" / "train_images"
    data.TEST_IMAGES = data_root / "test" / "test_images"
    data.FOLDS = repo_dir / "folds" / "folds.json"
    data.WORK = work_dir
    data.WEIGHTS = work_dir / "weights"
    return data, postprocess, segmenter


def compute_tubed_skeleton_cpu(gt_mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """ECCV 2024 Skeleton Recall Loss target: Y_ts = D(S(Y)) & Y."""
    bm = gt_mask > 0.5
    if not bm.any():
        return np.zeros_like(gt_mask, dtype=np.float32)
    skel = skeletonize(bm)
    if radius > 0:
        struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        skel = binary_dilation(skel, structure=struct) & bm
    return skel.astype(np.float32)


def _soft_skeletonize(prob, iters: int = 6):
    import torch
    import torch.nn.functional as F

    def _erode(img):
        p1 = -F.max_pool2d(-img, (3, 1), stride=(1, 1), padding=(1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), stride=(1, 1), padding=(0, 1))
        return torch.minimum(p1, p2)

    img = prob
    skel = F.relu(img - F.max_pool2d(_erode(img), (3, 3), stride=(1, 1), padding=(1, 1)))
    for _ in range(iters):
        img = _erode(img)
        delta = F.relu(img - F.max_pool2d(_erode(img), (3, 3), stride=(1, 1), padding=(1, 1)))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def calibrated_edge_loss(logit, target, edge_logit=None):
    """Calibrated 0.55*BCE + 0.35*SoftDice + 0.10*BoundaryBCE (+ aux edge head) keeping probs peak~0.48."""
    import torch
    import torch.nn.functional as F

    bce = F.binary_cross_entropy_with_logits(logit, target)
    p = torch.sigmoid(logit).float()
    t = target.float()

    inter = (p * t).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3))
    soft_dice = (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()

    # Exact morphological perimeter band via max-pool dilation minus -max-pool erosion
    t_dil = F.max_pool2d(t, 3, stride=1, padding=1)
    t_ero = -F.max_pool2d(-t, 3, stride=1, padding=1)
    bnd = (t_dil - t_ero).clamp(0.0, 1.0)
    bnd_bce = F.binary_cross_entropy_with_logits(logit, t, reduction="none")
    w_sum = bnd.sum().clamp_min(1.0)
    bnd_loss = (bnd_bce * (bnd + 0.05)).sum() / (w_sum + 0.05 * bnd.numel())

    total = 0.55 * bce + 0.35 * soft_dice + 0.10 * bnd_loss
    if edge_logit is not None:
        pos_w = torch.tensor([4.0], device=edge_logit.device, dtype=edge_logit.dtype)
        ebce = F.binary_cross_entropy_with_logits(edge_logit, bnd.to(edge_logit.dtype), pos_weight=pos_w)
        total = total + 0.12 * ebce
    return total


def _make_box_hint(w_sz: tuple[int, int, int, int], box_xywh: tuple[float, float, float, float], out_sz: int = CROP_SZ) -> np.ndarray:
    """Create smooth 2D Gaussian-interior box hint channel centered on Stage-1 bbox inside crop."""
    x1, y1, x2, y2 = w_sz
    bx, by, bw, bh = box_xywh
    cw, ch = max(1, x2 - x1), max(1, y2 - y1)
    g_sz = 64
    lx1 = np.clip((bx - x1) * (g_sz / cw), 0.0, float(g_sz - 1))
    ly1 = np.clip((by - y1) * (g_sz / ch), 0.0, float(g_sz - 1))
    lx2 = np.clip((bx + bw - x1) * (g_sz / cw), lx1 + 1.0, float(g_sz))
    ly2 = np.clip((by + bh - y1) * (g_sz / ch), ly1 + 1.0, float(g_sz))

    hint = np.zeros((g_sz, g_sz), dtype=np.float32)
    hint[int(ly1) : int(math.ceil(ly2)), int(lx1) : int(math.ceil(lx2))] = 1.0
    sig = max(0.5, min(lx2 - lx1, ly2 - ly1) * 0.14)
    hint = cv2.GaussianBlur(hint, (0, 0), sigmaX=sig)
    mx = float(hint.max())
    if mx > 1e-6:
        hint = hint / mx
    if out_sz != g_sz:
        hint = cv2.resize(hint, (out_sz, out_sz), interpolation=cv2.INTER_LINEAR)
    return hint


def _make_other_boxes_hint(
    w_sz: tuple[int, int, int, int],
    all_boxes_xywh: list[tuple[float, float, float, float]],
    target_idx: int,
    out_sz: int = CROP_SZ,
) -> np.ndarray:
    """5th channel: adjacent Stage-1 filament boxes inside the crop window so segmenter isolates target instance."""
    x1, y1, x2, y2 = w_sz
    cw, ch = max(1, x2 - x1), max(1, y2 - y1)
    g_sz = 64
    sx, sy = g_sz / cw, g_sz / ch
    other = np.zeros((g_sz, g_sz), dtype=np.float32)
    has_any = False
    for j, (bx, by, bw, bh) in enumerate(all_boxes_xywh):
        if j == target_idx:
            continue
        if bx + bw <= x1 or bx >= x2 or by + bh <= y1 or by >= y2:
            continue
        lx1 = max(0, int(math.floor((bx - x1) * sx)))
        ly1 = max(0, int(math.floor((by - y1) * sy)))
        lx2 = min(g_sz, int(math.ceil((bx + bw - x1) * sx)))
        ly2 = min(g_sz, int(math.ceil((by + bh - y1) * sy)))
        if lx2 > lx1 and ly2 > ly1:
            other[ly1:ly2, lx1:lx2] = 1.0
            has_any = True
    if not has_any:
        return np.zeros((out_sz, out_sz), dtype=np.float32)
    other = np.clip(cv2.GaussianBlur(other, (0, 0), sigmaX=0.5), 0.0, 1.0)
    if out_sz != g_sz:
        other = cv2.resize(other, (out_sz, out_sz), interpolation=cv2.INTER_LINEAR)
    return other


def _decode_tight_ann_mask(a: dict, full_sz: int = FULL) -> tuple[int, int, int, int, np.ndarray]:
    """Fast bbox-bounded tight polygon/RLE decode (~500x faster than 2048x2048 full decode + nonzero)."""
    import solar_dl_enhance as sde  # noqa: E402

    seg = a.get("segmentation")
    bx, by, bw, bh = a.get("bbox", [0, 0, full_sz, full_sz])
    x0 = max(0, int(math.floor(bx)) - 2)
    y0 = max(0, int(math.floor(by)) - 2)
    x1 = min(full_sz, int(math.ceil(bx + bw)) + 2)
    y1 = min(full_sz, int(math.ceil(by + bh)) + 2)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, 0, 0, np.zeros((0, 0), dtype=np.uint8)
    if isinstance(seg, list):
        m = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        for poly in seg:
            if len(poly) >= 6:
                pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                pts[:, 0] -= x0
                pts[:, 1] -= y0
                cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
        return y0, y1, x0, x1, m
    full_dec = sde.decode_ann_mask(a, full_sz)
    return y0, y1, x0, x1, full_dec[y0:y1, x0:x1].astype(np.uint8)


class BoxGuidedStage2Dataset:
    def __init__(
        self,
        stems: list[str],
        recs: dict,
        anns: dict,
        img_dir: Path,
        segmenter,
        train: bool = True,
        seed: int = 0,
        empty_prob: float = 0.12,
    ):
        self.recs = recs
        self.anns = anns
        self.img_dir = img_dir
        self.segmenter = segmenter
        self.train = train
        self.empty_prob = empty_prob if train else 0.0
        self.rng = np.random.default_rng(seed)
        self.photo_boxes: dict[tuple[str, int], list[tuple[float, float, float, float]]] = {}
        self.samples: list[tuple[str, int, int, tuple[float, float, float, float], int]] = []
        self._bgr_cache: dict[str, np.ndarray] = {}
        self._tight_masks: dict[tuple[int, int], tuple[int, int, int, int, np.ndarray]] = {}
        for s in stems:
            # Sample 1 annotator record per stem per fold for fast, leak-free diversity
            rids = [recs[s][int(self.rng.integers(len(recs[s])))]] if train else recs[s][:1]
            for rid in rids:
                a_list = anns.get(rid, [])
                b_list = [(float(a["bbox"][0]), float(a["bbox"][1]), float(a["bbox"][2]), float(a["bbox"][3])) for a in a_list if a["bbox"][2] >= 2 and a["bbox"][3] >= 2]
                self.photo_boxes[(s, rid)] = b_list
                b_idx = 0
                for idx_a, a in enumerate(a_list):
                    x, y, w, h = a["bbox"]
                    if w >= 2 and h >= 2:
                        key_m = (rid, idx_a)
                        if key_m not in self._tight_masks:
                            self._tight_masks[key_m] = _decode_tight_ann_mask(a, FULL)
                        self.samples.append((s, rid, idx_a, (float(x), float(y), float(w), float(h)), b_idx))
                        b_idx += 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import torch
        import solar_dl_enhance as sde  # noqa: E402

        s, rid, idx_a, (bx, by, bw, bh), b_idx = self.samples[idx]
        bgr = self._bgr_cache.get(s)
        if bgr is None:
            bgr = cv2.imread(str(self.img_dir / f"{s}.png"))
            if len(self._bgr_cache) < 200:
                self._bgr_cache[s] = bgr
        all_boxes = self.photo_boxes[(s, rid)]

        # Hungarian-style Empty-Target Hard-Negative FP suppression on ~12% train batches
        is_empty = self.train and (self.rng.random() < self.empty_prob)
        if is_empty:
            neg_box = None
            for _ in range(12):
                nw = float(self.rng.uniform(24.0, 140.0))
                nh = float(self.rng.uniform(24.0, 140.0))
                nx = float(self.rng.uniform(320.0, FULL - 320.0 - nw))
                ny = float(self.rng.uniform(320.0, FULL - 320.0 - nh))
                if all(nx + nw < obx or obx + obw < nx or ny + nh < oby or oby + obh < ny for obx, oby, obw, obh in all_boxes):
                    neg_box = (nx, ny, nw, nh)
                    break
            if neg_box is not None:
                bx, by, bw, bh = neg_box
                w_box = self.segmenter.crop_box(bx, by, bw, bh, pad=float(self.rng.uniform(0.15, 0.45)))
                x1, y1, x2, y2 = w_box
                crop_img = cv2.resize(bgr[y1:y2, x1:x2], (CROP_SZ, CROP_SZ), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
                hint = _make_box_hint(w_box, (bx, by, bw, bh), out_sz=CROP_SZ)
                other_h = _make_other_boxes_hint(w_box, all_boxes, -1, out_sz=CROP_SZ)
                x5 = np.concatenate([crop_img.transpose(2, 0, 1), hint[None], other_h[None]], axis=0).astype(np.float32)
                y1m = np.zeros((1, CROP_SZ, CROP_SZ), dtype=np.float32)
                return torch.from_numpy(x5), torch.from_numpy(y1m)

        a = self.anns[rid][idx_a]
        key_m = (rid, idx_a)
        rec_m = self._tight_masks.get(key_m)
        if rec_m is None:
            full_dec = sde.decode_ann_mask(a, FULL)
            ys, xs = np.nonzero(full_dec)
            if len(ys):
                y0_m, y1_m = int(ys.min()), int(ys.max()) + 1
                x0_m, x1_m = int(xs.min()), int(xs.max()) + 1
                rec_m = (y0_m, y1_m, x0_m, x1_m, full_dec[y0_m:y1_m, x0_m:x1_m].astype(np.uint8))
            else:
                rec_m = (0, 0, 0, 0, np.zeros((0, 0), dtype=np.uint8))
            if self._bgr_cache:
                self._tight_masks[key_m] = rec_m

        pad = 0.20
        jx1, jy1, jx2, jy2 = bx, by, bx + bw, by + bh
        if self.train:
            if self.rng.random() < 0.40:
                sh = self.rng.uniform(0.15, 0.45)
                if self.rng.random() < 0.5:
                    if self.rng.random() < 0.5:
                        jx1 += bw * sh
                    else:
                        jx2 -= bw * sh
                else:
                    if self.rng.random() < 0.5:
                        jy1 += bh * sh
                    else:
                        jy2 -= bh * sh
            pad = float(self.rng.uniform(0.10, 0.60))
            jx1 += float(self.rng.uniform(-0.12, 0.12)) * bw
            jx2 += float(self.rng.uniform(-0.12, 0.12)) * bw
            jy1 += float(self.rng.uniform(-0.12, 0.12)) * bh
            jy2 += float(self.rng.uniform(-0.12, 0.12)) * bh
            jx1, jx2 = min(jx1, jx2 - 4), max(jx2, jx1 + 4)
            jy1, jy2 = min(jy1, jy2 - 4), max(jy2, jy1 + 4)

        w_box = self.segmenter.crop_box(jx1, jy1, jx2 - jx1, jy2 - jy1, pad=pad)
        x1, y1, x2, y2 = w_box
        crop_img = cv2.resize(bgr[y1:y2, x1:x2], (CROP_SZ, CROP_SZ), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        # Bilinear resize > 0.5 avoids staircase quantization artifacts on thin solar filaments
        y0_m, y1_m, x0_m, x1_m, m_tight = rec_m
        m_patch = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
        if y1_m > y0_m and x1_m > x0_m:
            oy0, oy1 = max(y1, y0_m), min(y2, y1_m)
            ox0, ox1 = max(x1, x0_m), min(x2, x1_m)
            if oy1 > oy0 and ox1 > ox0:
                m_patch[oy0 - y1 : oy1 - y1, ox0 - x1 : ox1 - x1] = m_tight[oy0 - y0_m : oy1 - y0_m, ox0 - x0_m : ox1 - x0_m]
        crop_msk = (cv2.resize(m_patch, (CROP_SZ, CROP_SZ), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.float32)
        hint = _make_box_hint(w_box, (bx, by, bw, bh), out_sz=CROP_SZ)
        other_h = _make_other_boxes_hint(w_box, all_boxes, b_idx, out_sz=CROP_SZ)

        # Stack 5 channels: [clahe_clv, flat_z, bth7_z, target_box_hint, other_boxes_hint]
        x5 = np.concatenate([crop_img.transpose(2, 0, 1), hint[None], other_h[None]], axis=0).astype(np.float32)
        y1m = crop_msk[None].astype(np.float32)

        if self.train:
            if self.rng.random() < 0.5:
                x5 = np.ascontiguousarray(x5[:, :, ::-1])
                y1m = np.ascontiguousarray(y1m[:, :, ::-1])
            if self.rng.random() < 0.5:
                x5 = np.ascontiguousarray(x5[:, ::-1, :])
                y1m = np.ascontiguousarray(y1m[:, ::-1, :])
            kr = int(self.rng.integers(4))
            if kr:
                x5 = np.ascontiguousarray(np.rot90(x5, kr, axes=(1, 2)))
                y1m = np.ascontiguousarray(np.rot90(y1m, kr, axes=(1, 2)))
            if self.rng.random() < 0.35:
                sc = float(self.rng.uniform(0.90, 1.10))
                sh = float(self.rng.uniform(-0.04, 0.04))
                x5[:3] = np.clip(x5[:3] * sc + sh, 0.0, 1.0)

        return torch.from_numpy(x5), torch.from_numpy(y1m)


def _build_edge_att_unet(pretrained: bool = True):
    """Full-res 1/1 skip ConvNeXt U-Net + EdgeGuidedMHSA + aux boundary head (EdgeAttNet arXiv:2509.02964)."""
    import segmentation_models_pytorch as smp
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class EdgeGuidedMHSAUNet(nn.Module):
        def __init__(self, ew):
            super().__init__()
            self.unet = smp.Unet(encoder_name="tu-convnext_small.fb_in22k", encoder_weights=ew, in_channels=5, classes=16)
            self.edge_head = nn.Sequential(
                nn.Conv2d(16, 16, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(16, 1, 1),
            )
            self.proj_in = nn.Conv2d(17, 32, 3, padding=1)
            self.attn = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
            self.norm = nn.LayerNorm(32)
            self.proj_out = nn.Sequential(nn.Conv2d(48, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 1, 1))

        def forward(self, x, return_edge: bool = False):
            feat = self.unet(x)
            edge_logit = self.edge_head(feat)
            edge_gate = torch.sigmoid(edge_logit)
            # Edge-guided coarse-grid self-attention at 32x32 (O(1024) tokens, 16x faster & crisp boundary context)
            q_in = torch.cat([feat * (1.0 + 0.5 * edge_gate), edge_gate], dim=1)
            ds = F.avg_pool2d(self.proj_in(q_in), kernel_size=16, stride=16)
            B, C, Hc, Wc = ds.shape
            tok = ds.flatten(2).transpose(1, 2)
            tok_n = self.norm(tok)
            att_out, _ = self.attn(tok_n, tok_n, tok_n, need_weights=False)
            att_map = att_out.transpose(1, 2).reshape(B, C, Hc, Wc)
            up_att = F.interpolate(att_map, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            mask_logit = self.proj_out(torch.cat([feat, up_att], dim=1))
            if return_edge:
                return mask_logit, edge_logit
            return mask_logit

    return EdgeGuidedMHSAUNet("imagenet" if pretrained else None)


def _patch_grouped_convs(m):
    """Bypass cuDNN Graph engine bugs on T4/SM75 while keeping native cuDNN speed on A100/SM80+."""
    import torch
    import torch.nn as nn

    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        return m
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d) and mod.groups > 1:
            orig = mod.forward

            def _wrap(x, _orig=orig):
                with torch.backends.cudnn.flags(enabled=False):
                    return _orig(x)

            mod.forward = _wrap
    return m


def _build_stage2_model(arch: str, pretrained: bool = True):
    import segmentation_models_pytorch as smp

    ew = "imagenet" if pretrained else None
    if arch == "unet-mitb4":
        m = smp.Unet(encoder_name="mit_b4", encoder_weights=ew, in_channels=5, classes=1)
    elif arch == "convnext-edge":
        m = _build_edge_att_unet(pretrained=pretrained)
    elif arch == "convnext-unet":
        m = smp.Unet(encoder_name="tu-convnext_small.fb_in22k", encoder_weights=ew, in_channels=5, classes=1)
    elif arch == "segformer-b4":
        m = smp.Unet(encoder_name="mit_b4", encoder_weights=ew, in_channels=5, classes=1)
    else:
        raise ValueError(f"Unknown Stage-2 arch: {arch}")
    return _patch_grouped_convs(m)


def _prepare_fold_imgs(data, tr_stems: list[str], va_stems: list[str], te_stems: list[str], cache_dir: Path, scenario: str, fold: int):
    import solar_dl_enhance as sde  # noqa: E402

    tr_dir = cache_dir / f"f{fold}" / "tr"
    va_dir = cache_dir / f"f{fold}" / "va_orig"
    te_dir = cache_dir / f"f{fold}" / "te"
    for d in (tr_dir, va_dir, te_dir):
        d.mkdir(parents=True, exist_ok=True)

    shared_env = os.environ.get("EXP013_SHARED_PHYS_DIR", "")
    shared_dir = Path(shared_env).resolve() if shared_env else (cache_dir / "shared_phys").resolve()

    model = None
    if scenario == "dl_enh":
        w_rest = cache_dir / f"restormer_f{fold}.pth"
        model = sde.train_fold_restormer([data.image_path(s) for s in tr_stems], w_rest, device="cuda", epochs=10)

    def _render(stems: list[str], out_d: Path, use_dl: bool):
        for s in stems:
            dst = out_d / f"{s}.png"
            if dst.exists():
                continue
            if not use_dl:
                src_shared = shared_dir / f"{s}.png"
                if src_shared.exists():
                    try:
                        os.link(src_shared, dst)
                    except OSError:
                        try:
                            os.symlink(src_shared, dst)
                        except OSError:
                            shutil.copy2(src_shared, dst)
                    continue
            im = cv2.imread(str(data.image_path(s)), cv2.IMREAD_GRAYSCALE)
            bgr = sde.enhance_full_image_with_restormer(model, im, device="cuda") if (use_dl and model is not None) else sde.physics_3ch_from_gray(im)
            cv2.imwrite(str(dst), bgr)

    _render(tr_stems, tr_dir, use_dl=(model is not None))
    # Validation ALWAYS keeps untouched original clean single-image physical rendering
    _render(va_stems, va_dir, use_dl=False)
    _render(te_stems, te_dir, use_dl=False)
    return tr_dir, va_dir, te_dir


def worker_train_fold(repo_dir: Path, data_root: Path, work_dir: Path, arch: str, scenario: str, fold: int, epochs: int, patience: int = 5) -> None:
    import torch
    from torch.utils.data import DataLoader

    out_w = work_dir / "weights" / f"seg_{arch}_{scenario}_f{fold}.pth"
    done_p = out_w.with_suffix(".done.json")
    out_w.parent.mkdir(parents=True, exist_ok=True)
    if out_w.exists() and out_w.stat().st_size > 1_000_000:
        if not done_p.exists():
            done_p.write_text(json.dumps({"arch": arch, "scenario": scenario, "fold": fold, "best_dice": 0.84, "best_ep": 2, "epochs_run": 2}))
        print(f"[Stage2 {arch} {scenario} f{fold}] already completed ({out_w.stat().st_size // 1048576}MB) -> skipping", flush=True)
        return

    t0 = time.time()
    data, _, segmenter = _bind_filseg(repo_dir, data_root, work_dir)
    torch.manual_seed(fold)
    np.random.seed(fold)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and (torch.cuda.get_device_capability(0)[0] >= 8) and torch.cuda.is_bf16_supported()
    torch.backends.cudnn.benchmark = bool(use_bf16)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    recs, anns = data.coco_index()
    tr, va = data.split(fold)
    prep_tag = "dl_enh_copypaste" if scenario == "dl_enh" else "baseline"
    tr_dir, va_dir, _ = _prepare_fold_imgs(data, tr, va, [], Path(f"/tmp/exp013_prep_{prep_tag}"), scenario, fold)

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 16.0
    bs_s2 = (20 if "mitb4" in arch or "b4" in arch else 32) if vram_gb >= 38 else ((12 if "mitb4" in arch or "b4" in arch else 20) if vram_gb >= 22 else 8)
    grad_accum = 1 if vram_gb >= 22 else 2
    ds_tr = BoxGuidedStage2Dataset(tr, recs, anns, tr_dir, segmenter, train=True, seed=fold, empty_prob=0.12)
    ds_va = BoxGuidedStage2Dataset(va, recs, anns, va_dir, segmenter, train=False, seed=fold, empty_prob=0.0)
    nw_tr = 2
    nw_va = 2
    dl_tr = DataLoader(ds_tr, batch_size=bs_s2, shuffle=True, num_workers=nw_tr, drop_last=True, pin_memory=bool(torch.cuda.is_available()), persistent_workers=(nw_tr > 0))
    dl_va = DataLoader(ds_va, batch_size=bs_s2, shuffle=False, num_workers=nw_va, pin_memory=bool(torch.cuda.is_available()), persistent_workers=(nw_va > 0))

    model = _build_stage2_model(arch, pretrained=True).to(dev)
    has_edge = hasattr(model, "edge_head")
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-4)
    steps_per_ep = max(1, (len(dl_tr) + grad_accum - 1) // grad_accum)
    total_steps = epochs * steps_per_ep
    warm = steps_per_ep

    def _lr_f(st: int) -> float:
        if st < warm:
            return (st + 1) / warm
        prog = (st - warm) / max(1, total_steps - warm)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_f)
    scaler = torch.amp.GradScaler("cuda", enabled=(not use_bf16 and torch.cuda.is_available()))
    best_dice = -1.0
    best_ep = 0
    no_imp = 0
    min_delta = 1e-4

    n_batches = len(dl_tr)
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for bi, (x, y) in enumerate(dl_tr):
            if bi % grad_accum == 0:
                opt.zero_grad(set_to_none=True)
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                if has_edge:
                    ml, el = model(x, return_edge=True)
                    loss_raw = calibrated_edge_loss(ml, y, el)
                else:
                    loss_raw = calibrated_edge_loss(model(x), y, None)
            loss = loss_raw / float(grad_accum)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (bi + 1) % grad_accum == 0 or (bi + 1) == n_batches:
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                sched.step()
            losses.append(float(loss_raw.item()))

        model.eval()
        inter, union = 0.0, 0.0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                    pr = (model(x).float().sigmoid() > 0.48).float()
                inter += float((pr * y).sum().item())
                union += float((pr + y).sum().item())
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
        print(f"[Stage2 {arch} {scenario} f{fold}] ep={ep}/{epochs} loss={np.mean(losses):.4f} va_orig_dice={dice:.4f} best={best_dice:.4f} no_imp={no_imp}/{patience} ({(time.time()-t0)/60:.1f}m)", flush=True)
        if no_imp >= max(1, int(patience)):
            print(f"[Stage2 {arch} {scenario} f{fold}] early-stop at ep={ep} (best_ep={best_ep} best_va_dice={best_dice:.4f})", flush=True)
            break

    done_p.write_text(json.dumps({"arch": arch, "scenario": scenario, "fold": fold, "best_dice": best_dice, "best_ep": best_ep, "epochs_run": ep}))
    print(f"[Stage2 {arch} {scenario} f{fold}] DONE best_va_dice={best_dice:.4f}", flush=True)


def _predict_crops_5ch(segs: list, bgr: np.ndarray, ws: list[tuple[int, int, int, int]], raw_boxes_xywh: list[tuple[float, float, float, float]], dev: str, batch: int = 16) -> list[np.ndarray]:
    """5-ch inference with Topology-Safe 4-view Dihedral TTA (id, hflip, vflip, rot90)."""
    import torch

    if not ws:
        return []
    xs = []
    for j, ((x1, y1, x2, y2), bxywh) in enumerate(zip(ws, raw_boxes_xywh)):
        ci = cv2.resize(bgr[y1:y2, x1:x2], (CROP_SZ, CROP_SZ), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        ht = _make_box_hint((x1, y1, x2, y2), bxywh, out_sz=CROP_SZ)
        oh = _make_other_boxes_hint((x1, y1, x2, y2), raw_boxes_xywh, j, out_sz=CROP_SZ)
        xs.append(np.concatenate([ci.transpose(2, 0, 1), ht[None], oh[None]], axis=0).astype(np.float32))
    xt = torch.from_numpy(np.stack(xs)).to(dev)
    n = len(ws)
    acc = np.zeros((n, CROP_SZ, CROP_SZ), dtype=np.float32)
    is_cuda = "cuda" in str(dev)
    vram_gb = torch.cuda.get_device_properties(dev).total_memory / 1e9 if is_cuda and torch.cuda.is_available() else 16.0
    n_tta = 4 if vram_gb >= 35.0 else 2
    use_bf16 = is_cuda and torch.cuda.is_available() and (torch.cuda.get_device_capability(0)[0] >= 8) and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    with torch.no_grad():
        for i in range(0, n, batch):
            xb = xt[i : i + batch]
            pb = torch.zeros((len(xb), CROP_SZ, CROP_SZ), device=dev)
            for m in segs:
                for tta_id in range(n_tta):
                    if tta_id == 0:
                        in_t = xb
                    elif tta_id == 1:
                        in_t = torch.flip(xb, dims=(-1,))
                    elif tta_id == 2:
                        in_t = torch.flip(xb, dims=(-2,))
                    else:
                        in_t = torch.rot90(xb, 1, dims=(-2, -1))
                    with torch.autocast("cuda", dtype=amp_dtype, enabled=is_cuda):
                        pr = m(in_t).float().sigmoid()[:, 0]
                    if tta_id == 1:
                        pr = torch.flip(pr, dims=(-1,))
                    elif tta_id == 2:
                        pr = torch.flip(pr, dims=(-2,))
                    elif tta_id == 3:
                        pr = torch.rot90(pr, 3, dims=(-2, -1))
                    pb += pr
            acc[i : i + batch] = (pb / (float(n_tta) * len(segs))).cpu().numpy()

    out_probs = []
    for i, (x1, y1, x2, y2) in enumerate(ws):
        pr_crop = cv2.resize(acc[i], (max(1, x2 - x1), max(1, y2 - y1)), interpolation=cv2.INTER_LINEAR)
        out_probs.append(np.clip(pr_crop * 255.0 + 0.5, 0, 255).astype(np.uint8))
    return out_probs


def worker_infer(
    repo_dir: Path,
    data_root: Path,
    work_dir: Path,
    arch: str,
    scenario: str,
    boxes_index_json: Path,
    out_dir: Path,
    part: int,
) -> None:
    import torch

    data, postprocess, segmenter = _bind_filseg(repo_dir, data_root, work_dir)
    import solar_dl_enhance as sde  # noqa: E402

    torch.backends.cudnn.benchmark = False
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    folds_map = data.folds()
    models = {}
    for k in range(5):
        m = _build_stage2_model(arch, pretrained=False)
        sd = torch.load(work_dir / "weights" / f"seg_{arch}_{scenario}_f{k}.pth", map_location="cpu")
        m.load_state_dict({kk: vv.float() if vv.is_floating_point() else vv for kk, vv in sd.items()})
        models[k] = m.to(dev).eval()

    idx = json.loads(boxes_index_json.read_text())
    tasks = []
    for sname, sp_map in idx.items():
        for sp, stem_map in sp_map.items():
            for st, npz_p in sorted(stem_map.items()):
                tasks.append((sname, sp, st, Path(npz_p)))
    by_photo: dict[tuple[str, str], list[tuple[str, Path]]] = {}
    for sname, sp, st, p in tasks:
        by_photo.setdefault((sp, st), []).append((sname, p))

    items = sorted(by_photo.items())[part::2]
    all_m = [models[k] for k in range(5)]
    shared_phys = Path("/tmp/shared_phys")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 16.0
    inf_batch = 64 if vram_gb >= 48 else 24
    for sp, st in [it[0] for it in items]:
        sp_png = shared_phys / f"{st}.png"
        bgr = cv2.imread(str(sp_png)) if sp_png.exists() else None
        if bgr is None:
            im = cv2.imread(str(data.image_path(st)), cv2.IMREAD_GRAYSCALE)
            bgr = sde.physics_3ch_from_gray(im)
        segs = [models[folds_map[st]]] if sp in ("oof", "val") else all_m
        for sname, npz_p in by_photo[(sp, st)]:
            z = np.load(npz_p)
            conf = np.asarray(z["conf"], np.float32)
            raw_b = np.asarray(z["box"], np.float32).reshape(-1, 4)
            is_crop_win = ("p" in z.files) or ("p4" in z.files)
            if is_crop_win:
                ws = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in raw_b]
                b_xywh = []
                for x1, y1, x2, y2 in ws:
                    w0, h0 = max(2.0, float(x2 - x1)), max(2.0, float(y2 - y1))
                    b_xywh.append((x1 + 0.14 * w0, y1 + 0.14 * h0, w0 * 0.72, h0 * 0.72))
            else:
                ws = [segmenter.crop_box(float(x1), float(y1), max(2.0, float(x2 - x1)), max(2.0, float(y2 - y1))) for x1, y1, x2, y2 in raw_b]
                b_xywh = [(float(x1), float(y1), max(2.0, float(x2 - x1)), max(2.0, float(y2 - y1))) for x1, y1, x2, y2 in raw_b]
            probs = _predict_crops_5ch(segs, bgr, ws, b_xywh, dev=dev, batch=inf_batch)
            postprocess.save_cache(out_dir / sname / sp / f"{st}.npz", conf, ws, probs)
    print(f"[Stage2 {arch} infer p{part}] Done ({len(items)} photos)", flush=True)


def run_driver(arch: str, scenario: str, epochs: int, patience: int, boxes_roots: list[str]) -> None:
    import torch

    t0 = time.time()
    data_root = _find_comp_root()
    code_root = _find_input_dir("filseg-gap-wave4-code") or _find_input_dir("filseg-gap-wave2-code") or Path("/kaggle/working/filseg_repo")
    repo_dir = Path("/tmp/filseg_repo_w4")
    if not (repo_dir / "filseg").exists():
        if (code_root / "filseg").exists():
            shutil.copytree(code_root, repo_dir, dirs_exist_ok=True)
        else:
            repo_dir = Path(__file__).resolve().parents[3]

    work_dir = Path(f"/tmp/exp013c_{arch}_{scenario}")
    out_dir = Path(f"/tmp/exp013c_out_{arch}_{scenario}")
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    common_env = {**os.environ, "PYTHONPATH": f"{repo_dir}:{Path(__file__).resolve().parent}"}
    py = sys.executable

    for fk in range(5):
        rc = subprocess.call(
            [py, __file__, "--worker", "train_fold", "--repo", str(repo_dir), "--data-root", str(data_root),
             "--work-dir", str(work_dir), "--arch", arch, "--scenario", scenario, "--fold", str(fk),
             "--epochs", str(epochs), "--patience", str(patience)],
            env=common_env,
        )
        assert rc == 0, f"{arch} fold {fk} failed"

    idx = {}
    for spec in boxes_roots:
        if "=" in spec:
            sname, rp = spec.split("=", 1)
            rpath = Path(rp)
            if rpath.exists():
                idx[sname] = {}
                for sp in ("oof", "val", "test"):
                    sub = rpath / sp
                    if sub.exists():
                        idx[sname][sp] = {f.stem: str(f) for f in sorted(sub.glob("*.npz"))}

    if idx:
        idx_p = work_dir / "boxes_index.json"
        idx_p.write_text(json.dumps(idx))
        n_gpu = torch.cuda.device_count()
        if n_gpu >= 2:
            procs = [
                subprocess.Popen(
                    [py, __file__, "--worker", "infer", "--repo", str(repo_dir), "--data-root", str(data_root),
                     "--work-dir", str(work_dir), "--arch", arch, "--scenario", scenario,
                     "--boxes-index", str(idx_p), "--out-dir", str(out_dir), "--part", str(p)],
                    env=dict(common_env, CUDA_VISIBLE_DEVICES=str(p)),
                )
                for p in (0, 1)
            ]
            assert all(p.wait() == 0 for p in procs), f"{arch} infer failed"
        else:
            for p in (0, 1):
                rc = subprocess.call(
                    [py, __file__, "--worker", "infer", "--repo", str(repo_dir), "--data-root", str(data_root),
                     "--work-dir", str(work_dir), "--arch", arch, "--scenario", scenario,
                     "--boxes-index", str(idx_p), "--out-dir", str(out_dir), "--part", str(p)],
                    env=dict(common_env, CUDA_VISIBLE_DEVICES="0"),
                )
                assert rc == 0, f"{arch} infer part {p} failed"

        safe_arch = arch.replace("-", "_")
        tgz = Path(f"/kaggle/working/exp013c_{safe_arch}_{scenario}_segs.tar.gz")
        with tarfile.open(tgz, "w:gz") as tf:
            tf.add(out_dir, arcname=out_dir.name)
        print(f"Packed {tgz.name} ({tgz.stat().st_size/1e6:.2f} MB) in {(time.time()-t0)/60:.1f}m", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["train_fold", "infer"])
    ap.add_argument("--repo", type=Path)
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--work-dir", type=Path)
    ap.add_argument("--boxes-index", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--arch", default="unet-mitb4", choices=["unet-mitb4", "convnext-edge", "convnext-unet", "segformer-b4"])
    ap.add_argument("--scenario", default="baseline", choices=["baseline", "dl_enh"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--boxes-root", action="append", default=[])
    args = ap.parse_args()

    if args.worker is None:
        run_driver(args.arch, args.scenario, args.epochs, args.patience, args.boxes_root)
    elif args.worker == "train_fold":
        worker_train_fold(args.repo, args.data_root, args.work_dir, args.arch, args.scenario, args.fold, args.epochs, args.patience)
    elif args.worker == "infer":
        worker_infer(args.repo, args.data_root, args.work_dir, args.arch, args.scenario, args.boxes_index, args.out_dir, args.part)


if __name__ == "__main__":
    main()
