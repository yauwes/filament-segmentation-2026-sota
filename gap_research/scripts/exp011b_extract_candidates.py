"""EXP-011b Stage 1: Fast Symmetric Track-A++ Candidate Graph & Solar-Physics Extractor.

Single-pass .npz loader for 707-OOF <-> 180-Test symmetric streams:
  - Preserves EXACT Track-A Clean `ta_base` + `ta_rescue` masks (0 diff vs Track-A Clean `0.46613`)
  - Uses Track-A's calibrated high-SQ continuous segmenter weights (`0.55 sam2, 0.25 s1e_g2, 0.10 e4, 0.10 e4x` on ms;
    `0.30 e4, 0.50 s1e_g2, 0.20 e4x` on s/main; `0.25 each` on SAHI e4/e4_g1/s1e/sam2) down to conf=0.14
  - Emits consensus variants (`cons68`, `cons62`, `cons74`), `split` (Voronoi on >0.76 cores), `merge` (collinear gaps),
    and disjoint/alt `solo` rescues
  - Extracts 69 leak-free, Fold-invariant features including SAM-vs-CNN IoU, continuous prob percentiles, rotated
    spine geometry/solidity, 1024/512 H-alpha CLV/bth/bright-plage/Hessian dark-ridge physics, and Track-A anchor context.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import concurrent.futures
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

cv2.setNumThreads(1)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "gap_research" / "scripts"))

import build_gap_submission as bgs  # noqa: E402
import build_super_submission as bss  # noqa: E402
import eval_wave23_joint as ej  # noqa: E402
from filseg import data, metric  # noqa: E402

FULL = 2048
HALF = 1024
OBS_MAP = {"Bh": 0.0, "Ch": 1.0, "Lh": 2.0, "Mh": 3.0, "Th": 4.0, "Uh": 5.0}
KIND_MAP = {
    "ta_base": 0,
    "ta_rescue": 1,
    "cons68": 2,
    "cons62": 3,
    "cons74": 4,
    "split": 5,
    "merge": 6,
    "solo": 7,
}

FEATURE_NAMES = [
    # Consensus & stream confidences (15)
    "c_tracka",
    "weighted_vote",
    "n_streams",
    "ms_conf",
    "s_conf",
    "main_conf",
    "sahi_conf",
    "max_conf",
    "min_conf_nz",
    "std_conf_nz",
    "sahi_only",
    "ms_only",
    "sam_cnn_iou",
    "ms_s_iou",
    "stream_iou_mean",
    # Continuous probability map stats (14)
    "p_mean",
    "p_q10",
    "p_q25",
    "p_q50",
    "p_q75",
    "p_q90",
    "p_std",
    "p_core",
    "p_rim",
    "p_core_rim_diff",
    "p_frac_gt65",
    "p_frac_gt75",
    "p_frac_gt85",
    "n_cc_core76",
    # Geometry & rotated morphology (13)
    "area",
    "log_area",
    "bw",
    "bh",
    "max_side",
    "rot_len",
    "rot_wid",
    "rot_elong",
    "fill",
    "solidity",
    "compact",
    "mean_width",
    "n_cc_mask",
    # Solar H-alpha disk physics at 1024/512 (16)
    "rho_c",
    "rho_max",
    "darkness_z",
    "dark_p10",
    "flat_mean",
    "flat_p10",
    "bth_mean",
    "bth_p10",
    "bth_core_skirt_diff",
    "bright_ring_z",
    "core_dark_frac",
    "hess_ridge_mean",
    "hess_ridge_p75",
    "ring_std",
    "img_mad",
    "obs_code",
    # Graph & anchor context (11)
    "kind_id",
    "is_ta_base",
    "is_ta_rescue",
    "iou_with_ta",
    "c_vs_ta",
    "n_cands_img",
    "sum_c_img",
    "rank_c_img",
    "ov_cnt",
    "ov_max",
    "near_ta_dist",
]

_KERNEL_BTH = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
_KERNEL_3 = np.ones((3, 3), np.uint8)
_KERNEL_7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
_SPECS_CACHE: dict[str, dict] = {}
_OCC_BUF = np.zeros((FULL, FULL), dtype=bool)
_TA_BM_BUF = np.zeros((FULL, FULL), dtype=bool)
_PMAP_U8_BUF = np.zeros((FULL, FULL), dtype=np.uint8)
_HALF_Y, _HALF_X = np.mgrid[:HALF, :HALF]
_HALF_Y = _HALF_Y.astype(np.float32)
_HALF_X = _HALF_X.astype(np.float32)


def _get_split_specs(split: str) -> dict:
    """Return symmetric OOF<->Test specs matching exact Track-A Clean segmenter weights."""
    if split in _SPECS_CACHE:
        return _SPECS_CACHE[split]
    e6 = ej.WAVE2_ROOT / "exp006"
    e7 = ej.WAVE2_ROOT / "exp007"
    sp_dir = "oof" if split in ("oof", "val") else "test"
    ta_streams = {
        "ms": (
            1.35,
            0.28,
            [
                ("e4", REPO / "work" / "cache" / "ms" / sp_dir, 0.10, 4),
                ("sam2", e6 / "sam2" / "ms" / sp_dir, 0.55, 4),
                ("s1e_g2", e6 / "s1e_g2" / "ms" / sp_dir, 0.25, 4),
                ("e4x", e6 / "e4x" / "e4x" / "ms" / sp_dir, 0.10, 4),
            ],
        ),
        "s": (
            1.05,
            0.28,
            [
                ("e4", REPO / "work" / "cache" / "s" / sp_dir, 0.30, 4),
                ("s1e_g2", e6 / "s1e_g2" / "s" / sp_dir, 0.50, 4),
                ("e4x", e6 / "e4x" / "e4x" / "s" / sp_dir, 0.20, 4),
            ],
        ),
        "main": (
            0.95,
            0.25,
            [
                ("e4", REPO / "work" / "cache" / "main" / sp_dir, 0.30, 4),
                ("s1e_g2", e6 / "s1e_g2" / "main" / sp_dir, 0.50, 4),
                ("e4x", e6 / "e4x" / "e4x" / "main" / sp_dir, 0.20, 4),
            ],
        ),
    }
    sahi_specs = [
        ("e4", e7 / "sahi_e4" / sp_dir, 0.25, 4),
        ("e4_g1", e7 / "sahi_e4_g1" / sp_dir, 0.25, 4),
        ("s1e", e7 / "sahi_s1e" / sp_dir, 0.25, 4),
        ("sam2", e7 / "sahi_sam2" / sp_dir, 0.25, 4),
    ]
    _SPECS_CACHE[split] = {"ta_streams": ta_streams, "sahi": (1.15, 0.20, sahi_specs)}
    return _SPECS_CACHE[split]


def _unpack_npz(path: Path, tta_prefer: int = 4) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    with np.load(path) as d:
        conf = np.asarray(d["conf"], np.float32)
        box = np.asarray(d["box"], np.int32).reshape(-1, 4)
        if "p" in d:
            flat = np.asarray(d["p"], np.uint8)
        elif tta_prefer == 4 and "p4" in d:
            flat = np.asarray(d["p4"], np.uint8)
        elif "p1" in d:
            flat = np.asarray(d["p1"], np.uint8)
        else:
            flat = np.asarray(d["p4"], np.uint8)
    probs = []
    off = 0
    for x1, y1, x2, y2 in box:
        w, h = max(0, int(x2 - x1)), max(0, int(y2 - y1))
        sz = w * h
        probs.append(flat[off : off + sz].reshape(h, w) if sz > 0 else np.zeros((0, 0), np.uint8))
        off += sz
    return conf, box, probs


def _load_blended_crops_once(
    specs: list[tuple[str, Path, float, int]],
    stem: str,
    min_raw_floor: float = 0.12,
    small_scale: float = 1.2,
    small_px: int = 120,
    track_sam_cnn: bool = False,
    clamp_to_primary: bool = False,
) -> list[dict]:
    """Read each segmenter .npz ONCE per stem and return raw blended continuous crops sorted by -conf0."""
    loaded = []
    for sname, sdir, w, tta in specs:
        if w <= 0:
            continue
        p = sdir / f"{stem}.npz"
        if not p.exists():
            continue
        conf, box, probs = _unpack_npz(p, tta_prefer=tta)
        loaded.append((sname, float(w), conf, box, probs))
    if not loaded:
        return []
    _, _, conf0, box0, _ = loaded[0]
    n_boxes = len(conf0)
    if n_boxes == 0:
        return []

    crops = []
    for idx in np.argsort(-conf0):
        raw_c = float(conf0[idx])
        bx0, by0, bx1, by1 = [int(v) for v in box0[idx]]
        span = max(bx1 - bx0, by1 - by0)
        eff_c = raw_c * (small_scale if span < small_px else 1.0)
        if eff_c < min_raw_floor and raw_c < min_raw_floor:
            continue

        gx1, gy1, gx2, gy2 = bx0, by0, bx1, by1
        valid = []
        for sname, w, _, b_arr, p_list in loaded:
            if idx >= len(b_arr):
                continue
            sb = b_arr[idx]
            if sb[2] <= sb[0] or sb[3] <= sb[1]:
                continue
            if not clamp_to_primary:
                gx1 = min(gx1, int(sb[0]))
                gy1 = min(gy1, int(sb[1]))
                gx2 = max(gx2, int(sb[2]))
                gy2 = max(gy2, int(sb[3]))
            valid.append((sname, w, sb, p_list[idx]))
        if not valid or gx2 <= gx1 or gy2 <= gy1:
            continue

        uh, uw = gy2 - gy1, gx2 - gx1
        if (
            len(valid) == 1
            and tuple(int(v) for v in valid[0][2]) == (gx1, gy1, gx2, gy2)
            and valid[0][3].shape == (uh, uw)
        ):
            prob_u8 = valid[0][3]
            sam_cnn = 0.0
        else:
            acc = np.zeros((uh, uw), dtype=np.float32)
            wsum = np.zeros((uh, uw), dtype=np.float32)
            if track_sam_cnn:
                sam_acc = np.zeros((uh, uw), dtype=np.float32)
                sam_wsum = np.zeros((uh, uw), dtype=np.float32)
                cnn_acc = np.zeros((uh, uw), dtype=np.float32)
                cnn_wsum = np.zeros((uh, uw), dtype=np.float32)
            for sname, w, sb, pr in valid:
                sx1, sy1, sx2, sy2 = int(sb[0]), int(sb[1]), int(sb[2]), int(sb[3])
                if pr.shape != (sy2 - sy1, sx2 - sx1):
                    pr = cv2.resize(pr, (sx2 - sx1, sy2 - sy1), interpolation=cv2.INTER_LINEAR)
                ix1, iy1, ix2, iy2 = max(gx1, sx1), max(gy1, sy1), min(gx2, sx2), min(gy2, sy2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                ys_sl = slice(iy1 - gy1, iy2 - gy1)
                xs_sl = slice(ix1 - gx1, ix2 - gx1)
                pr_sl = pr[iy1 - sy1 : iy2 - sy1, ix1 - sx1 : ix2 - sx1]
                pf = (w / 255.0) * pr_sl.astype(np.float32)
                acc[ys_sl, xs_sl] += pf
                wsum[ys_sl, xs_sl] += w
                if track_sam_cnn:
                    if "sam2" in sname:
                        sam_acc[ys_sl, xs_sl] += pf
                        sam_wsum[ys_sl, xs_sl] += w
                    elif "e4" in sname or "s1e" in sname:
                        cnn_acc[ys_sl, xs_sl] += pf
                        cnn_wsum[ys_sl, xs_sl] += w
            if not (wsum > 0).any():
                continue
            prob_u8 = np.where(wsum > 0, (acc / np.maximum(wsum, 1e-6)) * 255.0 + 0.5, 0.0).astype(np.uint8)
            sam_cnn = 0.0
            if track_sam_cnn and sam_wsum.max() > 0 and cnn_wsum.max() > 0:
                sm = np.where(sam_wsum > 0, sam_acc / np.maximum(sam_wsum, 1e-6), 0.0) > 0.65
                cm = np.where(cnn_wsum > 0, cnn_acc / np.maximum(cnn_wsum, 1e-6), 0.0) > 0.65
                un = int((sm | cm).sum())
                if un > 0:
                    sam_cnn = float((sm & cm).sum()) / float(un)

        crops.append({
            "raw_c": raw_c,
            "eff_c": eff_c,
            "box": (gx1, gy1, gx2, gy2),
            "prob_u8": prob_u8,
            "sam_cnn": sam_cnn,
        })
    return crops


def _paint_exact_from_crops(
    crops: list[dict],
    thr: float,
    conf_floor: float,
    drop: float,
    min_area: int,
    use_raw_c: bool = False,
) -> list[tuple[float, tuple[int, int, int, int], np.ndarray]]:
    """Exact in-memory match to load_blended_clamped (use_raw_c=False) or load_blended_stream (use_raw_c=True)."""
    if not crops:
        return []
    t_u8 = int(thr * 255.0 + 0.5)
    occ = _OCC_BUF
    occ.fill(False)
    out = []
    # Match ordering: conf0 descending (crops already sorted by -conf0)
    for cr in crops:
        c = cr["raw_c"] if use_raw_c else cr["eff_c"]
        if c < conf_floor:
            continue
        gx1, gy1, gx2, gy2 = cr["box"]
        m = cr["prob_u8"] > t_u8
        msum = int(m.sum())
        if msum == 0:
            continue
        keep = m & ~occ[gy1:gy2, gx1:gx2]
        k = int(keep.sum())
        if (k > min_area if use_raw_c else k >= min_area) and k > 0 and k >= drop * msum:
            occ[gy1:gy2, gx1:gx2] |= keep
            v = bss.trim(c, (gx1, gy1, gx2, gy2), keep)
            if v is not None:
                out.append(v)
    return out


def _paint_continuous_from_crops(crops: list[dict], conf_floor: float = 0.14, drop: float = 0.88, min_area: int = 80) -> list[dict]:
    """Deduplicate crops at m62 while carrying continuous prob_u8 for clustering."""
    if not crops:
        return []
    t62 = int(0.62 * 255.0 + 0.5)
    t68 = int(0.68 * 255.0 + 0.5)
    occ = _OCC_BUF
    occ.fill(False)
    out = []
    for cr in sorted(crops, key=lambda x: -x["eff_c"]):
        if cr["eff_c"] < conf_floor:
            continue
        gx1, gy1, gx2, gy2 = cr["box"]
        prob_u8 = cr["prob_u8"]
        m62 = prob_u8 > t62
        msum = int(m62.sum())
        if msum < min_area:
            continue
        keep = m62 & ~occ[gy1:gy2, gx1:gx2]
        k = int(keep.sum())
        if k >= min_area and k >= drop * msum:
            occ[gy1:gy2, gx1:gx2] |= keep
            ys, xs = np.nonzero(keep)
            tx1, ty1, tx2, ty2 = gx1 + int(xs.min()), gy1 + int(ys.min()), gx1 + int(xs.max()) + 1, gy1 + int(ys.max()) + 1
            ys_s = slice(ty1 - gy1, ty2 - gy1)
            xs_s = slice(tx1 - gx1, tx2 - gx1)
            kp = keep[ys_s, xs_s]
            pu = prob_u8[ys_s, xs_s]
            out.append({
                "c": cr["eff_c"],
                "box": (tx1, ty1, tx2, ty2),
                "p": pu,
                "m62": kp,
                "m68": (pu > t68) & kp,
                "area": k,
                "sam_cnn": cr["sam_cnn"],
            })
    return out


def _crop_iou(ba: tuple[int, int, int, int], ma: np.ndarray, bb: tuple[int, int, int, int], mb: np.ndarray) -> tuple[float, int]:
    ax1, ay1, ax2, ay2 = ba
    bx1, by1, bx2, by2 = bb
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0, 0
    inter = int((ma[iy1 - ay1 : iy2 - ay1, ix1 - ax1 : ix2 - ax1] & mb[iy1 - by1 : iy2 - by1, ix1 - bx1 : ix2 - bx1]).sum())
    if inter == 0:
        return 0.0, 0
    return inter / float(int(ma.sum()) + int(mb.sum()) - inter), inter


def _trim_cand(box: tuple[int, int, int, int], m: np.ndarray, min_area: int = 90) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    ys, xs = np.nonzero(m)
    if len(ys) < min_area:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bx1, by1, _, _ = box
    return (bx1 + x0, by1 + y0, bx1 + x1, by1 + y1), m[y0:y1, x0:x1]


def _compute_img_physics(stem: str) -> dict:
    p = data.image_path(stem)
    im = cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_2)  # 1024x1024
    if im is None:
        raise FileNotFoundError(f"Missing image {p}")
    q99 = float(np.percentile(im, 99))
    ys, xs = np.nonzero(im > 0.25 * q99)
    if len(xs) > 100:
        cx, cy = float(xs.mean()), float(ys.mean())
        R = float(0.99 * np.sqrt(len(xs) / np.pi))
    else:
        cx, cy, R = 512.0, 512.0, 460.0
    disk = np.hypot(_HALF_Y - cy, _HALF_X - cx) < R

    xf = im.astype(np.float32)
    # 256x256 GaussianBlur(sigma=6.25) -> 1024x1024 is >99.99% identical to full 1024 sigma=25.0 and 50x faster
    sm256 = cv2.resize(xf, (256, 256), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.GaussianBlur(sm256, (0, 0), 6.25), (HALF, HALF), interpolation=cv2.INTER_LINEAR)
    flat = xf - bg
    bth = -cv2.morphologyEx(im, cv2.MORPH_BLACKHAT, _KERNEL_BTH).astype(np.float32)

    vf = flat[disk][::8]
    med_f = float(np.median(vf))
    mad_f = 1.4826 * float(np.median(np.abs(vf - med_f))) + 1e-6
    vb = bth[disk][::8]
    med_b = float(np.median(vb))
    mad_b = 1.4826 * float(np.median(np.abs(vb - med_b))) + 1e-6

    flat_z = np.where(disk, np.clip((flat - med_f) / mad_f, -12.0, 12.0), 0.0).astype(np.float32)
    bth_z = np.where(disk, np.clip((bth - med_b) / mad_b, -12.0, 12.0), 0.0).astype(np.float32)

    sm = cv2.resize(flat_z, (512, 512), interpolation=cv2.INTER_AREA)
    ridge = np.zeros((512, 512), dtype=np.float32)
    for sig in (1.2, 2.4):
        bl = cv2.GaussianBlur(sm, (0, 0), sig)
        ixx = cv2.Sobel(bl, cv2.CV_32F, 2, 0, ksize=3) * (sig * sig)
        iyy = cv2.Sobel(bl, cv2.CV_32F, 0, 2, ksize=3) * (sig * sig)
        ixy = cv2.Sobel(bl, cv2.CV_32F, 1, 1, ksize=3) * (sig * sig)
        tr = ixx + iyy
        det_term = np.sqrt(np.maximum(0.0, (ixx - iyy) ** 2 + 4.0 * ixy * ixy))
        l1 = 0.5 * (tr + det_term)
        l2 = 0.5 * (tr - det_term)
        ratio_guard = np.maximum(0.0, 1.0 - np.abs(l2) / (np.abs(l1) + 1e-3))
        ridge = np.maximum(ridge, np.maximum(0.0, l1) * ratio_guard)
    ridge_1024 = cv2.resize(ridge, (HALF, HALF), interpolation=cv2.INTER_LINEAR)

    return {
        "cx": cx * 2.0,
        "cy": cy * 2.0,
        "R": R * 2.0,
        "mad": mad_f,
        "flat_z": flat_z,
        "bth_z": bth_z,
        "ridge": ridge_1024,
    }


def _extract_stem(args: tuple[str, str, int, list]) -> dict:
    stem, split, fold, stem_gt = args
    sp = _get_split_specs(split)

    # 1. Single-pass loading of all 4 symmetric Track-A streams
    raw_crops = {}
    ta_stream_insts = []
    for sname in ("ms", "s", "main"):
        wt, cf_ta, specs = sp["ta_streams"][sname]
        crops = _load_blended_crops_once(
            specs, stem, min_raw_floor=0.14, track_sam_cnn=(sname == "ms"), clamp_to_primary=True
        )
        raw_crops[sname] = (wt, crops)
        insts = _paint_exact_from_crops(crops, thr=0.68, conf_floor=cf_ta, drop=0.90, min_area=100, use_raw_c=False)
        ta_stream_insts.append((sname, wt, insts))

    sahi_wt, _, sahi_specs = sp["sahi"]
    sahi_crops = _load_blended_crops_once(sahi_specs, stem, min_raw_floor=0.14, track_sam_cnn=True)
    raw_crops["SAHI"] = (sahi_wt, sahi_crops)

    # Exact Track-A consensus RLEs
    ta_base_rles = bgs.weighted_consensus_stem(ta_stream_insts, match_iou=0.40, vote_thr=0.52, c_floor=0.28, min_area=100, drop=0.90)
    ta_bm = _TA_BM_BUF
    ta_bm.fill(False)
    ta_base_items = []
    for rstr in ta_base_rles:
        mf = mu.decode({"size": [FULL, FULL], "counts": rstr.encode("ascii")}).astype(bool)
        ys, xs = np.nonzero(mf)
        if not len(ys):
            continue
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        cm = mf[y1:y2, x1:x2]
        ta_bm[y1:y2, x1:x2] |= cm
        ta_base_items.append(((x1, y1, x2, y2), cm, rstr))

    # Exact Track-A SAHI disjoint rescue matching eval_wave23_joint._append_disjoint_rescue
    # NOTE: bgs.load_blended_stream drops strictly via area > min_area (`k > 100`), use_raw_c=True, thr=0.68, conf_floor=0.20
    sahi_dec = _paint_exact_from_crops(sahi_crops, thr=0.68, conf_floor=0.20, drop=0.90, min_area=100, use_raw_c=True)
    ta_rescue_items = []
    for c, (x1, y1, x2, y2), m in sahi_dec:
        if c < 0.22:
            continue
        ma = int(m.sum())
        if ma < 100 or ma > 400:
            continue
        if int((m & ta_bm[y1:y2, x1:x2]).sum()) / ma <= 0.15:
            keep = m & ~ta_bm[y1:y2, x1:x2]
            ka = int(keep.sum())
            if ka >= 100 and ka >= 0.90 * ma:
                ta_bm[y1:y2, x1:x2] |= keep
                rstr = bss.encode_crop_ascii((x1, y1, x2, y2), keep)
                trimmed = _trim_cand((x1, y1, x2, y2), keep, min_area=100)
                if trimmed is not None:
                    ta_rescue_items.append((trimmed[0], trimmed[1], rstr, float(c)))

    # 2. Continuous low-floor stream items across ALL 4 streams (ms, s, main, SAHI)
    total_w = sum(w for w, _ in raw_crops.values())
    cont_streams = {
        sname: _paint_continuous_from_crops(crops, conf_floor=0.14, drop=0.88, min_area=80)
        for sname, (_, crops) in raw_crops.items()
    }

    flat_items = []
    for sname, (sw, _) in raw_crops.items():
        for it in cont_streams[sname]:
            flat_items.append((sname, sw, it))
    flat_items.sort(key=lambda x: -(x[2]["c"] * min(1.5, x[1])))

    clusters = []
    for sname, sw, it in flat_items:
        box, m = it["box"], it["m62"]
        best_j, best_iou = -1, 0.0
        for j, cl in enumerate(clusters):
            if sname in cl["seen"]:
                continue
            bx1, by1, bx2, by2 = box
            cx1, cy1, cx2, cy2 = cl["ubox"]
            if min(bx2, cx2) <= max(bx1, cx1) or min(by2, cy2) <= max(by1, cy1):
                continue
            iou, _ = _crop_iou(box, m, cl["rbox"], cl["rmask"])
            if iou >= 0.40 and iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j < 0:
            clusters.append({
                "seen": {sname},
                "items": [(sname, sw, it)],
                "rbox": box,
                "rmask": m,
                "ubox": [box[0], box[1], box[2], box[3]],
            })
        else:
            cl = clusters[best_j]
            cl["seen"].add(sname)
            cl["items"].append((sname, sw, it))
            ub = cl["ubox"]
            ub[0] = min(ub[0], box[0])
            ub[1] = min(ub[1], box[1])
            ub[2] = max(ub[2], box[2])
            ub[3] = max(ub[3], box[3])

    for cl in clusters:
        gx1, gy1, gx2, gy2 = cl["ubox"]
        uh, uw = gy2 - gy1, gx2 - gx1
        pacc = np.zeros((uh, uw), dtype=np.float32)
        pwsum = np.zeros((uh, uw), dtype=np.float32)
        vacc68 = np.zeros((uh, uw), dtype=np.float32)
        vacc62 = np.zeros((uh, uw), dtype=np.float32)
        w_present = 0.0
        sum_wc = 0.0
        sconf = {"ms": 0.0, "s": 0.0, "main": 0.0, "SAHI": 0.0}
        sc_vals = []
        sm_masks = {}
        for sname, sw, it in cl["items"]:
            bx1, by1, bx2, by2 = it["box"]
            ys_s = slice(by1 - gy1, by2 - gy1)
            xs_s = slice(bx1 - gx1, bx2 - gx1)
            pacc[ys_s, xs_s] += sw * (it["p"].astype(np.float32) / 255.0)
            pwsum[ys_s, xs_s] += sw
            vacc68[ys_s, xs_s] += sw * it["m68"].astype(np.float32)
            vacc62[ys_s, xs_s] += sw * it["m62"].astype(np.float32)
            w_present += sw
            sum_wc += sw * it["c"]
            sconf[sname] = max(sconf[sname], it["c"])
            if it["sam_cnn"] > 0:
                sc_vals.append(it["sam_cnn"])
            sm_masks[sname] = (it["box"], it["m62"])

        pmap = np.where(pwsum > 0, pacc / np.maximum(pwsum, 1e-6), 0.0).astype(np.float32)
        vmap68 = vacc68 / max(w_present, 1e-6)
        vmap62 = vacc62 / max(w_present, 1e-6)
        confs_nz = [v for v in sconf.values() if v > 0]
        ms_s_iou = _crop_iou(sm_masks["ms"][0], sm_masks["ms"][1], sm_masks["s"][0], sm_masks["s"][1])[0] if ("ms" in sm_masks and "s" in sm_masks) else 0.0
        pair_ious = []
        skeys = list(sm_masks.keys())
        for a in range(len(skeys)):
            for b in range(a + 1, len(skeys)):
                pair_ious.append(_crop_iou(sm_masks[skeys[a]][0], sm_masks[skeys[a]][1], sm_masks[skeys[b]][0], sm_masks[skeys[b]][1])[0])

        cl["meta"] = {
            "box": (gx1, gy1, gx2, gy2),
            "pmap": pmap,
            "vmap68": vmap68,
            "vmap62": vmap62,
            "c_tracka": sum_wc / total_w,
            "w_vote": w_present / total_w,
            "n_streams": len(cl["seen"]),
            "sconf": sconf,
            "max_conf": max(confs_nz) if confs_nz else 0.0,
            "min_conf_nz": min(confs_nz) if confs_nz else 0.0,
            "std_conf_nz": float(np.std(confs_nz)) if len(confs_nz) > 1 else 0.0,
            "sahi_only": 1.0 if (sconf["SAHI"] > 0 and max(sconf["ms"], sconf["s"], sconf["main"]) == 0) else 0.0,
            "ms_only": 1.0 if (sconf["ms"] > 0 and max(sconf["s"], sconf["main"], sconf["SAHI"]) == 0) else 0.0,
            "sam_cnn_iou": float(np.mean(sc_vals)) if sc_vals else 0.0,
            "ms_s_iou": float(ms_s_iou),
            "stream_iou_mean": float(np.mean(pair_ious)) if pair_ious else 0.0,
        }

    # Lightweight uint8 max-envelope probability map (reused buffer)
    full_pmap_u8 = _PMAP_U8_BUF
    full_pmap_u8.fill(0)
    for cl in clusters:
        gx1, gy1, gx2, gy2 = cl["meta"]["box"]
        pu8 = np.clip(cl["meta"]["pmap"] * 255.0 + 0.5, 0, 255).astype(np.uint8)
        full_pmap_u8[gy1:gy2, gx1:gx2] = np.maximum(full_pmap_u8[gy1:gy2, gx1:gx2], pu8)

    # 3. Emit candidates
    cands = []

    def _match_cluster(box: tuple[int, int, int, int], m: np.ndarray) -> dict | None:
        best_cl, best_iou = None, 0.15
        for cl in clusters:
            cb = cl["meta"]["box"]
            if min(box[2], cb[2]) <= max(box[0], cb[0]) or min(box[3], cb[3]) <= max(box[1], cb[1]):
                continue
            iou, _ = _crop_iou(box, m, cl["rbox"], cl["rmask"])
            if iou > best_iou:
                best_iou = iou
                best_cl = cl
        return best_cl

    def _default_meta(c_val: float, sname: str = "SAHI") -> dict:
        sc = {"ms": 0.0, "s": 0.0, "main": 0.0, "SAHI": 0.0}
        sc[sname] = c_val
        return {
            "c_tracka": c_val,
            "w_vote": 0.25,
            "n_streams": 1,
            "sconf": sc,
            "max_conf": c_val,
            "min_conf_nz": c_val,
            "std_conf_nz": 0.0,
            "sahi_only": 1.0 if sname == "SAHI" else 0.0,
            "ms_only": 1.0 if sname == "ms" else 0.0,
            "sam_cnn_iou": 0.0,
            "ms_s_iou": 0.0,
            "stream_iou_mean": 0.0,
        }

    def _add_cand(box: tuple[int, int, int, int], m: np.ndarray, kind: str, meta: dict, rle_override: str | None = None, dedup_thr: float = 0.94) -> bool:
        trimmed = _trim_cand(box, m, min_area=90)
        if trimmed is None:
            return False
        tb, tm = trimmed
        if dedup_thr <= 1.0:
            for ex in cands:
                eb = ex["box"]
                if min(tb[2], eb[2]) <= max(tb[0], eb[0]) or min(tb[3], eb[3]) <= max(tb[1], eb[1]):
                    continue
                iou, _ = _crop_iou(tb, tm, eb, ex["mask"])
                if iou >= dedup_thr:
                    return False
        rstr = rle_override if rle_override is not None else bss.encode_crop_ascii(tb, tm)
        cands.append({"box": tb, "mask": tm, "rle": rstr, "kind": kind, "meta": meta})
        return True

    # 3a. Exact Track-A Base & Rescue anchors
    for tb, tm, rstr in ta_base_items:
        cl = _match_cluster(tb, tm)
        meta = cl["meta"] if cl is not None else _default_meta(0.28, "ms")
        _add_cand(tb, tm, "ta_base", meta, rle_override=rstr, dedup_thr=1.01)

    for tb, tm, rstr, c_raw in ta_rescue_items:
        cl = _match_cluster(tb, tm)
        if cl is not None:
            meta = cl["meta"]
        else:
            meta = _default_meta(float(c_raw) * 1.15 / total_w, "SAHI")
            meta["sconf"]["SAHI"] = float(c_raw)
        _add_cand(tb, tm, "ta_rescue", meta, rle_override=rstr, dedup_thr=1.01)

    # 3b. Consensus variants & Voronoi core splits
    cons_for_merge = []
    for cl in clusters:
        cm = cl["meta"]
        gx1, gy1, gx2, gy2 = cm["box"]
        pmap, v68, v62 = cm["pmap"], cm["vmap68"], cm["vmap62"]

        m68 = v68 >= 0.52
        if int(m68.sum()) >= 90:
            _add_cand((gx1, gy1, gx2, gy2), m68, "cons68", cm, dedup_thr=0.94)
            t68 = _trim_cand((gx1, gy1, gx2, gy2), m68, min_area=90)
            if t68 is not None and cm["c_tracka"] >= 0.15:
                cons_for_merge.append((t68[0], t68[1], cm))

        m62 = (pmap >= 0.56) | (v62 >= 0.52)
        if int(m62.sum()) >= 90:
            _add_cand((gx1, gy1, gx2, gy2), m62, "cons62", cm, dedup_thr=0.93)

        m74 = pmap >= 0.72
        if int(m74.sum()) >= 90:
            _add_cand((gx1, gy1, gx2, gy2), m74, "cons74", cm, dedup_thr=0.93)

        base_m = m68 if int(m68.sum()) >= 180 else m62
        if int(base_m.sum()) >= 180:
            core = (pmap > 0.76) & base_m
            n_cc, lbl, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), connectivity=8)
            big_ids = [i for i in range(1, n_cc) if stats[i, cv2.CC_STAT_AREA] >= 55]
            if 2 <= len(big_ids) <= 4:
                dists = [cv2.distanceTransform((lbl != i).astype(np.uint8), cv2.DIST_L2, 3) for i in big_ids]
                assign = np.argmin(np.stack(dists, axis=0), axis=0)
                for k_idx in range(len(big_ids)):
                    child = base_m & (assign == k_idx)
                    if int(child.sum()) >= 90:
                        _add_cand((gx1, gy1, gx2, gy2), child, "split", cm, dedup_thr=0.93)

    # 3c. Single-stream disjoint / low-conf rescue
    for sname, (sw, _) in raw_crops.items():
        for it in cont_streams[sname]:
            if it["c"] < 0.14:
                continue
            m = it["p"] > int(0.66 * 255 + 0.5)
            if int(m.sum()) < 90:
                continue
            cl = _match_cluster(it["box"], m)
            meta = cl["meta"] if cl is not None else _default_meta(it["c"] * sw / total_w, sname)
            _add_cand(it["box"], m, "solo", meta, dedup_thr=0.90)

    # 3d. Collinear near-gap merge candidates
    merge_pool = cons_for_merge[:24]
    for i in range(len(merge_pool)):
        ba, ma, cma = merge_pool[i]
        ax1, ay1, ax2, ay2 = ba
        for j in range(i + 1, len(merge_pool)):
            bb, mb, cmb = merge_pool[j]
            bx1, by1, bx2, by2 = bb
            gx = max(0, max(ax1, bx1) - min(ax2, bx2))
            gy = max(0, max(ay1, by1) - min(ay2, by2))
            if max(gx, gy) > 16:
                continue
            ux1, uy1, ux2, uy2 = min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)
            if max(ux2 - ux1, uy2 - uy1) > 650:
                continue
            um = np.zeros((uy2 - uy1, ux2 - ux1), dtype=bool)
            um[ay1 - uy1 : ay2 - uy1, ax1 - ux1 : ax2 - ux1] |= ma
            um[by1 - uy1 : by2 - uy1, bx1 - ux1 : bx2 - ux1] |= mb
            dil = cv2.dilate(um.astype(np.uint8), _KERNEL_7) > 0
            n_before = cv2.connectedComponents(um.astype(np.uint8))[0]
            n_after = cv2.connectedComponents(dil.astype(np.uint8))[0]
            if n_before > 2 and n_after < n_before:
                um = um | (dil & (full_pmap_u8[uy1:uy2, ux1:ux2] > 90))
            merged_meta = cma if cma["c_tracka"] >= cmb["c_tracka"] else cmb
            _add_cand((ux1, uy1, ux2, uy2), um, "merge", merged_meta, dedup_thr=0.92)

    # 4. Exact Physics & Features
    if not cands:
        return {
            "stem": stem,
            "fold": fold,
            "obs": stem[-2:],
            "r_count": len(stem_gt),
            "rles": [],
            "kinds": [],
            "boxes": np.zeros((0, 4), dtype=np.int32),
            "areas": np.zeros(0, dtype=np.int32),
            "X": np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32),
            "h": np.zeros(0, dtype=np.float32),
            "m": np.zeros(0, dtype=np.float32),
            "u": np.zeros(0, dtype=np.float32),
            "edges": [],
            "patches": np.zeros((0, 4, 48, 48), dtype=np.uint8),
        }

    phys = _compute_img_physics(stem)
    ta_cands = [c for c in cands if c["kind"] in ("ta_base", "ta_rescue")]

    n = len(cands)
    n_cands_img = float(n)
    sum_c_img = float(sum(c["meta"]["c_tracka"] for c in cands))
    c_arr = np.array([c["meta"]["c_tracka"] for c in cands], dtype=np.float32)
    ranks = np.argsort(np.argsort(-c_arr)).astype(np.float32)

    ov_cnt = np.zeros(n, dtype=np.float32)
    ov_max = np.zeros(n, dtype=np.float32)
    edges = []
    areas_arr = np.array([int(c["mask"].sum()) for c in cands], dtype=np.int32)
    for i in range(n):
        bi, mi = cands[i]["box"], cands[i]["mask"]
        for j in range(i + 1, n):
            bj, mj = cands[j]["box"], cands[j]["mask"]
            if min(bi[2], bj[2]) <= max(bi[0], bj[0]) or min(bi[3], bj[3]) <= max(bi[1], bj[1]):
                continue
            iou, inter = _crop_iou(bi, mi, bj, mj)
            if inter > 0:
                if iou > 0.05:
                    ov_cnt[i] += 1.0
                    ov_cnt[j] += 1.0
                    if iou > ov_max[i]:
                        ov_max[i] = iou
                    if iou > ov_max[j]:
                        ov_max[j] = iou
                inter_frac = inter / float(max(1, min(int(areas_arr[i]), int(areas_arr[j]))))
                edges.append((i, j, float(inter_frac), float(iou)))

    obs_code = OBS_MAP.get(stem[-2:], -1.0)
    X = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    patches = np.zeros((n, 4, 48, 48), dtype=np.uint8)
    for i, cd in enumerate(cands):
        x1, y1, x2, y2 = cd["box"]
        msk = cd["mask"]
        area = float(areas_arr[i])
        meta = cd["meta"]
        bw, bh = float(x2 - x1), float(y2 - y1)

        pcrop = full_pmap_u8[y1:y2, x1:x2].astype(np.float32) / 255.0
        pv = pcrop[msk]
        if len(pv) == 0:
            pv = np.array([0.5], dtype=np.float32)
        pq = np.percentile(pv, [10, 25, 50, 75, 90])
        ero = cv2.erode(msk.astype(np.uint8), _KERNEL_3) > 0
        rim = msk & ~ero
        p_core = float(pcrop[ero].mean()) if ero.any() else float(pv.mean())
        p_rim = float(pcrop[rim].mean()) if rim.any() else float(pv.mean())
        core76 = (pcrop > 0.76) & msk
        n_cc76_raw, _, st76, _ = cv2.connectedComponentsWithStats(core76.astype(np.uint8), connectivity=8)
        n_cc_core76 = float(sum(1 for k_id in range(1, n_cc76_raw) if st76[k_id, cv2.CC_STAT_AREA] >= 40))

        m_u8 = msk.astype(np.uint8)
        n_cc_m = float(cv2.connectedComponents(m_u8, connectivity=8)[0] - 1)
        cnts, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            pts = np.vstack(cnts)
            (_, (rw, rh), _) = cv2.minAreaRect(pts)
            rot_len = float(max(rw, rh, 1.0))
            rot_wid = float(max(1.0, min(rw, rh)))
            hull_area = float(max(area, cv2.contourArea(cv2.convexHull(pts))))
            perim = float(sum(cv2.arcLength(ct, True) for ct in cnts))
        else:
            rot_len, rot_wid, hull_area, perim = max(bw, bh), min(bw, bh), area, 2.0 * (bw + bh)
        solidity = area / max(1.0, hull_area)
        compact = (perim * perim) / max(1.0, 4.0 * np.pi * area)
        mean_width = (2.0 * area) / max(1.0, perim)

        hx1, hy1 = max(0, x1 // 2), max(0, y1 // 2)
        hx2, hy2 = min(HALF, max(hx1 + 1, (x2 + 1) // 2)), min(HALF, max(hy1 + 1, (y2 + 1) // 2))
        mh = cv2.resize(m_u8, (hx2 - hx1, hy2 - hy1), interpolation=cv2.INTER_NEAREST) > 0
        if not mh.any():
            mh[0, 0] = True
        dil_h = cv2.dilate(mh.astype(np.uint8), _KERNEL_7) > 0
        skirt_h = dil_h & ~mh

        fz = phys["flat_z"][hy1:hy2, hx1:hx2]
        bz = phys["bth_z"][hy1:hy2, hx1:hx2]
        rz = phys["ridge"][hy1:hy2, hx1:hx2]

        f_m = fz[mh]
        b_m = bz[mh]
        r_m = rz[mh]
        if skirt_h.any():
            f_s = fz[skirt_h]
            b_s = bz[skirt_h]
            q_loc = float(np.median(f_s))
            bright_ring_z = float(np.percentile(f_s, 90))
            bth_cs_diff = float(b_m.mean() - b_s.mean())
            ring_std = float(f_s.std())
        else:
            q_loc, bright_ring_z, bth_cs_diff, ring_std = 0.0, 0.0, 0.0, 0.0

        cx_p, cy_p = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        rho_c = float(np.hypot(cx_p - phys["cx"], cy_p - phys["cy"]) / max(1.0, phys["R"]))
        rho_max = float(max(np.hypot(cx - phys["cx"], cy - phys["cy"]) / max(1.0, phys["R"]) for cx in (x1, x2) for cy in (y1, y2)))

        iou_ta, c_ta, near_d = 0.0, 0.0, 999.0
        if cd["kind"] in ("ta_base", "ta_rescue"):
            iou_ta, c_ta, near_d = 1.0, meta["c_tracka"], 0.0
        elif ta_cands:
            for tac in ta_cands:
                tb = tac["box"]
                d = float(np.hypot(max(0, max(x1, tb[0]) - min(x2, tb[2])), max(0, max(y1, tb[1]) - min(y2, tb[3]))))
                if d < near_d:
                    near_d = d
                if d == 0.0:
                    tiou, _ = _crop_iou(cd["box"], msk, tb, tac["mask"])
                    if tiou > iou_ta:
                        iou_ta = tiou
                        c_ta = tac["meta"]["c_tracka"]

        X[i] = np.array([
            meta["c_tracka"],
            meta["w_vote"],
            float(meta["n_streams"]),
            meta["sconf"]["ms"],
            meta["sconf"]["s"],
            meta["sconf"]["main"],
            meta["sconf"]["SAHI"],
            meta["max_conf"],
            meta["min_conf_nz"],
            meta["std_conf_nz"],
            meta["sahi_only"],
            meta["ms_only"],
            meta["sam_cnn_iou"],
            meta["ms_s_iou"],
            meta["stream_iou_mean"],
            float(pv.mean()),
            float(pq[0]),
            float(pq[1]),
            float(pq[2]),
            float(pq[3]),
            float(pq[4]),
            float(pv.std()),
            p_core,
            p_rim,
            p_core - p_rim,
            float((pv > 0.65).mean()),
            float((pv > 0.75).mean()),
            float((pv > 0.85).mean()),
            n_cc_core76,
            area,
            float(np.log1p(area)),
            bw,
            bh,
            max(bw, bh),
            rot_len,
            rot_wid,
            rot_len / max(1.0, rot_wid),
            area / max(1.0, bw * bh),
            solidity,
            compact,
            mean_width,
            n_cc_m,
            rho_c,
            rho_max,
            float(q_loc - np.percentile(f_m, 25)),
            float(q_loc - np.percentile(f_m, 10)),
            float(f_m.mean()),
            float(np.percentile(f_m, 10)),
            float(b_m.mean()),
            float(np.percentile(b_m, 10)),
            bth_cs_diff,
            bright_ring_z,
            float((f_m < (q_loc - 1.5)).mean()),
            float(r_m.mean()),
            float(np.percentile(r_m, 75)),
            ring_std,
            phys["mad"],
            obs_code,
            float(KIND_MAP[cd["kind"]]),
            1.0 if cd["kind"] == "ta_base" else 0.0,
            1.0 if cd["kind"] == "ta_rescue" else 0.0,
            float(iou_ta),
            float(meta["c_tracka"] - c_ta) if iou_ta >= 0.35 else float(meta["c_tracka"]),
            n_cands_img,
            sum_c_img,
            float(ranks[i]),
            ov_cnt[i],
            ov_max[i],
            near_d,
        ], dtype=np.float32)

        # Compact square aspect-preserving 4x48x48 multimodal visual patch [flat_z, bth_z, pmap, mask]
        side = max(32, int(round(max(bw, bh) * 1.25)) + 16)
        px1 = int(round(0.5 * (x1 + x2))) - side // 2
        py1 = int(round(0.5 * (y1 + y2))) - side // 2
        px2, py2 = px1 + side, py1 + side
        cx1, cy1, cx2, cy2 = max(0, px1), max(0, py1), min(FULL, px2), min(FULL, py2)
        interp = cv2.INTER_AREA if side > 48 else cv2.INTER_LINEAR

        fl_sq = np.full((side, side), 153, dtype=np.uint8)
        bt_sq = np.full((side, side), 153, dtype=np.uint8)
        pm_sq = np.zeros((side, side), dtype=np.uint8)
        mk_sq = np.zeros((side, side), dtype=np.uint8)
        if cx2 > cx1 and cy2 > cy1:
            phx1, phy1 = max(0, cx1 // 2), max(0, cy1 // 2)
            phx2, phy2 = min(HALF, max(phx1 + 1, (cx2 + 1) // 2)), min(HALF, max(phy1 + 1, (cy2 + 1) // 2))
            fz_u8 = np.clip((phys["flat_z"][phy1:phy2, phx1:phx2] + 6.0) * (255.0 / 10.0), 0.0, 255.0).astype(np.uint8)
            bz_u8 = np.clip((phys["bth_z"][phy1:phy2, phx1:phx2] + 6.0) * (255.0 / 10.0), 0.0, 255.0).astype(np.uint8)
            sy0, sy1, sx0, sx1 = cy1 - py1, cy2 - py1, cx1 - px1, cx2 - px1
            fl_sq[sy0:sy1, sx0:sx1] = cv2.resize(fz_u8, (cx2 - cx1, cy2 - cy1), interpolation=cv2.INTER_LINEAR)
            bt_sq[sy0:sy1, sx0:sx1] = cv2.resize(bz_u8, (cx2 - cx1, cy2 - cy1), interpolation=cv2.INTER_LINEAR)
            pm_sq[sy0:sy1, sx0:sx1] = full_pmap_u8[cy1:cy2, cx1:cx2]
        mx1, my1, mx2, my2 = max(x1, px1), max(y1, py1), min(x2, px2), min(y2, py2)
        if mx2 > mx1 and my2 > my1:
            mk_sq[my1 - py1 : my2 - py1, mx1 - px1 : mx2 - px1] = (
                msk[my1 - y1 : my2 - y1, mx1 - x1 : mx2 - x1].astype(np.uint8) * 255
            )
        patches[i, 0] = cv2.resize(fl_sq, (48, 48), interpolation=interp)
        patches[i, 1] = cv2.resize(bt_sq, (48, 48), interpolation=interp)
        patches[i, 2] = cv2.resize(pm_sq, (48, 48), interpolation=interp)
        patches[i, 3] = cv2.resize(mk_sq, (48, 48), interpolation=cv2.INTER_AREA if side > 48 else cv2.INTER_NEAREST)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    r_count = len(stem_gt)
    h_tgt = np.zeros(n, dtype=np.float32)
    m_tgt = np.zeros(n, dtype=np.float32)
    u_tgt = np.zeros(n, dtype=np.float32)
    if r_count > 0:
        pred_rles = [{"size": [FULL, FULL], "counts": c["rle"].encode("ascii")} for c in cands]
        hit_cnt = np.zeros(n, dtype=np.float32)
        iou_sum = np.zeros(n, dtype=np.float32)
        for gr in stem_gt:
            if not gr:
                continue
            mat = mu.iou(pred_rles, gr, [0] * len(gr))
            best = mat.max(axis=1)
            hit = best > 0.5
            hit_cnt += hit.astype(np.float32)
            iou_sum += np.where(hit, best, 0.0).astype(np.float32)
        h_tgt = hit_cnt / float(r_count)
        m_tgt = np.where(hit_cnt > 0, iou_sum / np.maximum(hit_cnt, 1.0), 0.0).astype(np.float32)
        u_tgt = iou_sum / float(r_count)

    return {
        "stem": stem,
        "fold": fold,
        "obs": stem[-2:],
        "r_count": r_count,
        "rles": [c["rle"] for c in cands],
        "kinds": [c["kind"] for c in cands],
        "boxes": np.array([c["box"] for c in cands], dtype=np.int32),
        "areas": areas_arr,
        "X": X,
        "h": h_tgt,
        "m": m_tgt,
        "u": u_tgt,
        "edges": edges,
        "patches": patches,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("/private/tmp/exp011b_pool.pkl"))
    args = ap.parse_args()

    t0 = time.time()
    folds = data.folds()
    oof_stems = sorted(folds)
    test_stems = sorted(data.test_stems())
    gt_recs = metric.load_records(oof_stems)

    if args.smoke > 0:
        oof_stems = oof_stems[: args.smoke]
        test_stems = test_stems[: min(args.smoke, len(test_stems))]

    oof_tasks = [(s, "oof", int(folds[s]), gt_recs[s]) for s in oof_stems]
    test_tasks = [(s, "test", -1, []) for s in test_stems]

    print(f"EXP-011b Symmetric Extraction: OOF={len(oof_tasks)}, Test={len(test_tasks)}, workers={args.workers}, features={len(FEATURE_NAMES)}", flush=True)
    oof_out = {}
    test_out = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for idx, res in enumerate(ex.map(_extract_stem, oof_tasks, chunksize=2), 1):
            oof_out[res["stem"]] = res
            if idx % 50 == 0 or idx == len(oof_tasks):
                print(f"  OOF {idx}/{len(oof_tasks)} ({time.time()-t0:.1f}s)", flush=True)
        for idx, res in enumerate(ex.map(_extract_stem, test_tasks, chunksize=2), 1):
            test_out[res["stem"]] = res
            if idx % 30 == 0 or idx == len(test_tasks):
                print(f"  Test {idx}/{len(test_tasks)} ({time.time()-t0:.1f}s)", flush=True)

    n_cands = sum(len(v["rles"]) for v in oof_out.values())
    n_test = sum(len(v["rles"]) for v in test_out.values())
    u_all = np.concatenate([v["u"] for v in oof_out.values() if len(v["u"])])
    h_all = np.concatenate([v["h"] for v in oof_out.values() if len(v["h"])])
    m_all = np.concatenate([v["m"] for v in oof_out.values() if len(v["m"])])
    ta_u = np.concatenate([v["u"][np.isin(v["kinds"], ["ta_base", "ta_rescue"])] for v in oof_out.values() if len(v["u"])])
    print(f"OOF cands={n_cands}, h>0={(h_all>0).mean():.3f}, mean_h={h_all.mean():.4f}, mean_u={u_all.mean():.4f}, mean_m|pos={m_all[h_all>0].mean():.4f}", flush=True)
    print(f"Track-A subset cands={len(ta_u)}, u>0={(ta_u>0).mean():.3f}, mean_u={ta_u.mean():.4f} | Test cands={n_test}", flush=True)

    # Extract visual crop patches into separate zero-overhead .npy files while keeping pool .pkl light
    patch_stem = args.out.stem
    oof_chunks = []
    off = 0
    for s in sorted(oof_out):
        p = oof_out[s].pop("patches")
        oof_chunks.append(p)
        oof_out[s]["patch_slice"] = (off, off + len(p))
        off += len(p)
    test_chunks = []
    off_t = 0
    for s in sorted(test_out):
        p = test_out[s].pop("patches")
        test_chunks.append(p)
        test_out[s]["patch_slice"] = (off_t, off_t + len(p))
        off_t += len(p)

    patches_oof_p = args.out.parent / f"{patch_stem}_patches_oof.npy"
    patches_test_p = args.out.parent / f"{patch_stem}_patches_test.npy"
    np.save(patches_oof_p, np.concatenate(oof_chunks, axis=0) if oof_chunks else np.zeros((0, 4, 48, 48), dtype=np.uint8))
    np.save(patches_test_p, np.concatenate(test_chunks, axis=0) if test_chunks else np.zeros((0, 4, 48, 48), dtype=np.uint8))

    payload = {
        "oof": oof_out,
        "test": test_out,
        "feature_names": FEATURE_NAMES,
        "patches_oof_path": str(patches_oof_p),
        "patches_test_path": str(patches_test_p),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {args.out} ({args.out.stat().st_size / 1e6:.2f} MB) + patches ({patches_oof_p.stat().st_size / 1e6:.1f} MB) in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
