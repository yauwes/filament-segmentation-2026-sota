"""EXP-011 Stage 1: Extract rich multi-variant candidate pool + H-alpha physics + multi-annotator Hurdle targets.

Outputs `/private/tmp/exp011_pool.pkl` containing ALL 707 OOF + 180 Test stems:
  Per stem:
    - `fold`, `obs`, `r_count` (R_i in {1,2,3} for train, 0 for test)
    - `rles`: list[str] (2048x2048 COCO compressed ASCII RLE for each candidate)
    - `boxes`: (N, 4) int32
    - `areas`: (N,) int32
    - `X`: (N, F) float32 leak-free features (consensus, per-stream conf/IoU, SAM2-vs-CNN IoU,
         soft prob quantiles/rim/core, morphology, real H-alpha ray-cast rho/darkness_z/bth7/flat,
         observatory & busy-frame context)
    - `h`, `m`, `u`: (N,) float32 exact official R_i-annotator hit rate, cond IoU, utility u=h*m
    - `edges`: list[(i, j)] of candidate pairs that share any pixel (>0 px overlap)
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "gap_research" / "scripts"))

from filseg import data, postprocess

WAVE2_ROOT = Path("/private/tmp/gap_wave2_cache")
E6 = WAVE2_ROOT / "exp006"
E7 = WAVE2_ROOT / "exp007"
FULL = 2048
OBS_MAP = {"Bh": 0, "Ch": 1, "Lh": 2, "Mh": 3, "Th": 4, "Uh": 5}
KIND_MAP = {"cons68": 0, "thr60": 1, "thr74": 2, "clamped": 3, "split": 4, "merge": 5, "solo": 6}


def _fit_disk(img: np.ndarray) -> tuple[float, float, float]:
    h, w = img.shape
    p99 = float(np.percentile(img, 99))
    mask = img > max(18.0, 0.25 * p99)
    ys, xs = np.nonzero(mask)
    if len(ys) < 1000:
        return w / 2.0, h / 2.0, min(w, h) * 0.43
    cx0, cy0 = float(xs.mean()), float(ys.mean())
    r0 = float(np.sqrt(len(xs) / np.pi))

    thetas = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
    rs = np.linspace(max(10.0, 0.85 * r0), min(w * 0.65, 1.15 * r0), 72)
    gx = np.clip(np.round(cx0 + np.cos(thetas)[:, None] * rs[None, :]).astype(np.int32), 0, w - 1)
    gy = np.clip(np.round(cy0 + np.sin(thetas)[:, None] * rs[None, :]).astype(np.int32), 0, h - 1)
    prof = img[gy, gx].astype(np.float32)
    best = np.argmax(prof[:, :-2] - prof[:, 2:], axis=1) + 1
    rl = rs[best]
    mr = float(np.median(rl))
    mad = 1.4826 * float(np.median(np.abs(rl - mr))) + 1e-3
    inl = np.abs(rl - mr) <= max(6.0, 2.5 * mad)
    if int(inl.sum()) < 60:
        return cx0, cy0, r0
    px = cx0 + rl[inl] * np.cos(thetas[inl])
    py = cy0 + rl[inl] * np.sin(thetas[inl])
    try:
        sol, _, _, _ = np.linalg.lstsq(np.c_[2.0 * px, 2.0 * py, np.ones(len(px))], px * px + py * py, rcond=None)
        cx, cy = float(sol[0]), float(sol[1])
        r = float(np.sqrt(max(1.0, sol[2] + cx * cx + cy * cy)))
        if abs(cx - cx0) < 40 and abs(cy - cy0) < 40 and abs(r - r0) < 40:
            return cx, cy, r
    except Exception:
        pass
    return cx0, cy0, r0


def _compute_img_physics(stem: str, split: str) -> dict:
    p = (data.TRAIN_IMAGES if split == "oof" else data.TEST_IMAGES) / f"{stem}.jpeg"
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"ok": False, "cx": 1024.0, "cy": 1024.0, "R": 880.0, "mad": 1.0}
    small = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
    cx_s, cy_s, r_s = _fit_disk(small)
    cx, cy, R = cx_s * 2.0, cy_s * 2.0, r_s * 2.0
    yy, xx = np.mgrid[:1024, :1024].astype(np.float32)
    disk_s = np.hypot(xx - cx_s, yy - cy_s) < 0.99 * r_s

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bth_s = -cv2.morphologyEx(small, cv2.MORPH_BLACKHAT, k3).astype(np.float32)
    flat_s = small.astype(np.float32) - cv2.medianBlur(small, 51).astype(np.float32)
    raw_s = small.astype(np.float32)

    def _znorm(a: np.ndarray) -> tuple[np.ndarray, float]:
        v = a[disk_s]
        med = float(np.median(v)) if len(v) else 0.0
        mad = 1.4826 * float(np.median(np.abs(v - med))) + 1e-6 if len(v) else 1.0
        return ((a - med) / mad).astype(np.float32), mad

    raw_z, raw_mad = _znorm(raw_s)
    flat_z, _ = _znorm(flat_s)
    bth_z, _ = _znorm(bth_s)
    return {
        "ok": True, "cx": cx, "cy": cy, "R": R,
        "raw_z": raw_z, "flat_z": flat_z, "bth_z": bth_z,
        "raw_mad": raw_mad, "disk_med": float(np.median(raw_s[disk_s])) if disk_s.any() else 128.0,
    }


def _unpack_npz(p: Path, tta_prefer: int = 4) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    z = np.load(p)
    conf, box = z["conf"], z["box"]
    flat = z[f"p{tta_prefer}"] if f"p{tta_prefer}" in z.files else z["p"]
    probs, o = [], 0
    for x1, y1, x2, y2 in box:
        n = int((y2 - y1) * (x2 - x1))
        probs.append(flat[o : o + n].reshape(int(y2 - y1), int(x2 - x1)))
        o += n
    return conf, box, probs


def _load_stream_prob_items(
    seg_specs: list[tuple[str, Path, float]],
    stem: str,
    conf_floor: float = 0.14,
    small_scale: float = 1.20,
    small_px: int = 120,
    border_g2: bool = True,
    g2_keys: tuple[str, ...] = ("e4x_g2", "s1e_g2", "sam2_g2", "e4_g1"),
) -> list[dict]:
    loaded = []
    for k, sdir, w in seg_specs:
        p = sdir / f"{stem}.npz"
        if w > 0 and p.exists():
            c, b, pr = _unpack_npz(p, tta_prefer=4)
            loaded.append((k, w, c, b, pr))
    if not loaded:
        return []
    anchor = 0
    for j, it in enumerate(loaded):
        if it[0] == "e4":
            anchor = j
            break
    _, _, conf0, box0, _ = loaded[anchor]
    if len(conf0) == 0:
        return []

    out = []
    for idx in np.argsort(-conf0):
        raw_c = float(conf0[idx])
        bx0, by0, bx1, by1 = [int(v) for v in box0[idx]]
        eff_c = raw_c * (small_scale if max(bx1 - bx0, by1 - by0) < small_px else 1.0)
        if eff_c < conf_floor or bx1 <= bx0 or by1 <= by0:
            continue

        ux0, uy0, ux1, uy1 = bx0, by0, bx1, by1
        if border_g2:
            for k, _, _, b_arr, _ in loaded:
                if k in g2_keys and idx < len(b_arr):
                    sx1, sy1, sx2, sy2 = [int(v) for v in b_arr[idx]]
                    if sx2 > sx1 and sy2 > sy1:
                        ux0, uy0 = min(ux0, sx1), min(uy0, sy1)
                        ux1, uy1 = max(ux1, sx2), max(uy1, sy2)

        uh, uw = uy1 - uy0, ux1 - ux0
        acc = np.zeros((uh, uw), dtype=np.float32)
        wsum = np.zeros((uh, uw), dtype=np.float32)
        sam_acc = np.zeros((uh, uw), dtype=np.float32)
        sam_w = np.zeros((uh, uw), dtype=np.float32)
        cnn_acc = np.zeros((uh, uw), dtype=np.float32)
        cnn_w = np.zeros((uh, uw), dtype=np.float32)
        g2_acc = np.zeros((uh, uw), dtype=np.float32)
        g2_w = np.zeros((uh, uw), dtype=np.float32)

        for k, w, _, b_arr, p_list in loaded:
            if idx >= len(b_arr):
                continue
            sx1, sy1, sx2, sy2 = [int(v) for v in b_arr[idx]]
            if sx2 <= sx1 or sy2 <= sy1:
                continue
            pr = p_list[idx]
            if pr.shape != (sy2 - sy1, sx2 - sx1):
                pr = cv2.resize(pr, (sx2 - sx1, sy2 - sy1), interpolation=cv2.INTER_LINEAR)
            ix1, iy1, ix2, iy2 = max(ux0, sx1), max(uy0, sy1), min(ux1, sx2), min(uy1, sy2)
            if ix2 > ix1 and iy2 > iy1:
                sl_y, sl_x = slice(iy1 - uy0, iy2 - uy0), slice(ix1 - ux0, ix2 - ux0)
                sub = (pr[iy1 - sy1 : iy2 - sy1, ix1 - sx1 : ix2 - sx1]).astype(np.float32) * (1.0 / 255.0)
                acc[sl_y, sl_x] += w * sub
                wsum[sl_y, sl_x] += w
                if "sam" in k:
                    sam_acc[sl_y, sl_x] += w * sub
                    sam_w[sl_y, sl_x] += w
                else:
                    cnn_acc[sl_y, sl_x] += w * sub
                    cnn_w[sl_y, sl_x] += w
                if k in g2_keys:
                    g2_acc[sl_y, sl_x] += w * sub
                    g2_w[sl_y, sl_x] += w

        if not (wsum > 0).any():
            continue
        p_u8 = np.where(wsum > 0, (acc / np.maximum(wsum, 1e-6)) * 255.0 + 0.5, 0.0).astype(np.uint8)
        if not (p_u8 > 140).any():
            continue
        ys, xs = np.nonzero(p_u8 > 128)
        if not len(ys):
            continue
        tx0, ty0, tx1, ty1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        out.append({
            "conf": eff_c,
            "raw_conf": raw_c,
            "box": (ux0 + tx0, uy0 + ty0, ux0 + tx1, uy0 + ty1),
            "box0": (bx0, by0, bx1, by1),
            "expanded": (ux0 < bx0 or uy0 < by0 or ux1 > bx1 or uy1 > by1),
            "p": np.ascontiguousarray(p_u8[ty0:ty1, tx0:tx1]),
            "sam_p": np.where(sam_w[ty0:ty1, tx0:tx1] > 0, sam_acc[ty0:ty1, tx0:tx1] / np.maximum(sam_w[ty0:ty1, tx0:tx1], 1e-6), -1.0).astype(np.float32),
            "cnn_p": np.where(cnn_w[ty0:ty1, tx0:tx1] > 0, cnn_acc[ty0:ty1, tx0:tx1] / np.maximum(cnn_w[ty0:ty1, tx0:tx1], 1e-6), -1.0).astype(np.float32),
            "g2_has": bool((g2_w > 0).any()),
        })
    return out


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(1.0, float(ua))


def _crop_iou(ba: tuple[int, int, int, int], ma: np.ndarray, bb: tuple[int, int, int, int], mb: np.ndarray) -> tuple[float, int]:
    ix1, iy1 = max(ba[0], bb[0]), max(ba[1], bb[1])
    ix2, iy2 = min(ba[2], bb[2]), min(ba[3], bb[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0, 0
    sa = ma[iy1 - ba[1] : iy2 - ba[1], ix1 - ba[0] : ix2 - ba[0]]
    sb = mb[iy1 - bb[1] : iy2 - bb[1], ix1 - bb[0] : ix2 - bb[0]]
    inter = int((sa & sb).sum())
    if inter == 0:
        return 0.0, 0
    union = int(ma.sum()) + int(mb.sum()) - inter
    return inter / max(1, union), inter


def _trim_bool(box: tuple[int, int, int, int], m: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    ys, xs = np.nonzero(m)
    if not len(ys):
        return None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    return (box[0] + x1, box[1] + y1, box[0] + x2, box[1] + y2), np.ascontiguousarray(m[y1:y2, x1:x2])


def _extract_stem(args: tuple[str, str, int, list]) -> dict:
    stem, split, fold, gt_anns = args
    cv2.setNumThreads(0)
    sp = "oof" if split == "oof" else "test"
    phys = _compute_img_physics(stem, split)

    streams_cfg = {
        "ms": (1.35, [
            ("e4", REPO / "work" / "cache" / "ms" / sp, 0.10),
            ("sam2", E6 / "sam2" / "ms" / sp, 0.30),
            ("sam2_g2", E6 / "sam2_g2" / "ms" / sp, 0.25),
            ("s1e_g2", E6 / "s1e_g2" / "ms" / sp, 0.25),
            ("e4x", E6 / "e4x" / "e4x" / "ms" / sp, 0.05),
            ("e4x_g2", E6 / "e4x" / "e4x_g2" / "ms" / sp, 0.05),
        ]),
        "s": (1.05, [
            ("e4", REPO / "work" / "cache" / "s" / sp, 0.25),
            ("s1e_g2", E6 / "s1e_g2" / "s" / sp, 0.45),
            ("e4x", E6 / "e4x" / "e4x" / "s" / sp, 0.15),
            ("e4x_g2", E6 / "e4x" / "e4x_g2" / "s" / sp, 0.15),
        ]),
        "main": (0.95, [
            ("e4", REPO / "work" / "cache" / "main" / sp, 0.25),
            ("s1e_g2", E6 / "s1e_g2" / "main" / sp, 0.45),
            ("e4x", E6 / "e4x" / "e4x" / "main" / sp, 0.15),
            ("e4x_g2", E6 / "e4x" / "e4x_g2" / "main" / sp, 0.15),
        ]),
        "SAHI": (0.55, [
            ("sam2", E7 / "sahi_sam2" / sp, 0.40),
            ("s1e", E7 / "sahi_s1e" / sp, 0.30),
            ("e4_g1", E7 / "sahi_e4_g1" / sp, 0.20),
            ("e4", E7 / "sahi_e4" / sp, 0.10),
        ]),
    }
    raw_items: dict[str, list[dict]] = {}
    for sname, (_, specs) in streams_cfg.items():
        raw_items[sname] = _load_stream_prob_items(specs, stem, conf_floor=0.14, small_scale=1.0 if sname == "SAHI" else 1.20)

    # Cluster across all detections from ms/s/main/SAHI by mask@0.65 overlap > 0.32 or box IoU > 0.40
    all_dets = []
    for sname in ("ms", "s", "main", "SAHI"):
        wt = streams_cfg[sname][0]
        for j, d in enumerate(raw_items[sname]):
            m65 = d["p"] > int(0.65 * 255)
            if int(m65.sum()) < 40:
                continue
            all_dets.append((sname, wt, d, m65))

    used = [False] * len(all_dets)
    clusters = []
    # Greedily anchor by confidence/stream priority
    order = sorted(range(len(all_dets)), key=lambda idx: -(all_dets[idx][2]["conf"] * all_dets[idx][1]))
    for idx in order:
        if used[idx]:
            continue
        used[idx] = True
        members = [all_dets[idx]]
        # Match best detection from OTHER streams
        for j in order:
            if used[j]:
                continue
            sname_j, _, d_j, m_j = all_dets[j]
            if any(m[0] == sname_j for m in members):
                continue
            ref_d, ref_m = members[0][2], members[0][3]
            if _box_iou(ref_d["box"], d_j["box"]) == 0.0:
                continue
            iou, _ = _crop_iou(ref_d["box"], ref_m, d_j["box"], m_j)
            if iou >= 0.30:
                used[j] = True
                members.append(all_dets[j])
        clusters.append(members)

    raw_cands = []
    for cl in clusters:
        ux0 = min(m[2]["box"][0] for m in cl)
        uy0 = min(m[2]["box"][1] for m in cl)
        ux1 = max(m[2]["box"][2] for m in cl)
        uy1 = max(m[2]["box"][3] for m in cl)
        uh, uw = uy1 - uy0, ux1 - ux0
        acc = np.zeros((uh, uw), dtype=np.float32)
        wsum = np.zeros((uh, uw), dtype=np.float32)
        s_confs = {"ms": 0.0, "s": 0.0, "main": 0.0, "SAHI": 0.0}
        vsum = 0.0
        bx0 = min(m[2]["box0"][0] for m in cl)
        by0 = min(m[2]["box0"][1] for m in cl)
        bx1 = max(m[2]["box0"][2] for m in cl)
        by1 = max(m[2]["box0"][3] for m in cl)

        for sname, wt, d, _ in cl:
            x1, y1, x2, y2 = d["box"]
            sl_y, sl_x = slice(y1 - uy0, y2 - uy0), slice(x1 - ux0, x2 - ux0)
            w_eff = wt * (0.4 + 0.6 * d["conf"])
            acc[sl_y, sl_x] += w_eff * (d["p"].astype(np.float32) / 255.0)
            wsum[sl_y, sl_x] += w_eff
            s_confs[sname] = max(s_confs[sname], d["conf"])
            vsum += wt * d["conf"]

        p_blend = np.where(wsum > 0, acc / np.maximum(wsum, 1e-6), 0.0).astype(np.float32)
        # Disagreement: ms vs s, sam2 vs cnn inside first anchor
        d0 = cl[0][2]
        sam_cnn_iou = -1.0
        if (d0["sam_p"] >= 0).any() and (d0["cnn_p"] >= 0).any():
            sm_b = d0["sam_p"] > 0.65
            cn_b = d0["cnn_p"] > 0.65
            un = int((sm_b | cn_b).sum())
            sam_cnn_iou = float((sm_b & cn_b).sum()) / un if un > 0 else 0.0
        ms_s_iou = -1.0
        m_ms = [m for m in cl if m[0] == "ms"]
        m_s = [m for m in cl if m[0] == "s"]
        if m_ms and m_s:
            ms_s_iou, _ = _crop_iou(m_ms[0][2]["box"], m_ms[0][3], m_s[0][2]["box"], m_s[0][3])

        meta = {
            "s_confs": s_confs,
            "vote_sum": sum(streams_cfg[m[0]][0] for m in cl),
            "weighted_vote": vsum,
            "n_streams": len(cl),
            "sam_cnn_iou": sam_cnn_iou,
            "ms_s_iou": ms_s_iou,
            "p_blend": p_blend,
            "ubox": (ux0, uy0, ux1, uy1),
            "box0": (bx0, by0, bx1, by1),
            "expanded": any(m[2]["expanded"] for m in cl),
        }

        def _push_mask(full_crop_m: np.ndarray, kind: str) -> None:
            tr = _trim_bool((ux0, uy0, ux1, uy1), full_crop_m)
            if tr is None:
                return
            tb, tm = tr
            area = int(tm.sum())
            if area < 90:
                return
            raw_cands.append((tb, tm, kind, meta))

        m68 = p_blend > 0.68
        _push_mask(m68, "cons68" if len(cl) >= 2 else "solo")
        m60 = p_blend > 0.60
        if int(m60.sum()) >= 90 and int((m60 & m68).sum()) / max(1, int((m60 | m68).sum())) < 0.94:
            _push_mask(m60, "thr60")
        m74 = p_blend > 0.74
        if int(m74.sum()) >= 90 and int((m74 & m68).sum()) / max(1, int((m74 | m68).sum())) < 0.94:
            _push_mask(m74, "thr74")

        # Clamped box0 variant when expanded
        if ux0 < bx0 or uy0 < by0 or ux1 > bx1 or uy1 > by1:
            m_cl = np.zeros_like(m68)
            ox0, oy0, ox1, oy1 = max(0, bx0 - ux0), max(0, by0 - uy0), min(uw, bx1 - ux0), min(uh, by1 - uy0)
            m_cl[oy0:oy1, ox0:ox1] = m68[oy0:oy1, ox0:ox1]
            if int(m_cl.sum()) >= 90 and int((m_cl & m68).sum()) / max(1, int((m_cl | m68).sum())) < 0.96:
                _push_mask(m_cl, "clamped")

        # Split alternative if >= 2 large CCs in m68
        n_cc, lab = cv2.connectedComponents(m68.astype(np.uint8), connectivity=8)
        big_ccs = [cid for cid in range(1, n_cc) if int((lab == cid).sum()) >= 100]
        if len(big_ccs) >= 2:
            for cid in big_ccs:
                _push_mask(lab == cid, "split")

    # Add nearby morphological merges (pair of cons68/solo with <= 8px gap)
    base_idxs = [i for i, c in enumerate(raw_cands) if c[2] in ("cons68", "solo")]
    for ia in range(len(base_idxs)):
        i = base_idxs[ia]
        ba, ma, _, ma_meta = raw_cands[i]
        for ib in range(ia + 1, len(base_idxs)):
            j = base_idxs[ib]
            bb, mb, _, mb_meta = raw_cands[j]
            if (max(ba[0], bb[0]) - min(ba[2], bb[2]) > 8) or (max(ba[1], bb[1]) - min(ba[3], bb[3]) > 8):
                continue
            ov_iou, _ = _crop_iou(ba, ma, bb, mb)
            if ov_iou > 0.25:
                continue
            mx0, my0 = min(ba[0], bb[0]), min(ba[1], bb[1])
            mx1, my1 = max(ba[2], bb[2]), max(ba[3], bb[3])
            mm = np.zeros((my1 - my0, mx1 - mx0), dtype=bool)
            mm[ba[1] - my0 : ba[3] - my0, ba[0] - mx0 : ba[2] - mx0] |= ma
            mm[bb[1] - my0 : bb[3] - my0, bb[0] - mx0 : bb[2] - mx0] |= mb
            # Pick richer cluster meta + synthetic prob
            pb = np.zeros((my1 - my0, mx1 - mx0), dtype=np.float32)
            for md in (ma_meta, mb_meta):
                ux0, uy0, ux1, uy1 = md["ubox"]
                ix1, iy1, ix2, iy2 = max(mx0, ux0), max(my0, uy0), min(mx1, ux1), min(my1, uy1)
                if ix2 > ix1 and iy2 > iy1:
                    pb[iy1 - my0 : iy2 - my0, ix1 - mx0 : ix2 - mx0] = np.maximum(
                        pb[iy1 - my0 : iy2 - my0, ix1 - mx0 : ix2 - mx0],
                        md["p_blend"][iy1 - uy0 : iy2 - uy0, ix1 - ux0 : ix2 - ux0],
                    )
            tr = _trim_bool((mx0, my0, mx1, my1), mm)
            if tr is not None:
                tb, tm = tr
                mmeta = dict(ma_meta if ma_meta["weighted_vote"] >= mb_meta["weighted_vote"] else mb_meta)
                mmeta["p_blend"] = pb
                mmeta["ubox"] = (mx0, my0, mx1, my1)
                raw_cands.append((tb, tm, "merge", mmeta))

    # Deduplicate near-identical masks (IoU >= 0.94), preferring base/cons68
    raw_cands.sort(key=lambda c: (KIND_MAP[c[2]], -c[3]["weighted_vote"]))
    dedup = []
    for cand in raw_cands:
        tb, tm, kind, meta = cand
        dup = False
        for ob, om, _, _ in dedup:
            if _box_iou(tb, ob) > 0.70 and _crop_iou(tb, tm, ob, om)[0] >= 0.94:
                dup = True
                break
        if not dup:
            dedup.append(cand)

    # Encode COCO RLEs, compute pairwise overlap edges, features, and R_i GT targets
    N = len(dedup)
    rle_objs = []
    rle_strs = []
    boxes = np.zeros((N, 4), dtype=np.int32)
    areas = np.zeros(N, dtype=np.int32)
    for k, (tb, tm, _, _) in enumerate(dedup):
        x1, y1, x2, y2 = tb
        boxes[k] = [x1, y1, x2, y2]
        areas[k] = int(tm.sum())
        fm = postprocess.full_mask(tb, tm)
        ro = mu.encode(np.asfortranarray(fm))
        rle_objs.append(ro)
        cnt = ro["counts"]
        rle_strs.append(cnt.decode("ascii") if isinstance(cnt, bytes) else cnt)

    edges = []
    ov_cnt = np.zeros(N, dtype=np.int32)
    ov_max = np.zeros(N, dtype=np.float32)
    for i in range(N):
        bi, mi = dedup[i][0], dedup[i][1]
        for j in range(i + 1, N):
            bj, mj = dedup[j][0], dedup[j][1]
            if _box_iou(bi, bj) == 0.0:
                continue
            iou, inter = _crop_iou(bi, mi, bj, mj)
            if inter > 0:
                edges.append((i, j))
                ov_cnt[i] += 1
                ov_cnt[j] += 1
                ov_max[i] = max(ov_max[i], iou)
                ov_max[j] = max(ov_max[j], iou)

    obs_code = OBS_MAP.get(stem[-2:], -1)
    cx, cy, R = phys["cx"], phys["cy"], phys["R"]
    # Build leak-free feature matrix X
    feats = []
    sc_list = [max(c[3]["s_confs"].values()) for c in dedup]
    ranks = np.argsort(np.argsort([-v for v in sc_list]))
    sum_hc = float(sum(1 for v in sc_list if v >= 0.35))

    k3 = np.ones((3, 3), np.uint8)
    for idx, ((x1, y1, x2, y2), tm, kind, meta) in enumerate(dedup):
        bw, bh = float(x2 - x1), float(y2 - y1)
        area = float(areas[idx])
        fill = area / max(1.0, bw * bh)
        asp = max(bw, bh) / max(1.0, min(bw, bh))
        er = cv2.erode(tm.astype(np.uint8), k3) > 0
        rim = tm & ~er
        perim = float(rim.sum())
        compact = (perim * perim) / max(1.0, 4.0 * np.pi * area)
        n_cc, _ = cv2.connectedComponents(tm.astype(np.uint8), connectivity=8)

        # Sample blended probability on mask, core, and rim
        ux0, uy0, ux1, uy1 = meta["ubox"]
        pb = meta["p_blend"]
        sub_p = np.zeros_like(tm, dtype=np.float32)
        ix1, iy1, ix2, iy2 = max(x1, ux0), max(y1, uy0), min(x2, ux1), min(y2, uy1)
        if ix2 > ix1 and iy2 > iy1:
            sub_p[iy1 - y1 : iy2 - y1, ix1 - x1 : ix2 - x1] = pb[iy1 - uy0 : iy2 - uy0, ix1 - ux0 : ix2 - ux0]
        p_vals = sub_p[tm]
        p_mean = float(p_vals.mean()) if len(p_vals) else 0.0
        pq25, pq50, pq75, pq90 = [float(v) for v in np.percentile(p_vals, [25, 50, 75, 90])] if len(p_vals) else [0.0]*4
        p_std = float(p_vals.std()) if len(p_vals) else 0.0
        p_rim = float(sub_p[rim].mean()) if rim.any() else p_mean
        p_core = float(sub_p[er].mean()) if er.any() else p_mean

        # Solar disk geometry + H-alpha physics on 1024 grid
        ys_m, xs_m = np.nonzero(tm)
        mcx, mcy = x1 + float(xs_m.mean()), y1 + float(ys_m.mean())
        rho_c = float(np.hypot(mcx - cx, mcy - cy) / max(1.0, R))
        if phys["ok"]:
            sx1, sy1 = max(0, x1 // 2 - 8), max(0, y1 // 2 - 8)
            sx2, sy2 = min(1024, max(sx1 + 1, (x2 + 1) // 2 + 8)), min(1024, max(sy1 + 1, (y2 + 1) // 2 + 8))
            ms_small = np.zeros((sy2 - sy1, sx2 - sx1), dtype=bool)
            tx1, ty1 = max(0, x1 // 2) - sx1, max(0, y1 // 2) - sy1
            tx2, ty2 = min(1024, max(x1 // 2 + 1, (x2 + 1) // 2)) - sx1, min(1024, max(y1 // 2 + 1, (y2 + 1) // 2)) - sy1
            if tx2 > tx1 and ty2 > ty1:
                ms_small[ty1:ty2, tx1:tx2] = cv2.resize(tm.astype(np.uint8), (tx2 - tx1, ty2 - ty1), interpolation=cv2.INTER_NEAREST) > 0
            if not ms_small.any():
                ms_small[ty1:ty2, tx1:tx2] = True
            ring_outer = cv2.dilate(ms_small.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
            ring_inner = cv2.dilate(ms_small.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            ring = ring_outer & ~ring_inner
            r_crop = phys["raw_z"][sy1:sy2, sx1:sx2]
            f_crop = phys["flat_z"][sy1:sy2, sx1:sx2]
            b_crop = phys["bth_z"][sy1:sy2, sx1:sx2]
            dark_m = float(r_crop[ms_small].mean())
            dark_p10 = float(np.percentile(r_crop[ms_small], 10))
            ring_med = float(np.median(r_crop[ring])) if ring.any() else dark_m
            ring_std = float(r_crop[ring].std()) + 1e-3 if ring.any() else 1.0
            darkness_z = (ring_med - dark_m) / ring_std
            flat_m = float(f_crop[ms_small].mean())
            bth_m = float(b_crop[ms_small].mean())
            bth_p10 = float(np.percentile(b_crop[ms_small], 10))
        else:
            darkness_z = flat_m = bth_m = bth_p10 = dark_p10 = ring_std = 0.0

        sc = meta["s_confs"]
        confs_nz = [v for v in sc.values() if v > 0]
        feats.append([
            meta["vote_sum"], meta["weighted_vote"], float(meta["n_streams"]),
            sc["ms"], sc["s"], sc["main"], sc["SAHI"],
            max(sc.values()), min(confs_nz) if confs_nz else 0.0, float(np.std(confs_nz)) if len(confs_nz) > 1 else 0.0,
            float(sc["SAHI"] > 0 and meta["n_streams"] == 1),
            float(sc["ms"] > 0 and meta["n_streams"] == 1),
            float(KIND_MAP[kind]),
            float(meta["expanded"]),
            meta["sam_cnn_iou"], meta["ms_s_iou"],
            p_mean, pq25, pq50, pq75, pq90, p_std, p_rim, p_core,
            float((p_vals > 0.75).mean()), float((p_vals > 0.55).mean()),
            area, float(np.log1p(area)), bw, bh, max(bw, bh), asp, fill, compact, float(n_cc - 1),
            rho_c, darkness_z, dark_p10, flat_m, bth_m, bth_p10, ring_std, float(phys["raw_mad"]),
            float(obs_code), float(N), sum_hc, float(ranks[idx]), float(ov_cnt[idx]), float(ov_max[idx]),
        ])

    X = np.asarray(feats, dtype=np.float32) if feats else np.zeros((0, 49), dtype=np.float32)

    # Exact official R_i multi-annotator targets
    h_tgt = np.zeros(N, dtype=np.float32)
    m_tgt = np.zeros(N, dtype=np.float32)
    u_tgt = np.zeros(N, dtype=np.float32)
    r_cnt = len(gt_anns)
    if split == "oof" and r_cnt > 0 and N > 0:
        hit_cnt = np.zeros(N, dtype=np.float32)
        iou_sum = np.zeros(N, dtype=np.float32)
        for gt_r in gt_anns:
            if not gt_r:
                continue
            mat = mu.iou(rle_objs, gt_r, [0] * len(gt_r))  # (N, n_gt)
            best_g = mat.max(axis=1)
            hit = best_g > 0.5
            hit_cnt += hit.astype(np.float32)
            iou_sum += np.where(hit, best_g, 0.0).astype(np.float32)
        h_tgt = hit_cnt / float(r_cnt)
        m_tgt = np.where(hit_cnt > 0, iou_sum / np.maximum(hit_cnt, 1.0), 0.0).astype(np.float32)
        u_tgt = (iou_sum / float(r_cnt)).astype(np.float32)

    return {
        "stem": stem,
        "fold": fold,
        "obs": stem[-2:],
        "r_count": r_cnt,
        "rles": rle_strs,
        "boxes": boxes,
        "areas": areas,
        "X": X,
        "h": h_tgt,
        "m": m_tgt,
        "u": u_tgt,
        "edges": edges,
    }


def main() -> None:
    t0 = time.time()
    folds = data.folds()
    recs, anns = data.coco_index()
    oof_stems = sorted(folds)
    test_stems = data.test_stems()
    assert len(test_stems) == 180, f"Expected 180 test stems, got {len(test_stems)}"

    tasks = []
    for s in oof_stems:
        gt = [[data.ann_rle(a) for a in anns.get(rid, [])] for rid in recs[s]]
        tasks.append((s, "oof", int(folds[s]), gt))
    for s in test_stems:
        tasks.append((s, "test", -1, []))

    print(f"EXP-011 Stage 1: Extracting candidates & H-alpha physics for {len(oof_stems)} OOF + {len(test_stems)} Test stems on 7 workers...", flush=True)
    pool = {"oof": {}, "test": {}}
    workers = min(7, os.cpu_count() or 4)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(_extract_stem, tasks, chunksize=8)):
            sp = "test" if res["fold"] == -1 else "oof"
            pool[sp][res["stem"]] = res
            if (i + 1) % 100 == 0 or i + 1 == len(tasks):
                print(f"  [{i+1}/{len(tasks)}] elapsed {time.time()-t0:.1f}s", flush=True)

    out_p = Path("/private/tmp/exp011_pool.pkl")
    with open(out_p, "wb") as f:
        pickle.dump(pool, f, protocol=pickle.HIGHEST_PROTOCOL)
    tot_oof = sum(len(v["rles"]) for v in pool["oof"].values())
    tot_test = sum(len(v["rles"]) for v in pool["test"].values())
    print(f"Saved {out_p} ({out_p.stat().st_size/1e6:.1f} MB) | OOF cands={tot_oof} | Test cands={tot_test} | {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
