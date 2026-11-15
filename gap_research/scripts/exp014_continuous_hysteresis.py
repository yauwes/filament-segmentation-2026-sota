"""EXP-014 (Local Wave-2 Continuous Fusion): Continuous 2D Cross-Stream Probability Clustering
+ Border-G2 Unlock + Hysteresis (thr_hi/thr_lo) + Peak/Mean Gating across all 707 OOF Stems.

Strictly leak-free:
  - Uses ONLY clean competition-only EXP-006/EXP-007 + work/cache streams (NO F1 / NO MAGFiLO_extra).
  - Evaluates parity against Track-A Clean (OOF PQ=0.46613) first, then sweeps post-processing
    under strict 5-Fold Nested Date-CV + Paired Date-Cluster Bootstrap (N=1000) + Sim_r80_PQ.
  - Can also export a verified submission CSV (--export) validated via submission.validate & test_shift_audit.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "gap_research" / "scripts"))

import build_super_submission as bss  # noqa: E402
import exp010_border_g2_and_sahi as e10  # noqa: E402
import exp011b_hurdle_antichain as e11b  # noqa: E402
from filseg import data, metric, submission  # noqa: E402

FULL = 2048


@dataclass(frozen=True)
class ModeSpec:
    name: str
    w_ms: dict[str, float]
    w_sm: dict[str, float]
    w_sahi: dict[str, float]
    border_g2: bool
    w_sahi_cons: float
    sahi_boost: float
    match_iou: float
    small_px: int = 120
    small_scale: float = 1.20


MODES: dict[str, ModeSpec] = {
    "M0_cont_tracka": ModeSpec(
        name="M0_cont_tracka",
        w_ms={"e4": 0.10, "sam2": 0.55, "s1e_g2": 0.25, "e4x": 0.10},
        w_sm={"e4": 0.30, "s1e_g2": 0.50, "e4x": 0.20},
        w_sahi={"e4": 0.25, "e4_g1": 0.25, "s1e": 0.25, "sam2": 0.25},
        border_g2=False,
        w_sahi_cons=0.0,
        sahi_boost=0.0,
        match_iou=0.40,
    ),
    "M1_g2_sahiwt": ModeSpec(
        name="M1_g2_sahiwt",
        w_ms={"e4": 0.08, "sam2": 0.32, "sam2_g2": 0.22, "s1e_g2": 0.24, "e4x": 0.07, "e4x_g2": 0.07},
        w_sm={"e4": 0.24, "s1e_g2": 0.46, "e4x": 0.15, "e4x_g2": 0.15},
        w_sahi={"sam2": 0.40, "s1e": 0.30, "e4_g1": 0.20, "e4": 0.10},
        border_g2=True,
        w_sahi_cons=0.0,
        sahi_boost=0.0,
        match_iou=0.40,
    ),
    "M2_g2_sahijoint": ModeSpec(
        name="M2_g2_sahijoint",
        w_ms={"e4": 0.08, "sam2": 0.32, "sam2_g2": 0.22, "s1e_g2": 0.24, "e4x": 0.07, "e4x_g2": 0.07},
        w_sm={"e4": 0.24, "s1e_g2": 0.46, "e4x": 0.15, "e4x_g2": 0.15},
        w_sahi={"sam2": 0.40, "s1e": 0.30, "e4_g1": 0.20, "e4": 0.10},
        border_g2=True,
        w_sahi_cons=0.65,
        sahi_boost=0.03,
        match_iou=0.40,
    ),
}


def _stream_continuous_props(
    seg_data: dict[str, tuple],
    weights: dict[str, float],
    pre_floor: float,
    small_px: int,
    small_scale: float,
    border_g2: bool,
    min_keep_u8: int = 85,
    ref_t_u8: int = 166,
) -> list[tuple[float, tuple[int, int, int, int], np.ndarray, np.ndarray, int]]:
    """Build continuous [0..255] crop proposals per detector stream without premature binarization or occ clipping."""
    loaded = [(k, weights[k], *seg_data[k]) for k in weights if weights[k] > 0 and k in seg_data]
    if not loaded:
        return []

    anchor_idx = next((j for j, item in enumerate(loaded) if item[0] == "e4"), 0)
    _, _, conf0, box0, _ = loaded[anchor_idx]
    if len(conf0) == 0:
        return []

    g2_keys = ("e4x_g2", "s1e_g2", "sam2_g2", "e4_g1")
    props = []
    for idx in np.argsort(-conf0):
        raw_c = float(conf0[idx])
        bx0, by0, bx1, by1 = [int(v) for v in box0[idx]]
        eff_c = raw_c * (small_scale if max(bx1 - bx0, by1 - by0) < small_px else 1.0)
        if eff_c < pre_floor or bx1 <= bx0 or by1 <= by0:
            continue

        ux0, uy0, ux1, uy1 = bx0, by0, bx1, by1
        if border_g2:
            for k, _, _, b_arr, _ in loaded:
                if k in g2_keys and idx < len(b_arr):
                    sx1, sy1, sx2, sy2 = [int(v) for v in b_arr[idx]]
                    if sx2 > sx1 and sy2 > sy1:
                        ux0 = min(ux0, sx1)
                        uy0 = min(uy0, sy1)
                        ux1 = max(ux1, sx2)
                        uy1 = max(uy1, sy2)

        uh, uw = uy1 - uy0, ux1 - ux0
        acc = np.zeros((uh, uw), dtype=np.float32)
        wsum = np.zeros((uh, uw), dtype=np.float32)
        g2_acc = np.zeros((uh, uw), dtype=np.float32) if border_g2 else None
        g2_wsum = np.zeros((uh, uw), dtype=np.float32) if border_g2 else None

        for k, w, _, b_arr, p_list in loaded:
            if idx >= len(b_arr):
                continue
            sx1, sy1, sx2, sy2 = [int(v) for v in b_arr[idx]]
            if sx2 <= sx1 or sy2 <= sy1:
                continue
            pr = p_list[idx]
            if pr.shape != (sy2 - sy1, sx2 - sx1):
                pr = cv2.resize(pr, (sx2 - sx1, sy2 - sy1), interpolation=cv2.INTER_LINEAR)
            if not border_g2:
                ix1, iy1, ix2, iy2 = max(bx0, sx1), max(by0, sy1), min(bx1, sx2), min(by1, sy2)
                if ix2 > ix1 and iy2 > iy1:
                    sl_u = (slice(iy1 - uy0, iy2 - uy0), slice(ix1 - ux0, ix2 - ux0))
                    sl_s = (slice(iy1 - sy1, iy2 - sy1), slice(ix1 - sx1, ix2 - sx1))
                    acc[sl_u] += (w / 255.0) * pr[sl_s]
                    wsum[sl_u] += w
            else:
                ix1, iy1, ix2, iy2 = max(ux0, sx1), max(uy0, sy1), min(ux1, sx2), min(uy1, sy2)
                if ix2 > ix1 and iy2 > iy1:
                    sl_u = (slice(iy1 - uy0, iy2 - uy0), slice(ix1 - ux0, ix2 - ux0))
                    sl_s = (slice(iy1 - sy1, iy2 - sy1), slice(ix1 - sx1, ix2 - sx1))
                    acc[sl_u] += (w / 255.0) * pr[sl_s]
                    wsum[sl_u] += w
                    if k in g2_keys:
                        g2_acc[sl_u] += (w / 255.0) * pr[sl_s]
                        g2_wsum[sl_u] += w

        if not (wsum > 0).any():
            continue

        prob_u8 = np.clip(np.where(wsum > 0, (acc / np.maximum(wsum, 1e-6)) * 255.0 + 0.5, 0.0), 0.0, 255.0).astype(np.uint8)
        ox0, oy0, ox1, oy1 = bx0 - ux0, by0 - uy0, bx1 - ux0, by1 - uy0
        if border_g2 and (ux0 < bx0 or uy0 < by0 or ux1 > bx1 or uy1 > by1):
            in_mask = np.zeros((uh, uw), dtype=bool)
            in_mask[oy0:oy1, ox0:ox1] = prob_u8[oy0:oy1, ox0:ox1] >= 140
            if in_mask.any() and (g2_wsum > 0).any():
                g2_u8 = np.clip(np.where(g2_wsum > 0, (g2_acc / np.maximum(g2_wsum, 1e-6)) * 255.0 + 0.5, 0.0), 0.0, 255.0).astype(np.uint8)
                # Smoothly populate exterior from _g2 blend, keep only components touching box0 interior
                prob_mix = prob_u8.copy()
                ext_zone = np.ones((uh, uw), dtype=bool)
                ext_zone[oy0:oy1, ox0:ox1] = False
                prob_mix[ext_zone] = g2_u8[ext_zone]
                sup = (prob_mix >= min_keep_u8) & (in_mask | ext_zone)
                bridge = cv2.dilate(in_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                _, labels = cv2.connectedComponents((sup | bridge).astype(np.uint8), connectivity=8)
                keep_cc = np.unique(labels[in_mask])
                keep_cc = keep_cc[keep_cc > 0]
                allowed = np.zeros((uh, uw), dtype=bool)
                allowed[oy0:oy1, ox0:ox1] = True
                if len(keep_cc):
                    allowed |= ext_zone & np.isin(labels, keep_cc)
                prob_u8[~allowed] = 0
            else:
                # Restrict cleanly to box0 if interior has no strong filament
                tmp = np.zeros_like(prob_u8)
                tmp[oy0:oy1, ox0:ox1] = prob_u8[oy0:oy1, ox0:ox1]
                prob_u8 = tmp

        ys, xs = np.nonzero(prob_u8 >= min_keep_u8)
        if not len(ys):
            continue
        x1, x2, y1, y2 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        p_trim = prob_u8[y1:y2, x1:x2]
        ref_m = p_trim > ref_t_u8
        ref_a = int(ref_m.sum())
        if ref_a == 0:
            # Keep if above min_keep for hysteresis tails? Still ensure valid ref using >=135 fallback for matching
            ref_m = p_trim > 135
            ref_a = int(ref_m.sum())
            if ref_a == 0:
                continue
        props.append((eff_c, (ux0 + x1, uy0 + y1, ux0 + x2, uy0 + y2), p_trim, ref_m, ref_a))
    return props


def _cluster_stem_continuous(stem: str, split: str, mode: ModeSpec) -> list[dict]:
    """Cluster continuous detector crops across ms/s/main (+ optional SAHI) and return cluster probability maps."""
    bundle = e10.load_stem_bundle(stem, split)
    stream_weights = {"ms": 1.35, "s": 1.05, "main": 0.95}
    base_total_w = sum(stream_weights.values())

    items = []
    streams_cfg = [
        ("ms", 1.35, mode.w_ms, 0.24),
        ("s", 1.05, mode.w_sm, 0.24),
        ("main", 0.95, mode.w_sm, 0.22),
    ]
    if mode.w_sahi_cons > 0:
        streams_cfg.append(("SAHI", mode.w_sahi_cons, mode.w_sahi, 0.18))

    for sname, sw, wdict, pfl in streams_cfg:
        props = _stream_continuous_props(
            bundle[sname], wdict, pre_floor=pfl,
            small_px=mode.small_px, small_scale=mode.small_scale, border_g2=mode.border_g2,
        )
        for c, box, p_u8, ref_m, ref_a in props:
            items.append((sname, sw, c, box, p_u8, ref_m, ref_a))

    items.sort(key=lambda x: -(x[2] * min(1.5, x[1])))
    clusters = []
    for sname, sw, c, box, p_u8, ref_m, ref_a in items:
        x1, y1, x2, y2 = box
        best_j, best_iou = -1, 0.0
        for j, cl in enumerate(clusters):
            if sname in cl["seen"]:
                continue
            rx1, ry1, rx2, ry2 = cl["rbox"]
            ix1, iy1, ix2, iy2 = max(x1, rx1), max(y1, ry1), min(x2, rx2), min(y2, ry2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = int(
                (ref_m[iy1 - y1 : iy2 - y1, ix1 - x1 : ix2 - x1]
                 & cl["rmask"][iy1 - ry1 : iy2 - ry1, ix1 - rx1 : ix2 - rx1]).sum()
            )
            if inter:
                iou = inter / (ref_a + cl["rarea"] - inter)
                if iou >= mode.match_iou and iou > best_iou:
                    best_iou = iou
                    best_j = j
        if best_j < 0:
            clusters.append({
                "seen": {sname},
                "items": [(sname, sw, c, box, p_u8)],
                "rbox": box,
                "rmask": ref_m,
                "rarea": ref_a,
                "ubox": [x1, y1, x2, y2],
            })
        else:
            cl = clusters[best_j]
            cl["seen"].add(sname)
            cl["items"].append((sname, sw, c, box, p_u8))
            ub = cl["ubox"]
            ub[0], ub[1], ub[2], ub[3] = min(ub[0], x1), min(ub[1], y1), max(ub[2], x2), max(ub[3], y2)

    out = []
    for cl in clusters:
        gx1, gy1, gx2, gy2 = cl["ubox"]
        acc = np.zeros((gy2 - gy1, gx2 - gx1), dtype=np.float32)
        w_base = 0.0
        c_base_sum = 0.0
        c_sahi = 0.0
        w_present = 0.0
        for sname, sw, c, (bx1, by1, bx2, by2), p_u8 in cl["items"]:
            acc[by1 - gy1 : by2 - gy1, bx1 - gx1 : bx2 - gx1] += (sw / 255.0) * p_u8
            w_present += sw
            if sname == "SAHI":
                c_sahi = max(c_sahi, c)
            else:
                w_base += sw
                c_base_sum += sw * c
        # Score normalization: base streams determine base score, SAHI match boosts score without penalizing large filaments
        c_score = (c_base_sum / base_total_w) + (mode.sahi_boost * c_sahi if c_sahi > 0 else 0.0)
        if mode.w_sahi_cons > 0 and w_base == 0 and c_sahi > 0:
            c_score = 0.90 * c_sahi  # pure SAHI cluster weighted appropriately; filtered later by rescue rules if small
        if c_score < 0.18:
            continue
        p_cl = np.clip(np.rint((acc / max(w_present, 1e-6)) * 255.0), 0.0, 255.0).astype(np.uint8)
        out.append({
            "score": float(c_score),
            "box": (gx1, gy1, gx2, gy2),
            "prob_u8": p_cl,
            "n_streams": len(cl["items"]),
            "is_sahi_only": (w_base == 0 and c_sahi > 0),
        })

    # Also build SAHI continuous props for disjoint rescue (small objects)
    sahi_props = _stream_continuous_props(
        bundle["SAHI"], mode.w_sahi, pre_floor=0.18,
        small_px=mode.small_px, small_scale=mode.small_scale, border_g2=False,
    )
    for c, box, p_u8, _, _ in sahi_props:
        out.append({
            "score": float(c),
            "box": box,
            "prob_u8": p_u8,
            "n_streams": 1,
            "is_sahi_only": True,
            "is_rescue_pool": True,
        })
    return out


def _build_cache_chunk(args: tuple[list[str], str, ModeSpec]) -> dict[str, list[dict]]:
    stems, split, mode = args
    cv2.setNumThreads(0)
    return {s: _cluster_stem_continuous(s, split, mode) for s in stems}


def build_mode_clusters(stems: list[str], split: str, mode: ModeSpec, workers: int = 7) -> dict[str, list[dict]]:
    chunks = [stems[i::workers] for i in range(workers)]
    tasks = [(ch, split, mode) for ch in chunks if ch]
    out: dict[str, list[dict]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(tasks)) as ex:
        for part in ex.map(_build_cache_chunk, tasks):
            out.update(part)
    return out


@dataclass(frozen=True)
class DecodeCfg:
    name: str
    mode: str
    thr_hi: float
    thr_lo: float
    c_floor: float
    min_peak: float = 0.0
    min_mean: float = 0.0
    drop: float = 0.90
    min_area: int = 100
    offset: int = 0
    sahi_rescue_conf: float = 0.22
    sahi_max_area: int = 400
    sahi_prob_thr: float = 0.68
    fill_holes: bool = False


def _hysteresis_mask(prob_u8: np.ndarray, t_hi: int, t_lo: int) -> np.ndarray | None:
    m_hi = prob_u8 > t_hi
    if not m_hi.any():
        return None
    if t_lo >= t_hi:
        return m_hi
    m_lo = prob_u8 > t_lo
    n_cc, labels = cv2.connectedComponents(m_lo.astype(np.uint8), connectivity=8)
    if n_cc <= 1:
        return m_hi
    keep_ids = np.unique(labels[m_hi])
    keep_ids = keep_ids[keep_ids > 0]
    if not len(keep_ids):
        return m_hi
    return np.isin(labels, keep_ids)


def decode_stem_clusters(clusters: list[dict], cfg: DecodeCfg) -> list[str]:
    taken = np.zeros((FULL, FULL), dtype=bool)
    t_hi = int(cfg.thr_hi * 255.0 + 0.5)
    t_lo = int(cfg.thr_lo * 255.0 + 0.5)
    t_sahi = int(cfg.sahi_prob_thr * 255.0 + 0.5)

    # First pass: consensus clusters (non-rescue)
    cons = [cl for cl in clusters if not cl.get("is_rescue_pool") and not (cl.get("is_sahi_only") and cl["score"] < cfg.c_floor)]
    cons.sort(key=lambda x: -x["score"])
    out_rles = []

    for cl in cons:
        if cl["score"] < cfg.c_floor:
            break
        m = _hysteresis_mask(cl["prob_u8"], t_hi, t_lo)
        if m is None:
            continue
        if cfg.offset:
            mu8 = (m).astype(np.uint8)
            m = (cv2.dilate if cfg.offset > 0 else cv2.erode)(mu8, bss._K[abs(cfg.offset)]).astype(bool)
        if cfg.fill_holes and m.any():
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros_like(m, dtype=np.uint8)
            cv2.drawContours(filled, cnts, -1, 1, thickness=-1)
            m = filled.astype(bool)
        if not m.any():
            continue
        if cfg.min_peak > 0.0 and float(cl["prob_u8"][m].max()) / 255.0 < cfg.min_peak:
            continue
        if cfg.min_mean > 0.0 and float(cl["prob_u8"][m].mean()) / 255.0 < cfg.min_mean:
            continue

        x1, y1, x2, y2 = cl["box"]
        keep = m & ~taken[y1:y2, x1:x2]
        area = int(keep.sum())
        if area >= cfg.min_area and area >= cfg.drop * int(m.sum()):
            taken[y1:y2, x1:x2] |= keep
            v = bss.trim(cl["score"], (x1, y1, x2, y2), keep)
            if v is not None:
                out_rles.append(bss.encode_crop_ascii(v[1], v[2]))

    # Second pass: SAHI small-object disjoint rescue
    if cfg.sahi_rescue_conf < 5.0:
        rescues = [cl for cl in clusters if cl.get("is_rescue_pool")]
        rescues.sort(key=lambda x: -x["score"])
        for cl in rescues:
            if cl["score"] < cfg.sahi_rescue_conf:
                break
            m = _hysteresis_mask(cl["prob_u8"], t_sahi, min(t_sahi, t_lo if cfg.sahi_prob_thr > cfg.thr_lo else t_sahi))
            if m is None:
                continue
            area_raw = int(m.sum())
            if area_raw < cfg.min_area or area_raw > cfg.sahi_max_area:
                continue
            if cfg.min_peak > 0.0 and float(cl["prob_u8"][m].max()) / 255.0 < max(0.70, cfg.min_peak - 0.06):
                continue
            x1, y1, x2, y2 = cl["box"]
            keep = m & ~taken[y1:y2, x1:x2]
            area = int(keep.sum())
            if area >= cfg.min_area and area >= 0.85 * area_raw:
                taken[y1:y2, x1:x2] |= keep
                v = bss.trim(cl["score"], (x1, y1, x2, y2), keep)
                if v is not None:
                    out_rles.append(bss.encode_crop_ascii(v[1], v[2]))
    return out_rles


def _eval_worker(args: tuple[list[str], dict[str, list[dict]], dict, list[DecodeCfg]]) -> dict[str, dict[str, dict]]:
    stems, sub_clusters, sub_recs, cfgs = args
    cv2.setNumThreads(0)
    out = {c.name: {} for c in cfgs}
    for s in stems:
        recs = sub_recs[s]
        cl_s = sub_clusters[s]
        for cfg in cfgs:
            rstrs = decode_stem_clusters(cl_s, cfg)
            out[cfg.name][s] = metric.counts([{"size": [FULL, FULL], "counts": r.encode()} for r in rstrs], recs)
    return out


def build_decode_grid() -> list[DecodeCfg]:
    cfgs = [
        # Baseline continuous parity against Track-A
        DecodeCfg("M0_base_68", "M0_cont_tracka", thr_hi=0.68, thr_lo=0.68, c_floor=0.28),
        DecodeCfg("M1_base_68", "M1_g2_sahiwt", thr_hi=0.68, thr_lo=0.68, c_floor=0.28),
        DecodeCfg("M2_base_68", "M2_g2_sahijoint", thr_hi=0.68, thr_lo=0.68, c_floor=0.28),
    ]
    for mode in ("M0_cont_tracka", "M1_g2_sahiwt", "M2_g2_sahijoint"):
        mshort = mode[:2]
        for hi, lo in ((0.68, 0.56), (0.68, 0.60), (0.70, 0.56), (0.72, 0.56), (0.72, 0.60), (0.74, 0.58), (0.66, 0.56), (0.66, 0.66)):
            for cf in (0.26, 0.28, 0.30, 0.32):
                cfgs.append(DecodeCfg(f"{mshort}_h{int(hi*100)}_l{int(lo*100)}_c{int(cf*100)}", mode, thr_hi=hi, thr_lo=lo, c_floor=cf))
        # Dariushafshar-inspired peak + mean gating + hysteresis on top region
        for hi, lo in ((0.70, 0.56), (0.72, 0.56), (0.74, 0.58)):
            for pk, mn in ((0.85, 0.72), (0.90, 0.74), (0.92, 0.76), (0.88, 0.0)):
                for cf in (0.26, 0.28):
                    cfgs.append(
                        DecodeCfg(
                            f"{mshort}_h{int(hi*100)}_l{int(lo*100)}_pk{int(pk*100)}_mn{int(mn*100)}_c{int(cf*100)}",
                            mode, thr_hi=hi, thr_lo=lo, c_floor=cf, min_peak=pk, min_mean=mn,
                        )
                    )
        # Drop + offset + fill_holes + sahi area sensitivity around top hysteresis
        for drp in (0.85, 0.90):
            for fill in (False, True):
                for sma in (400, 520):
                    cfgs.append(
                        DecodeCfg(
                            f"{mshort}_h70_l56_c28_d{int(drp*100)}_f{int(fill)}_sa{sma}",
                            mode, thr_hi=0.70, thr_lo=0.56, c_floor=0.28, drop=drp, fill_holes=fill, sahi_max_area=sma,
                        )
                    )
    return cfgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(7, os.cpu_count() or 4))
    ap.add_argument("--export", type=Path, default=None)
    args = ap.parse_args()

    t0 = time.time()
    folds = data.folds()
    all_stems = sorted(folds)
    gt_recs = metric.load_records(all_stems)
    ta_cache = pickle.load(open("/private/tmp/tracka_clean_oof_cache.pkl", "rb"))["counts"]
    ta_sm = metric.summarize([ta_cache[s] for s in all_stems])
    print(f"[Track-A Clean Ref] PQ={ta_sm['pq']:.5f} SQ={ta_sm['sq']:.5f} RQ={ta_sm['rq']:.5f} TP={ta_sm['tp']} FP={ta_sm['fp']} FN={ta_sm['fn']}", flush=True)

    cfgs = build_decode_grid()
    all_cnt = {c.name: {} for c in cfgs}
    chunks = [all_stems[i::args.workers] for i in range(args.workers)]

    # Process 1 mode at a time to keep RAM tiny on 8GB Mac and support spawn cleanly
    for mname, mspec in MODES.items():
        t1 = time.time()
        cl_map = build_mode_clusters(all_stems, "oof", mspec, workers=args.workers)
        n_cl = sum(len(v) for v in cl_map.values())
        m_cfgs = [c for c in cfgs if c.mode == mname]
        print(f"Mode [{mname}]: built {n_cl} clusters in {time.time()-t1:.1f}s; scoring {len(m_cfgs)} configs...", flush=True)
        tasks = [(ch, {s: cl_map[s] for s in ch}, {s: gt_recs[s] for s in ch}, m_cfgs) for ch in chunks if ch]
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(tasks)) as ex:
            for part in ex.map(_eval_worker, tasks):
                for cname, d in part.items():
                    all_cnt[cname].update(d)
    print(f"Evaluated all {len(cfgs)} configs in {time.time()-t0:.1f}s", flush=True)

    ranked = []
    for cfg in cfgs:
        sm = metric.summarize([all_cnt[cfg.name][s] for s in all_stems])
        ranked.append((sm["pq"], sm["pq"] - ta_sm["pq"], sm["sq"], sm["rq"], sm["tp"], sm["fp"], sm["fn"], cfg))
    ranked.sort(key=lambda x: -x[0])

    print("\nBaselines vs Top 15 Single Configs:")
    for cname in ("M0_base_68", "M1_base_68", "M2_base_68"):
        sm = metric.summarize([all_cnt[cname][s] for s in all_stems])
        print(f"  {cname:38s} PQ={sm['pq']:.5f} ({sm['pq']-ta_sm['pq']:+.5f}) SQ={sm['sq']:.5f} RQ={sm['rq']:.5f} TP={sm['tp']} FP={sm['fp']} FN={sm['fn']}")
    print("  ---")
    for pq, d, sq, rq, tp, fp, fn, cfg in ranked[:15]:
        print(f"  {cfg.name:38s} PQ={pq:.5f} ({d:+.5f}) SQ={sq:.5f} RQ={rq:.5f} TP={tp} FP={fp} FN={fn}")

    # Strict 5-Fold Nested Date-CV (rebuilds clusters only for chosen modes to keep RAM tiny)
    nested_summaries = {}
    chosen_modes_needed = set()
    fold_picks = {}
    for subset_name, filter_fn in (
        ("Nested-All", lambda c: True),
        ("Nested-M0", lambda c: c.mode == "M0_cont_tracka"),
        ("Nested-M1", lambda c: c.mode == "M1_g2_sahiwt"),
        ("Nested-M2", lambda c: c.mode == "M2_g2_sahijoint"),
    ):
        sub_cfgs = [c for c in cfgs if filter_fn(c)]
        chosen = {}
        for fk in range(5):
            tr = [s for s in all_stems if folds[s] != fk]
            best_c = max(sub_cfgs, key=lambda c: metric.summarize([all_cnt[c.name][s] for s in tr])["pq"])
            chosen[fk] = best_c.name
            chosen_modes_needed.add(best_c.mode)
        fold_picks[subset_name] = chosen

    cfg_map = {c.name: c for c in cfgs}
    # Build clusters only once per needed mode for Sim_r80
    needed_clusters = {m: build_mode_clusters(all_stems, "oof", MODES[m], workers=args.workers) for m in sorted(chosen_modes_needed)}
    for subset_name, chosen in fold_picks.items():
        nested_rows, nested_rles = {}, {}
        for fk in range(5):
            va = [s for s in all_stems if folds[s] == fk]
            best_c = cfg_map[chosen[fk]]
            for s in va:
                nested_rows[s] = all_cnt[best_c.name][s]
                nested_rles[s] = decode_stem_clusters(needed_clusters[best_c.mode][s], best_c)
        sm_n = metric.summarize(list(nested_rows.values()))
        boot = e11b._paired_date_bootstrap(nested_rows, ta_cache, all_stems, n_boot=1000, seed=42)
        s80m, s80s = e11b._sim_r80(all_stems, nested_rles, gt_recs, trials=10)
        print(
            f"\n[{subset_name:12s}] PQ={sm_n['pq']:.5f} ({sm_n['pq']-ta_sm['pq']:+.5f}) SQ={sm_n['sq']:.5f} RQ={sm_n['rq']:.5f} "
            f"TP={sm_n['tp']} FP={sm_n['fp']} FN={sm_n['fn']} | boot={boot['mean_gain']:+.5f} P={boot['p_win']:.4f} Sim80={s80m:.5f}+/-{s80s:.5f}"
        )
        print(f"  chosen={chosen}")
        nested_summaries[subset_name] = {
            **sm_n,
            "delta_vs_ta": sm_n["pq"] - ta_sm["pq"],
            "boot_mean": boot["mean_gain"],
            "boot_p": boot["p_win"],
            "sim80_mean": s80m,
            "sim80_std": s80s,
            "chosen": chosen,
        }

    Path("/private/tmp/exp014_summary.json").write_text(json.dumps({
        "tracka": ta_sm,
        "top15": [dict(name=c.name, pq=pq, delta=d, sq=sq, rq=rq, tp=tp, fp=fp, fn=fn) for pq, d, sq, rq, tp, fp, fn, c in ranked[:15]],
        "nested": nested_summaries,
    }, indent=2))

    best_overall = ranked[0][7]
    if args.export is not None:
        test_stems = list(data.test_stems())
        mspec = MODES[best_overall.mode]
        t_cl = build_mode_clusters(test_stems, "test", mspec, workers=args.workers)
        rles_test = {
            s: [{"size": [FULL, FULL], "counts": r.encode()} for r in decode_stem_clusters(t_cl[s], best_overall)]
            for s in test_stems
        }
        args.export.parent.mkdir(parents=True, exist_ok=True)
        submission.write_csv(args.export, rles_test)
        probs = submission.validate(args.export, test_stems)
        assert not probs, f"Validation errors: {probs}"
        print(f"\nExported verified submission {args.export} using {best_overall.name}", flush=True)


if __name__ == "__main__":
    main()
