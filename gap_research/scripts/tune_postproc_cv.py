"""5-Fold OOF & Nested Date-CV Post-Processing Research (Hysteresis, Island Clean, Drop/MinArea, Small Boost).

Evaluates strictly clean post-processing hypotheses across BOTH:
  - Track A: 707-frame 5-Fold OOF (`ms + s + main`)
  - Track B: 145-frame Fold-0 (`E19_no_F1`) with Nested Date-CV
Reports PQ, SQ, RQ, <580px bin PQ, and near-miss / merged-FP stats via filseg.metric.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "gap_research" / "scripts"))

import build_gap_submission as bgs  # noqa: E402
import build_super_submission as bss  # noqa: E402
import eval_harness as eh  # noqa: E402
import eval_wave2 as ew2  # noqa: E402
from filseg import data, metric, postprocess  # noqa: E402


def _hysteresis_crop(prob_u8: np.ndarray, thr_hi: float, thr_lo: float, keep_cc_min: int = 0) -> np.ndarray:
    hi = prob_u8 > int(thr_hi * 255.0 + 0.5)
    if not hi.any():
        return hi
    if thr_lo >= thr_hi:
        m = hi
    else:
        lo = (prob_u8 > int(thr_lo * 255.0 + 0.5)).astype(np.uint8)
        n_cc, lbl = cv2.connectedComponents(lo, connectivity=8)
        if n_cc <= 2 and n_cc == 1:
            return np.zeros_like(hi)
        keep_ids = np.unique(lbl[hi])
        keep_ids = keep_ids[keep_ids > 0]
        if len(keep_ids) == 0:
            return np.zeros_like(hi)
        m = np.isin(lbl, keep_ids)
    if keep_cc_min > 0:
        n_cc, lbl, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        if n_cc > 2:
            areas = stats[:, cv2.CC_STAT_AREA]
            valid = [i for i in range(1, n_cc) if areas[i] >= keep_cc_min]
            if valid:
                m = np.isin(lbl, valid)
    return m


def _paint_from_loaded(
    raw_inst: list[tuple[float, tuple[int, int, int, int], np.ndarray]],
    thr_hi: float,
    thr_lo: float,
    conf_floor: float,
    drop: float,
    min_area: int,
    small_scale: float = 1.0,
    small_px: int = 120,
    keep_cc_min: int = 0,
    offset: int = 0,
    large_thr: float | None = None,
    large_drop: float | None = None,
    large_px: int = 240,
):
    if not raw_inst:
        return []
    occ = np.zeros((2048, 2048), dtype=bool)
    out = []
    for raw_c, (x1, y1, x2, y2), p in raw_inst:
        if x2 <= x1 or y2 <= y1:
            continue
        span = max(x2 - x1, y2 - y1)
        eff_c = float(raw_c) * (small_scale if span < small_px else 1.0)
        if eff_c < conf_floor:
            continue
        use_hi = large_thr if (large_thr is not None and span >= large_px) else thr_hi
        use_lo = use_hi if (large_thr is not None and span >= large_px) else thr_lo
        use_drop = large_drop if (large_drop is not None and span >= large_px) else drop
        m = _hysteresis_crop(p, use_hi, use_lo, keep_cc_min=keep_cc_min)
        if offset:
            m_u8 = m.astype(np.uint8)
            m = (cv2.dilate if offset > 0 else cv2.erode)(m_u8, bss._K[abs(offset)]).astype(bool)
        if m.shape != (y2 - y1, x2 - x1):
            m = cv2.resize(m.astype(np.uint8), (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST).astype(bool)
        keep = m & ~occ[y1:y2, x1:x2]
        area = int(keep.sum())
        if area > min_area and area > 0 and area >= use_drop * int(m.sum()):
            occ[y1:y2, x1:x2] |= keep
            v = bss.trim(eff_c, (x1, y1, x2, y2), keep)
            if v is not None:
                out.append(v)
    return out


def eval_track_a_cfg(stems: list[str], preloaded: dict[str, dict], cfg: dict) -> tuple[dict, dict]:
    dec = {}
    for k in ("ms", "s", "main"):
        cf = {"ms": 0.30, "s": 0.30, "main": 0.25}[k]
        dec[k] = {
            s: _paint_from_loaded(
                preloaded[k][s],
                thr_hi=cfg["thr_hi"],
                thr_lo=cfg["thr_lo"],
                conf_floor=cf,
                drop=cfg["drop"],
                min_area=cfg["min_area"],
                small_scale=cfg["small_scale"],
                keep_cc_min=cfg["keep_cc_min"],
                offset=cfg["offset"],
                large_thr=cfg.get("large_thr"),
                large_drop=cfg.get("large_drop"),
                large_px=cfg.get("large_px", 240),
            )
            for s in stems
        }
    rles = {
        s: bgs.weighted_consensus_stem(
            [("ms", 1.25, dec["ms"][s]), ("s", 1.0, dec["s"][s]), ("main", 0.9, dec["main"][s])],
            match_iou=cfg["match_iou"],
            vote_thr=cfg["vote_thr"],
            c_floor=cfg["c_floor"],
            min_area=cfg["min_area"],
            drop=cfg["drop"],
        )
        for s in stems
    }
    cnt = eh.score_rle_strs_dict(rles, stems=stems)
    return eh.score_from_counts(cnt, stems), cnt


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", action="store_true", help="Run focused joint 707-frame post-proc sweep")
    ap.add_argument("--joint2", action="store_true", help="Run fine boundary check around thr64_drop82_sm25")
    ap.add_argument("--joint3", action="store_true", help="Test high drop (0.85..0.95) and thr (0.64..0.70)")
    ap.add_argument("--joint4", action="store_true", help="Fine check on c_floor (0.24..0.28) and thr (0.66..0.70)")
    ap.add_argument("--size-aware", action="store_true", help="Test crisp small vs connected large crop thr/drop")
    ap.add_argument("--joint5", action="store_true", help="Test vote_thr x c_floor x small_scale precision/recall frontier")
    args = ap.parse_args()

    t0 = time.time()
    stems_707 = sorted(eh.GT.keys())
    fold_of = data.folds()
    dirs_a = {k: REPO / "work" / "cache" / k / "oof" for k in ("ms", "s", "main")}
    preloaded = {k: {s: bss.load_raw_npz(dirs_a[k] / f"{s}.npz", tta=4) for s in stems_707} for k in ("ms", "s", "main")}
    print(f"Preloaded 707x3 caches in {time.time()-t0:.1f}s", flush=True)

    base_cfg = dict(thr_hi=0.60, thr_lo=0.60, drop=0.70, min_area=100, small_scale=1.0, keep_cc_min=0, offset=0, match_iou=0.40, vote_thr=0.45, c_floor=0.30)
    if args.joint5:
        ref = {**base_cfg, "thr_hi": 0.68, "thr_lo": 0.68, "drop": 0.90}
        cands = [
            ("t68_d90_sm30_v42_c28", {**ref, "small_scale": 1.30, "vote_thr": 0.42, "c_floor": 0.28}),
            ("t68_d90_sm30_v48_c28", {**ref, "small_scale": 1.30, "vote_thr": 0.48, "c_floor": 0.28}),
            ("t68_d90_sm30_v52_c28", {**ref, "small_scale": 1.30, "vote_thr": 0.52, "c_floor": 0.28}),
            ("t68_d90_sm30_v56_c28", {**ref, "small_scale": 1.30, "vote_thr": 0.56, "c_floor": 0.28}),
            ("t68_d90_sm30_v45_c30", {**ref, "small_scale": 1.30, "vote_thr": 0.45, "c_floor": 0.30}),
            ("t68_d90_sm35_v52_c30", {**ref, "small_scale": 1.35, "vote_thr": 0.52, "c_floor": 0.30}),
            ("t68_d90_sm20_v52_c28", {**ref, "small_scale": 1.20, "vote_thr": 0.52, "c_floor": 0.28}),
            ("t68_d90_sm25_v55_c29", {**ref, "small_scale": 1.25, "vote_thr": 0.55, "c_floor": 0.29}),
        ]
    elif args.size_aware:
        ref = {**base_cfg, "thr_hi": 0.68, "thr_lo": 0.68, "drop": 0.90, "small_scale": 1.30, "vote_thr": 0.42, "c_floor": 0.28}
        cands = [
            ("t68_d90_sm30_c28", ref),
            ("sa_lt240_lg60_d75", {**ref, "large_thr": 0.60, "large_drop": 0.75, "large_px": 240}),
            ("sa_lt240_lg62_d80", {**ref, "large_thr": 0.62, "large_drop": 0.80, "large_px": 240}),
            ("sa_lt300_lg60_d75", {**ref, "large_thr": 0.60, "large_drop": 0.75, "large_px": 300}),
            ("sa_lt180_lg62_d80", {**ref, "large_thr": 0.62, "large_drop": 0.80, "large_px": 180}),
            ("sa_lt180_lg64_d85", {**ref, "large_thr": 0.64, "large_drop": 0.85, "large_px": 180}),
        ]
    elif args.joint4:
        cands = [
            ("t66_d88_sm30_c28", {**base_cfg, "thr_hi": 0.66, "thr_lo": 0.66, "drop": 0.88, "small_scale": 1.30, "vote_thr": 0.42, "c_floor": 0.28}),
            ("t68_d90_sm30_c28", {**base_cfg, "thr_hi": 0.68, "thr_lo": 0.68, "drop": 0.90, "small_scale": 1.30, "vote_thr": 0.42, "c_floor": 0.28}),
            ("t68_d90_sm30_c26", {**base_cfg, "thr_hi": 0.68, "thr_lo": 0.68, "drop": 0.90, "small_scale": 1.30, "vote_thr": 0.42, "c_floor": 0.26}),
            ("t68_d90_sm30_c24", {**base_cfg, "thr_hi": 0.68, "thr_lo": 0.68, "drop": 0.90, "small_scale": 1.30, "vote_thr": 0.42, "c_floor": 0.24}),
            ("t66_d90_sm30_c26", {**base_cfg, "thr_hi": 0.66, "thr_lo": 0.66, "drop": 0.90, "small_scale": 1.30, "vote_thr": 0.42, "c_floor": 0.26}),
            ("t66_d90_sm35_c26", {**base_cfg, "thr_hi": 0.66, "thr_lo": 0.66, "drop": 0.90, "small_scale": 1.35, "vote_thr": 0.42, "c_floor": 0.26}),
            ("t70_d90_sm30_c26", {**base_cfg, "thr_hi": 0.70, "thr_lo": 0.70, "drop": 0.90, "small_scale": 1.30, "vote_thr": 0.42, "c_floor": 0.26}),
            ("t68_d92_sm30_c26", {**base_cfg, "thr_hi": 0.68, "thr_lo": 0.68, "drop": 0.92, "small_scale": 1.30, "vote_thr": 0.42, "c_floor": 0.26}),
        ]
    elif args.joint3:
        cands = [
            ("thr64_d85_sm30_vt42", {**base_cfg, "thr_hi": 0.64, "thr_lo": 0.64, "drop": 0.85, "small_scale": 1.30, "vote_thr": 0.42}),
            ("thr64_d90_sm30_vt42", {**base_cfg, "thr_hi": 0.64, "thr_lo": 0.64, "drop": 0.90, "small_scale": 1.30, "vote_thr": 0.42}),
            ("thr64_d95_sm30_vt42", {**base_cfg, "thr_hi": 0.64, "thr_lo": 0.64, "drop": 0.95, "small_scale": 1.30, "vote_thr": 0.42}),
            ("thr66_d85_sm30_vt42", {**base_cfg, "thr_hi": 0.66, "thr_lo": 0.66, "drop": 0.85, "small_scale": 1.30, "vote_thr": 0.42}),
            ("thr66_d90_sm30_vt42", {**base_cfg, "thr_hi": 0.66, "thr_lo": 0.66, "drop": 0.90, "small_scale": 1.30, "vote_thr": 0.42}),
            ("thr68_d90_sm30_vt42", {**base_cfg, "thr_hi": 0.68, "thr_lo": 0.68, "drop": 0.90, "small_scale": 1.30, "vote_thr": 0.42}),
            ("thr70_d95_sm30_vt42", {**base_cfg, "thr_hi": 0.70, "thr_lo": 0.70, "drop": 0.95, "small_scale": 1.30, "vote_thr": 0.42}),
            ("thr66_d88_sm30_c28", {**base_cfg, "thr_hi": 0.66, "thr_lo": 0.66, "drop": 0.88, "small_scale": 1.30, "vote_thr": 0.42, "c_floor": 0.28}),
        ]
    elif args.joint2:
        cands = [
            ("thr64_drop78_sm25", {**base_cfg, "thr_hi": 0.64, "thr_lo": 0.64, "drop": 0.78, "small_scale": 1.25}),
            ("thr64_drop82_sm25", {**base_cfg, "thr_hi": 0.64, "thr_lo": 0.64, "drop": 0.82, "small_scale": 1.25}),
            ("thr64_drop85_sm25", {**base_cfg, "thr_hi": 0.64, "thr_lo": 0.64, "drop": 0.85, "small_scale": 1.25}),
            ("thr65_drop82_sm25", {**base_cfg, "thr_hi": 0.65, "thr_lo": 0.65, "drop": 0.82, "small_scale": 1.25}),
            ("thr64_drop82_sm30", {**base_cfg, "thr_hi": 0.64, "thr_lo": 0.64, "drop": 0.82, "small_scale": 1.30}),
            ("thr64_drop82_sm35", {**base_cfg, "thr_hi": 0.64, "thr_lo": 0.64, "drop": 0.82, "small_scale": 1.35}),
            ("thr64_d82_sm30_vt42", {**base_cfg, "thr_hi": 0.64, "thr_lo": 0.64, "drop": 0.82, "small_scale": 1.30, "vote_thr": 0.42}),
            ("thr64_d82_sm30_vt48", {**base_cfg, "thr_hi": 0.64, "thr_lo": 0.64, "drop": 0.82, "small_scale": 1.30, "vote_thr": 0.48}),
        ]
    elif args.joint:
        cands = [
            ("baseline", base_cfg),
            ("thr64", {**base_cfg, "thr_hi": 0.64, "thr_lo": 0.64}),
            ("drop78_small25", {**base_cfg, "drop": 0.78, "small_scale": 1.25}),
            ("drop82_small25", {**base_cfg, "drop": 0.82, "small_scale": 1.25}),
            ("thr64_drop78_sm25", {**base_cfg, "thr_hi": 0.64, "thr_lo": 0.64, "drop": 0.78, "small_scale": 1.25}),
            ("thr64_drop78_sm25_mi48", {**base_cfg, "thr_hi": 0.64, "thr_lo": 0.64, "drop": 0.78, "small_scale": 1.25, "match_iou": 0.48}),
            ("thr63_drop78_sm18_c32", {**base_cfg, "thr_hi": 0.63, "thr_lo": 0.63, "drop": 0.78, "small_scale": 1.18, "match_iou": 0.45, "c_floor": 0.32}),
        ]
    else:
        cands = [("baseline", base_cfg)]
        for hi, lo in ((0.60, 0.45), (0.60, 0.50), (0.62, 0.48), (0.58, 0.45), (0.55, 0.55)):
            cands.append((f"hyst_{hi:.2f}_{lo:.2f}", {**base_cfg, "thr_hi": hi, "thr_lo": lo}))
        for cc in (16, 25):
            cands.append((f"cc_min_{cc}", {**base_cfg, "keep_cc_min": cc}))
            cands.append((f"hyst_60_48_cc{cc}", {**base_cfg, "thr_hi": 0.60, "thr_lo": 0.48, "keep_cc_min": cc}))
        for ma in (60, 80, 120):
            cands.append((f"min_area_{ma}", {**base_cfg, "min_area": ma}))
        for drp in (0.55, 0.62, 0.78):
            cands.append((f"drop_{drp:.2f}", {**base_cfg, "drop": drp}))
        for sc in (1.10, 1.25):
            cands.append((f"small_sc_{sc:.2f}", {**base_cfg, "small_scale": sc}))
        for mi in (0.35, 0.45, 0.50):
            cands.append((f"match_iou_{mi:.2f}", {**base_cfg, "match_iou": mi}))
        for off in (-1, 1):
            cands.append((f"offset_{off:+d}", {**base_cfg, "offset": off}))

    all_cnt = {}
    print(f"Running {len(cands)} clean post-processing configs across 707-frame OOF...", flush=True)
    for name, cfg in cands:
        rep, cnt = eval_track_a_cfg(stems_707, preloaded, cfg)
        all_cnt[name] = cnt
        print(f"  {name:20s}: PQ={rep['pq']:.5f} SQ={rep['sq']:.5f} RQ={rep['rq']:.5f} TP={rep['tp']} FP={rep['fp']} FN={rep['fn']}", flush=True)

    # Strict 5-fold nested selection across cands
    pooled = {}
    chosen = {}
    for fk in range(5):
        vs = [s for s in stems_707 if fold_of[s] == fk]
        ts = [s for s in stems_707 if fold_of[s] != fk]
        best = max(all_cnt, key=lambda k: eh.score_from_counts(all_cnt[k], ts)["pq"])
        chosen[fk] = best
        for s in vs:
            pooled[s] = all_cnt[best][s]
    cv = eh.score_from_counts(pooled, stems_707)
    print(f"\n[707 5F-Nested-CV] PQ={cv['pq']:.5f} SQ={cv['sq']:.5f} RQ={cv['rq']:.5f} chosen={chosen} ({(time.time()-t0):.1f}s)", flush=True)


if __name__ == "__main__":
    main()
