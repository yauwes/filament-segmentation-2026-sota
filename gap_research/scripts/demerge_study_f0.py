"""Fold-0 Track-B (E19_no_F1) De-Merge, Near-Miss IoU (0.30..0.50), and <580px Post-Processing Nested Date-CV Study."""
from __future__ import annotations

import json
import sys
import time
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
from filseg import data, metric  # noqa: E402


def _paint_custom(
    inst_list,
    thr: float = 0.60,
    offset: int = 0,
    conf: float = 0.25,
    drop: float = 0.70,
    min_area: int = 100,
    split_cc_min: int = 0,
    keep_cc_min: int = 0,
    small_scale: float = 1.0,
    small_px: int = 120,
    occ: np.ndarray | None = None,
):
    if not inst_list:
        return []
    if occ is None:
        occ = np.zeros((2048, 2048), dtype=bool)
    else:
        occ.fill(False)
    t_u8 = int(thr * 255.0 + 0.5)
    out = []
    # Sort descending after applying optional small-filament boost
    scored = []
    for c, box, p in inst_list:
        x1, y1, x2, y2 = box
        ec = float(c) * (small_scale if max(x2 - x1, y2 - y1) < small_px else 1.0)
        if ec >= conf:
            scored.append((ec, (x1, y1, x2, y2), p))
    scored.sort(key=lambda t: -t[0])

    for ec, (x1, y1, x2, y2), p in scored:
        m = (p > t_u8).astype(np.uint8)
        if offset:
            m = (cv2.dilate if offset > 0 else cv2.erode)(m, bss._K[abs(offset)])
        m_bool = m.astype(bool)
        if not m_bool.any():
            continue

        if keep_cc_min > 0:
            n_cc, lbl, stats, _ = cv2.connectedComponentsWithStats(m_bool.astype(np.uint8), connectivity=8)
            if n_cc > 2:
                val = [i for i in range(1, n_cc) if stats[i, cv2.CC_STAT_AREA] >= keep_cc_min]
                if val:
                    m_bool = np.isin(lbl, val)

        sub_masks = [m_bool]
        if split_cc_min > 0:
            n_cc, lbl, stats, _ = cv2.connectedComponentsWithStats(m_bool.astype(np.uint8), connectivity=8)
            big = [i for i in range(1, n_cc) if stats[i, cv2.CC_STAT_AREA] >= split_cc_min]
            if len(big) >= 2:
                sub_masks = [lbl == i for i in big]

        for sm in sub_masks:
            keep = sm & ~occ[y1:y2, x1:x2]
            area = int(keep.sum())
            if area > min_area and area > 0 and area >= drop * int(sm.sum()):
                occ[y1:y2, x1:x2] |= keep
                v = bss.trim(ec, (x1, y1, x2, y2), keep)
                if v is not None:
                    out.append(v)
    return out


def main() -> None:
    t0 = time.time()
    stems = list(eh.FOLD0_STEMS)
    fp = bss.FP_ROOT
    g19 = Path("/private/tmp/filseg_ens_cache/g19")
    yolo_raw = json.loads((fp / "yolo_val_raw.json").read_text())

    core_keys = ("G1", "A", "B", "P1", "P4a", "F0")
    extra_keys = ("ms", "s", "main", "F3", "P6")
    yolo_s = {s: bss.load_yolo_cropped(yolo_raw, s, 0.2, -1, 0.7) for s in stems}
    recs_by_stem = metric.load_records(stems=stems)

    # Fast reusable occupancy buffer avoids 2000x 4MB allocations
    _occ = np.zeros((2048, 2048), dtype=bool)
    configs = [
        ("baseline_E19_no_F1", dict(thr=0.60, offset=0, drop=0.70, min_area=100, keep_cc_min=0, split_cc_min=0, small_scale=1.0, rescue_thr=0.58)),
        ("thr64_drop85", dict(thr=0.64, offset=0, drop=0.85, min_area=100, keep_cc_min=0, split_cc_min=0, small_scale=1.0, rescue_thr=0.58)),
        ("thr64_drop90", dict(thr=0.64, offset=0, drop=0.90, min_area=100, keep_cc_min=0, split_cc_min=0, small_scale=1.0, rescue_thr=0.58)),
        ("thr66_drop88", dict(thr=0.66, offset=0, drop=0.88, min_area=100, keep_cc_min=0, split_cc_min=0, small_scale=1.0, rescue_thr=0.58)),
        ("thr66_drop92", dict(thr=0.66, offset=0, drop=0.92, min_area=100, keep_cc_min=0, split_cc_min=0, small_scale=1.0, rescue_thr=0.58)),
        ("thr66_drop95", dict(thr=0.66, offset=0, drop=0.95, min_area=100, keep_cc_min=0, split_cc_min=0, small_scale=1.0, rescue_thr=0.58)),
        ("thr68_drop90", dict(thr=0.68, offset=0, drop=0.90, min_area=100, keep_cc_min=0, split_cc_min=0, small_scale=1.0, rescue_thr=0.58)),
    ]

    # Separate unique paint parameter tuples so rescue_* / ensemble-only tweaks reuse painted masks at zero cost
    paint_cfgs = {}
    for _, cfg in configs:
        pk = (cfg["thr"], cfg["offset"], cfg["drop"], cfg["min_area"], cfg["keep_cc_min"], cfg["split_cc_min"], cfg["small_scale"])
        paint_cfgs[pk] = cfg

    painted_by_pk = {pk: {k: {} for k in core_keys + extra_keys} for pk in paint_cfgs}
    for k in core_keys + extra_keys:
        val_dir = eh.STREAM_META[k]["val"]
        tta = eh.STREAM_META[k]["tta"]
        conf_floor = eh.STREAM_META[k]["cell"][2]
        raw_k = {s: bss.load_raw_npz(val_dir / f"{s}.npz", tta=tta) for s in stems}
        for pk, cfg in paint_cfgs.items():
            painted_by_pk[pk][k] = {
                s: _paint_custom(
                    raw_k[s],
                    thr=cfg["thr"],
                    offset=cfg["offset"],
                    conf=conf_floor,
                    drop=cfg["drop"],
                    min_area=cfg["min_area"],
                    split_cc_min=cfg["split_cc_min"],
                    keep_cc_min=cfg["keep_cc_min"],
                    small_scale=cfg["small_scale"],
                    occ=_occ,
                )
                for s in stems
            }
        del raw_k
        print(f"Decoded stream {k} ({time.time()-t0:.1f}s)", flush=True)

    for cname, cfg in configs:
        pk = (cfg["thr"], cfg["offset"], cfg["drop"], cfg["min_area"], cfg["keep_cc_min"], cfg["split_cc_min"], cfg["small_scale"])
        rles = {}
        for s in stems:
            c_list = [painted_by_pk[pk][k][s] for k in core_keys] + [yolo_s[s]]
            ex = {k: painted_by_pk[pk][k][s] for k in extra_keys}
            ex_w = {"ms": 0.18, "s": 0.18, "main": 0.12, "F3": 0.06}
            rles[s] = bgs.super_ensemble_upgraded_stem(c_list, ex, extra_weights=ex_w, rescue_min_score=cfg["rescue_thr"])

        rep = bgs.evaluate_rles_with_nested_cv(rles, stems)
        ious, gdeg, pdeg = [], [], []
        for s in stems:
            pred_list = [{"size": [2048, 2048], "counts": c.encode()} for c in rles[s]]
            st = metric.overlap_stats(pred_list, recs_by_stem[s])
            ious.extend(st["iou"])
            gdeg.extend(st["gt_degree"])
            pdeg.extend(st["pred_degree"])
        os_rep = {
            "n_gt": len(gdeg),
            "n_pred": len(pdeg),
            "gt_split": sum(1 for d in gdeg if d > 1),
            "pred_merge": sum(1 for d in pdeg if d > 1),
            "mean_iou": round(sum(ious) / len(ious), 5) if ious else 0.0,
        }
        srep = rep.get("summary", rep)
        print(f"{cname:18s} | rep={json.dumps(srep)} | os={json.dumps(os_rep)}", flush=True)


if __name__ == "__main__":
    main()
