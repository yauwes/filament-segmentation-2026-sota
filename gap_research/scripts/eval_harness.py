"""EXP-001: Leak-free Fold-0 Evaluation, Date-Grouped Nested CV, and Date Split-Half Bootstrap Harness."""
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import build_super_submission as bss

FP_ROOT = Path("/private/tmp/filseg_ens_cache/fp/final")
G19_ROOT = Path("/private/tmp/filseg_ens_cache/g19")
W_ROOT = REPO / "work" / "cache"

with open(REPO / "folds" / "folds.json") as f:
    FOLDS = json.load(f)
with open("/private/tmp/gt_records.pkl", "rb") as f:
    GT = pickle.load(f)

FOLD0_STEMS = sorted(s for s, k in FOLDS.items() if k == 0)
TEST_STEMS = sorted(p.stem for p in (W_ROOT / "ms" / "test").glob("*.npz"))

STREAM_META = {
    "G1":   {"val": FP_ROOT / "avg/cache_G1/avgSAM2/val/1280",  "test": FP_ROOT / "avg/cache_G1/avgSAM2/test/1280",  "tta": 1, "family": "yolo11_t", "cell": (0.6, 0, 0.25, 0.7, 0)},
    "A":    {"val": FP_ROOT / "avg/cache_A/avgSAM2/val/1664",   "test": FP_ROOT / "avg/cache_A/avgSAM2/test/1664",   "tta": 4, "family": "yolo11_t", "cell": (0.6, 0, 0.20, 0.7, 0)},
    "B":    {"val": FP_ROOT / "avg/cache_B/avgSAM2/val/1280",   "test": FP_ROOT / "avg/cache_B/avgSAM2/test/1280",   "tta": 4, "family": "yolo11_t", "cell": (0.6, 0, 0.25, 0.7, 0)},
    "P1":   {"val": FP_ROOT / "avg/cache_P1/avgSAM2/val/1280",  "test": FP_ROOT / "avg/cache_P1/avgSAM2/test/1280",  "tta": 4, "family": "yolo11_s", "cell": (0.6, 0, 0.30, 0.7, 0)},
    "P4a":  {"val": FP_ROOT / "tsP4a/cache/S1E/val/1280",       "test": FP_ROOT / "tsP4a/cache/S1E/test/1280",       "tta": 4, "family": "yolo11_s", "cell": (0.6, 0, 0.35, 0.7, 0)},
    "P6":   {"val": G19_ROOT / "tsP6/cache/S1E/val/1280",       "test": G19_ROOT / "tsP6/cache/S1E/test/1280",       "tta": 4, "family": "yolo11_s", "cell": (0.6, 0, 0.30, 0.7, 0)},
    "F0":   {"val": FP_ROOT / "avg/cache_F0/avgSAM2/val/1280",  "test": FP_ROOT / "avg/cache_F0/avgSAM2/test/1280",  "tta": 4, "family": "frcnn_t",  "cell": (0.6, 0, 0.80, 0.7, 0)},
    "F1":   {"val": FP_ROOT / "avg/cache_F1/avgSAM2/val/1280",  "test": FP_ROOT / "avg/cache_F1/avgSAM2/test/1280",  "tta": 4, "family": "frcnn_s",  "cell": (0.6, 0, 0.85, 0.7, 0)},
    "F3":   {"val": G19_ROOT / "tsF3/cache/S1E/val/1280",       "test": G19_ROOT / "tsF3/cache/S1E/test/1280",       "tta": 4, "family": "frcnn_s",  "cell": (0.6, 0, 0.85, 0.7, 0)},
    "ms":   {"val": W_ROOT / "ms/oof",                          "test": W_ROOT / "ms/test",                          "tta": 4, "family": "5fold_ts", "cell": (0.6, 0, 0.30, 0.7, 0)},
    "s":    {"val": W_ROOT / "s/oof",                           "test": W_ROOT / "s/test",                           "tta": 4, "family": "5fold_s",  "cell": (0.6, 0, 0.30, 0.7, 0)},
    "main": {"val": W_ROOT / "main/oof",                        "test": W_ROOT / "main/test",                        "tta": 4, "family": "5fold_t",  "cell": (0.6, 0, 0.25, 0.7, 0)},
}


def make_nested_date_folds(stems=FOLD0_STEMS, n_folds=5, seed=42):
    """Assign each stem in Fold 0 to 5 date-grouped inner folds deterministically."""
    dates = sorted({s[:8] for s in stems})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(dates))
    date_to_fold = {dates[idx]: int(rank % n_folds) for rank, idx in enumerate(perm)}
    return {s: date_to_fold[s[:8]] for s in stems}


def score_rles_per_photo(pred_rles, gt_recs):
    """Exact official organizer matching + TP/FP/FN/IOU + per-instance diagnostics."""
    tp = fp = fn = 0
    iou_sum = 0.0
    n_p = len(pred_rles)
    for g_rles in gt_recs:
        n_g = len(g_rles)
        if n_p == 0:
            fn += n_g
            continue
        if n_g == 0:
            fp += n_p
            continue
        ious = mu.iou(pred_rles, g_rles, [0] * n_g)
        hit_g = np.zeros(n_g, bool)
        for i in range(n_p):
            best = False
            for j in range(n_g):
                v = float(ious[i, j])
                if v > 0.5:
                    tp += 1
                    iou_sum += v
                    hit_g[j] = True
                    best = True
            if not best:
                fp += 1
        fn += int((~hit_g).sum())
    return np.array([tp, fp, fn, iou_sum], dtype=np.float64)


def score_from_counts(counts_by_stem, stems=None):
    if stems is None:
        stems = sorted(counts_by_stem)
    tot = np.sum([counts_by_stem[s] for s in stems], axis=0)
    tp, fp, fn, iou = tot
    den = tp + 0.5 * (fp + fn)
    return {
        "pq": float(iou / den) if den else 0.0,
        "sq": float(iou / tp) if tp else 0.0,
        "rq": float(tp / den) if den else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "iou_sum": float(iou),
    }


def score_rle_strs_dict(rle_strs_by_stem, stems=FOLD0_STEMS):
    counts = {}
    for s in stems:
        rles = [{"size": [2048, 2048], "counts": str(c).encode("ascii")} for c in rle_strs_by_stem[s]]
        counts[s] = score_rles_per_photo(rles, GT[s])
    return counts


def paired_date_bootstrap(counts_new, counts_base, stems=FOLD0_STEMS, reps=200, seed=0):
    by_date = defaultdict(list)
    for s in stems:
        by_date[s[:8]].append(s)
    dates = sorted(by_date)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(reps):
        perm = rng.permutation(dates)
        h = len(dates) // 2
        for half in (perm[:h], perm[h:]):
            hs = [s for d in half for s in by_date[d]]
            pn = score_from_counts(counts_new, hs)["pq"]
            pb = score_from_counts(counts_base, hs)["pq"]
            diffs.append(pn - pb)
    diffs = np.asarray(diffs)
    return {
        "mean_gain": float(diffs.mean()),
        "p5": float(np.percentile(diffs, 5.0)),
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "win_rate": float((diffs > 0).mean()),
    }


def detailed_error_profile(rle_strs_by_stem, stems=FOLD0_STEMS):
    """Area-bucket and radial-ring recall/precision diagnostic on Fold 0."""
    area_bins = {"small(<580)": [0, 0, 0], "med(580-1582)": [0, 0, 0], "large(>=1582)": [0, 0, 0]}
    pred_rad = {"r<0.50": [0, 0], "0.50<=r<0.75": [0, 0], "r>=0.75": [0, 0]}
    for s in stems:
        rles = [{"size": [2048, 2048], "counts": str(c).encode("ascii")} for c in rle_strs_by_stem[s]]
        pa = mu.area(rles).astype(float) if rles else np.zeros(0)
        pcen = []
        for r in rles:
            x, y, w, h = mu.toBbox(r)
            pcen.append(float(np.hypot(x + 0.5 * w - 1024.0, y + 0.5 * h - 1024.0) / 920.0))
        pcen = np.asarray(pcen, dtype=float)
        for g_rles in GT[s]:
            ga = mu.area(g_rles).astype(float) if g_rles else np.zeros(0)
            ious = mu.iou(rles, g_rles, [0] * len(g_rles)) if (rles and g_rles) else np.zeros((len(rles), len(g_rles)))
            for j in range(len(g_rles)):
                a = ga[j]
                bkey = "small(<580)" if a < 580 else ("med(580-1582)" if a < 1582 else "large(>=1582)")
                best = float(ious[:, j].max()) if len(rles) else 0.0
                if best > 0.5:
                    area_bins[bkey][0] += 1
                elif best > 0:
                    area_bins[bkey][1] += 1
                else:
                    area_bins[bkey][2] += 1
            for i in range(len(rles)):
                r = pcen[i]
                rkey = "r<0.50" if r < 0.50 else ("0.50<=r<0.75" if r < 0.75 else "r>=0.75")
                pred_rad[rkey][1] += 1
                if len(g_rles) and float(ious[i].max()) > 0.5:
                    pred_rad[rkey][0] += 1
    return area_bins, pred_rad


def compute_e19_baseline():
    with open(FP_ROOT / "yolo_val_raw.json") as f:
        yolo_val = json.load(f)
    rle_strs = {}
    for s in FOLD0_STEMS:
        ps = {}
        for k, meta in STREAM_META.items():
            raw = bss.load_raw_npz(meta["val"] / f"{s}.npz", tta=meta["tta"])
            ps[k] = bss.paint_ts_cropped(raw, *meta["cell"])
        yolo_s = bss.load_yolo_cropped(yolo_val, s, 0.2, -1, 0.7)
        core = [ps[k] for k in ("G1", "A", "B", "P1", "P4a", "F0", "F1")] + [yolo_s]
        extra = {k: ps[k] for k in ("ms", "s", "main", "F3")}
        rle_strs[s] = bss.super_ensemble_stem(core, extra)
    counts = score_rle_strs_dict(rle_strs)
    return rle_strs, counts


if __name__ == "__main__":
    t0 = time.time()
    out_dir = REPO / "gap_research" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    rle_strs, counts = compute_e19_baseline()
    res = score_from_counts(counts)
    area_bins, pred_rad = detailed_error_profile(rle_strs)
    print(f"EXP-001 E19 Baseline verified in {time.time()-t0:.2f}s: PQ={res['pq']:.4f} SQ={res['sq']:.4f} RQ={res['rq']:.4f} TP={res['tp']} FP={res['fp']} FN={res['fn']}")
    print("  GT area buckets [match, nearmiss, untouched]:")
    for k, (mt, nm, un) in area_bins.items():
        tot = mt + nm + un
        print(f"    {k:15s}: n={tot:4d} match={100*mt/tot:5.1f}% nearmiss={100*nm/tot:5.1f}% untouched={100*un/tot:5.1f}%")
    print("  Pred radial precision (IoU>0.5):")
    for k, (tp_r, tot_r) in pred_rad.items():
        print(f"    {k:15s}: n={tot_r:4d} prec={100*tp_r/max(tot_r,1):5.1f}%")

    # Save locked E19 baseline counts for paired bootstrap across all subsequent experiments
    with open(out_dir / "e19_fold0_baseline.pkl", "wb") as f:
        pickle.dump({"rle_strs": rle_strs, "counts": counts, "metrics": res}, f)
