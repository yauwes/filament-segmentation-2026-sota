"""Reproduce E19 on Fold 0, run nested date-CV check, and diagnose errors & test confirmation shift."""
import json
import pickle
import sys
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
with open(FP_ROOT / "yolo_val_raw.json") as f:
    YOLO_VAL = json.load(f)
with open(FP_ROOT / "yolo_test_raw.json") as f:
    YOLO_TEST = json.load(f)

VAL_DIRS = {
    "G1": (FP_ROOT / "avg/cache_G1/avgSAM2/val/1280", 1, (0.6, 0, 0.25, 0.7, 0)),
    "A": (FP_ROOT / "avg/cache_A/avgSAM2/val/1664", 4, (0.6, 0, 0.20, 0.7, 0)),
    "B": (FP_ROOT / "avg/cache_B/avgSAM2/val/1280", 4, (0.6, 0, 0.25, 0.7, 0)),
    "P1": (FP_ROOT / "avg/cache_P1/avgSAM2/val/1280", 4, (0.6, 0, 0.30, 0.7, 0)),
    "P4a": (FP_ROOT / "tsP4a/cache/S1E/val/1280", 4, (0.6, 0, 0.35, 0.7, 0)),
    "F0": (FP_ROOT / "avg/cache_F0/avgSAM2/val/1280", 4, (0.6, 0, 0.80, 0.7, 0)),
    "F1": (FP_ROOT / "avg/cache_F1/avgSAM2/val/1280", 4, (0.6, 0, 0.85, 0.7, 0)),
    "F3": (G19_ROOT / "tsF3/cache/S1E/val/1280", 4, (0.6, 0, 0.85, 0.7, 0)),
    "P6": (G19_ROOT / "tsP6/cache/S1E/val/1280", 4, (0.6, 0, 0.30, 0.7, 0)),
    "ms": (W_ROOT / "ms/oof", 4, (0.6, 0, 0.30, 0.7, 0)),
    "s": (W_ROOT / "s/oof", 4, (0.6, 0, 0.30, 0.7, 0)),
    "main": (W_ROOT / "main/oof", 4, (0.6, 0, 0.25, 0.7, 0)),
}


def score_frame(rle_strs, gt_recs):
    pred_rles = [{"size": [2048, 2048], "counts": s.encode("ascii")} for s in rle_strs]
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
    return tp, fp, fn, iou_sum


def paint_to_rles(inst_list):
    taken = np.zeros((2048, 2048), bool)
    rles = []
    for c, (x1, y1, x2, y2), m in inst_list:
        keep = m & ~taken[y1:y2, x1:x2]
        area = int(keep.sum())
        if area >= 100 and area >= 0.7 * int(m.sum()):
            taken[y1:y2, x1:x2] |= keep
            rles.append(bss.encode_crop_ascii((x1, y1, x2, y2), keep))
    return rles


def main():
    f0_stems = sorted(s for s, k in FOLDS.items() if k == 0)
    painted = {k: {} for k in list(VAL_DIRS) + ["YOLO"]}
    for s in f0_stems:
        for k, (d, tta, cell) in VAL_DIRS.items():
            raw = bss.load_raw_npz(d / f"{s}.npz", tta=tta)
            painted[k][s] = bss.paint_ts_cropped(raw, *cell)
        painted["YOLO"][s] = bss.load_yolo_cropped(YOLO_VAL, s, 0.2, -1, 0.7)

    # 1. Single-stream Fold0 PQ
    print("=== SINGLE STREAM FOLD-0 PQ ===")
    for k in ["G1", "A", "B", "P1", "P4a", "F0", "F1", "F3", "P6", "ms", "s", "main", "YOLO"]:
        TP = FP = FN = 0
        IOU = 0.0
        for s in f0_stems:
            tp, fp, fn, iou = score_frame(paint_to_rles(painted[k][s]), GT[s])
            TP += tp; FP += fp; FN += fn; IOU += iou
        den = TP + 0.5 * (FP + FN)
        pq = IOU / den if den else 0.0
        sq = IOU / TP if TP else 0.0
        rq = TP / den if den else 0.0
        print(f"  {k:5s}: PQ={pq:.4f} SQ={sq:.4f} RQ={rq:.4f} TP={TP:4d} FP={FP:4d} FN={FN:4d}")

    # 2. Exact E19 baseline reproduction + breakdowns
    e19_rles = {}
    TP = FP = FN = 0
    IOU = 0.0
    by_nrec = defaultdict(lambda: [0, 0, 0, 0.0])
    by_stn = defaultdict(lambda: [0, 0, 0, 0.0])
    by_yr = defaultdict(lambda: [0, 0, 0, 0.0])
    rand1 = [0, 0, 0, 0.0]
    rng = np.random.default_rng(42)

    for s in f0_stems:
        core = [painted[k][s] for k in ("G1", "A", "B", "P1", "P4a", "F0", "F1")] + [painted["YOLO"][s]]
        extra = {k: painted[k][s] for k in ("ms", "s", "main", "F3")}
        rles = bss.super_ensemble_stem(core, extra)
        e19_rles[s] = rles
        tp, fp, fn, iou = score_frame(rles, GT[s])
        TP += tp; FP += fp; FN += fn; IOU += iou

        nrec = len(GT[s])
        for d_arr, key in [(by_nrec, nrec), (by_stn, s[14:-1]), (by_yr, int(s[:4]))]:
            d_arr[key][0] += tp; d_arr[key][1] += fp; d_arr[key][2] += fn; d_arr[key][3] += iou

        idx = int(rng.integers(nrec))
        tp1, fp1, fn1, iou1 = score_frame(rles, [GT[s][idx]])
        rand1[0] += tp1; rand1[1] += fp1; rand1[2] += fn1; rand1[3] += iou1

    den = TP + 0.5 * (FP + FN)
    print(f"\n=== E19 EXACT REPRODUCTION ON FOLD 0 ===")
    print(f"  E19 Fold0: PQ={IOU/den:.4f} SQ={IOU/TP:.4f} RQ={TP/den:.4f} TP={TP} FP={FP} FN={FN}")
    den1 = rand1[0] + 0.5 * (rand1[1] + rand1[2])
    print(f"  E19 Fold0 (1 random rec/frame sim): PQ={rand1[3]/den1:.4f} SQ={rand1[3]/rand1[0]:.4f} RQ={rand1[0]/den1:.4f}")

    print("\n=== E19 BREAKDOWN BY ANNOTATOR COUNT / STATION / YEAR ===")
    for nrec, (tp, fp, fn, iou) in sorted(by_nrec.items()):
        d = tp + 0.5 * (fp + fn)
        print(f"  n_rec={nrec}: PQ={iou/d:.4f} SQ={iou/tp:.4f} RQ={tp/d:.4f} TP={tp} FP={fp} FN={fn}")
    for stn, (tp, fp, fn, iou) in sorted(by_stn.items()):
        d = tp + 0.5 * (fp + fn)
        print(f"  stn={stn}: PQ={iou/d:.4f} SQ={iou/tp:.4f} RQ={tp/d:.4f} TP={tp} FP={fp} FN={fn}")
    for yr, (tp, fp, fn, iou) in sorted(by_yr.items()):
        d = tp + 0.5 * (fp + fn)
        print(f"  year={yr}: PQ={iou/d:.4f} TP={tp} FP={fp} FN={fn}")

    # 3. Check ms/s/F3 match rates on Val vs Test at current thresholds!
    test_dirs = {
        "G1": (FP_ROOT / "avg/cache_G1/avgSAM2/test/1280", 1, (0.6, 0, 0.25, 0.7, 0)),
        "A": (FP_ROOT / "avg/cache_A/avgSAM2/test/1664", 4, (0.6, 0, 0.20, 0.7, 0)),
        "B": (FP_ROOT / "avg/cache_B/avgSAM2/test/1280", 4, (0.6, 0, 0.25, 0.7, 0)),
        "P1": (FP_ROOT / "avg/cache_P1/avgSAM2/test/1280", 4, (0.6, 0, 0.30, 0.7, 0)),
        "P4a": (FP_ROOT / "tsP4a/cache/S1E/test/1280", 4, (0.6, 0, 0.35, 0.7, 0)),
        "F0": (FP_ROOT / "avg/cache_F0/avgSAM2/test/1280", 4, (0.6, 0, 0.80, 0.7, 0)),
        "F1": (FP_ROOT / "avg/cache_F1/avgSAM2/test/1280", 4, (0.6, 0, 0.85, 0.7, 0)),
        "F3": (G19_ROOT / "tsF3/cache/S1E/test/1280", 4, (0.6, 0, 0.85, 0.7, 0)),
        "ms": (W_ROOT / "ms/test", 4, (0.6, 0, 0.30, 0.7, 0)),
        "s": (W_ROOT / "s/test", 4, (0.6, 0, 0.30, 0.7, 0)),
        "main": (W_ROOT / "main/test", 4, (0.6, 0, 0.25, 0.7, 0)),
    }
    test_stems = sorted(p.stem for p in (W_ROOT / "ms/test").glob("*.npz"))
    val_ms_cnt = sum(len(painted["ms"][s]) for s in f0_stems) / len(f0_stems)
    val_s_cnt = sum(len(painted["s"][s]) for s in f0_stems) / len(f0_stems)
    val_main_cnt = sum(len(painted["main"][s]) for s in f0_stems) / len(f0_stems)
    t_cnts = {"ms": 0, "s": 0, "main": 0, "F3": 0, "F1": 0, "P1": 0}
    for s in test_stems:
        for k in t_cnts:
            d, tta, cell = test_dirs[k]
            t_cnts[k] += len(bss.paint_ts_cropped(bss.load_raw_npz(d / f"{s}.npz", tta=tta), *cell))
    print("\n=== PAINTED INSTANCES PER FRAME (VAL vs TEST) ===")
    print(f"  Val:  ms={val_ms_cnt:.2f} s={val_s_cnt:.2f} main={val_main_cnt:.2f} F3={sum(len(painted['F3'][s]) for s in f0_stems)/len(f0_stems):.2f} P1={sum(len(painted['P1'][s]) for s in f0_stems)/len(f0_stems):.2f}")
    print(f"  Test: ms={t_cnts['ms']/len(test_stems):.2f} s={t_cnts['s']/len(test_stems):.2f} main={t_cnts['main']/len(test_stems):.2f} F3={t_cnts['F3']/len(test_stems):.2f} P1={t_cnts['P1']/len(test_stems):.2f}")


if __name__ == "__main__":
    main()
