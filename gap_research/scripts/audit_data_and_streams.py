"""Empirical audit of MAGFiLO train vs test distribution, annotator plurality, radial/size errors, and stream caches."""
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mu

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from filseg import metric as M

FP_ROOT = Path("/private/tmp/filseg_ens_cache/fp/final")
G19_ROOT = Path("/private/tmp/filseg_ens_cache/g19")
W_ROOT = REPO / "work" / "cache"

with open(REPO / "folds" / "folds.json") as f:
    FOLDS = json.load(f)

with open("/private/tmp/gt_records.pkl", "rb") as f:
    GT = pickle.load(f)


def parse_stem(s):
    # YYYYMMDDHHMMSS<Station>h
    year = int(s[:4])
    date = s[:8]
    station = s[14:-1] if len(s) >= 16 else "?"
    return year, date, station


def bbox_center_r(box):
    x1, y1, x2, y2 = box
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    return float(np.hypot(cx - 1024.0, cy - 1024.0) / 920.0)


def main():
    f0_stems = sorted(s for s, k in FOLDS.items() if k == 0)
    test_stems = sorted(p.stem for p in (W_ROOT / "ms" / "test").glob("*.npz"))

    print("=== 1. DATASET & METADATA AUDIT ===")
    train_stems = sorted(FOLDS.keys())
    print(f"Train frames: {len(train_stems)}, Fold0 frames: {len(f0_stems)}, Test frames: {len(test_stems)}")

    for label, stems in [("TrainAll", train_stems), ("Fold0", f0_stems), ("Test", test_stems)]:
        yrs = Counter(parse_stem(s)[0] for s in stems)
        stns = Counter(parse_stem(s)[2] for s in stems)
        dates = len({parse_stem(s)[1] for s in stems})
        print(f"  {label:8s} ({len(stems):3d} fr, {dates:3d} dates): stations={dict(sorted(stns.items()))}")
        print(f"    years={dict(sorted(yrs.items()))}")

    # Overlap in dates between train and test?
    tr_dates = {parse_stem(s)[1] for s in train_stems}
    te_dates = {parse_stem(s)[1] for s in test_stems}
    print(f"  Date overlap train vs test: {len(tr_dates & te_dates)}")

    # GT properties on Fold 0
    recs_per_frame = Counter(len(GT[s]) for s in f0_stems)
    gt_areas = []
    gt_rs = []
    gt_per_rec = []
    for s in f0_stems:
        for r in GT[s]:
            gt_per_rec.append(len(r))
            for rle in r:
                a = int(mu.area(rle))
                x, y, w, h = mu.toBbox(rle)
                cx, cy = x + 0.5 * w, y + 0.5 * h
                gt_areas.append(a)
                gt_rs.append(float(np.hypot(cx - 1024.0, cy - 1024.0) / 920.0))
    gt_areas = np.array(gt_areas)
    gt_rs = np.array(gt_rs)
    print(f"  Fold0 records/frame dist: {dict(sorted(recs_per_frame.items()))}")
    print(f"  Fold0 GT instances/rec mean={np.mean(gt_per_rec):.2f}, median={np.median(gt_per_rec):.1f}, min={np.min(gt_per_rec)}, max={np.max(gt_per_rec)}")
    print(f"  Fold0 GT area px quantiles [5,10,25,50,75,90,95]: {np.percentile(gt_areas, [5,10,25,50,75,90,95]).round(0).tolist()}")
    print(f"  Fold0 GT r/R quantiles [10,25,50,75,90,95,99]: {np.percentile(gt_rs, [10,25,50,75,90,95,99]).round(3).tolist()}")

    # Compare baseline submission test predictions vs Fold0
    sub_df = pd.read_csv(REPO / "results" / "submission.csv")
    sub_df["stem"] = sub_df["filament_id"].str.split("_").str[0]
    sub_test = sub_df.groupby("stem").size().reindex(test_stems, fill_value=0)
    print(f"  Baseline E19 Test preds: total={len(sub_df)}, mean/frame={sub_test.mean():.2f}, empty_frames={(sub_test == 0).sum()}")

    # Cache inspection across all 12 streams on Fold0 vs Test
    streams = {
        "G1": (FP_ROOT / "avg/cache_G1/avgSAM2/val/1280", FP_ROOT / "avg/cache_G1/avgSAM2/test/1280"),
        "A": (FP_ROOT / "avg/cache_A/avgSAM2/val/1664", FP_ROOT / "avg/cache_A/avgSAM2/test/1664"),
        "B": (FP_ROOT / "avg/cache_B/avgSAM2/val/1280", FP_ROOT / "avg/cache_B/avgSAM2/test/1280"),
        "P1": (FP_ROOT / "avg/cache_P1/avgSAM2/val/1280", FP_ROOT / "avg/cache_P1/avgSAM2/test/1280"),
        "P4a": (FP_ROOT / "tsP4a/cache/S1E/val/1280", FP_ROOT / "tsP4a/cache/S1E/test/1280"),
        "F0": (FP_ROOT / "avg/cache_F0/avgSAM2/val/1280", FP_ROOT / "avg/cache_F0/avgSAM2/test/1280"),
        "F1": (FP_ROOT / "avg/cache_F1/avgSAM2/val/1280", FP_ROOT / "avg/cache_F1/avgSAM2/test/1280"),
        "F3": (G19_ROOT / "tsF3/cache/S1E/val/1280", G19_ROOT / "tsF3/cache/S1E/test/1280"),
        "P6": (G19_ROOT / "tsP6/cache/S1E/val/1280", G19_ROOT / "tsP6/cache/S1E/test/1280"),
        "ms": (W_ROOT / "ms/oof", W_ROOT / "ms/test"),
        "s": (W_ROOT / "s/oof", W_ROOT / "s/test"),
        "main": (W_ROOT / "main/oof", W_ROOT / "main/test"),
    }
    print("\n=== 2. STREAM CONFIDENCE & BOX STATISTICS (FOLD0 vs TEST) ===")
    for name, (vd, td) in streams.items():
        vc, vr, tc, tr = [], [], [], []
        for s in f0_stems:
            z = np.load(vd / f"{s}.npz")
            vc.extend(z["conf"].tolist())
            for b in z["box"]:
                vr.append(bbox_center_r(b))
        for s in test_stems:
            z = np.load(td / f"{s}.npz")
            tc.extend(z["conf"].tolist())
            for b in z["box"]:
                tr.append(bbox_center_r(b))
        vc, tc = np.array(vc), np.array(tc)
        print(
            f"  {name:4s}: Val N={len(vc):4d} ({len(vc)/len(f0_stems):.2f}/fr) q50={np.median(vc):.3f} q10={np.percentile(vc,10):.3f} q90={np.percentile(vc,90):.3f} r>0.75={np.mean(np.array(vr)>0.75)*100:.1f}%"
            f" | Test N={len(tc):4d} ({len(tc)/len(test_stems):.2f}/fr) q50={np.median(tc):.3f} q10={np.percentile(tc,10):.3f} q90={np.percentile(tc,90):.3f} r>0.75={np.mean(np.array(tr)>0.75)*100:.1f}%"
        )

    # Check if E19 submission test RLE areas and r/R differ from Fold0
    test_areas, test_rs = [], []
    for _, row in sub_df.iterrows():
        rle = {"size": [2048, 2048], "counts": row["segmentation_rle"].encode("ascii")}
        test_areas.append(int(mu.area(rle)))
        x, y, w, h = mu.toBbox(rle)
        cx, cy = x + 0.5 * w, y + 0.5 * h
        test_rs.append(float(np.hypot(cx - 1024.0, cy - 1024.0) / 920.0))
    test_areas = np.array(test_areas)
    test_rs = np.array(test_rs)
    print(f"  E19 Test Pred area quantiles [5,10,25,50,75,90,95]: {np.percentile(test_areas, [5,10,25,50,75,90,95]).round(0).tolist()}")
    print(f"  E19 Test Pred <580px frac: {np.mean(test_areas < 580)*100:.1f}% (vs Fold0 GT <580px: {np.mean(gt_areas < 580)*100:.1f}%)")
    print(f"  E19 Test Pred r/R quantiles [10,25,50,75,90,95,99]: {np.percentile(test_rs, [10,25,50,75,90,95,99]).round(3).tolist()}")
    print(f"  E19 Test Pred r>0.75 frac: {np.mean(test_rs > 0.75)*100:.1f}% (vs Fold0 GT r>0.75: {np.mean(gt_rs > 0.75)*100:.1f}%)")


if __name__ == "__main__":
    main()
