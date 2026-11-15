"""Structured OOF error breakdown by single/multi annotator, IoU failure mode, size bin, and observatory."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "gap_research" / "scripts"))

from filseg import data, metric  # noqa: E402


def _area_bin(a: float) -> str:
    if a < 250:
        return "<250"
    if a < 580:
        return "250-580"
    if a < 2000:
        return "580-2k"
    return ">2k"


def analyze_rles(rles_by_stem: dict[str, list], stems: list[str] | None = None) -> dict:
    folds = data.folds()
    stems = sorted(stems if stems is not None else folds)
    recs = metric.load_records(stems)

    overall_rows = []
    by_obs = defaultdict(list)
    multi = {"single_rec": dict(tp=0, fp=0, fn=0, iou=0.0, n=0), "multi_rec": dict(tp=0, fp=0, fn=0, iou=0.0, n=0)}
    fp_type = defaultdict(int)
    fn_type = defaultdict(int)
    fp_area = defaultdict(int)
    fn_area = defaultdict(int)

    for s in stems:
        preds = [r if isinstance(r, dict) else {"size": [2048, 2048], "counts": r.encode()} for r in rles_by_stem.get(s, [])]
        g_recs = recs[s]
        c = metric.counts(preds, g_recs)
        overall_rows.append(c)
        by_obs[s[-2:]].append(c)

        mk = "multi_rec" if len(g_recs) > 1 else "single_rec"
        multi[mk]["n"] += 1
        for k in ("tp", "fp", "fn", "iou"):
            multi[mk][k] += c[k]

        p_areas = mu.area(preds).tolist() if preds else []
        for gt in g_recs:
            if not preds and not gt:
                continue
            if not preds:
                for ga in mu.area(gt).tolist():
                    fn_area[_area_bin(ga)] += 1
                    fn_type["zero_ov"] += 1
                continue
            if not gt:
                for pa in p_areas:
                    fp_area[_area_bin(pa)] += 1
                    fp_type["empty_gt_rec"] += 1
                continue
            ious = mu.iou(preds, gt, [0] * len(gt))
            g_areas = mu.area(gt).tolist()
            hit = ious > 0.5
            ov0 = ious > 0
            for i in range(len(preds)):
                if not hit[i].any():
                    fp_area[_area_bin(p_areas[i])] += 1
                    mx = float(ious[i].max())
                    if mx == 0.0:
                        fp_type["zero_ov"] += 1
                    elif int(ov0[i].sum()) >= 2:
                        fp_type["merge_pred_deg2+"] += 1
                    elif mx >= 0.35:
                        fp_type["near_miss_35_50"] += 1
                    else:
                        fp_type["low_ov_0_35"] += 1
            for j in range(len(gt)):
                if not hit[:, j].any():
                    fn_area[_area_bin(g_areas[j])] += 1
                    mx = float(ious[:, j].max())
                    if mx == 0.0:
                        fn_type["zero_ov"] += 1
                    elif int(ov0[:, j].sum()) >= 2:
                        fn_type["frag_gt_deg2+"] += 1
                    elif mx >= 0.35:
                        fn_type["near_miss_35_50"] += 1
                    else:
                        fn_type["low_ov_0_35"] += 1

    sm_all = metric.summarize(overall_rows)
    sm_multi = {k: {**v, **metric.summarize([v])} for k, v in multi.items()}
    sm_obs = {k: metric.summarize(rows) for k, rows in sorted(by_obs.items())}
    return {
        "overall": sm_all,
        "single_vs_multi": sm_multi,
        "fp_type": dict(fp_type),
        "fn_type": dict(fn_type),
        "fp_area": dict(fp_area),
        "fn_area": dict(fn_area),
        "by_observatory": sm_obs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", type=Path, default=Path("/private/tmp/tracka_clean_oof_cache.pkl"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    d = pickle.load(open(args.pkl, "rb"))
    rles = d["rles"] if "rles" in d else d
    rep = analyze_rles(rles)
    txt = json.dumps(rep, indent=2)
    print(txt)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(txt)


if __name__ == "__main__":
    main()
