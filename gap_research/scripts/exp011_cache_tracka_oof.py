"""Cache Track-A Clean 707-stem OOF RLEs + exact per-stem counts + audit overlap with exp011_pool.pkl."""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

from pycocotools import mask as mu

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "gap_research" / "scripts"))

import eval_wave23_joint as ej  # noqa: E402
from filseg import data, metric  # noqa: E402

FULL = 2048


def main() -> None:
    t0 = time.time()
    folds = data.folds()
    all_stems = sorted(folds)
    recs, anns = data.coco_index()
    gt_recs = {s: [[data.ann_rle(a) for a in anns.get(rid, [])] for rid in recs[s]] for s in all_stems}

    # Build Track-A Clean OOF with default parameters matching exp006_007_tracka_clean.csv
    rles_str, _ = ej.build_track_a_rles(all_stems, "oof")
    rows_by_stem = {}
    for s in all_stems:
        pr = [{"size": [FULL, FULL], "counts": c.encode("ascii")} for c in rles_str[s]]
        rows_by_stem[s] = metric.counts(pr, gt_recs[s])

    sm_all = metric.summarize([rows_by_stem[s] for s in all_stems])
    sm_f0 = metric.summarize([rows_by_stem[s] for s in all_stems if folds[s] == 0])
    bins = ej.bin_pq_summary(rles_str, all_stems, gt_recs)
    print(f"[Track-A Clean 707 OOF]: PQ={sm_all['pq']:.5f} SQ={sm_all['sq']:.5f} RQ={sm_all['rq']:.5f} TP={sm_all['tp']} FP={sm_all['fp']} FN={sm_all['fn']} P={sum(len(rles_str[s]) for s in all_stems)} ({time.time()-t0:.1f}s)", flush=True)
    print(f"[Track-A Clean Fold-0 ]: PQ={sm_f0['pq']:.5f} SQ={sm_f0['sq']:.5f} RQ={sm_f0['rq']:.5f} TP={sm_f0['tp']} FP={sm_f0['fp']} FN={sm_f0['fn']}", flush=True)
    print("Bins:", bins, flush=True)

    # Audit how many Track-A Clean OOF masks exist inside exp011_pool.pkl (IoU > 0.85 or > 0.90)
    pool_p = Path("/private/tmp/exp011_pool.pkl")
    if pool_p.exists():
        pool = pickle.load(open(pool_p, "rb"))["oof"]
        tot_ta = 0
        cov_90 = 0
        cov_75 = 0
        miss_tp = 0
        miss_fp = 0
        ta_u_vals = []
        for s in all_stems:
            pr_ta = [{"size": [FULL, FULL], "counts": c.encode("ascii")} for c in rles_str[s]]
            pr_pl = [{"size": [FULL, FULL], "counts": c.encode("ascii")} for c in pool[s]["rles"]]
            if not pr_ta:
                continue
            tot_ta += len(pr_ta)
            if not pr_pl:
                continue
            mat = mu.iou(pr_ta, pr_pl, [0] * len(pr_pl))
            b_iou = mat.max(axis=1)
            cov_90 += int((b_iou >= 0.90).sum())
            cov_75 += int((b_iou >= 0.75).sum())
            # Exact u_i of Track-A masks
            rc = len(gt_recs[s])
            iu_sum = [0.0] * len(pr_ta)
            for gr in gt_recs[s]:
                if not gr:
                    continue
                gm = mu.iou(pr_ta, gr, [0] * len(gr)).max(axis=1)
                for k, v in enumerate(gm):
                    if v > 0.5:
                        iu_sum[k] += float(v)
            for k in range(len(pr_ta)):
                u_k = iu_sum[k] / rc
                ta_u_vals.append(u_k)
                if b_iou[k] < 0.75:
                    if u_k > 0.25:
                        miss_tp += 1
                    else:
                        miss_fp += 1
        print(f"Pool coverage of Track-A masks: IoU>=0.90: {cov_90}/{tot_ta} ({cov_90/tot_ta:.3f}) | IoU>=0.75: {cov_75}/{tot_ta} ({cov_75/tot_ta:.3f}) | miss(<0.75) high_u={miss_tp} low_u={miss_fp}", flush=True)

    out_p = Path("/private/tmp/tracka_clean_oof_cache.pkl")
    with open(out_p, "wb") as f:
        pickle.dump({"rles": rles_str, "counts": rows_by_stem, "sm_all": sm_all, "sm_f0": sm_f0, "bins": bins}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {out_p}", flush=True)


if __name__ == "__main__":
    main()
