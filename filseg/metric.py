"""Panoptic Quality exactly as the organisers' self-evaluation notebook counts it.

    PQ = sum of IoU over true positives / (|TP| + 0.5 |FP| + 0.5 |FN|)        (Kirillov et al., CVPR 2019)

A predicted and an annotated filament are a true positive when their IoU is strictly above 0.5. A photo annotated by
several people is scored against each annotator record separately and the counts are pooled, so a prediction that
agrees with one annotator but not the other is a TP for the first and an FP for the second.
"""
import numpy as np
from pycocotools import mask as mu

from . import data


def load_records(stems=None):
    """stem -> [[RLE, ...] per annotator record]."""
    recs, anns = data.coco_index()
    return {s: [[data.ann_rle(a) for a in anns.get(rid, [])] for rid in recs[s]]
            for s in (stems if stems is not None else recs)}


def counts(pred, records, iou_thr=0.5):
    """pred: [RLE] of one photo; records: [[RLE] per annotator]. -> dict(tp, fp, fn, iou) pooled over records."""
    c = dict(tp=0, fp=0, fn=0, iou=0.0)
    for gt in records:
        if not len(pred) or not len(gt):
            c["fp"] += len(pred)
            c["fn"] += len(gt)
            continue
        iou = mu.iou(pred, gt, [0] * len(gt))                # (n_pred, n_gt)
        hit = iou > iou_thr
        c["tp"] += int(hit.sum())
        c["iou"] += float(iou[hit].sum())
        c["fp"] += int((~hit.any(1)).sum())
        c["fn"] += int((~hit.any(0)).sum())
    return c


def summarize(rows):
    """Pool per-photo counts -> PQ and its factors SQ (mean IoU of matches) and RQ (F1 of matching)."""
    t = {k: sum(r[k] for r in rows) for k in ("tp", "fp", "fn", "iou")}
    den = t["tp"] + 0.5 * (t["fp"] + t["fn"])
    t.update(pq=t["iou"] / den if den else 0.0, sq=t["iou"] / t["tp"] if t["tp"] else 0.0, rq=t["tp"] / den if den else 0.0)
    return t


def overlap_stats(pred, records):
    """The other quantities of the organisers' self-evaluation notebook, for one photo and pooled over records:
    IoU and Dice of every overlapping annotation/prediction pair (their histograms are the "distribution of IoU /
    Dice scores"), and for every annotation / prediction the number of partners it overlaps at all (IoU > 0):
    0 = missed / spurious, 1 = one-to-one, 2+ = one-to-many (fragmented) / many-to-one (merged)."""
    out = dict(iou=[], dice=[], gt_degree=[], pred_degree=[])
    for gt in records:
        if not len(pred) or not len(gt):
            out["gt_degree"] += [0] * len(gt)
            out["pred_degree"] += [0] * len(pred)
            continue
        iou = mu.iou(pred, gt, [0] * len(gt))                # (n_pred, n_gt)
        hit = iou > 0
        out["iou"] += iou[hit].tolist()
        out["dice"] += (2 * iou[hit] / (1 + iou[hit])).tolist()
        out["gt_degree"] += hit.sum(0).tolist()
        out["pred_degree"] += hit.sum(1).tolist()
    return out


def bootstrap_ci(rows, groups, n=1000, seed=0):
    """95% interval of PQ when whole observation dates are resampled (photos of one date are not independent)."""
    by = {}
    for r, g in zip(rows, groups):
        by.setdefault(g, []).append(r)
    keys, rng, vals = sorted(by), np.random.default_rng(seed), []
    for _ in range(n):
        vals.append(summarize([r for k in rng.choice(len(keys), len(keys)) for r in by[keys[k]]])["pq"])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
