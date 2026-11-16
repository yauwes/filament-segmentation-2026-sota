#!/usr/bin/env python3
"""Self-contained MiT-B3 dense + CC complementarity feature extractor for Wave-5 candidate pools."""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mutils

REPO = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(REPO), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import exp015_wave5_fusion as e15  # noqa: E402

MIT_DP_ROOT = Path(os.environ.get("FILSEG_MIT_DP_ROOT", "/private/tmp/gap_wave5_cache/exp012_mitb3/exp012_mitb3_dense_probs/dense_probs"))
MIT_CC_ROOT = Path(os.environ.get("FILSEG_MIT_CC_ROOT", "/private/tmp/gap_wave5_cache/exp012_mitb3/exp012_mitb3_cc_proposals/cc_proposals"))

MIT_NAMES = [
    "mit_fg_mean", "mit_fg_q75", "mit_fg_core", "mit_fg_rim",
    "mit_edge_mean", "mit_edge_rim", "mit_fg_iou50",
    "mit_a100_fg_diff", "mit_a100_fg_prod",
    "mit_cc_conf", "mit_cc_iou",
]

PROVENANCE_COLS = {
    "c_tracka", "weighted_vote", "n_streams", "ms_conf", "s_conf", "main_conf", "sahi_conf",
    "max_conf", "min_conf_nz", "std_conf_nz", "sahi_only", "ms_only", "sam_cnn_iou", "ms_s_iou", "stream_iou_mean",
    "kind_id", "is_ta_base", "is_ta_rescue", "iou_with_ta", "c_vs_ta", "near_ta_dist", "is_w5_cand",
}
_KERNEL_3 = np.ones((3, 3), dtype=np.uint8)


def extract_stem_mitb3(stem: str, split: str, ent: dict, dfg_idx: int) -> np.ndarray:
    """Compute 11 MiT-B3 dense probability & CC agreement features for one solar disk entry."""
    n = len(ent["rles"])
    if n == 0:
        return np.zeros((0, len(MIT_NAMES)), dtype=np.float32)

    dp = np.load(MIT_DP_ROOT / split / f"{stem}.npz")["p"]
    fg_h = dp[0].astype(np.float32) / 255.0
    ed_h = dp[1].astype(np.float32) / 255.0
    fg50 = fg_h >= 0.50
    fg50_sum = float(fg50.sum())

    confs_m, boxes_m, crops_m = e15._load_crop_stream(MIT_CC_ROOT / split / f"{stem}.npz")
    mit_rles = []
    if boxes_m is not None and len(boxes_m) > 0:
        for b, c in zip(boxes_m, crops_m):
            x1, y1, x2, y2 = [int(v) for v in b]
            m = np.zeros((2048, 2048), dtype=np.uint8)
            m[y1:y2, x1:x2] = (c >= 128).astype(np.uint8)
            mit_rles.append(mutils.encode(np.asfortranarray(m)))

    cand_rles = [{"size": [2048, 2048], "counts": r.encode("ascii")} for r in ent["rles"]]
    if mit_rles:
        ious_cc = mutils.iou(cand_rles, mit_rles, np.zeros(len(mit_rles), dtype=np.uint8))
        best_idx = ious_cc.argmax(axis=1)
        best_iou = ious_cc[np.arange(n), best_idx].astype(np.float32)
        best_conf = np.where(best_iou >= 0.30, confs_m[best_idx], 0.0).astype(np.float32)
    else:
        best_iou = np.zeros(n, dtype=np.float32)
        best_conf = np.zeros(n, dtype=np.float32)

    dfg_mean_col = ent["X"][:, dfg_idx]
    out = np.zeros((n, len(MIT_NAMES)), dtype=np.float32)
    for i, (rle, box) in enumerate(zip(cand_rles, ent["boxes"])):
        x1, y1, x2, y2 = [int(v) for v in box]
        hx1, hy1 = max(0, x1 // 2 - 2), max(0, y1 // 2 - 2)
        hx2, hy2 = min(1024, (x2 + 1) // 2 + 2), min(1024, (y2 + 1) // 2 + 2)
        if hx2 <= hx1 or hy2 <= hy1:
            continue
        m_full = mutils.decode(rle)
        sx1, sy1, sx2, sy2 = hx1 * 2, hy1 * 2, min(2048, hx2 * 2), min(2048, hy2 * 2)
        sub_full = m_full[sy1:sy2, sx1:sx2]
        w_h = max(1, (sx2 - sx1) // 2)
        h_h = max(1, (sy2 - sy1) // 2)
        m_half = sub_full[::2, ::2][:h_h, :w_h] > 0
        if not m_half.any():
            continue
        fg_sub = fg_h[hy1 : hy1 + h_h, hx1 : hx1 + w_h]
        ed_sub = ed_h[hy1 : hy1 + h_h, hx1 : hx1 + w_h]
        fg50_sub = fg50[hy1 : hy1 + h_h, hx1 : hx1 + w_h]

        vals_fg = fg_sub[m_half]
        fg_m = float(vals_fg.mean())
        fg_q75 = float(np.percentile(vals_fg, 75))
        m_u8 = m_half.astype(np.uint8)
        core = cv2.erode(m_u8, _KERNEL_3) > 0
        dil = cv2.dilate(m_u8, _KERNEL_3) > 0
        rim = np.logical_and(dil, np.logical_not(core))
        fg_core = float(fg_sub[core].mean()) if core.any() else fg_m
        fg_rim = float(fg_sub[rim].mean()) if rim.any() else fg_m
        ed_m = float(ed_sub[m_half].mean())
        ed_rim = float(ed_sub[rim].mean()) if rim.any() else ed_m
        inter = float(np.logical_and(m_half, fg50_sub).sum())
        union = float(m_half.sum()) + fg50_sum - inter
        iou50 = inter / max(union, 1.0)
        a100_fg = float(dfg_mean_col[i])
        out[i] = [
            fg_m, fg_q75, fg_core, fg_rim,
            ed_m, ed_rim, iou50,
            fg_m - a100_fg, float(np.sqrt(max(0.0, fg_m * a100_fg))),
            float(best_conf[i]), float(best_iou[i]),
        ]
    return out


def get_mit_feats(pool: dict, cache_path: Path | None) -> tuple[dict, dict]:
    """Load cached MiT-B3 features or extract across OOF + test splits."""
    if cache_path is not None and cache_path.exists():
        with open(cache_path, "rb") as f:
            d = pickle.load(f)
        if len(d["oof"]) == len(pool["oof"]) and len(d["test"]) == len(pool["test"]):
            print(f"[mit cache hit] {cache_path}", flush=True)
            return d["oof"], d["test"]
    t0 = time.time()
    dfg_idx = list(pool["feature_names"]).index("dfg_mean")
    oof_mit = {s: extract_stem_mitb3(s, "oof", pool["oof"][s], dfg_idx) for s in sorted(pool["oof"])}
    test_mit = {s: extract_stem_mitb3(s, "test", pool["test"][s], dfg_idx) for s in sorted(pool["test"])}
    if cache_path is not None:
        with open(cache_path, "wb") as f:
            pickle.dump({"oof": oof_mit, "test": test_mit}, f)
    print(f"[mit extract] {len(oof_mit)}+{len(test_mit)} stems in {time.time()-t0:.1f}s", flush=True)
    return oof_mit, test_mit
