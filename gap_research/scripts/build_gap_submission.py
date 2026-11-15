"""Cross-Family Multi-Segmenter Fusion, Nested Date-CV Verification, and Strict Submission Generator.

Key architectural advances over baseline E19 (build_super_submission.py):
  1. Stage-2 Continuous Crop Fusion: combines orthogonal segmenter families (e4, e4x/grow, S1E morphology,
     SAM2.1 Hiera-B+, ConvNeXt-Topo clDice) per detected box at continuous [0, 1] pixel probability BEFORE
     binarization or cross-detector clustering — handling grown boxes cleanly in their union bbox.
  2. Clean Non-Transductive Detector Ensemble: supports E19 clean streams + upgraded ms/s/main/P6/P4a/F3
     + native-res 5-fold SAHI@1024 YOLO11m (size-calibrated <580px vs >=580px) + clean 5-fold Faster R-CNN,
     replacing F1 (MAGFiLO_extra) cleanly with competition-only detectors.
  3. Full 5-Fold 707-image OOF Evaluation AND Fold-0 5-Fold Nested Date-CV + 200 paired bootstraps vs E19.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "gap_research" / "scripts"))

import build_super_submission as bss  # noqa: E402
import eval_harness as eh  # noqa: E402
from filseg import postprocess, submission  # noqa: E402

FULL = 2048


def _unpack_npz(path: Path, tta_prefer: int = 4) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Load (conf, box, probs_u8_list) supporting both {'p'} and {'p1','p4'} npz schemas."""
    with np.load(path) as d:
        conf = np.asarray(d["conf"], np.float32)
        box = np.asarray(d["box"], np.int32).reshape(-1, 4)
        if "p" in d:
            flat = np.asarray(d["p"], np.uint8)
        elif tta_prefer == 4 and "p4" in d:
            flat = np.asarray(d["p4"], np.uint8)
        elif "p1" in d:
            flat = np.asarray(d["p1"], np.uint8)
        else:
            flat = np.asarray(d["p4"], np.uint8)
    probs = []
    off = 0
    for x1, y1, x2, y2 in box:
        w, h = max(0, int(x2 - x1)), max(0, int(y2 - y1))
        sz = w * h
        probs.append(flat[off : off + sz].reshape(h, w) if sz > 0 else np.zeros((0, 0), np.uint8))
        off += sz
    return conf, box, probs


def load_blended_stream(
    seg_dirs: list[tuple[str, Path, float, int]],
    stem: str,
    thr: float = 0.60,
    offset: int = 0,
    conf_floor: float = 0.25,
    drop: float = 0.70,
    min_area: int = 0,
    conf_scale: float = 1.0,
    small_scale: float = 1.0,
    small_px: int = 580,
) -> list[tuple[float, tuple[int, int, int, int], np.ndarray]]:
    """Decode & blend multiple Stage-2 segmenters per box at continuous [0,1] BEFORE greedy non-overlap painting.

    seg_dirs: list of (seg_name, dir_path, weight, tta_prefer)
    Returns list of (score, (tx1, ty1, tx2, ty2), sub_bool_mask) trimmed to foreground bbox (same interface as bss).
    Also returns continuous crop maps if needed by continuous clustering.
    """
    loaded = []
    for sname, sdir, w, tta in seg_dirs:
        if w <= 0:
            continue
        p = sdir / f"{stem}.npz"
        if not p.exists():
            # Also check <imgsz>/<stem>.npz fallback
            cands = sorted(sdir.glob(f"*/{stem}.npz"))
            if not cands:
                continue
            p = cands[0]
        conf, box, probs = _unpack_npz(p, tta_prefer=tta)
        loaded.append((sname, w, conf, box, probs))
    if not loaded:
        return []

    # Primary confidence and box ordering come from the first loaded segmenter
    _, _, conf0, box0, _ = loaded[0]
    n_boxes = len(conf0)
    if n_boxes == 0:
        return []

    occ = np.zeros((FULL, FULL), dtype=bool)
    t_u8 = int(thr * 255.0 + 0.5)
    out = []

    order = np.argsort(-conf0)
    for idx in order:
        raw_c = float(conf0[idx])
        bx0, by0, bx1, by1 = [int(v) for v in box0[idx]]
        max_side = max(bx1 - bx0, by1 - by0)
        eff_c = raw_c * conf_scale * (small_scale if max_side < small_px else 1.0)
        if eff_c < conf_floor:
            continue

        # Compute union bbox across all segmenters matching same box count
        gx1, gy1, gx2, gy2 = bx0, by0, bx1, by1
        valid_segs = []
        for sname, w, c_arr, b_arr, p_list in loaded:
            if idx >= len(b_arr):
                continue
            sb = b_arr[idx]
            if sb[2] <= sb[0] or sb[3] <= sb[1]:
                continue
            gx1 = min(gx1, int(sb[0]))
            gy1 = min(gy1, int(sb[1]))
            gx2 = max(gx2, int(sb[2]))
            gy2 = max(gy2, int(sb[3]))
            valid_segs.append((w, sb, p_list[idx]))

        if not valid_segs or gx2 <= gx1 or gy2 <= gy1:
            continue

        uh, uw = gy2 - gy1, gx2 - gx1
        if len(valid_segs) == 1 and valid_segs[0][2].shape == (uh, uw):
            prob_u8 = valid_segs[0][2]
        else:
            acc = np.zeros((uh, uw), dtype=np.float32)
            wsum = np.zeros((uh, uw), dtype=np.float32)
            for w, sb, pr in valid_segs:
                sx1, sy1, sx2, sy2 = int(sb[0]), int(sb[1]), int(sb[2]), int(sb[3])
                if pr.shape != (sy2 - sy1, sx2 - sx1):
                    pr = cv2.resize(pr, (sx2 - sx1, sy2 - sy1), interpolation=cv2.INTER_LINEAR)
                acc[sy1 - gy1 : sy2 - gy1, sx1 - gx1 : sx2 - gx1] += (w / 255.0) * pr
                wsum[sy1 - gy1 : sy2 - gy1, sx1 - gx1 : sx2 - gx1] += w
            prob_u8 = np.where(wsum > 0, (acc / np.maximum(wsum, 1e-6)) * 255.0 + 0.5, 0.0).astype(np.uint8)

        if offset:
            m_u8 = (prob_u8 > t_u8).astype(np.uint8)
            m = (cv2.dilate if offset > 0 else cv2.erode)(m_u8, bss._K[abs(offset)]).astype(bool)
        else:
            m = prob_u8 > t_u8
        keep = m & ~occ[gy1:gy2, gx1:gx2]
        k = int(keep.sum())
        if k > min_area and k > 0 and k >= drop * int(m.sum()):
            occ[gy1:gy2, gx1:gx2] |= keep
            v = bss.trim(eff_c, (gx1, gy1, gx2, gy2), keep)
            if v is not None:
                out.append(v)
    return out


def weighted_consensus_stem(
    streams: list[tuple[str, float, list[tuple[float, tuple[int, int, int, int], np.ndarray]]]],
    match_iou: float = 0.40,
    vote_thr: float = 0.425,
    c_floor: float = 0.30,
    min_area: int = 100,
    drop: float = 0.70,
) -> list[str]:
    """Unified weighted soft-mask clustering across any combination of orthogonal detector streams.

    Unlike E19's fixed core/donor split, EVERY detector stream with w > 0 can propose filaments,
    allowing 5-fold ms, SAHI@1024, and FRCNN-5F to rescue False Negatives missed by old Fold-0 cores.
    Cluster score = sum(w_k * c_k for matched streams) / sum(w_k for ALL streams) * normalization OR
    weighted mean matching E19 scale cleanly.
    """
    total_w = sum(w for _, w, _ in streams if w > 0)
    if total_w <= 0:
        return []

    # Each cluster tracks: (sum_wc, sum_w_present, items[(w, c, box, mask)], ux1, uy1, ux2, uy2, u_mask)
    items = []
    for sname, sw, insts in streams:
        if sw <= 0:
            continue
        for c, box, mask in insts:
            area = int(mask.sum())
            if area > 0:
                items.append((sname, sw, float(c), box, mask, area))
    # Highest weighted confidence proposes clusters first
    items.sort(key=lambda x: -(x[2] * min(1.5, x[1])))

    clusters = []
    for sname, sw, c, box, mask, area in items:
        x1, y1, x2, y2 = box
        best_j, best_iou = -1, 0.0
        for j, cl in enumerate(clusters):
            if sname in cl["seen"]:
                continue
            cx1, cy1, cx2, cy2 = cl["ubox"]
            ix1, iy1, ix2, iy2 = max(x1, cx1), max(y1, cy1), min(x2, cx2), min(y2, cy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            # Check IoU against representative top mask in cluster
            rmask = cl["rmask"]
            rx1, ry1, rx2, ry2 = cl["rbox"]
            rix1, riy1, rix2, riy2 = max(x1, rx1), max(y1, ry1), min(x2, rx2), min(y2, ry2)
            if rix2 <= rix1 or riy2 <= riy1:
                continue
            inter = int(
                (mask[riy1 - y1 : riy2 - y1, rix1 - x1 : rix2 - x1] & rmask[riy1 - ry1 : riy2 - ry1, rix1 - rx1 : rix2 - rx1]).sum()
            )
            if inter:
                iou = inter / (area + cl["rarea"] - inter)
                if iou >= match_iou and iou > best_iou:
                    best_iou = iou
                    best_j = j
        if best_j < 0:
            clusters.append({
                "seen": {sname},
                "items": [(sw, c, box, mask)],
                "rbox": box,
                "rmask": mask,
                "rarea": area,
                "ubox": [x1, y1, x2, y2],
            })
        else:
            cl = clusters[best_j]
            cl["seen"].add(sname)
            cl["items"].append((sw, c, box, mask))
            ub = cl["ubox"]
            ub[0] = min(ub[0], x1)
            ub[1] = min(ub[1], y1)
            ub[2] = max(ub[2], x2)
            ub[3] = max(ub[3], y2)

    mod_list = []
    for cl in clusters:
        sum_wc = sum(sw * c for sw, c, _, _ in cl["items"])
        c_score = sum_wc / total_w
        if c_score < c_floor:
            continue
        gx1, gy1, gx2, gy2 = cl["ubox"]
        acc = np.zeros((gy2 - gy1, gx2 - gx1), dtype=np.float32)
        for sw, _, (bx1, by1, bx2, by2), bm in cl["items"]:
            acc[by1 - gy1 : by2 - gy1, bx1 - gx1 : bx2 - gx1] += sw * bm
        w_present = sum(sw for sw, _, _, _ in cl["items"])
        rsub = (acc / max(w_present, 1e-6)) >= vote_thr
        if not rsub.any():
            continue
        v = bss.trim(c_score, (gx1, gy1, gx2, gy2), rsub)
        if v is not None:
            mod_list.append(v)

    mod_list.sort(key=lambda x: -x[0])
    taken = np.zeros((FULL, FULL), dtype=bool)
    rle_strs = []
    for c_new, (x1, y1, x2, y2), m in mod_list:
        if c_new < c_floor:
            break
        keep = m & ~taken[y1:y2, x1:x2]
        area = int(keep.sum())
        if area >= min_area and area >= drop * int(m.sum()):
            taken[y1:y2, x1:x2] |= keep
            rle_strs.append(bss.encode_crop_ascii((x1, y1, x2, y2), keep))
    return rle_strs


def super_ensemble_upgraded_stem(
    core_models: list[list[tuple[float, tuple[int, int, int, int], np.ndarray]]],
    extra_dict: dict[str, list[tuple[float, tuple[int, int, int, int], np.ndarray]]],
    extra_weights: dict[str, float] | None = None,
    match: float = 0.40,
    vote: float = 0.40,
    vr: float = 0.425,
    c_floor: float = 0.32,
    boost_keys: tuple[str, ...] = ("F3", "ms", "s"),
    boost_both: float = 0.02,
    rescue_streams: list[tuple[str, float, list[tuple[float, tuple[int, int, int, int], np.ndarray]]]] | None = None,
    rescue_min_score: float = 0.55,
    rescue_max_iou: float = 0.15,
    drop: float = 0.70,
    min_area: int = 100,
) -> list[str]:
    """Generalization of E19 super_ensemble_stem supporting upgraded cores, donor weights, and disjoint rescue."""
    fused = bss.fuse_with_continuous_acc(core_models, match=match, vote=vote)
    weights = extra_weights or {"ms": 0.18, "s": 0.18, "main": 0.12, "F3": 0.06}
    mod_list = []
    for c, (ux1, uy1, ux2, uy2), fsub, vmap in fused:
        farea = int(fsub.sum())
        if farea == 0:
            continue
        matches = {}
        for k, ex_list in extra_dict.items():
            best_iou, best_item = 0.0, None
            for ec, (ex1, ey1, ex2, ey2), esub in ex_list:
                earea = int(esub.sum())
                if not earea:
                    continue
                ix1, iy1, ix2, iy2 = max(ux1, ex1), max(uy1, ey1), min(ux2, ex2), min(uy2, ey2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = int((fsub[iy1 - uy1 : iy2 - uy1, ix1 - ux1 : ix2 - ux1] & esub[iy1 - ey1 : iy2 - ey1, ix1 - ex1 : ix2 - ex1]).sum())
                    if inter:
                        iou = inter / (farea + earea - inter)
                        if iou > best_iou:
                            best_iou, best_item = iou, (ec, (ex1, ey1, ex2, ey2), esub, iou)
            if best_item is not None and best_iou >= 0.30:
                matches[k] = best_item

        has_frcnn = any((k in matches and matches[k][3] >= 0.40) for k in ("F3", "F5F") if k in matches)
        has_ms = any((k in matches and matches[k][3] >= 0.40) for k in ("ms", "s", "SAHI") if k in matches)
        c_new = c + (boost_both if (has_frcnn and has_ms) else 0.0)
        if c_new < c_floor:
            continue

        donors = [(dw, matches[k][1], matches[k][2]) for k, dw in weights.items() if dw > 0 and k in matches]
        if donors:
            gx1 = min([ux1] + [b[0] for _, b, _ in donors])
            gy1 = min([uy1] + [b[1] for _, b, _ in donors])
            gx2 = max([ux2] + [b[2] for _, b, _ in donors])
            gy2 = max([uy2] + [b[3] for _, b, _ in donors])
            acc = np.zeros((gy2 - gy1, gx2 - gx1), dtype=np.float32)
            acc[uy1 - gy1 : uy2 - gy1, ux1 - gx1 : ux2 - gx1] += 1.0 * vmap
            w_map = np.ones((gy2 - gy1, gx2 - gx1), dtype=np.float32)
            for dw, db, dsub in donors:
                acc[db[1] - gy1 : db[3] - gy1, db[0] - gx1 : db[2] - gx1] += dw * dsub
                w_map[db[1] - gy1 : db[3] - gy1, db[0] - gx1 : db[2] - gx1] += dw
            rsub = (acc / w_map) >= vr
            box_out, sub_out = ((gx1, gy1, gx2, gy2), rsub) if rsub.any() else ((ux1, uy1, ux2, uy2), fsub)
        else:
            box_out, sub_out = (ux1, uy1, ux2, uy2), fsub
        mod_list.append((c_new, box_out, sub_out))

    mod_list.sort(key=lambda x: -x[0])
    taken = np.zeros((FULL, FULL), dtype=bool)
    painted = []
    for c_new, (x1, y1, x2, y2), m in mod_list:
        if c_new < c_floor:
            break
        keep = m & ~taken[y1:y2, x1:x2]
        area = int(keep.sum())
        if area >= min_area and area >= drop * int(m.sum()):
            taken[y1:y2, x1:x2] |= keep
            painted.append((c_new, (x1, y1, x2, y2), keep))

    # Optional additive disjoint recall rescue (e.g. high-conf SAHI<580px / FRCNN5F completely missed by core)
    if rescue_streams:
        rcands = []
        for sname, sw, insts in rescue_streams:
            for c, (rx1, ry1, rx2, ry2), rm in insts:
                sc = float(c) * sw
                if sc < rescue_min_score or int(rm.sum()) < min_area:
                    continue
                ov = int((rm & taken[ry1:ry2, rx1:rx2]).sum())
                if ov / max(1, int(rm.sum())) <= rescue_max_iou:
                    rcands.append((sc, (rx1, ry1, rx2, ry2), rm))
        rcands.sort(key=lambda x: -x[0])
        for sc, (rx1, ry1, rx2, ry2), rm in rcands:
            keep = rm & ~taken[ry1:ry2, rx1:rx2]
            area = int(keep.sum())
            if area >= min_area and area >= drop * int(rm.sum()):
                taken[ry1:ry2, rx1:rx2] |= keep
                painted.append((sc, (rx1, ry1, rx2, ry2), keep))

    return [bss.encode_crop_ascii(box, keep) for _, box, keep in painted]


def evaluate_rles_with_nested_cv(
    rles_by_stem: dict[str, list[str]],
    stems: list[str],
    baseline_counts_pkl: Path | None = REPO / "gap_research" / "predictions" / "e19_fold0_baseline.pkl",
) -> dict:
    counts = eh.score_rle_strs_dict(rles_by_stem, stems=stems)
    tot = np.sum([counts[s] for s in stems], axis=0)
    tp, fp, fn, iou_sum = tot
    pq = float(iou_sum / (tp + 0.5 * (fp + fn))) if (tp + fp + fn) else 0.0
    sq = float(iou_sum / tp) if tp else 0.0
    rq = float(tp / (tp + 0.5 * (fp + fn))) if (tp + fp + fn) else 0.0
    res = {
        "n_stems": len(stems),
        "pq": round(pq, 5),
        "sq": round(sq, 5),
        "rq": round(rq, 5),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }
    if baseline_counts_pkl and baseline_counts_pkl.exists() and set(stems) == set(eh.FOLD0_STEMS):
        base = pickle.load(open(baseline_counts_pkl, "rb"))["counts"]
        boot = eh.paired_date_bootstrap(counts, base, stems=stems, reps=200, seed=0)
        stem_to_f = eh.make_nested_date_folds(stems)
        n_folds = max(stem_to_f.values()) + 1
        fold_wins = 0
        for fi in range(n_folds):
            f_stems = [s for s in stems if stem_to_f[s] == fi]
            m_cand = eh.score_from_counts(counts, f_stems)["pq"]
            m_base = eh.score_from_counts(base, f_stems)["pq"]
            if m_cand > m_base:
                fold_wins += 1
        res["vs_e19_delta"] = round(boot["mean_gain"], 5)
        res["vs_e19_ci95"] = [round(boot["ci95"][0], 5), round(boot["ci95"][1], 5)]
        res["vs_e19_p_win"] = round(boot["win_rate"], 4)
        res["nested_date_fold_wins"] = f"{fold_wins}/{n_folds}"
    return res


def verify_e19_parity() -> None:
    with open(bss.FP_ROOT / "yolo_val_raw.json") as f:
        yolo_val = json.load(f)
    rles_e19 = {}
    for s in eh.FOLD0_STEMS:
        ps = {k: bss.paint_ts_cropped(bss.load_raw_npz(m["val"] / f"{s}.npz", tta=m["tta"]), *m["cell"]) for k, m in eh.STREAM_META.items()}
        yolo_s = bss.load_yolo_cropped(yolo_val, s, 0.2, -1, 0.7)
        core = [ps[k] for k in ("G1", "A", "B", "P1", "P4a", "F0", "F1")] + [yolo_s]
        extra = {k: ps[k] for k in ("ms", "s", "main", "F3")}
        rles_e19[s] = bss.super_ensemble_stem(core, extra)
    rep = evaluate_rles_with_nested_cv(rles_e19, list(eh.FOLD0_STEMS))
    print("E19 parity check:", json.dumps(rep, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["verify_e19"], default="verify_e19")
    args = ap.parse_args()
    if args.mode == "verify_e19":
        verify_e19_parity()


if __name__ == "__main__":
    main()
