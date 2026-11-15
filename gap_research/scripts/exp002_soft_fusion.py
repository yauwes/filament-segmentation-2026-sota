"""EXP-002: Continuous Soft Probability Fusion + P6 + Family Precision Calibration (5-Fold Nested Date CV)."""
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.isotonic import IsotonicRegression

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gap_research" / "scripts"))
sys.path.insert(0, str(REPO / "scripts"))
import build_super_submission as bss
import eval_harness as eh

CACHE_PKL = Path("/private/tmp/gap_soft_fold0.pkl")
TRIM_U8 = 15


def build_calibrated_paint(inst, t_paint=0.60, conf_min=0.10, drop=0.70, soft=True, cal_fn=None):
    """Per-stream de-duplication returning (calibrated_conf, raw_conf, box, mask_or_u8_float)."""
    occ = np.zeros((2048, 2048), bool)
    t_u8 = int(t_paint * 255 + 0.5)
    out = []
    for raw_c, (x1, y1, x2, y2), p_u8 in inst:
        if raw_c < conf_min:
            break
        m_bin = p_u8 > t_u8
        m_sum = int(m_bin.sum())
        if m_sum == 0:
            continue
        keep_bin = m_bin & ~occ[y1:y2, x1:x2]
        k = int(keep_bin.sum())
        if k > 0 and k >= drop * m_sum:
            occ[y1:y2, x1:x2] |= keep_bin
            if soft:
                # Zero out any pixels already claimed by higher-conf same-stream instance, retain full continuous probs elsewhere
                clean_mask = (p_u8 > TRIM_U8) & ~occ[y1:y2, x1:x2] | keep_bin
                p_out = np.where(clean_mask, p_u8, 0).astype(np.float32) / 255.0
                ys, xs = np.nonzero(p_u8 > TRIM_U8)
                # use keep_bin tight bounds union or full nonzero? Keep p_u8 > 25 tight box for speed
                nz = (p_out > (TRIM_U8 / 255.0))
                ys, xs = np.nonzero(nz)
                if not len(ys):
                    continue
                bx1, bx2 = int(xs.min()), int(xs.max()) + 1
                by1, by2 = int(ys.min()), int(ys.max()) + 1
                sub_p = np.ascontiguousarray(p_out[by1:by2, bx1:bx2], dtype=np.float32)
                sub_bin = np.ascontiguousarray(keep_bin[by1:by2, bx1:bx2], dtype=bool)
                box_t = (x1 + bx1, y1 + by1, x1 + bx2, y1 + by2)
            else:
                ys, xs = np.nonzero(keep_bin)
                bx1, bx2 = int(xs.min()), int(xs.max()) + 1
                by1, by2 = int(ys.min()), int(ys.max()) + 1
                sub_bin = np.ascontiguousarray(keep_bin[by1:by2, bx1:bx2], dtype=bool)
                sub_p = sub_bin.astype(np.float32)
                box_t = (x1 + bx1, y1 + by1, x1 + bx2, y1 + by2)
            sc = float(cal_fn(raw_c)) if cal_fn is not None else float(raw_c)
            out.append((sc, float(raw_c), box_t, sub_bin, sub_p))
    return out


def fuse_soft_clusters(models, weights=None, match=0.40, vote=0.40, use_prob_weight=True):
    """Greedy IoU clustering on binary seed masks while preserving TRUE continuous soft probability vmap."""
    n = float(sum(weights)) if weights is not None else float(len(models))
    items = []
    for m_idx, ms in enumerate(models):
        mw = float(weights[m_idx]) if weights is not None else 1.0
        for sc, raw_c, b, sub_bin, sub_p in ms:
            area = int(sub_bin.sum())
            if area > 0:
                items.append((sc, raw_c, b, sub_bin, sub_p, area, m_idx, mw))
    items.sort(key=lambda x: -x[0])

    clusters = []
    for sc, raw_c, b, sub_bin, sub_p, area, m_idx, mw in items:
        x1, y1, x2, y2 = b
        best, bi = match, -1
        for j, cl in enumerate(clusters):
            cx1, cy1, cx2, cy2 = cl["box"]
            ix1, iy1, ix2, iy2 = max(x1, cx1), max(y1, cy1), min(x2, cx2), min(y2, cy2)
            if ix2 > ix1 and iy2 > iy1:
                inter = int((cl["bin"][iy1 - cy1:iy2 - cy1, ix1 - cx1:ix2 - cx1] & sub_bin[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1]).sum())
                if inter:
                    iou = inter / (cl["area"] + area - inter)
                    if iou > best:
                        best, bi = iou, j
        w_eff = (sc * mw) if use_prob_weight else mw
        if bi < 0:
            clusters.append({
                "scores": [sc], "raws": [raw_c], "weights": [w_eff], "mweights": [mw],
                "items": [(b, sub_p, sub_bin)], "members": [m_idx],
                "box": b, "bin": sub_bin, "area": area, "vmap": sub_p.copy(),
            })
            continue
        cl = clusters[bi]
        cl["scores"].append(sc)
        cl["raws"].append(raw_c)
        cl["weights"].append(w_eff)
        cl["mweights"].append(mw)
        cl["items"].append((b, sub_p, sub_bin))
        cl["members"].append(m_idx)
        ux1, uy1 = min(cl["box"][0], x1), min(cl["box"][1], y1)
        ux2, uy2 = max(cl["box"][2], x2), max(cl["box"][3], y2)
        acc = np.zeros((uy2 - uy1, ux2 - ux1), np.float32)
        for ew, (mb, mp, _) in zip(cl["weights"], cl["items"]):
            acc[mb[1] - uy1:mb[3] - uy1, mb[0] - ux1:mb[2] - ux1] += ew * mp
        vmap = acc / max(sum(cl["weights"]), 1e-6)
        fbin = vmap >= vote
        cl["box"] = (ux1, uy1, ux2, uy2)
        cl["vmap"] = vmap
        cl["bin"] = fbin
        cl["area"] = int(fbin.sum())

    out = []
    for cl in clusters:
        if cl["area"] > 0:
            # Distinct member weighted support
            seen = {}
            for m_idx, mw in zip(cl["members"], cl["mweights"]):
                seen[m_idx] = max(seen.get(m_idx, 0.0), mw)
            sup_distinct = sum(seen.values())
            mean_s = float(np.sum(np.array(cl["scores"]) * np.array(cl["mweights"])) / sum(cl["mweights"]))
            c_clust = mean_s * min(sup_distinct, n) / n
            out.append((c_clust, cl["box"], cl["bin"], cl["vmap"], cl["members"]))
    return sorted(out, key=lambda x: -x[0])


def run_super_variant(data_by_stem, core_names, donor_weights, core_weights=None,
                      match=0.40, vote=0.40, vr=0.425, c_floor=0.32, boost_both=0.02,
                      use_soft=True, donor_soft=True, drop_paint=0.70, min_area=100,
                      conf_overrides=None, cal_maps=None):
    """Run E19-style or generalized Continuous Soft Super-Ensemble across stems."""
    conf_ov = conf_overrides or {}
    rles_by_stem = {}
    for s, stem_dict in data_by_stem.items():
        painted_s = {}
        for k in set(core_names) | set(donor_weights):
            if k == "YOLO":
                # load_yolo_soft already built normalized conf & u8 mask
                occ = np.zeros((2048, 2048), bool)
                ylist = []
                for c, (x1, y1, x2, y2), p_u8 in stem_dict["YOLO"]:
                    if c < conf_ov.get("YOLO", 0.20):
                        break
                    m_bin = p_u8 > 153
                    msum = int(m_bin.sum())
                    if not msum:
                        continue
                    keep = m_bin & ~occ[y1:y2, x1:x2]
                    area = int(keep.sum())
                    if area >= 100 and area >= 0.7 * msum:
                        occ[y1:y2, x1:x2] |= keep
                        sc = float(cal_maps["YOLO"](c)) if (cal_maps and "YOLO" in cal_maps) else float(c)
                        p_sub = (p_u8.astype(np.float32) / 255.0) if use_soft else keep.astype(np.float32)
                        ylist.append((sc, c, (x1, y1, x2, y2), keep, p_sub))
                painted_s["YOLO"] = ylist
            else:
                meta_cell = eh.STREAM_META[k]["cell"]
                cmin = conf_ov.get(k, meta_cell[2])
                cal_fn = cal_maps[k] if (cal_maps and k in cal_maps) else None
                painted_s[k] = build_calibrated_paint(
                    stem_dict[k], t_paint=0.60, conf_min=cmin, drop=meta_cell[3], soft=use_soft, cal_fn=cal_fn
                )

        core_models = [painted_s[k] for k in core_names]
        fused = fuse_soft_clusters(core_models, weights=core_weights, match=match, vote=vote)
        extra_dict = {k: painted_s[k] for k in donor_weights}

        mod_list = []
        for c, (ux1, uy1, ux2, uy2), fsub, vmap, members in fused:
            farea = int(fsub.sum())
            if not farea:
                continue
            matches = {}
            for k, ex_list in extra_dict.items():
                best_iou, best_item = 0.0, None
                for ec, raw_ec, (ex1, ey1, ex2, ey2), esub, ep in ex_list:
                    earea = int(esub.sum())
                    if not earea:
                        continue
                    ix1, iy1, ix2, iy2 = max(ux1, ex1), max(uy1, ey1), min(ux2, ex2), min(uy2, ey2)
                    if ix2 > ix1 and iy2 > iy1:
                        inter = int((fsub[iy1 - uy1:iy2 - uy1, ix1 - ux1:ix2 - ux1] & esub[iy1 - ey1:iy2 - ey1, ix1 - ex1:ix2 - ex1]).sum())
                        if inter:
                            iou = inter / (farea + earea - inter)
                            if iou > best_iou:
                                best_iou = iou
                                best_item = (ec, (ex1, ey1, ex2, ey2), esub, ep, iou)
                if best_item is not None and best_iou >= 0.30:
                    matches[k] = best_item

            has_f3 = "F3" in matches and matches["F3"][4] >= 0.40
            has_ms = "ms" in matches and matches["ms"][4] >= 0.40
            has_s = "s" in matches and matches["s"][4] >= 0.40
            c_new = c + (boost_both if (has_f3 and (has_ms or has_s)) else 0.0)
            if c_new < c_floor:
                continue

            donors = []
            for k, dw in donor_weights.items():
                if dw > 0 and k in matches:
                    dp = matches[k][3] if donor_soft else matches[k][2].astype(np.float32)
                    donors.append((dw, matches[k][1], dp))
            if donors:
                gx1 = min([ux1] + [b[0] for _, b, _ in donors])
                gy1 = min([uy1] + [b[1] for _, b, _ in donors])
                gx2 = max([ux2] + [b[2] for _, b, _ in donors])
                gy2 = max([uy2] + [b[3] for _, b, _ in donors])
                acc = np.zeros((gy2 - gy1, gx2 - gx1), np.float32)
                acc[uy1 - gy1:uy2 - gy1, ux1 - gx1:ux2 - gx1] += 1.0 * vmap
                w_map = np.ones((gy2 - gy1, gx2 - gx1), np.float32)
                for dw, db, dp in donors:
                    acc[db[1] - gy1:db[3] - gy1, db[0] - gx1:db[2] - gx1] += dw * dp
                    w_map[db[1] - gy1:db[3] - gy1, db[0] - gx1:db[2] - gx1] += dw
                rsub = (acc / w_map) >= vr
                box_out, sub_out = ((gx1, gy1, gx2, gy2), rsub) if rsub.any() else ((ux1, uy1, ux2, uy2), fsub)
            else:
                box_out, sub_out = (ux1, uy1, ux2, uy2), fsub
            mod_list.append((c_new, box_out, sub_out))

        mod_list.sort(key=lambda x: -x[0])
        taken = np.zeros((2048, 2048), bool)
        rle_strs = []
        for c_new, (x1, y1, x2, y2), m in mod_list:
            if c_new < c_floor:
                break
            keep = m & ~taken[y1:y2, x1:x2]
            area = int(keep.sum())
            if area >= min_area and area >= drop_paint * int(m.sum()):
                taken[y1:y2, x1:x2] |= keep
                rle_strs.append(bss.encode_crop_ascii((x1, y1, x2, y2), keep))
        rles_by_stem[s] = rle_strs
    return rles_by_stem


def main():
    t0 = time.time()
    data = pickle.load(open(CACHE_PKL, "rb"))
    base_pkl = pickle.load(open(REPO / "gap_research" / "predictions" / "e19_fold0_baseline.pkl", "rb"))
    base_counts = base_pkl["counts"]
    data_by_stem = {s: {k: data[k][s] for k in data} for s in eh.FOLD0_STEMS}
    print(f"Loaded soft cache in {time.time()-t0:.2f}s")

    core_e19 = ("G1", "A", "B", "P1", "P4a", "F0", "F1", "YOLO")
    donors_e19 = {"ms": 0.18, "s": 0.18, "main": 0.12, "F3": 0.06}

    configs = [
        ("EXP-002a E19-hard-repro", core_e19, donors_e19, False, False, 0.425, 0.32, None),
        ("EXP-002b TrueSoft-prob", core_e19, donors_e19, True, True, 0.425, 0.32, None),
        ("EXP-002c TrueSoft+vr0.45", core_e19, donors_e19, True, True, 0.450, 0.32, None),
        ("EXP-002d TrueSoft+vr0.475", core_e19, donors_e19, True, True, 0.475, 0.32, None),
        ("EXP-002e TrueSoft+vr0.50", core_e19, donors_e19, True, True, 0.500, 0.32, None),
        ("EXP-002f TrueSoft+P6core(vr=0.45)", core_e19 + ("P6",), donors_e19, True, True, 0.450, 0.30, None),
        ("EXP-002g TrueSoft+P6donor(vr=0.45)", core_e19, {**donors_e19, "P6": 0.12}, True, True, 0.450, 0.32, None),
        ("EXP-002h TrueSoft+RelaxFRCNN(F0:0.65,F1:0.70,F3:0.70)+P6", core_e19 + ("P6",), donors_e19, True, True, 0.475, 0.30, {"F0": 0.65, "F1": 0.70, "F3": 0.70}),
    ]
    for label, cn, dw, us, ds, vr, cf, cov in configs:
        ts = time.time()
        rles = run_super_variant(data_by_stem, cn, dw, vr=vr, c_floor=cf, use_soft=us, donor_soft=ds, conf_overrides=cov)
        counts = eh.score_rle_strs_dict(rles)
        res = eh.score_from_counts(counts)
        boot = eh.paired_date_bootstrap(counts, base_counts)
        print(
            f"{label:52s}: PQ={res['pq']:.4f} (SQ={res['sq']:.4f} RQ={res['rq']:.4f} TP={res['tp']} FP={res['fp']} FN={res['fn']})"
            f" | vsE19 mean={boot['mean_gain']:+.4f} CI95=[{boot['ci95'][0]:+.4f},{boot['ci95'][1]:+.4f}] P={boot['win_rate']:.3f} ({time.time()-ts:.1f}s)"
        )


if __name__ == "__main__":
    main()
