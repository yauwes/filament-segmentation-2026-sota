"""EXP-002b: Decouple clustering/seed matching threshold (0.60) from soft vmap emission threshold (0.525..0.65) + test P6 & logit fusion."""
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gap_research" / "scripts"))
sys.path.insert(0, str(REPO / "scripts"))
import build_super_submission as bss
import eval_harness as eh
import exp002_soft_fusion as e2

CACHE_PKL = Path("/private/tmp/gap_soft_fold0.pkl")


def run_soft_v2(data_by_stem, core_names, donor_weights, match=0.40, vote_seed=0.40, vr=0.60,
                c_floor=0.32, boost_both=0.02, p_mode="prob", p6_in_confirm=False,
                drop_paint=0.70, conf_overrides=None, core_weights=None):
    """Clusters match using crisp binary masks (p>0.60, vote_seed=0.40), while preserving TRUE continuous
    probability maps (or clipped logits) for vmap refinement and final vr thresholding!"""
    conf_ov = conf_overrides or {}
    rles_by_stem = {}
    for s, stem_dict in data_by_stem.items():
        painted_s = {}
        for k in set(core_names) | set(donor_weights):
            if k == "YOLO":
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
                        p_sub = np.where(keep, 0.85, np.where((p_u8 > 15) & ~occ[y1:y2, x1:x2], p_u8 / 255.0, 0.0)).astype(np.float32)
                        ylist.append((float(c), float(c), (x1, y1, x2, y2), keep, p_sub))
                painted_s["YOLO"] = ylist
            else:
                meta_cell = eh.STREAM_META[k]["cell"]
                cmin = conf_ov.get(k, meta_cell[2])
                painted_s[k] = e2.build_calibrated_paint(
                    stem_dict[k], t_paint=0.60, conf_min=cmin, drop=meta_cell[3], soft=True, cal_fn=None
                )

        # Cluster cleanly on BINARY seed masks (identical topology to E19!) while accumulating continuous soft vmap
        n = float(sum(core_weights)) if core_weights is not None else float(len(core_names))
        items = []
        for m_idx, name in enumerate(core_names):
            mw = float(core_weights[m_idx]) if core_weights is not None else 1.0
            for sc, raw_c, b, sub_bin, sub_p in painted_s[name]:
                area = int(sub_bin.sum())
                if area:
                    items.append((sc, b, sub_bin, sub_p, area, m_idx, mw))
        items.sort(key=lambda x: -x[0])

        clusters = []
        for sc, b, sub_bin, sub_p, area, m_idx, mw in items:
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
            w_eff = sc * mw
            if bi < 0:
                if p_mode == "logit":
                    val = np.log(np.clip(sub_p, 0.02, 0.98) / (1.0 - np.clip(sub_p, 0.02, 0.98)))
                else:
                    val = sub_p
                clusters.append({
                    "scores": [sc], "weights": [w_eff], "mweights": [mw],
                    "bin_items": [(b, sub_bin)], "soft_items": [(b, val)],
                    "box": b, "bin": sub_bin, "area": area, "vmap": val.copy(), "w_box": b,
                })
                continue
            cl = clusters[bi]
            cl["scores"].append(sc)
            cl["weights"].append(w_eff)
            cl["mweights"].append(mw)
            cl["bin_items"].append((b, sub_bin))
            val = np.log(np.clip(sub_p, 0.02, 0.98) / (1.0 - np.clip(sub_p, 0.02, 0.98))) if p_mode == "logit" else sub_p
            cl["soft_items"].append((b, val))
            ux1, uy1 = min(cl["box"][0], x1), min(cl["box"][1], y1)
            ux2, uy2 = max(cl["box"][2], x2), max(cl["box"][3], y2)
            # Crisp seed cluster update for IoU matching
            acc_b = np.zeros((uy2 - uy1, ux2 - ux1), np.float32)
            for ew, (mb, mbin) in zip(cl["weights"], cl["bin_items"]):
                acc_b[mb[1] - uy1:mb[3] - uy1, mb[0] - ux1:mb[2] - ux1] += ew * mbin
            fbin = (acc_b / max(sum(cl["weights"]), 1e-6)) >= vote_seed
            # Soft prob/logit accumulator: outside each member's trimmed box, logit=-3.89 (p=0.02) and prob=0.0
            bg_val = -3.8918 if p_mode == "logit" else 0.0
            w_tot = max(sum(cl["weights"]), 1e-6)
            acc_s = np.full((uy2 - uy1, ux2 - ux1), bg_val * w_tot, np.float32)
            for ew, (mb, ms) in zip(cl["weights"], cl["soft_items"]):
                acc_s[mb[1] - uy1:mb[3] - uy1, mb[0] - ux1:mb[2] - ux1] += ew * (ms - bg_val)
            vmap = acc_s / w_tot
            cl["box"] = (ux1, uy1, ux2, uy2)
            cl["bin"] = fbin
            cl["area"] = int(fbin.sum())
            cl["vmap"] = vmap

        fused = []
        for cl in clusters:
            if cl["area"] > 0:
                c_clust = float(np.sum(np.array(cl["scores"]) * np.array(cl["mweights"])) / sum(cl["mweights"])) * min(sum(cl["mweights"]), n) / n
                # Careful: if one stream contributes once per cluster this equals len(cs), keeping exact E19 formula:
                c_e19 = float(np.mean(cl["scores"])) * min(len(cl["scores"]), n) / n
                fused.append((c_e19, cl["box"], cl["bin"], cl["vmap"]))
        fused.sort(key=lambda x: -x[0])

        extra_dict = {k: painted_s[k] for k in donor_weights}
        mod_list = []
        for c, (ux1, uy1, ux2, uy2), fsub, vmap in fused:
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
            has_p6 = "P6" in matches and matches["P6"][4] >= 0.40
            confirmed = has_f3 and (has_ms or has_s or (p6_in_confirm and has_p6))
            c_new = c + (boost_both if confirmed else 0.0)
            if c_new < c_floor:
                continue

            donors = []
            for k, dw in donor_weights.items():
                if dw > 0 and k in matches:
                    dp = matches[k][3]
                    if p_mode == "logit":
                        dp = np.log(np.clip(dp, 0.02, 0.98) / (1.0 - np.clip(dp, 0.02, 0.98)))
                    donors.append((dw, matches[k][1], dp))
            if donors:
                gx1 = min([ux1] + [b[0] for _, b, _ in donors])
                gy1 = min([uy1] + [b[1] for _, b, _ in donors])
                gx2 = max([ux2] + [b[2] for _, b, _ in donors])
                gy2 = max([uy2] + [b[3] for _, b, _ in donors])
                bg_val = -3.8918 if p_mode == "logit" else 0.0
                acc = np.full((gy2 - gy1, gx2 - gx1), bg_val, np.float32)
                acc[uy1 - gy1:uy2 - gy1, ux1 - gx1:ux2 - gx1] += (1.0 * vmap - bg_val)
                w_map = np.ones((gy2 - gy1, gx2 - gx1), np.float32)
                for dw, db, dp in donors:
                    acc[db[1] - gy1:db[3] - gy1, db[0] - gx1:db[2] - gx1] += dw * (dp - bg_val)
                    w_map[db[1] - gy1:db[3] - gy1, db[0] - gx1:db[2] - gx1] += dw
                v_final = acc / w_map
                if p_mode == "logit":
                    v_final = 1.0 / (1.0 + np.exp(-v_final))
                rsub = v_final >= vr
                box_out, sub_out = ((gx1, gy1, gx2, gy2), rsub) if rsub.any() else ((ux1, uy1, ux2, uy2), fsub)
            else:
                v_final = (1.0 / (1.0 + np.exp(-vmap))) if p_mode == "logit" else vmap
                rsub = v_final >= vr
                box_out, sub_out = ((ux1, uy1, ux2, uy2), rsub) if rsub.any() else ((ux1, uy1, ux2, uy2), fsub)
            mod_list.append((c_new, box_out, sub_out))

        mod_list.sort(key=lambda x: -x[0])
        taken = np.zeros((2048, 2048), bool)
        rle_strs = []
        for c_new, (x1, y1, x2, y2), m in mod_list:
            if c_new < c_floor:
                break
            keep = m & ~taken[y1:y2, x1:x2]
            area = int(keep.sum())
            if area >= 100 and area >= drop_paint * int(m.sum()):
                taken[y1:y2, x1:x2] |= keep
                rle_strs.append(bss.encode_crop_ascii((x1, y1, x2, y2), keep))
        rles_by_stem[s] = rle_strs
    return rles_by_stem


def main():
    data = pickle.load(open(CACHE_PKL, "rb"))
    base_counts = pickle.load(open(REPO / "gap_research" / "predictions" / "e19_fold0_baseline.pkl", "rb"))["counts"]
    data_by_stem = {s: {k: data[k][s] for k in data} for s in eh.FOLD0_STEMS}
    core_e19 = ("G1", "A", "B", "P1", "P4a", "F0", "F1", "YOLO")
    donors_e19 = {"ms": 0.18, "s": 0.18, "main": 0.12, "F3": 0.06}

    tests = [
        ("Hard+P6core(vr=0.425,c=0.30)", core_e19 + ("P6",), donors_e19, "prob", 0.425, 0.30, False, True),
        ("SoftV2-prob(vr=0.525)", core_e19, donors_e19, "prob", 0.525, 0.32, False, False),
        ("SoftV2-prob(vr=0.550)", core_e19, donors_e19, "prob", 0.550, 0.32, False, False),
        ("SoftV2-prob(vr=0.575)", core_e19, donors_e19, "prob", 0.575, 0.32, False, False),
        ("SoftV2-prob(vr=0.600)", core_e19, donors_e19, "prob", 0.600, 0.32, False, False),
        ("SoftV2-prob(vr=0.625)", core_e19, donors_e19, "prob", 0.625, 0.32, False, False),
        ("SoftV2-prob+P6donor(vr=0.575)", core_e19, {**donors_e19, "P6": 0.12}, "prob", 0.575, 0.32, False, False),
        ("SoftV2-prob+P6core(vr=0.575,c=0.30)", core_e19 + ("P6",), donors_e19, "prob", 0.575, 0.30, True, False),
        ("SoftV2-logit(vr=0.575)", core_e19, donors_e19, "logit", 0.575, 0.32, False, False),
        ("SoftV2-logit(vr=0.625)", core_e19, donors_e19, "logit", 0.625, 0.32, False, False),
    ]
    for label, cn, dw, pmode, vr, cf, p6conf, hard_flag in tests:
        if hard_flag:
            rles = e2.run_super_variant(data_by_stem, cn, dw, vr=vr, c_floor=cf, use_soft=False, donor_soft=False)
        else:
            rles = run_soft_v2(data_by_stem, cn, dw, p_mode=pmode, vr=vr, c_floor=cf, p6_in_confirm=p6conf)
        counts = eh.score_rle_strs_dict(rles)
        res = eh.score_from_counts(counts)
        boot = eh.paired_date_bootstrap(counts, base_counts)
        print(
            f"{label:38s}: PQ={res['pq']:.4f} (SQ={res['sq']:.4f} RQ={res['rq']:.4f} TP={res['tp']} FP={res['fp']} FN={res['fn']})"
            f" | vsE19 mean={boot['mean_gain']:+.4f} CI=[{boot['ci95'][0]:+.4f},{boot['ci95'][1]:+.4f}] P={boot['win_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
