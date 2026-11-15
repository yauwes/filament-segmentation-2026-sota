"""EXP-001b / EXP-003 / EXP-004 / EXP-005: Local Gap Research Suite with strict 5-Fold Date-Grouped Nested CV on Fold 0."""
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gap_research" / "scripts"))
sys.path.insert(0, str(REPO / "scripts"))
import build_super_submission as bss
import eval_harness as eh
import exp002_soft_fusion as e2

CACHE_PKL = Path("/private/tmp/gap_soft_fold0.pkl")
CLUSTER_PKL = Path("/private/tmp/gap_e19_clusters_fold0.pkl")

ALL_STREAMS = ("G1", "A", "B", "P1", "P4a", "P6", "F0", "F1", "F3", "ms", "s", "main", "YOLO")
CORE_E19 = ("G1", "A", "B", "P1", "P4a", "F0", "F1", "YOLO")
DONORS_E19 = {"ms": 0.18, "s": 0.18, "main": 0.12, "F3": 0.06}
_K1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def mask_iou_box(b1, m1, a1, b2, m2, a2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = int((m1[iy1 - b1[1]:iy2 - b1[1], ix1 - b1[0]:ix2 - b1[0]] & m2[iy1 - b2[1]:iy2 - b2[1], ix1 - b2[0]:ix2 - b2[0]]).sum())
    if not inter:
        return 0.0
    return inter / (a1 + a2 - inter)


def extract_all_candidates(c_min_extract=0.14):
    """Extract every E19 fused+donor candidate down to c>=0.14 with exact binary + soft vmap + 13-stream match & morphology features."""
    t0 = time.time()
    data = pickle.load(open(CACHE_PKL, "rb"))
    by_stem = {}
    for s in eh.FOLD0_STEMS:
        # 1. Paint exact E19 binary + relaxed lower-conf donor candidates
        painted_e19 = {}
        painted_relaxed = {}
        for k in ALL_STREAMS:
            if k == "YOLO":
                occ = np.zeros((2048, 2048), bool)
                ylist, yrel = [], []
                for c, (x1, y1, x2, y2), p_u8 in data["YOLO"][s]:
                    m_bin = p_u8 > 153
                    msum = int(m_bin.sum())
                    if not msum:
                        continue
                    keep = m_bin & ~occ[y1:y2, x1:x2]
                    area = int(keep.sum())
                    if area >= 100 and area >= 0.7 * msum:
                        if c >= 0.20:
                            occ[y1:y2, x1:x2] |= keep
                            ylist.append((float(c), float(c), (x1, y1, x2, y2), keep, (p_u8.astype(np.float32) / 255.0)))
                        if c >= 0.10:
                            yrel.append((float(c), (x1, y1, x2, y2), keep, int(keep.sum())))
                painted_e19["YOLO"] = ylist
                painted_relaxed["YOLO"] = yrel
            else:
                meta_cell = eh.STREAM_META[k]["cell"]
                painted_e19[k] = e2.build_calibrated_paint(
                    data[k][s], t_paint=0.60, conf_min=meta_cell[2], drop=meta_cell[3], soft=True, cal_fn=None
                )
                cmin_rel = 0.45 if k in ("F0", "F1", "F3") else 0.12
                # Simple top-box matchers down to cmin_rel for meta features
                rel_list = []
                for raw_c, (x1, y1, x2, y2), p_u8 in data[k][s]:
                    if raw_c < cmin_rel:
                        break
                    mb = p_u8 > 153
                    a = int(mb.sum())
                    if a >= 30:
                        rel_list.append((float(raw_c), (x1, y1, x2, y2), mb, a))
                painted_relaxed[k] = rel_list

        # 2. Cluster E19 exact 8-core (using crisp binary M_i for vmap matching exact E19!)
        n = float(len(CORE_E19))
        items = []
        for m_idx, name in enumerate(CORE_E19):
            for sc, raw_c, b, sub_bin, sub_p in painted_e19[name]:
                area = int(sub_bin.sum())
                if area:
                    items.append((sc, b, sub_bin, sub_p, area, name))
        items.sort(key=lambda x: -x[0])

        clusters = []
        for sc, b, sub_bin, sub_p, area, name in items:
            x1, y1, x2, y2 = b
            best, bi = 0.40, -1
            for j, cl in enumerate(clusters):
                iou = mask_iou_box(cl["box"], cl["bin"], cl["area"], b, sub_bin, area)
                if iou > best:
                    best, bi = iou, j
            if bi < 0:
                clusters.append({
                    "scores": [sc], "names": [name], "bin_items": [(b, sub_bin)],
                    "soft_items": [(b, sub_p)], "box": b, "bin": sub_bin, "area": area,
                    "vmap_bin": sub_bin.astype(np.float32), "vmap_soft": sub_p.copy(),
                })
                continue
            cl = clusters[bi]
            cl["scores"].append(sc)
            cl["names"].append(name)
            cl["bin_items"].append((b, sub_bin))
            cl["soft_items"].append((b, sub_p))
            ux1, uy1 = min(cl["box"][0], x1), min(cl["box"][1], y1)
            ux2, uy2 = max(cl["box"][2], x2), max(cl["box"][3], y2)
            w_tot = max(sum(cl["scores"]), 1e-6)
            acc_b = np.zeros((uy2 - uy1, ux2 - ux1), np.float32)
            acc_s = np.zeros((uy2 - uy1, ux2 - ux1), np.float32)
            for mc, (mb, mbin), (_, msoft) in zip(cl["scores"], cl["bin_items"], cl["soft_items"]):
                acc_b[mb[1] - uy1:mb[3] - uy1, mb[0] - ux1:mb[2] - ux1] += mc * mbin
                acc_s[mb[1] - uy1:mb[3] - uy1, mb[0] - ux1:mb[2] - ux1] += mc * msoft
            vb = acc_b / w_tot
            vs = acc_s / w_tot
            fb = vb >= 0.40
            cl["box"] = (ux1, uy1, ux2, uy2)
            cl["vmap_bin"] = vb
            cl["vmap_soft"] = vs
            cl["bin"] = fb
            cl["area"] = int(fb.sum())

        cands = []
        extra_e19 = {k: painted_e19[k] for k in ("ms", "s", "main", "F3")}
        gt_rles_list = eh.GT[s]

        for cl in clusters:
            if not cl["area"]:
                continue
            c_base = float(np.mean(cl["scores"])) * min(len(cl["scores"]), n) / n
            if c_base < c_min_extract:
                continue
            ux1, uy1, ux2, uy2 = cl["box"]
            fsub, farea = cl["bin"], cl["area"]

            # Exact E19 donor match + refinement
            matches = {}
            for k, ex_list in extra_e19.items():
                best_iou, best_item = 0.0, None
                for ec, raw_ec, eb, esub, ep in ex_list:
                    ea = int(esub.sum())
                    if not ea:
                        continue
                    iou = mask_iou_box((ux1, uy1, ux2, uy2), fsub, farea, eb, esub, ea)
                    if iou > best_iou:
                        best_iou = iou
                        best_item = (ec, eb, esub, ep, iou)
                if best_item is not None and best_iou >= 0.30:
                    matches[k] = best_item

            has_f3 = "F3" in matches and matches["F3"][4] >= 0.40
            has_ms = "ms" in matches and matches["ms"][4] >= 0.40
            has_s = "s" in matches and matches["s"][4] >= 0.40
            c_e19 = c_base + (0.02 if (has_f3 and (has_ms or has_s)) else 0.0)

            donors = [(dw, matches[k][1], matches[k][2], matches[k][3]) for k, dw in DONORS_E19.items() if dw > 0 and k in matches]
            if donors:
                gx1 = min([ux1] + [b[0] for _, b, _, _ in donors])
                gy1 = min([uy1] + [b[1] for _, b, _, _ in donors])
                gx2 = max([ux2] + [b[2] for _, b, _, _ in donors])
                gy2 = max([uy2] + [b[3] for _, b, _, _ in donors])
                acc_b = np.zeros((gy2 - gy1, gx2 - gx1), np.float32)
                acc_s = np.zeros((gy2 - gy1, gx2 - gx1), np.float32)
                acc_b[uy1 - gy1:uy2 - gy1, ux1 - gx1:ux2 - gx1] += cl["vmap_bin"]
                acc_s[uy1 - gy1:uy2 - gy1, ux1 - gx1:ux2 - gx1] += cl["vmap_soft"]
                w_map = np.ones((gy2 - gy1, gx2 - gx1), np.float32)
                for dw, db, dbin, dsoft in donors:
                    acc_b[db[1] - gy1:db[3] - gy1, db[0] - gx1:db[2] - gx1] += dw * dbin
                    acc_s[db[1] - gy1:db[3] - gy1, db[0] - gx1:db[2] - gx1] += dw * dsoft
                    w_map[db[1] - gy1:db[3] - gy1, db[0] - gx1:db[2] - gx1] += dw
                vb_fin = acc_b / w_map
                vs_fin = acc_s / w_map
                rsub = vb_fin >= 0.425
                box_out, mask_e19, vb_out, vs_out = ((gx1, gy1, gx2, gy2), rsub, vb_fin, vs_fin) if rsub.any() else ((ux1, uy1, ux2, uy2), fsub, cl["vmap_bin"], cl["vmap_soft"])
            else:
                box_out, mask_e19, vb_out, vs_out = (ux1, uy1, ux2, uy2), fsub, cl["vmap_bin"], cl["vmap_soft"]

            area_e19 = int(mask_e19.sum())
            if area_e19 < 50:
                continue

            # Rich ALL-13 relaxed stream matches against mask_e19
            stream_c = np.zeros(len(ALL_STREAMS), dtype=np.float32)
            stream_iou = np.zeros(len(ALL_STREAMS), dtype=np.float32)
            for idx_k, k in enumerate(ALL_STREAMS):
                biou, bc = 0.0, 0.0
                for rc, rb, rbin, ra in painted_relaxed[k]:
                    iou = mask_iou_box(box_out, mask_e19, area_e19, rb, rbin, ra)
                    if iou > biou:
                        biou, bc = iou, rc
                if biou >= 0.25:
                    stream_c[idx_k] = bc
                    stream_iou[idx_k] = biou

            # Morphology + solar radius + continuous vmap_soft quality stats
            x1, y1, x2, y2 = box_out
            bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            r_norm = float(np.hypot(cx - 1024.0, cy - 1024.0) / 920.0)
            # Max radial extent of mask corners
            r_max = float(max(
                np.hypot(x1 - 1024.0, y1 - 1024.0), np.hypot(x2 - 1024.0, y1 - 1024.0),
                np.hypot(x1 - 1024.0, y2 - 1024.0), np.hypot(x2 - 1024.0, y2 - 1024.0),
            ) / 920.0)
            m_u8 = mask_e19.astype(np.uint8)
            cnts, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            perim = float(sum(cv2.arcLength(cnt, True) for cnt in cnts))
            thinness = float((perim ** 2) / (4.0 * np.pi * max(area_e19, 1)))
            fill_ratio = float(area_e19 / (bw * bh))
            aspect = float(min(bw, bh) / max(bw, bh))
            mean_vsoft = float(vs_out[mask_e19].mean())
            q25_vsoft = float(np.percentile(vs_out[mask_e19], 25))
            halo_ratio = float((vs_out >= 0.35).sum() / max(area_e19, 1))
            core_ratio = float((vs_out >= 0.75).sum() / max(area_e19, 1))
            mean_vbin = float(vb_out[mask_e19].mean())

            n_yolo11_t = float(sum(stream_iou[i] >= 0.30 for i in (0, 1, 2)))
            n_yolo11_s = float(sum(stream_iou[i] >= 0.30 for i in (3, 4, 5)))
            n_frcnn = float(sum(stream_iou[i] >= 0.30 for i in (6, 7, 8)))
            n_5fold = float(sum(stream_iou[i] >= 0.30 for i in (9, 10, 11)))
            has_yoloseg = float(stream_iou[12] >= 0.30)
            n_families = float((n_yolo11_t > 0) + (n_yolo11_s > 0) + (n_frcnn > 0) + (n_5fold > 0) + (has_yoloseg > 0))

            feat = np.concatenate([
                np.array([
                    c_base, c_e19, float(len(cl["scores"])), np.std(cl["scores"]),
                    np.log1p(area_e19), np.log1p(bw * bh), fill_ratio, aspect, thinness,
                    r_norm, r_max, mean_vbin, mean_vsoft, q25_vsoft, halo_ratio, core_ratio,
                    n_yolo11_t, n_yolo11_s, n_frcnn, n_5fold, has_yoloseg, n_families,
                ], dtype=np.float32),
                stream_c,
                stream_iou,
            ])

            # Compute exact multi-record targets (only used cleanly in Nested OOF fit!)
            rle_e19 = bss.encode_crop_ascii(box_out, mask_e19)
            rle_obj = [{"size": [2048, 2048], "counts": rle_e19.encode("ascii")}]
            iou_hits = []
            for g_rles in gt_rles_list:
                if not g_rles:
                    iou_hits.append(0.0)
                else:
                    vals = mu.iou(rle_obj, g_rles, [0] * len(g_rles))[0]
                    vmax = float(vals.max())
                    iou_hits.append(vmax if vmax > 0.50 else 0.0)
            y_iou_per_rec = float(np.mean(iou_hits))
            y_tp_rate = float(np.mean([1.0 if v > 0 else 0.0 for v in iou_hits]))

            cands.append({
                "c_base": c_base,
                "c_e19": c_e19,
                "box": box_out,
                "mask": mask_e19,
                "vb": vb_out,
                "vs": vs_out,
                "feat": feat,
                "y_iou": y_iou_per_rec,
                "y_tp": y_tp_rate,
            })

        by_stem[s] = cands

    with open(CLUSTER_PKL, "wb") as f:
        pickle.dump(by_stem, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Extracted {sum(len(v) for v in by_stem.values())} candidate clusters in {time.time()-t0:.2f}s -> {CLUSTER_PKL}")
    return by_stem


def paint_cands_to_rles(cand_list, score_fn=None, floor=0.32, mask_fn=None, stitch_gap=0, min_area=100, drop=0.70):
    """Order candidates by score_fn, optionally apply mask_fn or collinear spine fragment stitching, then greedy paint."""
    items = []
    for idx, c in enumerate(cand_list):
        sc = float(c["c_e19"]) if score_fn is None else float(score_fn(idx, c))
        if sc < floor:
            continue
        b, m = (c["box"], c["mask"]) if mask_fn is None else mask_fn(c)
        if m is not None and int(m.sum()) >= 50:
            items.append((sc, b, m))
    items.sort(key=lambda x: -x[0])

    # Optional EXP-004: Collinear Spine Fragment Stitching before greedy painting
    if stitch_gap > 0 and len(items) > 1:
        used = [False] * len(items)
        merged = []
        for i in range(len(items)):
            if used[i]:
                continue
            sc_i, (x1, y1, x2, y2), m_i = items[i]
            ys_i, xs_i = np.nonzero(m_i)
            if len(ys_i) < 30:
                merged.append(items[i])
                used[i] = True
                continue
            pts_i = np.column_stack([xs_i + x1, ys_i + y1]).astype(np.float32)
            _, _, vt_i = np.linalg.svd(pts_i - pts_i.mean(0), full_matrices=False)
            dir_i = vt_i[0]

            # Look for one best compatible collinear piece within bounding proximity
            best_j = -1
            for j in range(i + 1, len(items)):
                if used[j]:
                    continue
                sc_j, (jx1, jy1, jx2, jy2), m_j = items[j]
                # Coarse bbox proximity check
                gx = max(0, max(x1, jx1) - min(x2, jx2))
                gy = max(0, max(y1, jy1) - min(y2, jy2))
                if max(gx, gy) > stitch_gap:
                    continue
                # Must NOT heavily overlap already (overlap is deduplication, not fragmented split)
                ix1, iy1, ix2, iy2 = max(x1, jx1), max(y1, jy1), min(x2, jx2), min(y2, jy2)
                inter = 0
                if ix2 > ix1 and iy2 > iy1:
                    inter = int((m_i[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1] & m_j[iy1 - jy1:iy2 - jy1, ix1 - jx1:ix2 - jx1]).sum())
                iou = inter / max(int(m_i.sum()) + int(m_j.sum()) - inter, 1)
                if iou > 0.25:
                    continue
                ys_j, xs_j = np.nonzero(m_j)
                if len(ys_j) < 20:
                    continue
                pts_j = np.column_stack([xs_j + jx1, ys_j + jy1]).astype(np.float32)
                _, _, vt_j = np.linalg.svd(pts_j - pts_j.mean(0), full_matrices=False)
                dir_j = vt_j[0]
                if abs(float(np.dot(dir_i, dir_j))) < 0.88:  # ~28 deg max principal axis angle
                    continue
                # Minimum pixel-to-pixel endpoint distance (subsampled for speed)
                p1 = pts_i[::max(1, len(pts_i) // 60)]
                p2 = pts_j[::max(1, len(pts_j) // 60)]
                d2 = ((p1[:, None, :] - p2[None, :, :]) ** 2).sum(-1)
                idx_min = int(d2.argmin())
                min_dist = float(np.sqrt(d2.flat[idx_min]))
                if min_dist <= stitch_gap:
                    r_a, r_b = divmod(idx_min, len(p2))
                    gap_vec = p2[r_b] - p1[r_a]
                    gn = float(np.linalg.norm(gap_vec))
                    if gn < 2.0 or abs(float(np.dot(gap_vec / gn, dir_i))) >= 0.80:
                        best_j = j
                        break

            if best_j >= 0:
                sc_j, (jx1, jy1, jx2, jy2), m_j = items[best_j]
                ux1, uy1 = min(x1, jx1), min(y1, jy1)
                ux2, uy2 = max(x2, jx2), max(y2, jy2)
                um = np.zeros((uy2 - uy1, ux2 - ux1), dtype=bool)
                um[y1 - uy1:y2 - uy1, x1 - ux1:x2 - ux1] |= m_i
                um[jy1 - uy1:jy2 - uy1, jx1 - ux1:jx2 - ux1] |= m_j
                merged.append((max(sc_i, sc_j), (ux1, uy1, ux2, uy2), um))
                used[i] = True
                used[best_j] = True
            else:
                merged.append(items[i])
                used[i] = True
        items = merged

    taken = np.zeros((2048, 2048), bool)
    rles = []
    for sc, (x1, y1, x2, y2), m in items:
        keep = m & ~taken[y1:y2, x1:x2]
        area = int(keep.sum())
        if area >= min_area and area >= drop * int(m.sum()):
            taken[y1:y2, x1:x2] |= keep
            rles.append(bss.encode_crop_ascii((x1, y1, x2, y2), keep))
    return rles


def main():
    by_stem = pickle.load(open(CLUSTER_PKL, "rb")) if CLUSTER_PKL.exists() else extract_all_candidates(0.14)
    base_counts = pickle.load(open(REPO / "gap_research" / "predictions" / "e19_fold0_baseline.pkl", "rb"))["counts"]
    nested_folds = eh.make_nested_date_folds(eh.FOLD0_STEMS, n_folds=5, seed=42)

    # 1. Exact candidate-cache repro check
    r_base = {s: paint_cands_to_rles(by_stem[s], floor=0.32) for s in eh.FOLD0_STEMS}
    c_base = eh.score_rle_strs_dict(r_base)
    m_base = eh.score_from_counts(c_base)
    print(f"E19 Repro via ClusterCache: PQ={m_base['pq']:.4f} (TP={m_base['tp']} FP={m_base['fp']} FN={m_base['fn']})")

    # 2. EXP-001b: Strict Nested 5-Fold Date-CV over E19 floor alone
    r_oof_floor = {}
    picked_floors = []
    for k in range(5):
        tr = [s for s in eh.FOLD0_STEMS if nested_folds[s] != k]
        va = [s for s in eh.FOLD0_STEMS if nested_folds[s] == k]
        best_fl, best_pq = 0.32, -1.0
        for fl in (0.28, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35):
            rc = eh.score_rle_strs_dict({s: paint_cands_to_rles(by_stem[s], floor=fl) for s in tr}, stems=tr)
            pq = eh.score_from_counts(rc, stems=tr)["pq"]
            if pq > best_pq:
                best_pq, best_fl = pq, fl
        picked_floors.append(best_fl)
        for s in va:
            r_oof_floor[s] = paint_cands_to_rles(by_stem[s], floor=best_fl)
    c_1b = eh.score_rle_strs_dict(r_oof_floor)
    m_1b = eh.score_from_counts(c_1b)
    print(f"EXP-001b E19 Nested Date-CV floor ({picked_floors}): PQ={m_1b['pq']:.4f} TP={m_1b['tp']} FP={m_1b['fp']} FN={m_1b['fn']}")

    # 3. EXP-003: Scale-Adaptive Mask Refinement (tighten small blobs using vs_out / vb_out or erode, expand large tails)
    print("\n=== EXP-003: SCALE-ADAPTIVE MASK CALIBRATION ===")
    for name, fn in [
        ("Small<700 tighter (vb>=0.52)", lambda c: (c["box"], c["vb"] >= 0.52) if int(c["mask"].sum()) < 700 and (c["vb"] >= 0.52).sum() >= 100 else (c["box"], c["mask"])),
        ("Small<700 tighter (vs>=0.60)", lambda c: (c["box"], c["vs"] >= 0.60) if int(c["mask"].sum()) < 700 and (c["vs"] >= 0.60).sum() >= 100 else (c["box"], c["mask"])),
        ("Small<550 erode1 (min90)", lambda c: (c["box"], cv2.erode(c["mask"].astype(np.uint8), _K1).astype(bool)) if int(c["mask"].sum()) < 550 and cv2.erode(c["mask"].astype(np.uint8), _K1).sum() >= 90 else (c["box"], c["mask"])),
        ("Small<600 vb>=0.50 & Large>=1600 vb>=0.38", lambda c: (c["box"], c["vb"] >= (0.50 if int(c["mask"].sum()) < 600 else (0.38 if int(c["mask"].sum()) >= 1600 else 0.425)))),
        ("Hybrid 0.7*vb+0.3*vs >= 0.46", lambda c: (c["box"], (0.7 * c["vb"] + 0.3 * c["vs"]) >= 0.46)),
        ("Hybrid 0.7*vb+0.3*vs >= 0.475", lambda c: (c["box"], (0.7 * c["vb"] + 0.3 * c["vs"]) >= 0.475)),
    ]:
        rr = {s: paint_cands_to_rles(by_stem[s], floor=0.32, mask_fn=fn) for s in eh.FOLD0_STEMS}
        cc = eh.score_rle_strs_dict(rr)
        mm = eh.score_from_counts(cc)
        bt = eh.paired_date_bootstrap(cc, base_counts)
        print(f"  {name:42s}: PQ={mm['pq']:.4f} SQ={mm['sq']:.4f} RQ={mm['rq']:.4f} TP={mm['tp']} FP={mm['fp']} | d={bt['mean_gain']:+.4f} P={bt['win_rate']:.3f}")

    # 4. EXP-004: Collinear Spine Fragment Re-Stitching
    print("\n=== EXP-004: COLLINEAR FRAGMENT STITCHING ===")
    for gap in (8, 12, 18):
        rr = {s: paint_cands_to_rles(by_stem[s], floor=0.32, stitch_gap=gap) for s in eh.FOLD0_STEMS}
        cc = eh.score_rle_strs_dict(rr)
        mm = eh.score_from_counts(cc)
        bt = eh.paired_date_bootstrap(cc, base_counts)
        print(f"  stitch_gap={gap:2d}px                          : PQ={mm['pq']:.4f} SQ={mm['sq']:.4f} RQ={mm['rq']:.4f} TP={mm['tp']} FP={mm['fp']} | d={bt['mean_gain']:+.4f} P={bt['win_rate']:.3f}")

    # 5. EXP-005: Strict 5-Fold Date-Grouped Nested CV Meta-Rescoring (HGBR + Ridge on y_iou & y_tp!)
    print("\n=== EXP-005: NESTED 5-FOLD DATE-CV META-RESCORING (ALL 13 STREAMS + R_SUN + MORPH) ===")
    X_by = {s: np.stack([c["feat"] for c in by_stem[s]]) if by_stem[s] else np.zeros((0, 48), np.float32) for s in eh.FOLD0_STEMS}
    Y_iou = {s: np.array([c["y_iou"] for c in by_stem[s]], np.float32) for s in eh.FOLD0_STEMS}
    Y_tp = {s: np.array([c["y_tp"] for c in by_stem[s]], np.float32) for s in eh.FOLD0_STEMS}

    oof_pred_iou = {s: np.zeros(len(by_stem[s]), np.float32) for s in eh.FOLD0_STEMS}
    oof_pred_tp = {s: np.zeros(len(by_stem[s]), np.float32) for s in eh.FOLD0_STEMS}
    oof_pred_ridge = {s: np.zeros(len(by_stem[s]), np.float32) for s in eh.FOLD0_STEMS}

    for k in range(5):
        tr = [s for s in eh.FOLD0_STEMS if nested_folds[s] != k]
        va = [s for s in eh.FOLD0_STEMS if nested_folds[s] == k]
        X_tr = np.concatenate([X_by[s] for s in tr])
        y_tr_i = np.concatenate([Y_iou[s] for s in tr])
        y_tr_t = np.concatenate([Y_tp[s] for s in tr])

        m_iou = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.04, max_leaf_nodes=12, min_samples_leaf=35, l2_regularization=2.0, random_state=42)
        m_iou.fit(X_tr, y_tr_i)
        m_tp = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.04, max_leaf_nodes=12, min_samples_leaf=35, l2_regularization=2.0, random_state=42)
        m_tp.fit(X_tr, y_tr_t)

        # Standardized linear baseline on selected non-leaky stats
        mu_x = X_tr.mean(0)
        sd_x = X_tr.std(0) + 1e-5
        m_rg = Ridge(alpha=50.0).fit((X_tr - mu_x) / sd_x, y_tr_i)

        for s in va:
            if len(by_stem[s]):
                oof_pred_iou[s] = m_iou.predict(X_by[s])
                oof_pred_tp[s] = m_tp.predict(X_by[s])
                oof_pred_ridge[s] = m_rg.predict((X_by[s] - mu_x) / sd_x)

    # Evaluate STRICTLY on Nested OOF predictions! Also select floor inside Nested Date-CV!
    for label, preds in [("HGBR(y_iou)", oof_pred_iou), ("HGBR(y_tp)", oof_pred_tp), ("Ridge(y_iou)", oof_pred_ridge)]:
        for alpha in (0.0, 0.3, 0.5, 0.7, 1.0):
            r_nested = {}
            fls = []
            # Blend normalized score s_blend = alpha * pred + (1-alpha) * c_e19
            sc_map = {s: (alpha * preds[s] + (1.0 - alpha) * np.array([c["c_e19"] for c in by_stem[s]], np.float32)) for s in eh.FOLD0_STEMS}
            grid = np.linspace(float(np.percentile(np.concatenate(list(sc_map.values())), 20)),
                               float(np.percentile(np.concatenate(list(sc_map.values())), 70)), 13)
            for k in range(5):
                tr = [s for s in eh.FOLD0_STEMS if nested_folds[s] != k]
                va = [s for s in eh.FOLD0_STEMS if nested_folds[s] == k]
                bfl, bpq = float(grid[0]), -1.0
                for fl in grid:
                    rc = eh.score_rle_strs_dict({s: paint_cands_to_rles(by_stem[s], score_fn=(lambda i, c, s=s: sc_map[s][i]), floor=float(fl)) for s in tr}, stems=tr)
                    pq = eh.score_from_counts(rc, stems=tr)["pq"]
                    if pq > bpq:
                        bpq, bfl = pq, float(fl)
                fls.append(round(bfl, 3))
                for s in va:
                    r_nested[s] = paint_cands_to_rles(by_stem[s], score_fn=(lambda i, c, s=s: sc_map[s][i]), floor=bfl)
            cc = eh.score_rle_strs_dict(r_nested)
            mm = eh.score_from_counts(cc)
            bt = eh.paired_date_bootstrap(cc, base_counts)
            print(f"  {label:12s} a={alpha:.1f} (nested CV): PQ={mm['pq']:.4f} SQ={mm['sq']:.4f} RQ={mm['rq']:.4f} TP={mm['tp']} FP={mm['fp']} | vsE19={bt['mean_gain']:+.4f} CI=[{bt['ci95'][0]:+.4f},{bt['ci95'][1]:+.4f}] P={bt['win_rate']:.3f} fls={fls}")


if __name__ == "__main__":
    main()
