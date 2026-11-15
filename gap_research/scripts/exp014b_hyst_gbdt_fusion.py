"""EXP-014b (Local Wave-2 Hysteresis + GBDT/ViT + MWIS Fusion):
Combines EXP-014 Continuous 2D Hysteresis masks (h74_l57, h72_l55) with EXP-011b GBDT + ViT
utility estimation and strict 5-fold nested Date-CV MWIS Antichain DP + Observatory calibration.

Strictly leak-free:
  - Uses ONLY clean competition-only EXP-006/EXP-007 + work/cache streams.
  - Fits 5-fold GBDT (LightGBM + XGBoost + CatBoost, 2 seeds) strictly out-of-fold.
  - Evaluates all policy & observatory offsets under strict 5-fold nested Date-CV + Sim_r80 + Bootstrap.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "gap_research" / "scripts"))

import build_super_submission as bss  # noqa: E402
import exp011b_extract_candidates as e11x  # noqa: E402
import exp011b_hurdle_antichain as e11b  # noqa: E402
import exp011c_transformer_rescorer as e11c  # noqa: E402
import exp014_continuous_hysteresis as e14  # noqa: E402
from filseg import data, metric, submission  # noqa: E402

FULL = e11x.FULL
HALF = e11x.HALF
KIND_HYST = dict(e11x.KIND_MAP, hyst74=3.0, hyst72=2.0)  # reuse GBDT ordinal neighborhood cleanly


def _build_full_pmap_u8(clusters_s: list[dict]) -> np.ndarray:
    pm = np.zeros((FULL, FULL), dtype=np.uint8)
    for cl in clusters_s:
        x1, y1, x2, y2 = cl["box"]
        pu = cl["prob_u8"]
        pm[y1:y2, x1:x2] = np.maximum(pm[y1:y2, x1:x2], pu)
    return pm


def _hyst_mask_features(
    stem: str,
    box: tuple[int, int, int, int],
    msk: np.ndarray,
    area: int,
    full_pmap_u8: np.ndarray,
    phys: dict,
    ta_items: list[tuple[tuple[int, int, int, int], np.ndarray, float]],
    base_X15: np.ndarray,
    kind_name: str,
    n_cands_img: float,
    sum_c_img: float,
    rank_c: float,
    ov_cnt_i: float,
    ov_max_i: float,
) -> np.ndarray:
    x1, y1, x2, y2 = box
    bw, bh = float(x2 - x1), float(y2 - y1)
    pcrop = full_pmap_u8[y1:y2, x1:x2].astype(np.float32) / 255.0
    pv = pcrop[msk]
    if not len(pv):
        pv = np.array([0.5], dtype=np.float32)
    pq = np.percentile(pv, [10, 25, 50, 75, 90])
    ero = cv2.erode(msk.astype(np.uint8), e11x._KERNEL_3) > 0
    rim = msk & ~ero
    p_core = float(pcrop[ero].mean()) if ero.any() else float(pv.mean())
    p_rim = float(pcrop[rim].mean()) if rim.any() else float(pv.mean())

    core76 = (pcrop > 0.76) & msk
    n_cc76_raw, _, st76, _ = cv2.connectedComponentsWithStats(core76.astype(np.uint8), connectivity=8)
    n_cc_core76 = float(sum(1 for k_id in range(1, n_cc76_raw) if st76[k_id, cv2.CC_STAT_AREA] >= 40))

    m_u8 = msk.astype(np.uint8)
    n_cc_m = float(cv2.connectedComponents(m_u8, connectivity=8)[0] - 1)
    cnts, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        pts = np.vstack(cnts)
        (_, (rw, rh), _) = cv2.minAreaRect(pts)
        rot_len = float(max(rw, rh, 1.0))
        rot_wid = float(max(1.0, min(rw, rh)))
        hull_area = float(max(float(area), cv2.contourArea(cv2.convexHull(pts))))
        perim = float(sum(cv2.arcLength(ct, True) for ct in cnts))
    else:
        rot_len, rot_wid, hull_area, perim = max(bw, bh), min(bw, bh), float(area), 2.0 * (bw + bh)
    solidity = float(area) / max(1.0, hull_area)
    compact = (perim * perim) / max(1.0, 4.0 * np.pi * float(area))
    mean_width = (2.0 * float(area)) / max(1.0, perim)

    hx1, hy1 = max(0, x1 // 2), max(0, y1 // 2)
    hx2, hy2 = min(HALF, max(hx1 + 1, (x2 + 1) // 2)), min(HALF, max(hy1 + 1, (y2 + 1) // 2))
    mh = cv2.resize(m_u8, (hx2 - hx1, hy2 - hy1), interpolation=cv2.INTER_NEAREST) > 0
    if not mh.any():
        mh[0, 0] = True
    dil_h = cv2.dilate(mh.astype(np.uint8), e11x._KERNEL_7) > 0
    skirt_h = dil_h & ~mh

    fz = phys["flat_z"][hy1:hy2, hx1:hx2]
    bz = phys["bth_z"][hy1:hy2, hx1:hx2]
    rz = phys["ridge"][hy1:hy2, hx1:hx2]
    f_m, b_m, r_m = fz[mh], bz[mh], rz[mh]
    if skirt_h.any():
        f_s, b_s = fz[skirt_h], bz[skirt_h]
        q_loc = float(np.median(f_s))
        bright_ring_z = float(np.percentile(f_s, 90))
        bth_cs_diff = float(b_m.mean() - b_s.mean())
        ring_std = float(f_s.std())
    else:
        q_loc, bright_ring_z, bth_cs_diff, ring_std = 0.0, 0.0, 0.0, 0.0

    cx_p, cy_p = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    rho_c = float(np.hypot(cx_p - phys["cx"], cy_p - phys["cy"]) / max(1.0, phys["R"]))
    rho_max = float(max(np.hypot(cx - phys["cx"], cy - phys["cy"]) / max(1.0, phys["R"]) for cx in (x1, x2) for cy in (y1, y2)))

    iou_ta, c_ta, near_d = 0.0, 0.0, 999.0
    for tb, tm, tc in ta_items:
        d = float(np.hypot(max(0, max(x1, tb[0]) - min(x2, tb[2])), max(0, max(y1, tb[1]) - min(y2, tb[3]))))
        if d < near_d:
            near_d = d
        if d == 0.0:
            tiou, _ = e11x._crop_iou(box, msk, tb, tm)
            if tiou > iou_ta:
                iou_ta, c_ta = tiou, tc

    obs_code = e11x.OBS_MAP.get(stem[-2:], -1.0)
    c_tr = float(base_X15[0])
    tail = np.array([
        float(pv.mean()), float(pq[0]), float(pq[1]), float(pq[2]), float(pq[3]), float(pq[4]), float(pv.std()),
        p_core, p_rim, p_core - p_rim,
        float((pv > 0.65).mean()), float((pv > 0.75).mean()), float((pv > 0.85).mean()), n_cc_core76,
        float(area), float(np.log1p(area)), bw, bh, max(bw, bh), rot_len, rot_wid,
        rot_len / max(1.0, rot_wid), float(area) / max(1.0, bw * bh),
        solidity, compact, mean_width, n_cc_m, rho_c, rho_max,
        float(q_loc - np.percentile(f_m, 25)), float(q_loc - np.percentile(f_m, 10)),
        float(f_m.mean()), float(np.percentile(f_m, 10)),
        float(b_m.mean()), float(np.percentile(b_m, 10)), bth_cs_diff, bright_ring_z,
        float((f_m < (q_loc - 1.5)).mean()), float(r_m.mean()), float(np.percentile(r_m, 75)),
        ring_std, phys["mad"], obs_code,
        float(KIND_HYST[kind_name]), 0.0, 0.0,
        float(iou_ta), float(c_tr - c_ta) if iou_ta >= 0.35 else c_tr,
        n_cands_img, sum_c_img, rank_c, ov_cnt_i, ov_max_i, near_d,
    ], dtype=np.float32)
    out = np.concatenate([base_X15.astype(np.float32), tail], axis=0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _augment_stem_worker(task: tuple[str, dict, list[dict], list]) -> tuple[str, dict, np.ndarray]:
    cv2.setNumThreads(0)
    stem, ent_in, clusters_s, stem_gt = task
    ent = dict(ent_in)
    orig_n = len(ent["rles"])

    if not clusters_s:
        parent_map = np.arange(orig_n, dtype=np.int32)
        return stem, ent, parent_map

    # Decode original crop masks cleanly for deduplication and edge construction
    orig_boxes = [tuple(int(v) for v in b) for b in ent["boxes"]]
    orig_masks = []
    for rstr, (x1, y1, x2, y2) in zip(ent["rles"], orig_boxes):
        fm = mu.decode({"size": [FULL, FULL], "counts": rstr.encode("ascii")}).astype(bool)
        orig_masks.append(fm[y1:y2, x1:x2])
    ta_items = [
        (orig_boxes[i], orig_masks[i], float(ent["X"][i, 0]))
        for i, k in enumerate(ent["kinds"])
        if k in ("ta_base", "ta_rescue")
    ]

    full_pmap_u8 = _build_full_pmap_u8(clusters_s)
    added = []
    all_boxes = list(orig_boxes)
    all_masks = list(orig_masks)

    for cl in clusters_s:
        sc = float(cl["score"])
        if sc < 0.22:
            continue
        is_rescue = bool(cl.get("is_rescue_pool"))
        specs = (("hyst74", 174, 145),) if is_rescue else (("hyst74", 189, 145), ("hyst72", 184, 140))
        for kname, t_hi, t_lo in specs:
            m = e14._hysteresis_mask(cl["prob_u8"], t_hi, t_lo)
            if m is None:
                continue
            ma_raw = int(m.sum())
            if ma_raw < 95 or (is_rescue and ma_raw > 420):
                continue
            trimmed = e11x._trim_cand(cl["box"], m, min_area=90)
            if trimmed is None:
                continue
            tb, tm = trimmed
            best_orig_j, best_orig_iou, dup_any = -1, 0.0, False
            for j, (ob, om) in enumerate(zip(all_boxes, all_masks)):
                if min(tb[2], ob[2]) <= max(tb[0], ob[0]) or min(tb[3], ob[3]) <= max(tb[1], ob[1]):
                    continue
                iou, _ = e11x._crop_iou(tb, tm, ob, om)
                if iou >= 0.94:
                    dup_any = True
                    break
                if j < orig_n and iou > best_orig_iou:
                    best_orig_iou, best_orig_j = iou, j
            if dup_any:
                continue
            rstr = bss.encode_crop_ascii(tb, tm)
            if best_orig_j >= 0 and best_orig_iou >= 0.25:
                base15 = ent["X"][best_orig_j, :15].copy()
                parent_j = best_orig_j if best_orig_iou >= 0.60 else -1
            else:
                base15 = np.zeros(15, dtype=np.float32)
                base15[0] = sc
                base15[1] = 0.25
                base15[2] = float(cl.get("n_streams", 1))
                base15[6 if is_rescue else 3] = sc
                base15[7] = sc
                base15[8] = sc
                base15[10] = 1.0 if is_rescue else 0.0
                parent_j = -1
            added.append((tb, tm, int(tm.sum()), rstr, kname, base15, parent_j))
            all_boxes.append(tb)
            all_masks.append(tm)

    parent_map = np.array([*range(orig_n), *[a[6] for a in added]], dtype=np.int32)
    if not added:
        return stem, ent, parent_map

    phys = e11x._compute_img_physics(stem)
    n_tot = len(all_boxes)
    new_edges = list(ent["edges"])
    orig_ov_cnt = ent["X"][:, 66].copy() if orig_n else np.zeros(0, dtype=np.float32)
    orig_ov_max = ent["X"][:, 67].copy() if orig_n else np.zeros(0, dtype=np.float32)
    ov_cnt = np.zeros(n_tot, dtype=np.float32)
    ov_max = np.zeros(n_tot, dtype=np.float32)
    if orig_n:
        ov_cnt[:orig_n] = orig_ov_cnt
        ov_max[:orig_n] = orig_ov_max
    all_areas = list(ent["areas"]) + [a[2] for a in added]

    for i_new in range(orig_n, n_tot):
        bi, mi = all_boxes[i_new], all_masks[i_new]
        ai = int(all_areas[i_new])
        for j in range(i_new):
            bj, mj = all_boxes[j], all_masks[j]
            if min(bi[2], bj[2]) <= max(bi[0], bj[0]) or min(bi[3], bj[3]) <= max(bi[1], bj[1]):
                continue
            iou, inter = e11x._crop_iou(bi, mi, bj, mj)
            if inter > 0:
                if iou > 0.05:
                    ov_cnt[i_new] += 1.0
                    ov_cnt[j] += 1.0
                    ov_max[i_new] = max(ov_max[i_new], iou)
                    ov_max[j] = max(ov_max[j], iou)
                ifrac = inter / float(max(1, min(ai, int(all_areas[j]))))
                new_edges.append((j, i_new, float(ifrac), float(iou)))

    c_all = np.concatenate([ent["X"][:, 0] if orig_n else np.zeros(0, np.float32), np.array([a[5][0] for a in added], np.float32)])
    n_cands_f = float(n_tot)
    sum_c_f = float(c_all.sum())
    ranks = np.argsort(np.argsort(-c_all)).astype(np.float32)

    X_new_rows = []
    for k_i, (tb, tm, area_i, rstr, kname, base15, _) in enumerate(added):
        idx_g = orig_n + k_i
        X_new_rows.append(
            _hyst_mask_features(
                stem, tb, tm, area_i, full_pmap_u8, phys, ta_items, base15, kname,
                n_cands_f, sum_c_f, float(ranks[idx_g]), float(ov_cnt[idx_g]), float(ov_max[idx_g]),
            )
        )
    X_add = np.stack(X_new_rows, axis=0)
    X_all = np.vstack([ent["X"], X_add]) if orig_n else X_add
    X_all[:, 63] = n_cands_f
    X_all[:, 64] = sum_c_f
    X_all[:, 65] = ranks
    X_all[:, 66] = ov_cnt
    X_all[:, 67] = ov_max

    m_add = len(added)
    h_add = np.zeros(m_add, dtype=np.float32)
    m_add_arr = np.zeros(m_add, dtype=np.float32)
    u_add = np.zeros(m_add, dtype=np.float32)
    if stem_gt:
        pred_rles = [{"size": [FULL, FULL], "counts": a[3].encode("ascii")} for a in added]
        hit_cnt = np.zeros(m_add, dtype=np.float32)
        iou_sum = np.zeros(m_add, dtype=np.float32)
        for gr in stem_gt:
            if not gr:
                continue
            mat = mu.iou(pred_rles, gr, [0] * len(gr))
            best = mat.max(axis=1)
            hit = best > 0.5
            hit_cnt += hit.astype(np.float32)
            iou_sum += np.where(hit, best, 0.0).astype(np.float32)
        rc = float(len(stem_gt))
        h_add = hit_cnt / rc
        m_add_arr = np.where(hit_cnt > 0, iou_sum / np.maximum(hit_cnt, 1.0), 0.0).astype(np.float32)
        u_add = iou_sum / rc

    ent["rles"] = list(ent["rles"]) + [a[3] for a in added]
    ent["kinds"] = list(ent["kinds"]) + [a[4] for a in added]
    ent["boxes"] = np.array(all_boxes, dtype=np.int32)
    ent["areas"] = np.array(all_areas, dtype=np.int32)
    ent["X"] = X_all
    ent["h"] = np.concatenate([ent["h"], h_add]) if orig_n else h_add
    ent["m"] = np.concatenate([ent["m"], m_add_arr]) if orig_n else m_add_arr
    ent["u"] = np.concatenate([ent["u"], u_add]) if orig_n else u_add
    ent["edges"] = new_edges
    return stem, ent, parent_map


def _build_hyst_augmented_pool(
    base_pool: dict, gt_recs: dict, workers: int, out_cache: Path
) -> tuple[dict, dict, dict, dict]:
    if out_cache.exists():
        print(f"Loading cached augmented pool: {out_cache}", flush=True)
        with open(out_cache, "rb") as f:
            d = pickle.load(f)
        return d["oof"], d["test"], d["pmap_oof"], d["pmap_test"]

    oof_stems = sorted(base_pool["oof"])
    test_stems = sorted(base_pool["test"])
    t0 = time.time()
    cl_oof = e14.build_mode_clusters(oof_stems, "oof", e14.MODES["M0_cont_tracka"], workers=workers)
    cl_test = e14.build_mode_clusters(test_stems, "test", e14.MODES["M0_cont_tracka"], workers=workers)
    print(f"Built M0 continuous clusters for OOF({len(cl_oof)}) + Test({len(cl_test)}) in {time.time()-t0:.1f}s", flush=True)

    aug_oof, pmap_oof = {}, {}
    oof_tasks = [(s, base_pool["oof"][s], cl_oof[s], gt_recs.get(s, [])) for s in oof_stems]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for s, ent, pm in ex.map(_augment_stem_worker, oof_tasks, chunksize=8):
            aug_oof[s] = ent
            pmap_oof[s] = pm

    aug_test, pmap_test = {}, {}
    test_tasks = [(s, base_pool["test"][s], cl_test[s], []) for s in test_stems]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for s, ent, pm in ex.map(_augment_stem_worker, test_tasks, chunksize=8):
            aug_test[s] = ent
            pmap_test[s] = pm

    n0 = sum(len(v["rles"]) for v in base_pool["oof"].values())
    n1 = sum(len(v["rles"]) for v in aug_oof.values())
    nt0 = sum(len(v["rles"]) for v in base_pool["test"].values())
    nt1 = sum(len(v["rles"]) for v in aug_test.values())
    print(f"Augmented pool in {time.time()-t0:.1f}s | OOF cands: {n0} -> {n1} (+{n1-n0} hyst) | Test cands: {nt0} -> {nt1} (+{nt1-nt0})", flush=True)

    with open(out_cache, "wb") as f:
        pickle.dump({"oof": aug_oof, "test": aug_test, "pmap_oof": pmap_oof, "pmap_test": pmap_test}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return aug_oof, aug_test, pmap_oof, pmap_test


def _compute_weights_obs(
    entry: dict,
    u_score: np.ndarray,
    params: tuple[float, float, float, float, bool],
    obs_delta: dict[str, float] | None = None,
) -> np.ndarray:
    tau_keep, tau_add, anchor_bonus, tau_base, allow_solo = params
    delta = obs_delta.get(entry["obs"], 0.0) if obs_delta else 0.0
    tb = tau_base + delta
    ta = tau_add + delta
    tk = max(0.02, tau_keep + 0.5 * delta)
    return e11c._node_weights_policy(entry, u_score, (tk, ta, anchor_bonus, tb, allow_solo))


def _nested_eval_policy(
    label: str,
    pool_oof: dict,
    adj_map: dict,
    stems: list[str],
    gt_recs: dict,
    cand_u_by_alpha: dict[float, dict[str, np.ndarray]],
    grid: list[tuple[float, float, float, float, bool]],
    obs_deltas: tuple[dict[str, float], ...] | None = None,
) -> dict:
    # Precompute approximate stem tuple stats per (alpha, params, obs_delta_id) ONCE for instant 5-fold selection!
    od_list = obs_deltas if obs_deltas is not None else ({},)
    cache_stats: dict[tuple, dict[str, tuple[float, float, float]]] = {}
    for alp, u_map in cand_u_by_alpha.items():
        for p in grid:
            for od_i, od in enumerate(od_list):
                per_stem = {}
                for s in stems:
                    ent = pool_oof[s]
                    if not len(ent["u"]):
                        per_stem[s] = (0.0, 0.0, 0.0)
                        continue
                    w = _compute_weights_obs(ent, u_map[s], p, od)
                    sel = e11b._solve_mwis_exact(w, adj_map[s])
                    rc = float(ent["r_count"])
                    tp_s = fp_s = iou_s = 0.0
                    for idx in sel:
                        tp_s += rc * float(ent["h"][idx])
                        fp_s += rc * (1.0 - float(ent["h"][idx]))
                        iou_s += rc * float(ent["u"][idx])
                    per_stem[s] = (tp_s, fp_s, iou_s)
                cache_stats[(alp, p, od_i)] = per_stem

    # Exact total GT across all annotators matching r_count scaling
    gt_tot_stem = {s: float(sum(len(gr) for gr in gt_recs[s])) for s in stems}
    rstr_oof, rdict_oof, chosen = {}, {}, []
    exact_stem_cache: dict[tuple, dict[str, dict]] = {}
    for fk in range(5):
        tr = [s for s in stems if pool_oof[s]["fold"] != fk]
        va = [s for s in stems if pool_oof[s]["fold"] == fk]
        tg_tr = sum(gt_tot_stem[s] for s in tr)
        ranked = []
        for key, st_dict in cache_stats.items():
            tp_a = sum(st_dict[s][0] for s in tr)
            fp_a = sum(st_dict[s][1] for s in tr)
            iou_a = sum(st_dict[s][2] for s in tr)
            den = tp_a + 0.5 * (fp_a + (tg_tr - tp_a))
            sc = iou_a / den if den > 0 else 0.0
            ranked.append((sc, key))
        ranked.sort(reverse=True, key=lambda x: x[0])

        # Refine top-8 soft candidates by exact filseg.metric PQ strictly on inner training folds
        best_ex_pq, best_key = -1.0, ranked[0][1]
        for _, key in ranked[:8]:
            alp_k, p_k, od_i_k = key
            if key not in exact_stem_cache:
                u_k = cand_u_by_alpha[alp_k]
                od_k = od_list[od_i_k]
                ex_rows = {}
                for s in stems:
                    w = _compute_weights_obs(pool_oof[s], u_k[s], p_k, od_k)
                    sel = e11b._solve_mwis_exact(w, adj_map[s])
                    rstrs = e11b._decode_rles_disjoint(pool_oof[s], sel, w)
                    ex_rows[s] = metric.counts([{"size": [FULL, FULL], "counts": r.encode("ascii")} for r in rstrs], gt_recs[s])
                exact_stem_cache[key] = ex_rows
            ex_pq = metric.summarize([exact_stem_cache[key][s] for s in tr])["pq"]
            if ex_pq > best_ex_pq:
                best_ex_pq, best_key = ex_pq, key

        alp_c, p_c, od_i = best_key
        chosen.append((alp_c, p_c, od_list[od_i]))
        u_map = cand_u_by_alpha[alp_c]
        for s in va:
            w = _compute_weights_obs(pool_oof[s], u_map[s], p_c, od_list[od_i])
            sel = e11b._solve_mwis_exact(w, adj_map[s])
            rstrs = e11b._decode_rles_disjoint(pool_oof[s], sel, w)
            rstr_oof[s] = rstrs
            rdict_oof[s] = [{"size": [FULL, FULL], "counts": r.encode("ascii")} for r in rstrs]

    rows = {s: metric.counts(rdict_oof[s], gt_recs[s]) for s in stems}
    sm = metric.summarize(list(rows.values()))
    sim_m, sim_s = e11b._sim_r80(stems, rstr_oof, gt_recs, trials=80)
    print(
        f"[{label:22s}] PQ={sm['pq']:.5f} SQ={sm['sq']:.5f} RQ={sm['rq']:.5f} "
        f"TP={sm['tp']} FP={sm['fp']} FN={sm['fn']} Sim80={sim_m:.5f}+/-{sim_s:.5f}",
        flush=True,
    )
    return {"label": label, "sm": sm, "rows": rows, "rstrs": rstr_oof, "sim_m": sim_m, "sim_s": sim_s, "chosen": chosen}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--base-pool", type=Path, default=Path("/private/tmp/exp011b_pool_clamped.pkl"))
    ap.add_argument("--p11c", type=Path, default=Path("/private/tmp/exp011c_branch_preds.pkl"))
    ap.add_argument("--aug-pool", type=Path, default=Path("/private/tmp/exp014b_pool_hyst.pkl"))
    ap.add_argument("--export", type=Path, default=REPO / "gap_research" / "submissions" / "exp014b_hyst_gbdt_clean.csv")
    args = ap.parse_args()

    t0 = time.time()
    folds = data.folds()
    all_stems = sorted(folds)
    gt_recs = metric.load_records(all_stems)

    with open(args.base_pool, "rb") as f:
        base_pool = pickle.load(f)
    with open(args.p11c, "rb") as f:
        p11c = pickle.load(f)

    # 1. Exact Track-A Clean reference
    adj_base = {s: e11b._build_adj(base_pool["oof"][s]) for s in all_stems}

    # 2. Check fixed-tg GBDT vs 0.75*GBDT+0.25*ViT on original base pool first
    grid_base = [
        (tk, ta, ab, tb, solo_ok)
        for tk in (0.04, 0.08, 0.12)
        for ta in (0.25, 0.28, 0.31)
        for ab in (0.02, 0.05, 0.09)
        for tb in (0.23, 0.25)
        for solo_ok in (False, True)
    ]

    def _make_blend_u(wg: float, alp: float, p_oof: dict, map_oof: dict | None = None, g_pred: dict | None = None):
        u_out = {}
        for s in all_stems:
            n_s = len(p_oof[s]["rles"])
            if g_pred is not None:
                gh, gm, gu = g_pred["h"][s], g_pred["m"][s], g_pred["u"][s]
            else:
                gh, gm, gu = p11c["oof"]["gbdt"]["oof_h"][s], p11c["oof"]["gbdt"]["oof_m"][s], p11c["oof"]["gbdt"]["oof_u"][s]
            vh_o, vm_o, vu_o = p11c["oof"]["vit"]["oof_h"][s], p11c["oof"]["vit"]["oof_m"][s], p11c["oof"]["vit"]["oof_u"][s]
            if map_oof is not None:
                pm = map_oof[s]
                valid = pm >= 0
                vh = np.where(valid, vh_o[np.maximum(0, pm)], gh)
                vm = np.where(valid, vm_o[np.maximum(0, pm)], gm)
                vu = np.where(valid, vu_o[np.maximum(0, pm)], gu)
            else:
                vh, vm, vu = vh_o, vm_o, vu_o
            up = wg * (gh * gm) + (1.0 - wg) * (vh * vm)
            ud = wg * gu + (1.0 - wg) * vu
            u_out[s] = (alp * up + (1.0 - alp) * ud).astype(np.float32) if n_s else np.zeros(0, dtype=np.float32)
        return u_out

    res_gbdt_orig = _nested_eval_policy(
        "A1_GBDT_Base_FixTG", base_pool["oof"], adj_base, all_stems, gt_recs,
        {alp: _make_blend_u(1.0, alp, base_pool["oof"]) for alp in (0.60, 0.80, 1.0)},
        grid_base,
    )
    res_b75_orig = _nested_eval_policy(
        "A2_B75_ViT_Base", base_pool["oof"], adj_base, all_stems, gt_recs,
        {alp: _make_blend_u(0.75, alp, base_pool["oof"]) for alp in (0.60, 0.80, 1.0)},
        grid_base,
    )

    # 3. Build EXP-014 continuous hysteresis-augmented candidate pool & fit OOF GBDT cleanly
    aug_oof, aug_test, pmap_oof, pmap_test = _build_hyst_augmented_pool(base_pool, gt_recs, args.workers, args.aug_pool)
    test_stems = sorted(aug_test)
    adj_aug = {s: e11b._build_adj(aug_oof[s]) for s in all_stems}
    adj_aug_test = {s: e11b._build_adj(aug_test[s]) for s in test_stems}

    gh_oof, gm_oof, gu_oof = {}, {}, {}
    test_h_list, test_m_list, test_u_list = [], [], []
    print("Training 5-fold GBDT on hysteresis-augmented pool...", flush=True)
    tg0 = time.time()
    for fk in range(5):
        tr_s = [s for s in all_stems if folds[s] != fk and len(aug_oof[s]["X"]) > 0]
        va_s = [s for s in all_stems if folds[s] == fk]
        X_tr = np.vstack([aug_oof[s]["X"] for s in tr_s])
        y_h = np.concatenate([aug_oof[s]["h"] for s in tr_s])
        y_m = np.concatenate([aug_oof[s]["m"] for s in tr_s])
        y_u = np.concatenate([aug_oof[s]["u"] for s in tr_s])
        w_r = np.concatenate([np.full(len(aug_oof[s]["h"]), max(1, aug_oof[s]["r_count"]), dtype=np.float32) for s in tr_s])
        mod = e11c._fit_fold_gbdt_fast(X_tr, y_h, y_m, y_u, w_r, seeds=(0, 42))
        vp = e11c._predict_gbdt_batch(mod, aug_oof, va_s)
        for s in va_s:
            gh_oof[s], gm_oof[s], gu_oof[s] = vp[s]
        tp = e11c._predict_gbdt_batch(mod, aug_test, test_stems)
        test_h_list.append({s: tp[s][0] for s in test_stems})
        test_m_list.append({s: tp[s][1] for s in test_stems})
        test_u_list.append({s: tp[s][2] for s in test_stems})
    print(f"5-Fold GBDT finished in {time.time()-tg0:.1f}s", flush=True)

    g_pred_aug = {"h": gh_oof, "m": gm_oof, "u": gu_oof}
    res_hyst_gbdt = _nested_eval_policy(
        "B1_Hyst_GBDT", aug_oof, adj_aug, all_stems, gt_recs,
        {alp: _make_blend_u(1.0, alp, aug_oof, pmap_oof, g_pred_aug) for alp in (0.60, 0.80, 1.0)},
        grid_base,
    )
    res_hyst_b75 = _nested_eval_policy(
        "B2_Hyst_B75_ViT", aug_oof, adj_aug, all_stems, gt_recs,
        {alp: _make_blend_u(0.75, alp, aug_oof, pmap_oof, g_pred_aug) for alp in (0.60, 0.80, 1.0)},
        grid_base,
    )
    obs_deltas = (
        {},
        {"Mh": -0.02, "Lh": -0.01},
        {"Mh": -0.03, "Lh": -0.015},
        {"Mh": -0.02, "Lh": -0.01, "Ch": 0.01},
    )
    res_hyst_obs = _nested_eval_policy(
        "B3_Hyst_B75_ObsCal", aug_oof, adj_aug, all_stems, gt_recs,
        {alp: _make_blend_u(0.75, alp, aug_oof, pmap_oof, g_pred_aug) for alp in (0.60, 0.80, 1.0)},
        grid_base, obs_deltas=obs_deltas,
    )

    candidates_res = [res_gbdt_orig, res_b75_orig, res_hyst_gbdt, res_hyst_b75, res_hyst_obs]
    # Select winner strictly on nested Sim80 & OOF PQ gain
    winner = max(candidates_res, key=lambda r: (r["sim_m"], r["sm"]["pq"]))
    boot_vs_a1 = e11b._paired_date_bootstrap(winner["rows"], res_gbdt_orig["rows"], all_stems, n_boot=1000, seed=42)
    print(
        f"\n[EXP-014b Winner: {winner['label']}] PQ={winner['sm']['pq']:.5f} Sim80={winner['sim_m']:.5f}+/-{winner['sim_s']:.5f} | "
        f"Boot vs A1_GBDT: mean={boot_vs_a1['mean_gain']:+.5f}, p={boot_vs_a1['p_win']:.4f}, CI95={boot_vs_a1['ci95']}",
        flush=True,
    )

    # Save diagnostic summary
    summary = {
        r["label"]: {
            "sm": r["sm"],
            "sim80_mean": r["sim_m"],
            "sim80_std": r["sim_s"],
            "chosen": r["chosen"],
        }
        for r in candidates_res
    }
    summary["winner_boot_vs_A1"] = boot_vs_a1
    (REPO / "gap_research" / "scripts" / "exp014b_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Export test CSV strictly for winning configuration (recomputing majority/mode test policy cleanly)
    if winner["label"].startswith("B"):
        te_gh = {s: np.mean([d[s] for d in test_h_list], axis=0) for s in test_stems}
        te_gm = {s: np.mean([d[s] for d in test_m_list], axis=0) for s in test_stems}
        te_gu = {s: np.mean([d[s] for d in test_u_list], axis=0) for s in test_stems}
        wg = 1.0 if winner["label"] == "B1_Hyst_GBDT" else 0.75
        pool_te, adj_te, pm_te = aug_test, adj_aug_test, pmap_test
    else:
        te_gh = {s: p11c["test"]["gbdt"][s][0] for s in test_stems}
        te_gm = {s: p11c["test"]["gbdt"][s][1] for s in test_stems}
        te_gu = {s: p11c["test"]["gbdt"][s][2] for s in test_stems}
        wg = 1.0 if winner["label"] == "A1_GBDT_Base_FixTG" else 0.75
        pool_te, adj_te, pm_te = base_pool["test"], {s: e11b._build_adj(base_pool["test"][s]) for s in test_stems}, None

    # Use median/most-common fold parameters cleanly across test set
    alps = sorted(c[0] for c in winner["chosen"])
    alp_te = alps[len(alps) // 2]
    # Use majority vote on (params, obs_delta)
    from collections import Counter
    p_te, od_te = Counter((c[1], json.dumps(c[2], sort_keys=True)) for c in winner["chosen"]).most_common(1)[0][0]
    od_te_d = json.loads(od_te)

    # Check exact vit test structure
    vit_t_raw = p11c["test"]["vit"]
    rles_te = {}
    for ti, s in enumerate(test_stems):
        gh, gm, gu = te_gh[s], te_gm[s], te_gu[s]
        vh_o, vm_o, vu_o = vit_t_raw[s]
        if pm_te is not None:
            pm = pm_te[s]
            valid = pm >= 0
            vh = np.where(valid, vh_o[np.maximum(0, pm)], gh)
            vm = np.where(valid, vm_o[np.maximum(0, pm)], gm)
            vu = np.where(valid, vu_o[np.maximum(0, pm)], gu)
        else:
            vh, vm, vu = vh_o, vm_o, vu_o
        up = wg * (gh * gm) + (1.0 - wg) * (vh * vm)
        ud = wg * gu + (1.0 - wg) * vu
        u_te = (alp_te * up + (1.0 - alp_te) * ud).astype(np.float32) if len(gh) else np.zeros(0, np.float32)
        w = _compute_weights_obs(pool_te[s], u_te, p_te, od_te_d)
        sel = e11b._solve_mwis_exact(w, adj_te[s])
        rstrs = e11b._decode_rles_disjoint(pool_te[s], sel, w)
        rles_te[s] = [{"size": [FULL, FULL], "counts": r.encode("ascii")} for r in rstrs]

    args.export.parent.mkdir(parents=True, exist_ok=True)
    submission.write_csv(args.export, rles_te)
    probs = submission.validate(args.export, test_stems)
    print(f"Exported {args.export} | Validate={probs} | Elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
