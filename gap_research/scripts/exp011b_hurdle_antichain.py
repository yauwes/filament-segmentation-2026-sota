"""EXP-011b Stage 2: Hurdle Expected-PQ Rescorer + Track-A Anchored Soft-Trim MWIS Antichain.

Trained strictly out-of-fold across all 707 OOF stems (date-grouped 5-fold CV) using LightGBM
(cross_entropy + regression), XGBoost (reg:logistic + squarederror), and CatBoost (CrossEntropy + RMSE).
Zero HistGBR (avoids macOS libomp spinlock). Uses Nested 5-Fold CV to tune:
  - u_blend alpha: mix of calibrated hurdle (h_hat * m_hat) and direct u_hat
  - tau_keep: prune low-utility Track-A FPs
  - tau_add: threshold for disjoint rescue proposals
  - anchor_bonus: margin required for cons62/cons74/split/merge variants to replace Track-A anchors
  - tau_base: Dinkelbach PQ cutoff (~0.23-0.25)

Evaluates exact filseg.metric 707-OOF PQ, Fold-0 PQ, area bins, Sim_r80 annotator subsampling,
paired N=1000 date-cluster bootstrap vs Track-A Clean, and exports + validates Test CSV.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMRegressor
from pycocotools import mask as mu
from xgboost import XGBRegressor

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "gap_research" / "scripts"))

import eval_wave23_joint as ej  # noqa: E402
import test_shift_audit as tsa  # noqa: E402
from filseg import data, metric, submission  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
import build_super_submission as bss  # noqa: E402

FULL = 2048
_TAKEN_BUF = np.zeros((FULL, FULL), dtype=bool)


def _solve_mwis_exact(weights: np.ndarray, adj: list[list[int]]) -> list[int]:
    """Exact MWIS on active positive nodes per connected component (fallback greedy if comp > 22)."""
    active = [i for i, w in enumerate(weights) if w > 0.0]
    if not active:
        return []
    act_set = set(active)
    visited = set()
    chosen: list[int] = []

    for start in active:
        if start in visited:
            continue
        comp = []
        stack = [start]
        visited.add(start)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v in act_set and v not in visited:
                    visited.add(v)
                    stack.append(v)

        if len(comp) == 1:
            chosen.append(comp[0])
            continue

        comp.sort(key=lambda idx: -float(weights[idx]))
        if len(comp) > 22:
            blocked = set()
            for u in comp:
                if u not in blocked:
                    chosen.append(u)
                    blocked.update(adj[u])
            continue

        pos = {u: idx for idx, u in enumerate(comp)}
        k_n = len(comp)
        comp_w = [float(weights[u]) for u in comp]
        comp_mask = [0] * k_n
        for idx, u in enumerate(comp):
            msk = 0
            for v in adj[u]:
                if v in pos:
                    msk |= 1 << pos[v]
            comp_mask[idx] = msk

        suf = [0.0] * (k_n + 1)
        for i in range(k_n - 1, -1, -1):
            suf[i] = suf[i + 1] + comp_w[i]

        best_val = 0.0
        best_pick = 0

        def _dfs(idx: int, cur_val: float, forb_mask: int, pick_mask: int) -> None:
            nonlocal best_val, best_pick
            if cur_val + suf[idx] <= best_val + 1e-12:
                return
            if idx == k_n:
                if cur_val > best_val:
                    best_val = cur_val
                    best_pick = pick_mask
                return
            if not (forb_mask & (1 << idx)):
                _dfs(idx + 1, cur_val + comp_w[idx], forb_mask | comp_mask[idx], pick_mask | (1 << idx))
            _dfs(idx + 1, cur_val, forb_mask, pick_mask)

        _dfs(0, 0.0, 0, 0)
        for idx, u in enumerate(comp):
            if best_pick & (1 << idx):
                chosen.append(u)
    return chosen


def _build_adj(entry: dict, hard_inter_frac: float = 0.12, hard_iou: float = 0.08) -> list[list[int]]:
    n = len(entry["rles"])
    adj = [[] for _ in range(n)]
    for e in entry["edges"]:
        i, j = int(e[0]), int(e[1])
        if len(e) >= 4:
            ifrac, iou_ij = float(e[2]), float(e[3])
            if ifrac < hard_inter_frac and iou_ij < hard_iou:
                continue
        adj[i].append(j)
        adj[j].append(i)
    return adj


def _fit_fold_models(
    X_tr: np.ndarray,
    y_h: np.ndarray,
    y_m: np.ndarray,
    y_u: np.ndarray,
    w_r: np.ndarray,
    seeds: tuple[int, ...] = (0, 42),
) -> dict:
    pos = y_h > 0
    X_pos = X_tr[pos]
    y_m_pos = y_m[pos]
    w_m_pos = (y_h[pos] * w_r[pos]).astype(np.float32)

    mh_list = []
    mm_list = []
    mu_list = []
    for sd in seeds:
        mh_lgb = LGBMRegressor(
            objective="cross_entropy",
            n_estimators=420,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=1.0,
            reg_lambda=3.0,
            n_jobs=4,
            random_state=sd,
            verbose=-1,
        )
        mh_lgb.fit(X_tr, y_h, sample_weight=w_r)

        mh_xgb = XGBRegressor(
            objective="reg:logistic",
            n_estimators=420,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=15,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=1.0,
            reg_lambda=3.0,
            tree_method="hist",
            n_jobs=4,
            random_state=sd,
        )
        mh_xgb.fit(X_tr, y_h, sample_weight=w_r, verbose=False)

        mh_cat = CatBoostClassifier(
            loss_function="CrossEntropy",
            iterations=550,
            learning_rate=0.035,
            depth=6,
            l2_leaf_reg=5.0,
            thread_count=4,
            random_seed=sd,
            verbose=0,
        )
        mh_cat.fit(X_tr, y_h, sample_weight=w_r)
        mh_list.extend([("lgb", mh_lgb), ("xgb", mh_xgb), ("cat", mh_cat)])

        mm_lgb = LGBMRegressor(
            objective="regression",
            n_estimators=360,
            learning_rate=0.03,
            num_leaves=23,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.5,
            reg_lambda=3.0,
            n_jobs=4,
            random_state=sd + 100,
            verbose=-1,
        )
        mm_lgb.fit(X_pos, y_m_pos, sample_weight=w_m_pos)

        mm_xgb = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=360,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=12,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=0.5,
            reg_lambda=3.0,
            tree_method="hist",
            n_jobs=4,
            random_state=sd + 100,
        )
        mm_xgb.fit(X_pos, y_m_pos, sample_weight=w_m_pos, verbose=False)

        mm_cat = CatBoostRegressor(
            loss_function="RMSE",
            iterations=480,
            learning_rate=0.035,
            depth=5,
            l2_leaf_reg=5.0,
            thread_count=4,
            random_seed=sd + 100,
            verbose=0,
        )
        mm_cat.fit(X_pos, y_m_pos, sample_weight=w_m_pos)
        mm_list.extend([("lgb", mm_lgb), ("xgb", mm_xgb), ("cat", mm_cat)])

        mu_lgb = LGBMRegressor(
            objective="regression",
            n_estimators=420,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=1.0,
            reg_lambda=3.0,
            n_jobs=4,
            random_state=sd + 200,
            verbose=-1,
        )
        mu_lgb.fit(X_tr, y_u, sample_weight=w_r)

        mu_xgb = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=420,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=15,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=1.0,
            reg_lambda=3.0,
            tree_method="hist",
            n_jobs=4,
            random_state=sd + 200,
        )
        mu_xgb.fit(X_tr, y_u, sample_weight=w_r, verbose=False)
        mu_list.extend([("lgb", mu_lgb), ("xgb", mu_xgb)])

    return {"mh": mh_list, "mm": mm_list, "mu": mu_list}


def _predict_models(models: dict, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(X) == 0:
        z = np.zeros(0, dtype=np.float32)
        return z, z, z
    ph = np.mean(
        [np.clip(m.predict_proba(X)[:, 1] if k == "cat" else m.predict(X), 0.0, 1.0) for k, m in models["mh"]],
        axis=0,
    )
    pm = np.mean([np.clip(m.predict(X), 0.50, 1.0) for _, m in models["mm"]], axis=0)
    pu_dir = np.mean([np.clip(m.predict(X), 0.0, 1.0) for _, m in models["mu"]], axis=0)
    return ph.astype(np.float32), pm.astype(np.float32), pu_dir.astype(np.float32)


def _compute_node_weights(entry: dict, u_score: np.ndarray, params: tuple[float, float, float, float]) -> np.ndarray:
    """Apply (tau_keep, tau_add, anchor_bonus, tau_base) policy weights for MWIS solver."""
    tau_keep, tau_add, anchor_bonus, tau_base = params
    kinds = entry["kinds"]
    n = len(kinds)
    w = np.full(n, -1.0, dtype=np.float32)
    is_anchor = np.array([k in ("ta_base", "ta_rescue") for k in kinds], dtype=bool)
    iou_ta = entry["X"][:, 61] if len(entry["X"]) else np.zeros(0, dtype=np.float32)  # iou_with_ta index=61

    for i in range(n):
        ui = float(u_score[i])
        if is_anchor[i]:
            if ui >= tau_keep:
                val = (ui - tau_base) + anchor_bonus
                if tau_keep <= 0.0:
                    val = max(1e-4 + 1e-3 * ui, val)
                w[i] = val
        elif iou_ta[i] >= 0.25:
            # Variant overlapping Track-A anchor (cons62/cons74/split/merge)
            if anchor_bonus < 5.0 and ui >= tau_base:
                w[i] = ui - tau_base
        else:
            # Strictly new disjoint rescue (cons68/cons62/solo)
            if ui >= tau_add:
                w[i] = max(1e-4 + 1e-3 * ui, ui - tau_base)
    return w


def _decode_rles_disjoint(entry: dict, picked: list[int], weights: np.ndarray) -> list[str]:
    if not picked:
        return []
    if len(picked) == 1:
        return [entry["rles"][picked[0]]]
    picked_sorted = sorted(picked, key=lambda idx: -float(weights[idx]))
    # Fast check: if boxes of selected items are pairwise disjoint, skip mask painting completely
    boxes = entry["boxes"]
    rles = entry["rles"]
    has_box_ov = False
    for a in range(len(picked_sorted)):
        ia = picked_sorted[a]
        ba = boxes[ia]
        for b in range(a + 1, len(picked_sorted)):
            ib = picked_sorted[b]
            bb = boxes[ib]
            if min(ba[2], bb[2]) > max(ba[0], bb[0]) and min(ba[3], bb[3]) > max(ba[1], bb[1]):
                has_box_ov = True
                break
        if has_box_ov:
            break
    if not has_box_ov:
        return [rles[i] for i in picked_sorted]

    _TAKEN_BUF.fill(False)
    out = []
    areas = entry["areas"]
    for idx in picked_sorted:
        rstr = rles[idx]
        m = mu.decode({"size": [FULL, FULL], "counts": rstr.encode("ascii")}).astype(bool)
        x1, y1, x2, y2 = [int(v) for v in boxes[idx]]
        sub = m[y1:y2, x1:x2]
        keep = sub & ~_TAKEN_BUF[y1:y2, x1:x2]
        ka = int(keep.sum())
        if ka >= 90 and ka >= 0.90 * int(areas[idx]):
            _TAKEN_BUF[y1:y2, x1:x2] |= keep
            if ka == int(sub.sum()):
                out.append(rstr)
            else:
                ys, xs = np.nonzero(keep)
                rx1, ry1, rx2, ry2 = x1 + int(xs.min()), y1 + int(ys.min()), x1 + int(xs.max()) + 1, y1 + int(ys.max()) + 1
                out.append(bss.encode_crop_ascii((rx1, ry1, rx2, ry2), keep[ry1 - y1 : ry2 - y1, rx1 - x1 : rx2 - x1]))
    return out


def _fast_approx_pq(pool_oof: dict, adj_map: dict, stems: list[str], u_map: dict[str, np.ndarray], params: tuple[float, float, float, float], total_gt: float) -> tuple[float, int]:
    tp_a, fp_a, iou_a, n_pred = 0.0, 0.0, 0.0, 0
    for s in stems:
        ent = pool_oof[s]
        if len(ent["u"]) == 0:
            continue
        w = _compute_node_weights(ent, u_map[s], params)
        sel = _solve_mwis_exact(w, adj_map[s])
        rc = float(ent["r_count"])
        for idx in sel:
            tp_a += rc * float(ent["h"][idx])
            fp_a += rc * (1.0 - float(ent["h"][idx]))
            iou_a += rc * float(ent["u"][idx])
            n_pred += 1
    fn_a = total_gt - tp_a
    den = tp_a + 0.5 * (fp_a + fn_a)
    return (iou_a / den if den > 0 else 0.0), n_pred


def _eval_exact_stems(
    pool_oof: dict,
    adj_map: dict,
    stems: list[str],
    gt_recs: dict,
    u_map: dict[str, np.ndarray],
    params_by_stem: dict[str, tuple[float, float, float, float]],
) -> tuple[dict, dict[str, dict], dict[str, list[str]]]:
    rows = {}
    rles_out = {}
    for s in stems:
        ent = pool_oof[s]
        w = _compute_node_weights(ent, u_map[s], params_by_stem[s])
        sel = _solve_mwis_exact(w, adj_map[s])
        rstrs = _decode_rles_disjoint(ent, sel, w)
        rles_out[s] = rstrs
        pr = [{"size": [FULL, FULL], "counts": r.encode("ascii")} for r in rstrs]
        rows[s] = metric.counts(pr, gt_recs[s])
    return metric.summarize(list(rows.values())), rows, rles_out


def _paired_date_bootstrap(rows_new: dict[str, dict], rows_base: dict[str, dict], stems: list[str], n_boot: int = 1000, seed: int = 0) -> dict:
    by_date: dict[str, list[str]] = {}
    for s in stems:
        by_date.setdefault(s[:8], []).append(s)
    dates = sorted(by_date)
    rng = np.random.default_rng(seed)
    gains = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.choice(len(dates), size=len(dates), replace=True)
        rn, rb = [], []
        for idx in pick:
            for s in by_date[dates[idx]]:
                rn.append(rows_new[s])
                rb.append(rows_base[s])
        gains[b] = metric.summarize(rn)["pq"] - metric.summarize(rb)["pq"]
    return {
        "mean_gain": float(gains.mean()),
        "ci95": [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))],
        "p_win": float((gains > 0).mean()),
    }


def _sim_r80(stems: list[str], rles_map: dict[str, list[str]], gt_recs: dict, trials: int = 10) -> tuple[float, float]:
    rng = np.random.default_rng(123)
    vals = []
    for _ in range(trials):
        rows = []
        for s in stems:
            rc = len(gt_recs[s])
            pick_rec = gt_recs[s] if rc <= 1 else [gt_recs[s][rng.integers(0, rc)]]
            pr = [{"size": [FULL, FULL], "counts": r.encode("ascii")} for r in rles_map[s]]
            rows.append(metric.counts(pr, pick_rec))
        vals.append(metric.summarize(rows)["pq"])
    return float(np.mean(vals)), float(np.std(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=Path("/private/tmp/exp011b_pool.pkl"))
    ap.add_argument("--out-csv", type=Path, default=REPO / "gap_research" / "submissions" / "exp011b_tracka_hurdle_clean.csv")
    args = ap.parse_args()

    t0 = time.time()
    pool = pickle.load(open(args.pool, "rb"))
    oof = pool["oof"]
    test = pool["test"]
    fnames = pool["feature_names"]
    iou_ta_idx = fnames.index("iou_with_ta")
    assert iou_ta_idx == 61, f"Expected iou_with_ta at idx 61, got {iou_ta_idx}"

    folds = data.folds()
    all_stems = sorted(oof)
    f0_stems = [s for s in all_stems if folds[s] == 0]
    gt_recs = metric.load_records(all_stems)

    adj_oof = {s: _build_adj(oof[s]) for s in all_stems}
    adj_test = {s: _build_adj(test[s]) for s in sorted(test)}

    # Check exact Track-A sanity directly from candidate pool (`ta_base` + `ta_rescue`)
    ta_u_dummy = {s: np.array([1.0 if k in ("ta_base", "ta_rescue") else 0.0 for k in oof[s]["kinds"]], dtype=np.float32) for s in all_stems}
    ta_params = {s: (0.5, 9.9, 9.9, 0.24) for s in all_stems}
    sm_ta_pool, rows_ta_pool, rles_ta_pool = _eval_exact_stems(oof, adj_oof, all_stems, gt_recs, ta_u_dummy, ta_params)
    print(f"[Sanity Track-A inside pool]: PQ={sm_ta_pool['pq']:.5f} SQ={sm_ta_pool['sq']:.5f} RQ={sm_ta_pool['rq']:.5f} TP={sm_ta_pool['tp']} FP={sm_ta_pool['fp']} FN={sm_ta_pool['fn']} ({time.time()-t0:.1f}s)", flush=True)

    ta_cache = pickle.load(open("/private/tmp/tracka_clean_oof_cache.pkl", "rb"))
    print(f"[Track-A Clean Baseline Cache]: PQ={ta_cache['sm_all']['pq']:.5f} | Fold-0={ta_cache['sm_f0']['pq']:.5f}", flush=True)

    # 1. Strictly Date-Grouped 5-Fold OOF Hurdle + Direct Utility Modeling
    pred_h_oof, pred_m_oof, pred_u_dir_oof = {}, {}, {}
    fold_models = {}
    for fk in range(5):
        tr_s = [s for s in all_stems if folds[s] != fk and len(oof[s]["X"]) > 0]
        va_s = [s for s in all_stems if folds[s] == fk]
        X_tr = np.vstack([oof[s]["X"] for s in tr_s])
        y_h = np.concatenate([oof[s]["h"] for s in tr_s])
        y_m = np.concatenate([oof[s]["m"] for s in tr_s])
        y_u = np.concatenate([oof[s]["u"] for s in tr_s])
        w_r = np.concatenate([np.full(len(oof[s]["X"]), float(oof[s]["r_count"]), dtype=np.float32) for s in tr_s])

        md = _fit_fold_models(X_tr, y_h, y_m, y_u, w_r, seeds=(0, 42))
        fold_models[fk] = md
        for s in va_s:
            ph, pm, pud = _predict_models(md, oof[s]["X"])
            pred_h_oof[s], pred_m_oof[s], pred_u_dir_oof[s] = ph, pm, pud
        print(f"  Fitted Fold {fk} ({len(X_tr)} train cands) elapsed={time.time()-t0:.1f}s", flush=True)

    # Check calibration on OOF
    all_ht = np.concatenate([oof[s]["h"] for s in all_stems if len(oof[s]["X"])])
    all_hp = np.concatenate([pred_h_oof[s] for s in all_stems if len(oof[s]["X"])])
    all_ut = np.concatenate([oof[s]["u"] for s in all_stems if len(oof[s]["X"])])
    all_uh = np.concatenate([pred_h_oof[s] * pred_m_oof[s] for s in all_stems if len(oof[s]["X"])])
    all_ud = np.concatenate([pred_u_dir_oof[s] for s in all_stems if len(oof[s]["X"])])
    print(f"OOF Calibration: true_h={all_ht.mean():.4f} pred_h={all_hp.mean():.4f} | corr(u_hurdle)={np.corrcoef(all_ut, all_uh)[0,1]:.4f} corr(u_direct)={np.corrcoef(all_ut, all_ud)[0,1]:.4f}", flush=True)

    # Feature importance summary from Fold 0 lgb
    lgb_h0 = fold_models[0]["mh"][0][1]
    top_f = sorted(zip(lgb_h0.feature_importances_, fnames), reverse=True)[:15]
    print("Top-15 h_model features:", [(f, int(imp)) for imp, f in top_f], flush=True)

    # 2. Strictly leak-free Nested 5-Fold CV hyperparameter search (including alpha inside inner folds!)
    alphas = (0.50, 0.70, 0.90)
    grid = [
        (tk, ta_v, ab, tb)
        for tk in (0.00, 0.08, 0.12, 0.15, 0.18, 0.21)
        for ta_v in (0.24, 0.27, 0.30, 0.33)
        for ab in (0.02, 0.05, 0.09, 9.9)
        for tb in (0.24,)
        if tk <= tb
    ]
    total_gt_by_fold = {
        fk: sum(float(sum(len(gr) for gr in gt_recs[s])) for s in all_stems if folds[s] == fk)
        for fk in range(5)
    }
    u_oof_by_alp = {
        alp: {s: alp * (pred_h_oof[s] * pred_m_oof[s]) + (1.0 - alp) * pred_u_dir_oof[s] for s in all_stems}
        for alp in alphas
    }

    u_nested = {}
    nested_params_by_stem = {}
    fold_chosen = []
    for fk in range(5):
        inner_stems = [s for s in all_stems if folds[s] != fk]
        inner_gt = sum(v for k, v in total_gt_by_fold.items() if k != fk)
        best_pq_in, best_alp, best_p = -1.0, alphas[1], grid[0]
        for alp in alphas:
            umap = u_oof_by_alp[alp]
            for p in grid:
                pq_in, _ = _fast_approx_pq(oof, adj_oof, inner_stems, umap, p, inner_gt)
                if pq_in > best_pq_in:
                    best_pq_in, best_alp, best_p = pq_in, alp, p
        fold_chosen.append((best_alp, best_p))
        for s in all_stems:
            if folds[s] == fk:
                u_nested[s] = u_oof_by_alp[best_alp][s]
                nested_params_by_stem[s] = best_p
    print("Nested 5-Fold selected (alpha, params):", fold_chosen, f"(elapsed={time.time()-t0:.1f}s)", flush=True)

    best_alpha = float(np.median([x[0] for x in fold_chosen]))
    u_oof = u_oof_by_alp[best_alpha] if best_alpha in u_oof_by_alp else {s: best_alpha * (pred_h_oof[s] * pred_m_oof[s]) + (1.0 - best_alpha) * pred_u_dir_oof[s] for s in all_stems}

    sm_nested, rows_nested, rles_nested = _eval_exact_stems(oof, adj_oof, all_stems, gt_recs, u_nested, nested_params_by_stem)
    sm_n_f0 = metric.summarize([rows_nested[s] for s in f0_stems])
    boot = _paired_date_bootstrap(rows_nested, ta_cache["counts"], all_stems, n_boot=1000, seed=42)
    sim80_mean, sim80_std = _sim_r80(all_stems, rles_nested, gt_recs, trials=10)
    bins_nested = ej.bin_pq_summary(rles_nested, all_stems, gt_recs)
    n_preds_oof = sum(len(rles_nested[s]) for s in all_stems)

    print(
        f"\n[NESTED 5-FOLD EXACT OOF]: PQ={sm_nested['pq']:.5f} SQ={sm_nested['sq']:.5f} RQ={sm_nested['rq']:.5f} "
        f"TP={sm_nested['tp']} FP={sm_nested['fp']} FN={sm_nested['fn']} P={n_preds_oof}",
        flush=True,
    )
    print(f"[NESTED FOLD-0]:          PQ={sm_n_f0['pq']:.5f} SQ={sm_n_f0['sq']:.5f} RQ={sm_n_f0['rq']:.5f} TP={sm_n_f0['tp']} FP={sm_n_f0['fp']} FN={sm_n_f0['fn']}", flush=True)
    print(f"[PAIRED BOOT vs Track-A]: delta={boot['mean_gain']:+.5f} CI95={boot['ci95']} P_win={boot['p_win']:.4f}", flush=True)
    print(f"[Sim_r80_PQ (10x 1-ann)]: {sim80_mean:.5f} +/- {sim80_std:.5f}", flush=True)
    print("Area Bins:", bins_nested, flush=True)

    # Also inspect global best fixed mode & "lock anchors (ab=9.9)" ablation cleanly
    for label, subset_ab in [("Lock-TA + Prune/Add", (9.9,)), ("Allow-Variants", (0.02, 0.05, 0.09))]:
        sub_grid = [p for p in grid if p[2] in subset_ab]
        best_p, best_v = None, -1.0
        for p in sub_grid:
            v, _ = _fast_approx_pq(oof, adj_oof, all_stems, u_oof, p, sum(total_gt_by_fold.values()))
            if v > best_v:
                best_v, best_p = v, p
        sm_g, rows_g, rles_g = _eval_exact_stems(oof, adj_oof, all_stems, gt_recs, u_oof, {s: best_p for s in all_stems})
        boot_g = _paired_date_bootstrap(rows_g, ta_cache["counts"], all_stems, n_boot=1000, seed=42)
        print(f"[Global Ablation {label:20s} {best_p}]: PQ={sm_g['pq']:.5f} TP={sm_g['tp']} FP={sm_g['fp']} FN={sm_g['fn']} boot={boot_g['mean_gain']:+.5f} P={boot_g['p_win']:.4f}", flush=True)

    # Consensus params for Test prediction (median of fold-chosen parameters)
    cons_p = (
        float(np.median([x[1][0] for x in fold_chosen])),
        float(np.median([x[1][1] for x in fold_chosen])),
        float(np.median([x[1][2] for x in fold_chosen])),
        float(np.median([x[1][3] for x in fold_chosen])),
    )
    print(f"Consensus Test params (alpha={best_alpha}): {cons_p}", flush=True)

    test_stems = sorted(test)
    pred_h_test, pred_m_test, pred_u_dir_test = {}, {}, {}
    rles_test_csv = {}
    for s in test_stems:
        Xt = test[s]["X"]
        if len(Xt) == 0:
            z = np.zeros(0, dtype=np.float32)
            pred_h_test[s], pred_m_test[s], pred_u_dir_test[s] = z, z, z
            rles_test_csv[s] = []
            continue
        phs, pms, puds = zip(*[_predict_models(fold_models[fk], Xt) for fk in range(5)])
        ph_t = np.mean(phs, axis=0).astype(np.float32)
        pm_t = np.mean(pms, axis=0).astype(np.float32)
        pu_t = np.mean(puds, axis=0).astype(np.float32)
        pred_h_test[s], pred_m_test[s], pred_u_dir_test[s] = ph_t, pm_t, pu_t
        u_t = best_alpha * (ph_t * pm_t) + (1.0 - best_alpha) * pu_t
        wt = _compute_node_weights(test[s], u_t, cons_p)
        sel = _solve_mwis_exact(wt, adj_test[s])
        rstrs = _decode_rles_disjoint(test[s], sel, wt)
        rles_test_csv[s] = [{"size": [FULL, FULL], "counts": r.encode("ascii")} for r in rstrs]

    with open("/private/tmp/exp011b_preds.pkl", "wb") as f:
        pickle.dump(
            {
                "pred_h_oof": pred_h_oof,
                "pred_m_oof": pred_m_oof,
                "pred_u_dir_oof": pred_u_dir_oof,
                "pred_h_test": pred_h_test,
                "pred_m_test": pred_m_test,
                "pred_u_dir_test": pred_u_dir_test,
                "fold_chosen": fold_chosen,
                "best_alpha": best_alpha,
                "cons_p": cons_p,
                "rles_nested": rles_nested,
                "rows_nested": rows_nested,
                "sm_nested": sm_nested,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.write_csv(args.out_csv, rles_test_csv)
    probs = submission.validate(args.out_csv, data.test_stems())
    assert probs == [], f"Validation failed: {probs}"
    audit = tsa.audit_csv(args.out_csv, REPO / "gap_research" / "submissions" / "exp006_007_tracka_clean.csv")
    print(f"Saved {args.out_csv} (validate=[] OK), Test shift audit:\n{json.dumps(audit, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
