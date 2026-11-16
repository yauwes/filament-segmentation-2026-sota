#!/usr/bin/env python3
"""EXP-016 Full Wave-5 5-Family Fusion (LGBM + LGBM-MiT + CatBoost + XGBoost + ViT + PurePhysViT + NovelSpec).

Strictly 5-fold date-grouped OOF + Nested 5-Fold Date-CV across both Raw PQ and Sparse-PQ (r=0.76).
Runs CatBoost+XGBoost in an isolated child worker so macOS OpenMP runtimes never collide with LightGBM.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faulthandler

faulthandler.enable()

import torch  # noqa: F401, E402

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "gap_research" / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from filseg import data, metric, submission  # noqa: E402
import exp011b_hurdle_antichain as e11b  # noqa: E402
import exp011c_transformer_rescorer as e11c  # noqa: E402
import exp015_wave5_fusion as e15  # noqa: E402
import mit_features as e15e  # noqa: E402
import test_shift_audit as tsa  # noqa: E402


def _logit(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-4, 1.0 - 1e-4)
    return np.log(pc / (1.0 - pc))


def _sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _pq_sparse(sm: dict, r: float = 0.76) -> float:
    tp = float(sm["tp"])
    fp = float(sm["fp"])
    fn = float(sm["fn"])
    si = float(sm.get("iou", sm["sq"] * tp))
    den = r * tp + 0.5 * fp + 0.5 * (1.0 - r) * tp + 0.5 * r * fn
    return (r * si / den) if den > 0 else 0.0


def _augment_97(pool: dict, mit_cache: Path) -> tuple[dict, dict, list[str], np.ndarray]:
    oof, test = pool["oof"], pool["test"]
    stems, test_stems = sorted(oof), sorted(test)
    base_names = list(pool["feature_names"])
    all_names = base_names + list(e15e.MIT_NAMES)
    nov_idx = np.array([i for i, n in enumerate(all_names) if n not in e15e.PROVENANCE_COLS], dtype=np.int64)
    mit_oof, mit_test = e15e.get_mit_feats(pool, mit_cache)
    oof_aug = {
        s: {**oof[s], "X": np.hstack([oof[s]["X"], mit_oof[s]]).astype(np.float32) if len(oof[s]["X"]) else np.zeros((0, len(all_names)), dtype=np.float32)}
        for s in stems
    }
    test_aug = {
        s: {**test[s], "X": np.hstack([test[s]["X"], mit_test[s]]).astype(np.float32) if len(test[s]["X"]) else np.zeros((0, len(all_names)), dtype=np.float32)}
        for s in test_stems
    }
    return oof_aug, test_aug, all_names, nov_idx


def _cbxgb_worker(pool_path: Path, mit_cache: Path, out_path: Path) -> None:
    from catboost import CatBoostClassifier, CatBoostRegressor
    from xgboost import XGBRegressor

    t0 = time.time()
    pool = pickle.load(open(pool_path, "rb"))
    oof_aug, test_aug, _, _ = _augment_97(pool, mit_cache)
    folds = data.folds()
    stems, test_stems = sorted(oof_aug), sorted(test_aug)
    res = {k: {"oof_h": {}, "oof_m": {}, "oof_u": {}, "test_h": [], "test_m": [], "test_u": []} for k in ("cb_mit", "xgb_mit")}

    for fk in range(5):
        tr_s = [s for s in stems if folds[s] != fk and len(oof_aug[s]["X"]) > 0]
        va_s = [s for s in stems if folds[s] == fk]
        X_tr = np.vstack([oof_aug[s]["X"] for s in tr_s])
        y_h = np.concatenate([oof_aug[s]["h"] for s in tr_s])
        y_m = np.concatenate([oof_aug[s]["m"] for s in tr_s])
        y_u = np.concatenate([oof_aug[s]["u"] for s in tr_s])
        w_r = np.concatenate([np.full(len(oof_aug[s]["X"]), float(oof_aug[s]["r_count"]), dtype=np.float32) for s in tr_s])
        pos = y_h > 0
        w_pos = w_r[pos] * y_h[pos]
        yh_c = np.clip(y_h, 0.0, 1.0)

        cb_models, xgb_models = [], []
        for sd in (0, 42):
            cb_models.append((
                CatBoostClassifier(loss_function="CrossEntropy", iterations=600, depth=6, learning_rate=0.04, l2_leaf_reg=6.0, random_seed=sd, verbose=0, thread_count=4).fit(X_tr, yh_c, sample_weight=w_r),
                CatBoostRegressor(loss_function="RMSE", iterations=500, depth=6, learning_rate=0.04, l2_leaf_reg=6.0, random_seed=sd + 1, verbose=0, thread_count=4).fit(X_tr[pos], y_m[pos], sample_weight=w_pos),
                CatBoostRegressor(loss_function="RMSE", iterations=600, depth=6, learning_rate=0.04, l2_leaf_reg=6.0, random_seed=sd + 2, verbose=0, thread_count=4).fit(X_tr, y_u, sample_weight=w_r),
            ))
            xgb_models.append((
                XGBRegressor(objective="reg:logistic", n_estimators=550, max_depth=5, learning_rate=0.035, subsample=0.80, colsample_bytree=0.75, min_child_weight=15, reg_lambda=5.0, random_state=sd, n_jobs=4, tree_method="hist").fit(X_tr, yh_c, sample_weight=w_r),
                XGBRegressor(objective="reg:squarederror", n_estimators=450, max_depth=5, learning_rate=0.035, subsample=0.80, colsample_bytree=0.75, min_child_weight=10, reg_lambda=5.0, random_state=sd + 1, n_jobs=4, tree_method="hist").fit(X_tr[pos], y_m[pos], sample_weight=w_pos),
                XGBRegressor(objective="reg:squarederror", n_estimators=550, max_depth=5, learning_rate=0.035, subsample=0.80, colsample_bytree=0.75, min_child_weight=15, reg_lambda=5.0, random_state=sd + 2, n_jobs=4, tree_method="hist").fit(X_tr, y_u, sample_weight=w_r),
            ))

        def _pred(models, split_d, ss):
            oh, om, ou = {}, {}, {}
            for s in ss:
                Xs = split_d[s]["X"]
                if not len(Xs):
                    z = np.zeros(0, dtype=np.float32)
                    oh[s], om[s], ou[s] = z, z, z
                    continue
                hs, ms, us = [], [], []
                for mh, mm, mu_m in models:
                    ph = mh.predict_proba(Xs)[:, 1] if hasattr(mh, "predict_proba") else mh.predict(Xs)
                    hs.append(np.clip(ph, 1e-4, 1.0 - 1e-4))
                    ms.append(np.clip(mm.predict(Xs), 0.0, 1.0))
                    us.append(np.clip(mu_m.predict(Xs), 0.0, 1.0))
                oh[s] = np.mean(hs, axis=0).astype(np.float32)
                om[s] = np.mean(ms, axis=0).astype(np.float32)
                ou[s] = np.mean(us, axis=0).astype(np.float32)
            return oh, om, ou

        for name, mdls in (("cb_mit", cb_models), ("xgb_mit", xgb_models)):
            vh, vm, vu = _pred(mdls, oof_aug, va_s)
            res[name]["oof_h"].update(vh); res[name]["oof_m"].update(vm); res[name]["oof_u"].update(vu)
            th, tm, tu = _pred(mdls, test_aug, test_stems)
            res[name]["test_h"].append(th); res[name]["test_m"].append(tm); res[name]["test_u"].append(tu)
        print(f"  [CB+XGB-MiT Fold {fk}] elapsed={time.time()-t0:.1f}s", flush=True)

    with open(out_path, "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved CB+XGB-MiT preds -> {out_path}", flush=True)


def _train_lgb_vit_branches(
    pool_path: Path,
    mit_cache: Path,
    preds_cache: Path,
    vit_epochs: int,
    pp_epochs: int,
    tab_only: bool = False,
    vit_seeds: int = 1,
    skip_pp_vit: bool = False,
) -> dict:
    e11c.KIND_LIST = e15.KIND_LIST_W5
    e11c.KIND_TO_IDX = {k: i for i, k in enumerate(e15.KIND_LIST_W5)}

    pool = pickle.load(open(pool_path, "rb"))
    oof, test = pool["oof"], pool["test"]
    folds = data.folds()
    stems, test_stems = sorted(oof), sorted(test)
    oof_aug, test_aug, all_names, nov_idx = _augment_97(pool, mit_cache)
    iou_ta_idx = all_names.index("iou_with_ta")

    oof_nov = {s: {**oof_aug[s], "X": oof_aug[s]["X"][:, nov_idx].copy() if len(oof_aug[s]["X"]) else np.zeros((0, len(nov_idx)), dtype=np.float32)} for s in stems}
    test_nov = {s: {**test_aug[s], "X": test_aug[s]["X"][:, nov_idx].copy() if len(test_aug[s]["X"]) else np.zeros((0, len(nov_idx)), dtype=np.float32)} for s in test_stems}

    preds = pickle.load(open(preds_cache, "rb")) if preds_cache.exists() else {}
    want = ["gbdt", "gbdt_mit", "novel_gbdt"]
    if not tab_only:
        want.append("vit_mit")
        if not skip_pp_vit:
            want.append("pp_vit")
    missing = [b for b in want if b not in preds or len(preds[b].get("oof_h", {})) != len(stems) or len(preds[b].get("test_h", [])) != 5]
    if not missing:
        print(f"[preds cache complete] {preds_cache} keys={list(preds.keys())}", flush=True)
        return preds

    print(f"[EXP-016 training branches] {missing} | vit_ep={vit_epochs} pp_ep={pp_epochs}", flush=True)
    for b in missing:
        preds[b] = {"oof_h": {}, "oof_m": {}, "oof_u": {}, "test_h": [], "test_m": [], "test_u": []}

    need_torch = any(b in missing for b in ("vit_mit", "pp_vit"))
    dev = e11c._get_device() if need_torch else None
    patches_oof = np.load(pool["patches_oof_path"], mmap_mode="r") if need_torch else None
    patches_test = np.load(pool["patches_test_path"], mmap_mode="r") if need_torch else None

    oof_pp, test_pp = {}, {}
    if "pp_vit" in missing:
        for src_d, dst_d, ks in ((oof_aug, oof_pp, stems), (test_aug, test_pp, test_stems)):
            for s in ks:
                Xp = src_d[s]["X"].copy()
                if len(Xp):
                    for idx_c, nm in enumerate(all_names):
                        if nm in e15e.PROVENANCE_COLS:
                            Xp[:, idx_c] = 0.0
                dst_d[s] = {**src_d[s], "X": Xp}

    for fk in range(5):
        tf0 = time.time()
        tr_s = [s for s in stems if folds[s] != fk and len(oof[s]["X"]) > 0]
        va_s = [s for s in stems if folds[s] == fk]

        if any(b in missing for b in ("gbdt", "gbdt_mit", "novel_gbdt")):
            X_tr_b = np.vstack([oof[s]["X"] for s in tr_s])
            X_tr_a = np.vstack([oof_aug[s]["X"] for s in tr_s])
            yh = np.concatenate([oof[s]["h"] for s in tr_s])
            ym = np.concatenate([oof[s]["m"] for s in tr_s])
            yu = np.concatenate([oof[s]["u"] for s in tr_s])
            wr = np.concatenate([np.full(len(oof[s]["X"]), float(oof[s]["r_count"]), dtype=np.float32) for s in tr_s])
            nov_mask = np.concatenate([
                (~np.array([k in ("ta_base", "ta_rescue") for k in oof[s]["kinds"]], dtype=bool))
                & (oof_aug[s]["X"][:, iou_ta_idx] < 0.25)
                for s in tr_s
            ])

            if "gbdt" in missing:
                m_gb = e11c._fit_fold_gbdt_fast(X_tr_b, yh, ym, yu, wr, seeds=(0, 42))
                va_gb = e11c._predict_gbdt_batch(m_gb, oof, va_s)
                te_gb = e11c._predict_gbdt_batch(m_gb, test, test_stems)
                for s in va_s:
                    preds["gbdt"]["oof_h"][s], preds["gbdt"]["oof_m"][s], preds["gbdt"]["oof_u"][s] = va_gb[s]
                for d_i, key in enumerate(("test_h", "test_m", "test_u")):
                    preds["gbdt"][key].append({s: te_gb[s][d_i] for s in test_stems})

            if "gbdt_mit" in missing:
                m_gm = e11c._fit_fold_gbdt_fast(X_tr_a, yh, ym, yu, wr, seeds=(0, 42))
                va_gm = e11c._predict_gbdt_batch(m_gm, oof_aug, va_s)
                te_gm = e11c._predict_gbdt_batch(m_gm, test_aug, test_stems)
                for s in va_s:
                    preds["gbdt_mit"]["oof_h"][s], preds["gbdt_mit"]["oof_m"][s], preds["gbdt_mit"]["oof_u"][s] = va_gm[s]
                for d_i, key in enumerate(("test_h", "test_m", "test_u")):
                    preds["gbdt_mit"][key].append({s: te_gm[s][d_i] for s in test_stems})

            if "novel_gbdt" in missing:
                X_tr_n = X_tr_a[nov_mask][:, nov_idx]
                m_nv = e11c._fit_fold_gbdt_fast(X_tr_n, yh[nov_mask], ym[nov_mask], yu[nov_mask], wr[nov_mask], seeds=(10, 52))
                va_nv = e11c._predict_gbdt_batch(m_nv, oof_nov, va_s)
                te_nv = e11c._predict_gbdt_batch(m_nv, test_nov, test_stems)
                for s in va_s:
                    preds["novel_gbdt"]["oof_h"][s], preds["novel_gbdt"]["oof_m"][s], preds["novel_gbdt"]["oof_u"][s] = va_nv[s]
                for d_i, key in enumerate(("test_h", "test_m", "test_u")):
                    preds["novel_gbdt"][key].append({s: te_nv[s][d_i] for s in test_stems})

        if "vit_mit" in missing:
            v_seeds = (42,) if vit_seeds <= 1 else (0, 42)
            vh, vm, vu, v_te = e11c._fit_predict_vit_fold(
                patches_oof, patches_test, oof_aug, test_aug, tr_s, va_s, test_stems, dev,
                epochs=vit_epochs, seeds=v_seeds,
            )
            for s in va_s:
                preds["vit_mit"]["oof_h"][s], preds["vit_mit"]["oof_m"][s], preds["vit_mit"]["oof_u"][s] = vh[s], vm[s], vu[s]
            for d_i, key in enumerate(("test_h", "test_m", "test_u")):
                preds["vit_mit"][key].append({s: v_te[s][d_i] for s in test_stems})

        if "pp_vit" in missing:
            ph, pm, pu, p_te = e11c._fit_predict_vit_fold(
                patches_oof, patches_test, oof_pp, test_pp, tr_s, va_s, test_stems, dev,
                epochs=pp_epochs, seeds=(10,),
            )
            for s in va_s:
                preds["pp_vit"]["oof_h"][s], preds["pp_vit"]["oof_m"][s], preds["pp_vit"]["oof_u"][s] = ph[s], pm[s], pu[s]
            for d_i, key in enumerate(("test_h", "test_m", "test_u")):
                preds["pp_vit"][key].append({s: p_te[s][d_i] for s in test_stems})

        print(f"  [EXP-016 Fold {fk}] done in {time.time()-tf0:.1f}s", flush=True)
        with open(preds_cache, "wb") as f:
            pickle.dump(preds, f, protocol=pickle.HIGHEST_PROTOCOL)

    return preds


def _get_hmu(preds: dict, br: str, s: str, is_oof: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if is_oof:
        return preds[br]["oof_h"][s], preds[br]["oof_m"][s], preds[br]["oof_u"][s]
    return (
        np.mean([d[s] for d in preds[br]["test_h"]], axis=0),
        np.mean([d[s] for d in preds[br]["test_m"]], axis=0),
        np.mean([d[s] for d in preds[br]["test_u"]], axis=0),
    )


def _blend_hmu(preds: dict, w_dict: list[tuple[str, float]], split_d: dict, ss: list[str], is_oof: bool, alp: float, iou_ta_idx: int, pp_w: float) -> dict[str, np.ndarray]:
    w_tot = sum(w for _, w in w_dict)
    out = {}
    for s in ss:
        ent = split_d[s]
        if not len(ent["X"]):
            out[s] = np.zeros(0, dtype=np.float32)
            continue
        lh = m = u = None
        for br, w in w_dict:
            ww = w / w_tot
            hb, mb, ub = _get_hmu(preds, br, s, is_oof)
            lh = _logit(hb) * ww if lh is None else (lh + _logit(hb) * ww)
            m = mb * ww if m is None else (m + mb * ww)
            u = ub * ww if u is None else (u + ub * ww)
        sc = alp * (_sig(lh) * m) + (1.0 - alp) * u
        if pp_w > 0.0 and "pp_vit" in preds and "novel_gbdt" in preds:
            is_ta = np.array([k in ("ta_base", "ta_rescue") for k in ent["kinds"]], dtype=bool)
            nov = (~is_ta) & (ent["X"][:, iou_ta_idx] < 0.25)
            if nov.any():
                nh, nm, nu = _get_hmu(preds, "novel_gbdt", s, is_oof)
                ph, pm, pu = _get_hmu(preds, "pp_vit", s, is_oof)
                hn = _sig((1.0 - pp_w) * _logit(nh) + pp_w * _logit(ph))
                mn = (1.0 - pp_w) * nm + pp_w * pm
                un = (1.0 - pp_w) * nu + pp_w * pu
                sc_n = alp * (hn * mn) + (1.0 - alp) * un
                sc[nov] = 0.50 * sc[nov] + 0.50 * sc_n[nov]
        out[s] = np.clip(sc, 0.0, 1.0).astype(np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--mit-cache", type=Path, default=Path("/private/tmp/exp016_mit.pkl"))
    ap.add_argument("--preds-cache", type=Path, default=Path("/private/tmp/exp016_preds.pkl"))
    ap.add_argument("--cbxgb-cache", type=Path, default=Path("/private/tmp/exp016_cbxgb.pkl"))
    ap.add_argument("--out-prefix", type=Path, default=REPO / "gap_research/submissions/exp016_wave5")
    ap.add_argument("--summary-json", type=Path, default=REPO / "gap_research/scripts/exp016_summary.json")
    ap.add_argument("--vit-epochs", type=int, default=8)
    ap.add_argument("--vit-seeds", type=int, default=1)
    ap.add_argument("--pp-epochs", type=int, default=8)
    ap.add_argument("--skip-pp-vit", action="store_true")
    ap.add_argument("--tab-only", action="store_true")
    ap.add_argument("--cbxgb-only", action="store_true")
    args = ap.parse_args()

    if args.cbxgb_only:
        _cbxgb_worker(args.pool, args.mit_cache, args.cbxgb_cache)
        return

    t0 = time.time()
    # Step 1: Fit/load CatBoost + XGBoost in an isolated child process before LightGBM is ever invoked.
    if not args.cbxgb_cache.exists():
        subprocess.check_call([sys.executable, __file__, "--pool", str(args.pool), "--mit-cache", str(args.mit_cache), "--cbxgb-cache", str(args.cbxgb_cache), "--cbxgb-only"])
    cbxgb = pickle.load(open(args.cbxgb_cache, "rb"))

    # Step 2: Fit/load LightGBM (base + MiT + Novel) + optional ViT-MiT + PurePhysViT.
    preds = _train_lgb_vit_branches(
        args.pool, args.mit_cache, args.preds_cache, args.vit_epochs, args.pp_epochs,
        tab_only=args.tab_only, vit_seeds=args.vit_seeds, skip_pp_vit=args.skip_pp_vit,
    )
    preds.update(cbxgb)

    pool = pickle.load(open(args.pool, "rb"))
    oof, test = pool["oof"], pool["test"]
    folds = data.folds()
    stems, test_stems = sorted(oof), sorted(test)
    gt_recs = metric.load_records(stems)
    adj_oof = {s: e11b._build_adj(oof[s]) for s in stems}
    adj_test = {s: e11b._build_adj(test[s]) for s in test_stems}
    stems_by_fold = {fk: [s for s in stems if folds[s] == fk] for fk in range(5)}
    iou_ta_idx = list(pool["feature_names"]).index("iou_with_ta")

    ta_u = {s: np.array([1.0 if k in ("ta_base", "ta_rescue") else 0.0 for k in oof[s]["kinds"]], dtype=np.float32) for s in stems}
    rows_ta = {
        s: metric.counts(
            [{"size": [2048, 2048], "counts": r.encode("ascii")}
             for r in e11b._decode_rles_disjoint(oof[s], e11b._solve_mwis_exact(e15._node_weights_w5(oof[s], ta_u[s], (0.5, 9.9, 9.9, 0.24, 0.0, False)), adj_oof[s]), e15._node_weights_w5(oof[s], ta_u[s], (0.5, 9.9, 9.9, 0.24, 0.0, False)))],
            gt_recs[s],
        )
        for s in stems
    }
    print(f"[Track-A Clean]: PQ={metric.summarize(list(rows_ta.values()))['pq']:.5f}", flush=True)

    specs = [
        ("GBDT_MiT",          [("gbdt_mit", 1.0)], 0.0),
        ("Tab3Fam_MiT",       [("gbdt_mit", 0.45), ("cb_mit", 0.35), ("xgb_mit", 0.20)], 0.0),
        ("Tab3Fam_HeavyLGB",  [("gbdt_mit", 0.55), ("cb_mit", 0.28), ("xgb_mit", 0.17)], 0.0),
    ]
    if "vit_mit" in preds:
        specs += [
            ("GV_MiT_65_35",      [("gbdt_mit", 0.65), ("vit_mit", 0.35)], 0.0),
            ("4Fam_MiT_Bal",      [("gbdt_mit", 0.32), ("cb_mit", 0.23), ("xgb_mit", 0.13), ("vit_mit", 0.32)], 0.0),
            ("4Fam_MiT_HeavyLGB", [("gbdt_mit", 0.42), ("cb_mit", 0.23), ("xgb_mit", 0.12), ("vit_mit", 0.23)], 0.0),
            ("4Fam_MiT_ViT20",    [("gbdt_mit", 0.44), ("cb_mit", 0.22), ("xgb_mit", 0.14), ("vit_mit", 0.20)], 0.0),
        ]
    if "vit_mit" in preds and "pp_vit" in preds:
        specs += [
            ("4Fam_MiT_PP25",     [("gbdt_mit", 0.30), ("cb_mit", 0.22), ("xgb_mit", 0.12), ("vit_mit", 0.26), ("pp_vit", 0.10)], 0.25),
            ("4Fam_MiT_NovPP35",  [("gbdt_mit", 0.28), ("cb_mit", 0.22), ("xgb_mit", 0.12), ("vit_mit", 0.28), ("pp_vit", 0.10)], 0.35),
        ]
    alphas = (0.60, 0.72)
    grid = [
        (tk, ta, ab, tb, w5b, allow)
        for tk in (0.195, 0.215, 0.242)
        for ta in (0.255, 0.280, 0.308)
        for ab in (9.9, 0.06, 0.025)
        for tb in (0.210, 0.235, 0.260, 0.285)
        for w5b in (0.0, 0.015)
        for allow in (True, False)
    ]

    exact_cache = {}
    reports = []
    for sname, wdict, ppw in specs:
        u_cache = {alp: _blend_hmu(preds, wdict, oof, stems, True, alp, iou_ta_idx, ppw) for alp in alphas}
        fast_ps = {}
        for alp, um in u_cache.items():
            for p in grid:
                ps = {}
                for s in stems:
                    ent = oof[s]
                    if not len(ent["X"]):
                        ps[s] = (0.0, 0.0)
                        continue
                    wt = e15._node_weights_w5(ent, um[s], p)
                    sel = e11b._solve_mwis_exact(wt, adj_oof[s])
                    rc = float(max(1, ent["r_count"]))
                    ps[s] = (float(ent["u"][sel].sum()) * rc, float(len(sel)) * rc)
                fast_ps[(alp, p)] = ps

        for obj in ("raw_pq", "sparse_76", "anc_raw", "anc_s76"):
            is_sparse = obj in ("sparse_76", "anc_s76")
            is_anc = obj in ("anc_raw", "anc_s76")
            k_pen = 0.272 if is_sparse else 0.235
            items_f = [
                (k, ps) for k, ps in fast_ps.items()
                if (not is_anc) or (k[1][2] in (9.9, 0.06) and bool(k[1][5]))
            ]
            chosen, u_n, p_n = [], {}, {}
            for fk in range(5):
                in_k = [s for s in stems if folds[s] != fk]
                va_k = stems_by_fold[fk]
                cands = sorted(((sum(ps[s][0] - k_pen * ps[s][1] for s in in_k), k) for k, ps in items_f), key=lambda z: -z[0])
                best_v, best_k = -1.0, cands[0][1]
                for _, k in cands[:20]:
                    ek = (sname, k)
                    if ek not in exact_cache:
                        alp, p = k
                        um = u_cache[alp]
                        rm = {}
                        for s in stems:
                            wt = e15._node_weights_w5(oof[s], um[s], p)
                            sel = e11b._solve_mwis_exact(wt, adj_oof[s])
                            rstrs = e11b._decode_rles_disjoint(oof[s], sel, wt)
                            rm[s] = metric.counts([{"size": [2048, 2048], "counts": r.encode("ascii")} for r in rstrs], gt_recs[s])
                        exact_cache[ek] = rm
                    sm_in = metric.summarize([exact_cache[ek][s] for s in in_k])
                    v = _pq_sparse(sm_in, 0.76) if is_sparse else sm_in["pq"]
                    if v > best_v:
                        best_v, best_k = v, k
                chosen.append(best_k)
                for s in va_k:
                    u_n[s], p_n[s] = u_cache[best_k[0]][s], best_k[1]

            rows_f, rles_f, rows_1r = {}, {}, []
            for s in stems:
                wt = e15._node_weights_w5(oof[s], u_n[s], p_n[s])
                sel = e11b._solve_mwis_exact(wt, adj_oof[s])
                rstrs = e11b._decode_rles_disjoint(oof[s], sel, wt)
                rles_f[s] = rstrs
                c = metric.counts([{"size": [2048, 2048], "counts": r.encode("ascii")} for r in rstrs], gt_recs[s])
                rows_f[s] = c
                if oof[s]["r_count"] == 1:
                    rows_1r.append(c)
            sm = metric.summarize(list(rows_f.values()))
            sm1 = metric.summarize(rows_1r)
            s80_m, s80_s = e11b._sim_r80(stems, rles_f, gt_recs, trials=15)
            boot = e11b._paired_date_bootstrap(rows_f, rows_ta, stems, n_boot=1500, seed=42)
            rep = {
                "obj": obj, "spec_name": sname, "wdict": wdict, "ppw": ppw,
                "pq": round(sm["pq"], 5), "sq": round(sm["sq"], 5), "rq": round(sm["rq"], 5),
                "tp": sm["tp"], "fp": sm["fp"], "fn": sm["fn"],
                "1rec": round(sm1["pq"], 5), "r75": round(_pq_sparse(sm, 0.75), 5),
                "r76": round(_pq_sparse(sm, 0.76), 5), "r77": round(_pq_sparse(sm, 0.77), 5),
                "sim80": round(s80_m, 5), "sim80_std": round(s80_s, 5),
                "vs_ta": round(boot["mean_gain"], 5), "vs_ta_p": round(boot["p_win"], 4),
                "chosen": chosen,
            }
            reports.append(rep)
            print(
                f"[{obj:9s} | {sname:16s}] OOF={rep['pq']:.5f} SQ={rep['sq']:.4f} "
                f"TP={rep['tp']} FP={rep['fp']} | 1rec={rep['1rec']:.5f} r75={rep['r75']:.5f} "
                f"r77={rep['r77']:.5f} sim80={rep['sim80']:.5f} p={rep['vs_ta_p']:.4f}",
                flush=True,
            )

    best_raw = max((r for r in reports if r["obj"] == "raw_pq"), key=lambda x: (x["pq"], x["sim80"]))
    best_s76 = max((r for r in reports if r["obj"] == "sparse_76"), key=lambda x: (x["r76"], x["pq"]))
    best_anc_raw = max((r for r in reports if r["obj"] == "anc_raw"), key=lambda x: (x["pq"], x["sim80"]))
    best_anc_s76 = max((r for r in reports if r["obj"] == "anc_s76"), key=lambda x: (x["r76"], x["pq"]))

    def _emit(rep: dict, suffix: str) -> dict:
        ch = rep["chosen"]
        alp_c = float(np.median([c[0] for c in ch]))
        cons_p = (
            float(np.median([c[1][0] for c in ch])),
            float(np.median([c[1][1] for c in ch])),
            float(np.median([c[1][2] for c in ch])),
            float(np.median([c[1][3] for c in ch])),
            float(np.median([c[1][4] for c in ch])),
            bool(np.mean([1.0 if c[1][5] else 0.0 for c in ch]) >= 0.5),
        )
        u_t = _blend_hmu(preds, rep["wdict"], test, test_stems, False, alp_c, iou_ta_idx, rep["ppw"])
        rles_t = {}
        for s in test_stems:
            if not len(test[s]["X"]):
                rles_t[s] = []
                continue
            wt = e15._node_weights_w5(test[s], u_t[s], cons_p)
            sel = e11b._solve_mwis_exact(wt, adj_test[s])
            rstrs = e11b._decode_rles_disjoint(test[s], sel, wt)
            rles_t[s] = [{"size": [2048, 2048], "counts": r.encode("ascii")} for r in rstrs]
        out_p = Path(f"{args.out_prefix}_{suffix}_clean.csv")
        out_p.parent.mkdir(parents=True, exist_ok=True)
        submission.write_csv(out_p, rles_t)
        assert submission.validate(out_p, data.test_stems()) == []
        audit = tsa.audit_csv(out_p, REPO / "results/submission.csv")
        return {"csv": str(out_p), "cons_alp": alp_c, "cons_p": cons_p, "shift": audit}

    raw_emit = _emit(best_raw, "raw")
    s76_emit = _emit(best_s76, "s76")
    anc_raw_emit = _emit(best_anc_raw, "anc_raw")
    anc_s76_emit = _emit(best_anc_s76, "anc_s76")
    out = {
        "best_raw": best_raw, "best_s76": best_s76,
        "best_anc_raw": best_anc_raw, "best_anc_s76": best_anc_s76,
        "raw_emit": raw_emit, "s76_emit": s76_emit,
        "anc_raw_emit": anc_raw_emit, "anc_s76_emit": anc_s76_emit,
        "all": reports, "elapsed_s": round(time.time() - t0, 1),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(out, indent=2, default=str))
    print(f"EXP-016 COMPLETE in {time.time()-t0:.1f}s -> {args.summary_json}", flush=True)


if __name__ == "__main__":
    main()
