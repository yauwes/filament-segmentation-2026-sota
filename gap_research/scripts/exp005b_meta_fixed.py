"""EXP-005b: Fixed Nested 5-Fold Date-CV Meta-Rescoring + P6/Relaxed Recovery over ClusterCache."""
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gap_research" / "scripts"))
import eval_harness as eh  # noqa: E402
import exp003_local_suite as els  # noqa: E402

CLUSTER_PKL = Path("/private/tmp/gap_e19_clusters_fold0.pkl")


def main():
    by_stem = pickle.load(open(CLUSTER_PKL, "rb"))
    base_counts = pickle.load(open(REPO / "gap_research" / "predictions" / "e19_fold0_baseline.pkl", "rb"))["counts"]
    nested_folds = eh.make_nested_date_folds(eh.FOLD0_STEMS, n_folds=5, seed=42)

    X_by = {s: np.stack([c["feat"] for c in by_stem[s]]) if by_stem[s] else np.zeros((0, 48), np.float32) for s in eh.FOLD0_STEMS}
    Y_iou = {s: np.array([c["y_iou"] for c in by_stem[s]], np.float32) for s in eh.FOLD0_STEMS}
    Y_tp = {s: np.array([c["y_tp"] for c in by_stem[s]], np.float32) for s in eh.FOLD0_STEMS}
    C_e19 = {s: np.array([c["c_e19"] for c in by_stem[s]], np.float32) for s in eh.FOLD0_STEMS}

    oof = {m: {s: np.zeros(len(by_stem[s]), np.float32) for s in eh.FOLD0_STEMS} for m in ("hgbr_iou", "hgbr_tp", "logreg_tp", "ridge_iou")}
    for k in range(5):
        tr = [s for s in eh.FOLD0_STEMS if nested_folds[s] != k]
        va = [s for s in eh.FOLD0_STEMS if nested_folds[s] == k]
        X_tr = np.concatenate([X_by[s] for s in tr])
        y_i = np.concatenate([Y_iou[s] for s in tr])
        y_t = np.concatenate([Y_tp[s] for s in tr])

        m_i = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.04, max_leaf_nodes=8, min_samples_leaf=45, l2_regularization=5.0, random_state=42).fit(X_tr, y_i)
        m_t = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.04, max_leaf_nodes=8, min_samples_leaf=45, l2_regularization=5.0, random_state=42).fit(X_tr, y_t)
        mu_x = X_tr.mean(0)
        sd_x = np.clip(X_tr.std(0), 1e-4, None)
        Xs = (X_tr - mu_x) / sd_x
        m_lr = LogisticRegression(C=0.05, max_iter=1000).fit(Xs, (y_t > 0.5).astype(int))
        m_rg = Ridge(alpha=200.0).fit(Xs, y_i)

        for s in va:
            if len(by_stem[s]):
                Xv = X_by[s]
                Xvs = (Xv - mu_x) / sd_x
                oof["hgbr_iou"][s] = m_i.predict(Xv)
                oof["hgbr_tp"][s] = m_t.predict(Xv)
                oof["logreg_tp"][s] = m_lr.predict_proba(Xvs)[:, 1]
                oof["ridge_iou"][s] = m_rg.predict(Xvs)

    # First check EXACT Nested Date-CV on c_e19 (including 0.25..0.38!)
    print("=== EXP-005b: FIXED FULL-RANGE NESTED DATE-CV ===")
    for label, preds in [("c_e19_only", C_e19)] + [(m, oof[m]) for m in ("logreg_tp", "ridge_iou", "hgbr_tp", "hgbr_iou")]:
        alphas = [0.0] if label == "c_e19_only" else [0.10, 0.20, 0.35, 0.50, 0.70, 1.0]
        for a in alphas:
            sc_map = {s: (C_e19[s] if label == "c_e19_only" else (1.0 - a) * C_e19[s] + a * preds[s]) for s in eh.FOLD0_STEMS}
            all_vals = np.concatenate(list(sc_map.values()))
            grid = np.linspace(float(np.percentile(all_vals, 2)), float(np.percentile(all_vals, 45)), 17)
            if label == "c_e19_only":
                grid = np.unique(np.append(grid, [0.30, 0.31, 0.32, 0.33, 0.34]))
            r_nested, fls = {}, []
            for k in range(5):
                tr = [s for s in eh.FOLD0_STEMS if nested_folds[s] != k]
                va = [s for s in eh.FOLD0_STEMS if nested_folds[s] == k]
                bfl, bpq = float(grid[0]), -1.0
                for fl in grid:
                    rc = eh.score_rle_strs_dict({s: els.paint_cands_to_rles(by_stem[s], score_fn=(lambda i, c, s=s: sc_map[s][i]), floor=float(fl)) for s in tr}, stems=tr)
                    pq = eh.score_from_counts(rc, stems=tr)["pq"]
                    if pq > bpq:
                        bpq, bfl = pq, float(fl)
                fls.append(round(bfl, 3))
                for s in va:
                    r_nested[s] = els.paint_cands_to_rles(by_stem[s], score_fn=(lambda i, c, s=s: sc_map[s][i]), floor=bfl)
            cc = eh.score_rle_strs_dict(r_nested)
            mm = eh.score_from_counts(cc)
            bt = eh.paired_date_bootstrap(cc, base_counts)
            print(f"  {label:12s} a={a:.2f}: PQ={mm['pq']:.4f} SQ={mm['sq']:.4f} RQ={mm['rq']:.4f} TP={mm['tp']} FP={mm['fp']} | vsE19={bt['mean_gain']:+.4f} CI=[{bt['ci95'][0]:+.4f},{bt['ci95'][1]:+.4f}] P={bt['win_rate']:.3f} fls={fls}", flush=True)


if __name__ == "__main__":
    main()
