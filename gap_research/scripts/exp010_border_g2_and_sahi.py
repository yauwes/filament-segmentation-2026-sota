"""EXP-010 (Wave-2.5): All-707 OOF Ablations on Clean EXP-006/007 Streams (Zero F1, 5-Fold Nested CV).

Tests orthogonal hypotheses missed in Track-A Clean (0.46613):
  H1. Border-Conditional _g2 Relaxation: unlock sam2_g2/e4x_g2/s1e_g2 outside box0 ONLY where the
      non-g2 mask touches a clipped box0 edge (directly targeting 1-to-1 truncated filament FNs).
  H2. SAHI Calibration & Consensus Participation: optimal segmenter weights for SAHI + sahi_e4_g1
      extension + direct weighted participation inside weighted_consensus_stem + rescue thresholds.
  H3. Joint Post-Proc Calibration across all 707 stems with strict 5-Fold Nested Date-CV +
      paired date-cluster bootstrap (N=1000) vs Track-A Clean parity baseline.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "gap_research" / "scripts"))

from filseg import data, metric
import build_gap_submission as bgs
from scripts import build_super_submission as bss

WAVE2_ROOT = Path("/private/tmp/gap_wave2_cache")
E6 = WAVE2_ROOT / "exp006"
E7 = WAVE2_ROOT / "exp007"
FULL = 2048


def _unpack_raw(path: Path) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    return bgs._unpack_npz(path, tta_prefer=4)


def load_stem_bundle(stem: str, split: str) -> dict[str, dict[str, tuple]]:
    sp = "oof" if split in ("oof", "val") else "test"
    paths = {
        "ms": {
            "e4": REPO / "work" / "cache" / "ms" / sp / f"{stem}.npz",
            "e4x": E6 / "e4x" / "e4x" / "ms" / sp / f"{stem}.npz",
            "e4x_g2": E6 / "e4x" / "e4x_g2" / "ms" / sp / f"{stem}.npz",
            "s1e_g2": E6 / "s1e_g2" / "ms" / sp / f"{stem}.npz",
            "sam2": E6 / "sam2" / "ms" / sp / f"{stem}.npz",
            "sam2_g2": E6 / "sam2_g2" / "ms" / sp / f"{stem}.npz",
        },
        "s": {
            "e4": REPO / "work" / "cache" / "s" / sp / f"{stem}.npz",
            "e4x": E6 / "e4x" / "e4x" / "s" / sp / f"{stem}.npz",
            "e4x_g2": E6 / "e4x" / "e4x_g2" / "s" / sp / f"{stem}.npz",
            "s1e_g2": E6 / "s1e_g2" / "s" / sp / f"{stem}.npz",
        },
        "main": {
            "e4": REPO / "work" / "cache" / "main" / sp / f"{stem}.npz",
            "e4x": E6 / "e4x" / "e4x" / "main" / sp / f"{stem}.npz",
            "e4x_g2": E6 / "e4x" / "e4x_g2" / "main" / sp / f"{stem}.npz",
            "s1e_g2": E6 / "s1e_g2" / "main" / sp / f"{stem}.npz",
        },
        "SAHI": {
            "e4": E7 / "sahi_e4" / sp / f"{stem}.npz",
            "e4_g1": E7 / "sahi_e4_g1" / sp / f"{stem}.npz",
            "s1e": E7 / "sahi_s1e" / sp / f"{stem}.npz",
            "sam2": E7 / "sahi_sam2" / sp / f"{stem}.npz",
        },
    }
    out = {}
    for stream, smap in paths.items():
        d = {}
        for seg, p in smap.items():
            if p.exists():
                d[seg] = _unpack_raw(p)
        out[stream] = d
    return out


def blend_stream_from_bundle(
    seg_data: dict[str, tuple],
    weights: dict[str, float],
    thr: float = 0.68,
    conf_floor: float = 0.28,
    drop: float = 0.90,
    min_area: int = 100,
    small_scale: float = 1.20,
    small_px: int = 120,
    border_g2: bool = False,
    thr_ext: float = 0.62,
    edge_px: int = 2,
    g2_keys: tuple[str, ...] = ("e4x_g2", "s1e_g2", "sam2_g2", "e4_g1"),
) -> list[tuple[float, tuple[int, int, int, int], np.ndarray]]:
    loaded = [(k, weights[k], *seg_data[k]) for k in weights if weights[k] > 0 and k in seg_data]
    if not loaded:
        return []
    # Anchor box0 on primary base segmenter ("e4" if present else first)
    anchor_idx = 0
    for j, item in enumerate(loaded):
        if item[0] == "e4":
            anchor_idx = j
            break
    _, _, conf0, box0, _ = loaded[anchor_idx]
    if len(conf0) == 0:
        return []

    occ = np.zeros((FULL, FULL), dtype=bool)
    t_u8 = int(thr * 255.0 + 0.5)
    t_ext_u8 = int(thr_ext * 255.0 + 0.5)
    out = []

    for idx in np.argsort(-conf0):
        raw_c = float(conf0[idx])
        bx0, by0, bx1, by1 = [int(v) for v in box0[idx]]
        eff_c = raw_c * (small_scale if max(bx1 - bx0, by1 - by0) < small_px else 1.0)
        if eff_c < conf_floor or bx1 <= bx0 or by1 <= by0:
            continue

        # Determine working canvas: clamped box0 vs union of expanded _g2 boxes
        ux0, uy0, ux1, uy1 = bx0, by0, bx1, by1
        if border_g2:
            for k, _, _, b_arr, _ in loaded:
                if k in g2_keys and idx < len(b_arr):
                    sx1, sy1, sx2, sy2 = [int(v) for v in b_arr[idx]]
                    if sx2 > sx1 and sy2 > sy1:
                        ux0 = min(ux0, sx1)
                        uy0 = min(uy0, sy1)
                        ux1 = max(ux1, sx2)
                        uy1 = max(uy1, sy2)

        uh, uw = uy1 - uy0, ux1 - ux0
        acc = np.zeros((uh, uw), dtype=np.float32)
        wsum = np.zeros((uh, uw), dtype=np.float32)
        g2_acc = np.zeros((uh, uw), dtype=np.float32) if border_g2 else None
        g2_wsum = np.zeros((uh, uw), dtype=np.float32) if border_g2 else None

        for k, w, _, b_arr, p_list in loaded:
            if idx >= len(b_arr):
                continue
            sx1, sy1, sx2, sy2 = [int(v) for v in b_arr[idx]]
            if sx2 <= sx1 or sy2 <= sy1:
                continue
            pr = p_list[idx]
            if pr.shape != (sy2 - sy1, sx2 - sx1):
                pr = cv2.resize(pr, (sx2 - sx1, sy2 - sy1), interpolation=cv2.INTER_LINEAR)
            if not border_g2:
                ix1, iy1, ix2, iy2 = max(bx0, sx1), max(by0, sy1), min(bx1, sx2), min(by1, sy2)
                if ix2 > ix1 and iy2 > iy1:
                    acc[iy1 - uy0 : iy2 - uy0, ix1 - ux0 : ix2 - ux0] += (w / 255.0) * pr[iy1 - sy1 : iy2 - sy1, ix1 - sx1 : ix2 - sx1]
                    wsum[iy1 - uy0 : iy2 - uy0, ix1 - ux0 : ix2 - ux0] += w
            else:
                ix1, iy1, ix2, iy2 = max(ux0, sx1), max(uy0, sy1), min(ux1, sx2), min(uy1, sy2)
                if ix2 > ix1 and iy2 > iy1:
                    acc[iy1 - uy0 : iy2 - uy0, ix1 - ux0 : ix2 - ux0] += (w / 255.0) * pr[iy1 - sy1 : iy2 - sy1, ix1 - sx1 : ix2 - sx1]
                    wsum[iy1 - uy0 : iy2 - uy0, ix1 - ux0 : ix2 - ux0] += w
                    if k in g2_keys:
                        g2_acc[iy1 - uy0 : iy2 - uy0, ix1 - ux0 : ix2 - ux0] += (w / 255.0) * pr[iy1 - sy1 : iy2 - sy1, ix1 - sx1 : ix2 - sx1]
                        g2_wsum[iy1 - uy0 : iy2 - uy0, ix1 - ux0 : ix2 - ux0] += w

        if not (wsum > 0).any():
            continue

        ox0, oy0, ox1, oy1 = bx0 - ux0, by0 - uy0, bx1 - ux0, by1 - uy0
        prob_u8 = np.where(wsum > 0, (acc / np.maximum(wsum, 1e-6)) * 255.0 + 0.5, 0.0).astype(np.uint8)
        m_in = np.zeros((uh, uw), dtype=bool)
        m_in[oy0:oy1, ox0:ox1] = prob_u8[oy0:oy1, ox0:ox1] > t_u8

        if border_g2 and (ux0 < bx0 or uy0 < by0 or ux1 > bx1 or uy1 > by1) and m_in.any():
            # Allow _g2 exterior where 8-connected (up to edge_px=4px fade bridge) to interior mask
            if (g2_wsum > 0).any():
                g2_u8 = np.where(g2_wsum > 0, (g2_acc / np.maximum(g2_wsum, 1e-6)) * 255.0 + 0.5, 0.0).astype(np.uint8)
                ext = np.zeros((uh, uw), dtype=bool)
                ext_mask = g2_u8 > t_ext_u8
                ext[:, :ox0] = ext_mask[:, :ox0]
                ext[:, ox1:] = ext_mask[:, ox1:]
                ext[:oy0, :] = ext_mask[:oy0, :]
                ext[oy1:, :] = ext_mask[oy1:, :]
                if ext.any():
                    bridge = cv2.dilate(m_in.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                    seed = m_in | (bridge & (g2_u8 > t_ext_u8))
                    num_cc, labels = cv2.connectedComponents((seed | ext).astype(np.uint8), connectivity=8)
                    keep_cc = np.unique(labels[m_in])
                    keep_cc = keep_cc[keep_cc > 0]
                    if len(keep_cc):
                        m = (seed | ext) & np.isin(labels, keep_cc)
                    else:
                        m = m_in
                else:
                    m = m_in
            else:
                m = m_in
        else:
            m = m_in

        v = bss.trim(eff_c, (ux0, uy0, ux1, uy1), m)
        if v is None:
            continue
        c_v, (x1, y1, x2, y2), mm = v
        keep = mm & ~occ[y1:y2, x1:x2]
        area = int(keep.sum())
        if area >= min_area and area >= drop * int(mm.sum()):
            occ[y1:y2, x1:x2] |= keep
            vv = bss.trim(c_v, (x1, y1, x2, y2), keep)
            if vv is not None:
                out.append(vv)
    return out


def _append_rescue_rles(
    rle_strs: list[str],
    rescue_insts: list[tuple[float, tuple[int, int, int, int], np.ndarray]],
    min_conf: float,
    max_area: int,
    min_area: int = 100,
    drop: float = 0.90,
) -> list[str]:
    if not rescue_insts or max_area <= 0:
        return rle_strs
    bm = np.zeros((FULL, FULL), dtype=bool)
    for r in rle_strs:
        bm |= mu.decode({"size": [FULL, FULL], "counts": r.encode()}).astype(bool)
    cur = list(rle_strs)
    for c, (x1, y1, x2, y2), m in rescue_insts:
        if c < min_conf:
            continue
        ma = int(m.sum())
        if ma < min_area or ma > max_area:
            continue
        if int((m & bm[y1:y2, x1:x2]).sum()) / ma <= 0.15:
            keep = m & ~bm[y1:y2, x1:x2]
            if int(keep.sum()) >= min_area and int(keep.sum()) >= drop * ma:
                bm[y1:y2, x1:x2] |= keep
                cur.append(bss.encode_crop_ascii((x1, y1, x2, y2), keep))
    return cur


@dataclass(frozen=True)
class ExpCfg:
    name: str
    ms_w: dict[str, float]
    s_w: dict[str, float]
    main_w: dict[str, float]
    sahi_w: dict[str, float]
    stream_wt: dict[str, float]  # ms, s, main, (optional) SAHI
    thr: float = 0.68
    drop: float = 0.90
    small_scale: float = 1.20
    small_px: int = 120
    vote_thr: float = 0.52
    c_floor: float = 0.28
    match_iou: float = 0.40
    border_g2: bool = False
    thr_ext: float = 0.62
    sahi_rescue_conf: float = 0.22
    sahi_rescue_max_area: int = 400
    sahi_floor: float = 0.20


def build_configs() -> list[ExpCfg]:
    w_ms_base = {"e4": 0.10, "sam2": 0.55, "s1e_g2": 0.25, "e4x": 0.10}
    w_sm_base = {"e4": 0.30, "s1e_g2": 0.50, "e4x": 0.20}
    w_sahi_eq = {"e4": 0.25, "e4_g1": 0.25, "s1e": 0.25, "sam2": 0.25}

    w_ms_g2 = {"e4": 0.10, "sam2": 0.30, "sam2_g2": 0.25, "s1e_g2": 0.25, "e4x": 0.05, "e4x_g2": 0.05}
    w_sm_g2 = {"e4": 0.25, "s1e_g2": 0.45, "e4x": 0.15, "e4x_g2": 0.15}
    w_sahi_wt = {"sam2": 0.40, "s1e": 0.30, "e4_g1": 0.20, "e4": 0.10}
    st_base = {"ms": 1.35, "s": 1.05, "main": 0.95}

    cfgs = [
        ExpCfg("B0_tracka_exact", w_ms_base, w_sm_base, w_sm_base, w_sahi_eq, st_base, border_g2=False),
        ExpCfg("B1_add_g2_clamped", w_ms_g2, w_sm_g2, w_sm_g2, w_sahi_eq, st_base, border_g2=False),
        ExpCfg("E1a_border_g2_t58", w_ms_g2, w_sm_g2, w_sm_g2, w_sahi_eq, st_base, border_g2=True, thr_ext=0.58),
        ExpCfg("E1b_border_g2_t62", w_ms_g2, w_sm_g2, w_sm_g2, w_sahi_eq, st_base, border_g2=True, thr_ext=0.62),
        ExpCfg("E1c_border_g2_t66", w_ms_g2, w_sm_g2, w_sm_g2, w_sahi_eq, st_base, border_g2=True, thr_ext=0.66),
        ExpCfg("E2a_sahi_wt_only", w_ms_base, w_sm_base, w_sm_base, w_sahi_wt, st_base, border_g2=False),
        ExpCfg("E2b_border_g2_plus_sahi_wt", w_ms_g2, w_sm_g2, w_sm_g2, w_sahi_wt, st_base, border_g2=True, thr_ext=0.62),
    ]

    # H2: SAHI direct participation inside consensus + precision trim on rescue
    for ws, vt in ((0.35, 0.56), (0.55, 0.58), (0.75, 0.62)):
        st = {"ms": 1.35, "s": 1.05, "main": 0.95, "SAHI": ws}
        cfgs.append(
            ExpCfg(
                f"E3_sahi_cons_w{int(ws*100)}_v{int(vt*100)}",
                w_ms_g2, w_sm_g2, w_sm_g2, w_sahi_wt, st,
                vote_thr=vt, border_g2=True, thr_ext=0.62,
                sahi_rescue_conf=0.24, sahi_rescue_max_area=480,
            )
        )

    # H3: Threshold, small_scale, vote_thr, c_floor, rescue calibration around E2b/E3
    for thr in (0.64, 0.66, 0.68):
        for cf in (0.26, 0.28, 0.30, 0.32):
            for vt in (0.48, 0.52, 0.56):
                cfgs.append(
                    ExpCfg(
                        f"E4_g2_thr{int(thr*100)}_cf{int(cf*100)}_vt{int(vt*100)}",
                        w_ms_g2, w_sm_g2, w_sm_g2, w_sahi_wt, st_base,
                        thr=thr, c_floor=cf, vote_thr=vt,
                        border_g2=True, thr_ext=0.62,
                        sahi_rescue_conf=0.25, sahi_rescue_max_area=520,
                    )
                )
    for rc in (0.20, 0.24, 0.28, 0.32, 9.99):
        for ma in (400, 580):
            if rc == 9.99 and ma == 580:
                continue
            cfgs.append(
                ExpCfg(
                    f"E4_resc_c{int(rc*100) if rc<5 else 999}_a{ma}",
                    w_ms_g2, w_sm_g2, w_sm_g2, w_sahi_wt, st_base,
                    border_g2=True, thr_ext=0.62,
                    sahi_rescue_conf=rc, sahi_rescue_max_area=ma,
                )
            )
    return cfgs


def _run_chunk(args: tuple[int, list[str], list[ExpCfg]]) -> dict[str, dict[str, dict]]:
    wid, stems, cfgs = args
    cv2.setNumThreads(0)
    recs, anns = data.coco_index()
    res: dict[str, dict[str, dict]] = {c.name: {} for c in cfgs}
    t0 = time.time()
    for i, s in enumerate(stems):
        bundle = load_stem_bundle(s, "oof")
        gt = [[data.ann_rle(a) for a in anns.get(rid, [])] for rid in recs[s]]
        blend_cache: dict[tuple, list] = {}
        for cfg in cfgs:
            st_items = []
            for sname in ("ms", "s", "main", "SAHI"):
                wt = cfg.stream_wt.get(sname, 0.0)
                if wt <= 0:
                    continue
                wdict = getattr(cfg, f"{sname.lower()}_w")
                cf = cfg.sahi_floor if sname == "SAHI" else (0.25 if sname == "main" else 0.28)
                key = (sname, tuple(sorted(wdict.items())), cfg.thr, cf, cfg.drop, cfg.small_scale, cfg.small_px, cfg.border_g2, cfg.thr_ext)
                if key not in blend_cache:
                    blend_cache[key] = blend_stream_from_bundle(
                        bundle[sname], wdict,
                        thr=cfg.thr, conf_floor=cf, drop=cfg.drop, min_area=100,
                        small_scale=cfg.small_scale, small_px=cfg.small_px,
                        border_g2=cfg.border_g2, thr_ext=cfg.thr_ext,
                    )
                st_items.append((sname, wt, blend_cache[key]))

            rle_strs = bgs.weighted_consensus_stem(
                st_items,
                match_iou=cfg.match_iou,
                vote_thr=cfg.vote_thr,
                c_floor=cfg.c_floor,
                min_area=100,
                drop=cfg.drop,
            )
            if cfg.sahi_rescue_conf < 5.0:
                skey = ("SAHI", tuple(sorted(cfg.sahi_w.items())), cfg.thr, cfg.sahi_floor, cfg.drop, 1.0, 120, cfg.border_g2, cfg.thr_ext)
                if skey not in blend_cache:
                    blend_cache[skey] = blend_stream_from_bundle(
                        bundle["SAHI"], cfg.sahi_w,
                        thr=cfg.thr, conf_floor=cfg.sahi_floor, drop=cfg.drop, min_area=100,
                        small_scale=1.0, small_px=120,
                        border_g2=cfg.border_g2, thr_ext=cfg.thr_ext,
                    )
                rle_strs = _append_rescue_rles(
                    rle_strs, blend_cache[skey],
                    min_conf=cfg.sahi_rescue_conf,
                    max_area=cfg.sahi_rescue_max_area,
                    min_area=100, drop=cfg.drop,
                )

            rle_dicts = [{"size": [FULL, FULL], "counts": r.encode() if isinstance(r, str) else r} for r in rle_strs]
            cnt = metric.counts(rle_dicts, gt)
            cnt["n_pred"] = len(rle_dicts)
            res[cfg.name][s] = cnt
        if wid == 0 and ((i + 1) % 20 == 0 or i + 1 == len(stems)):
            print(f"  [worker 0] {i+1}/{len(stems)} stems ({time.time()-t0:.1f}s)", flush=True)
    return res


def eval_all(cfgs: list[ExpCfg], stems: list[str], workers: int = 7) -> dict[str, dict[str, dict]]:
    chunks = [stems[i::workers] for i in range(workers)]
    tasks = [(i, ch, cfgs) for i, ch in enumerate(chunks) if ch]
    merged: dict[str, dict[str, dict]] = {c.name: {} for c in cfgs}
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(tasks)) as ex:
        for part in ex.map(_run_chunk, tasks):
            for k, v in part.items():
                merged[k].update(v)
    return merged


def main() -> None:
    t0 = time.time()
    folds = data.folds()
    all_stems = sorted(folds)
    dates = [s[:8] for s in all_stems]
    cfgs = build_configs()
    print(f"EXP-010: Evaluating {len(cfgs)} configs across {len(all_stems)} OOF stems on 7 workers...", flush=True)
    counts_by_cfg = eval_all(cfgs, all_stems, workers=min(7, os.cpu_count() or 4))
    print(f"Done in {time.time()-t0:.1f}s\n", flush=True)

    base = counts_by_cfg["B0_tracka_exact"]
    base_rows = [base[s] for s in all_stems]
    base_sum = metric.summarize(base_rows)

    ranked = []
    for cfg in cfgs:
        rows = [counts_by_cfg[cfg.name][s] for s in all_stems]
        sm = metric.summarize(rows)
        npred = sum(r["n_pred"] for r in rows)
        delta = sm["pq"] - base_sum["pq"]
        ranked.append((sm["pq"], delta, sm["sq"], sm["rq"], sm["tp"], sm["fp"], sm["fn"], npred, cfg.name))

    ranked.sort(key=lambda x: -x[0])
    print(f"Baseline parity B0_tracka_exact: PQ={base_sum['pq']:.5f} SQ={base_sum['sq']:.5f} RQ={base_sum['rq']:.5f} TP={base_sum['tp']} FP={base_sum['fp']} FN={base_sum['fn']} preds={sum(r['n_pred'] for r in base_rows)}")
    print("\nTop 20 Configs (All 707 OOF Stems):")
    for pq, d, sq, rq, tp, fp, fn, npred, name in ranked[:20]:
        print(f"  {name:34s} PQ={pq:.5f} ({d:+.5f}) SQ={sq:.5f} RQ={rq:.5f} TP={tp} FP={fp} FN={fn} P={npred}")

    # Strict 5-Fold Nested CV over ALL candidate configs using official data.folds()
    nested_rows = []
    chosen_names = []
    for f in range(5):
        tr = [s for s in all_stems if folds[s] != f]
        va = [s for s in all_stems if folds[s] == f]
        best_cfg_name = max(counts_by_cfg, key=lambda k: metric.summarize([counts_by_cfg[k][s] for s in tr])["pq"])
        chosen_names.append((f, best_cfg_name))
        for s in va:
            nested_rows.append(counts_by_cfg[best_cfg_name][s])
    nested_map = {s: counts_by_cfg[dict(chosen_names)[folds[s]]][s] for s in all_stems}
    nested_sum = metric.summarize([nested_map[s] for s in all_stems])

    # Paired date-cluster bootstrap N=1000 vs B0
    rng = np.random.default_rng(0)
    by_date = {}
    for s in all_stems:
        by_date.setdefault(s[:8], []).append(s)
    dkeys = sorted(by_date)
    best_name = ranked[0][8]
    diffs_best, diffs_nested = [], []
    for _ in range(1000):
        idx = rng.integers(0, len(dkeys), size=len(dkeys))
        samp = [s for k in idx for s in by_date[dkeys[k]]]
        pb0 = metric.summarize([base[s] for s in samp])["pq"]
        p1 = metric.summarize([counts_by_cfg[best_name][s] for s in samp])["pq"]
        pn = metric.summarize([nested_map[s] for s in samp])["pq"]
        diffs_best.append(p1 - pb0)
        diffs_nested.append(pn - pb0)

    print(f"\n5-Fold Nested CV ({chosen_names}):")
    print(f"  Nested PQ={nested_sum['pq']:.5f} ({nested_sum['pq']-base_sum['pq']:+.5f}) | Boot P(>B0)={np.mean(np.array(diffs_nested)>0):.3f} CI95=[{np.percentile(diffs_nested,2.5):+.5f}, {np.percentile(diffs_nested,97.5):+.5f}]")
    print(f"  Top-1 ({best_name}): PQ={ranked[0][0]:.5f} ({ranked[0][1]:+.5f}) | Boot P(>B0)={np.mean(np.array(diffs_best)>0):.3f} CI95=[{np.percentile(diffs_best,2.5):+.5f}, {np.percentile(diffs_best,97.5):+.5f}]")

    out_dir = REPO / "gap_research" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "baseline": base_sum,
        "top20": [dict(name=n, pq=pq, delta=d, sq=sq, rq=rq, tp=tp, fp=fp, fn=fn, n_pred=np_) for pq, d, sq, rq, tp, fp, fn, np_, n in ranked[:20]],
        "nested_cv": {
            "pq": nested_sum["pq"],
            "delta_vs_b0": nested_sum["pq"] - base_sum["pq"],
            "chosen_per_fold": chosen_names,
            "boot_p": float(np.mean(np.array(diffs_nested) > 0)),
            "ci95": [float(np.percentile(diffs_nested, 2.5)), float(np.percentile(diffs_nested, 97.5))],
        },
        "best_boot": {
            "name": best_name,
            "boot_p": float(np.mean(np.array(diffs_best) > 0)),
            "ci95": [float(np.percentile(diffs_best, 2.5)), float(np.percentile(diffs_best, 97.5))],
        },
    }
    (out_dir / "exp010_results.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
