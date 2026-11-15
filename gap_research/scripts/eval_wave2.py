"""Wave-2 GPU Cache Unpacker, Multi-Segmenter CV Optimizer, 707-OOF & Fold-0 Nested Date-CV Evaluator."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pickle
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "gap_research" / "scripts"))

import build_gap_submission as bgs  # noqa: E402
import build_super_submission as bss  # noqa: E402
import eval_harness as eh  # noqa: E402
import test_shift_audit  # noqa: E402
from filseg import data, submission  # noqa: E402

WAVE2_ROOT = Path("/private/tmp/gap_wave2_cache")
G19_ROOT = Path("/private/tmp/filseg_ens_cache/g19")


def unpack_archives(wave2_root: Path = WAVE2_ROOT) -> dict[str, list[str]]:
    """Idempotently extract compact exp006..exp008b .tar.gz bundles (skipping heavy weight tarballs)."""
    extracted = {}
    if not wave2_root.exists():
        return extracted
    for exp_dir in sorted(p for p in wave2_root.iterdir() if p.is_dir()):
        done = []
        for tb in sorted(exp_dir.glob("*.tar.gz")):
            if "weights" in tb.name:
                continue
            marker = exp_dir / f".extracted_{tb.stem}"
            if marker.exists():
                done.append(tb.name)
                continue
            with tarfile.open(tb, "r:gz") as tf:
                tf.extractall(exp_dir)
            marker.write_text("ok\n")
            done.append(tb.name)
        extracted[exp_dir.name] = done
    return extracted


def _find_subdir(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists() and any(p.glob("*.npz")):
            return p
    return None


def resolve_stream_seg_dirs(
    stream_name: str,
    split: str,
    wave2_root: Path = WAVE2_ROOT,
) -> dict[str, Path]:
    """Return available {seg_family: split_dir} for a detector stream ('oof'/'val'/'test')."""
    sp_cands = ["oof", "val"] if split in ("oof", "val") else ["test"]
    out: dict[str, Path] = {}

    if stream_name in ("ms", "s", "main"):
        p_e4 = _find_subdir([REPO / "work" / "cache" / stream_name / sp for sp in sp_cands])
        if p_e4:
            out["e4"] = p_e4
    elif stream_name == "F3":
        p = _find_subdir([G19_ROOT / "tsF3" / "cache" / "S1E" / sp / "1280" for sp in sp_cands] + [G19_ROOT / "tsF3" / "cache" / "S1E" / sp for sp in sp_cands])
        if p:
            out["s1e"] = p
    elif stream_name == "P6":
        p = _find_subdir([G19_ROOT / "tsP6" / "cache" / "S1E" / sp / "1280" for sp in sp_cands] + [G19_ROOT / "tsP6" / "cache" / "S1E" / sp for sp in sp_cands])
        if p:
            out["s1e"] = p
    elif stream_name == "P4a":
        p = _find_subdir([bss.FP_ROOT / "tsP4a" / "cache" / "S1E" / sp / "1280" for sp in sp_cands])
        if p:
            out["s1e"] = p
    elif stream_name in ("A", "B", "F0", "G1", "P1"):
        for sz in ("1280", "1664"):
            p = _find_subdir([bss.AVG_ROOT / f"cache_{stream_name}" / "avgSAM2" / sp / sz for sp in sp_cands])
            if p:
                out[f"sam2_{sz}"] = p
                break
    elif stream_name == "SAHI":
        e007 = wave2_root / "exp007"
        for key, folder in (("e4", "sahi_e4"), ("e4_g1", "sahi_e4_g1"), ("s1e", "sahi_s1e"), ("sam2", "sahi_sam2")):
            p = _find_subdir([e007 / folder / sp for sp in sp_cands])
            if p:
                out[key] = p
    elif stream_name == "F5F":
        e008a = wave2_root / "exp008a"
        for key, folder in (("e4", "frcnn_e4"), ("e4_g1", "frcnn_e4_g1"), ("s1e", "frcnn_s1e"), ("sam2", "frcnn_sam2")):
            p = _find_subdir([e008a / folder / sp for sp in sp_cands])
            if p:
                out[key] = p

    # Overlay Wave-2 multi-segmenter upgrades for ms/F3/P6/P4a/s/main
    if stream_name in ("ms", "F3", "P6", "P4a", "s", "main"):
        e006 = wave2_root / "exp006"
        for key, sub in (
            ("sam2", e006 / "sam2" / stream_name),
            ("s1e", e006 / "s1e" / stream_name),
            ("e4x", e006 / "e4x" / "e4x" / stream_name),
            ("e4x_g2", e006 / "e4x" / "e4x_g2" / stream_name),
            ("sam2_g2", e006 / "sam2_g2" / stream_name),
            ("s1e_g2", e006 / "s1e_g2" / stream_name),
        ):
            p = _find_subdir([sub / sp for sp in sp_cands])
            if p:
                out[key] = p
        cnx = _find_subdir([wave2_root / "exp008b" / "cnxtopo" / stream_name / sp for sp in sp_cands])
        if cnx:
            out["cnxtopo"] = cnx
    return out


def decode_stream_dict(
    seg_specs: list[tuple[str, Path, float, int]],
    stems: list[str],
    thr: float = 0.60,
    offset: int = 0,
    conf_floor: float = 0.25,
    drop: float = 0.70,
    min_area: int = 0,
    conf_scale: float = 1.0,
    small_scale: float = 1.0,
) -> dict[str, list[tuple[float, tuple[int, int, int, int], np.ndarray]]]:
    """Decode blended stream for all requested stems into memory."""
    out = {}
    for s in stems:
        out[s] = bgs.load_blended_stream(
            seg_specs,
            s,
            thr=thr,
            offset=offset,
            conf_floor=conf_floor,
            drop=drop,
            min_area=min_area,
            conf_scale=conf_scale,
            small_scale=small_scale,
        )
    return out


def insts_to_rles(insts_by_stem: dict[str, list], conf_thr: float = 0.0) -> dict[str, list[str]]:
    rles = {}
    for s, insts in insts_by_stem.items():
        rles[s] = [bss.encode_crop_ascii(box, mask) for c, box, mask in insts if c >= conf_thr]
    return rles


def score_and_profile(rles_by_stem: dict[str, list[str]], stems: list[str], label: str) -> dict:
    counts = eh.score_rle_strs_dict(rles_by_stem, stems=stems)
    rep = eh.score_from_counts(counts, stems)
    prof = eh.detailed_error_profile(rles_by_stem, stems=stems)
    sm = prof["bins"]["<580"]
    print(
        f"[{label}] N={len(stems)} PQ={rep['pq']:.5f} SQ={rep['sq']:.5f} RQ={rep['rq']:.5f} "
        f"TP={rep['tp']} FP={rep['fp']} FN={rep['fn']} | "
        f"<580px: TP={sm['tp']} FP={sm['fp']} FN={sm['fn']} PQ_bin={sm['pq']:.4f}",
        flush=True,
    )
    return {"summary": rep, "profile": prof, "counts": counts}


def run_profile(stems: list[str], split: str) -> dict:
    streams = ["ms", "s", "main", "SAHI", "F5F"] + (["F3", "P6", "P4a", "G1", "A", "B", "F0", "P1"] if split == "val" else [])
    results = {}
    for sname in streams:
        dirs = resolve_stream_seg_dirs(sname, split)
        if not dirs:
            continue
        print(f"\n=== Stream {sname} ({split}) segs={sorted(dirs.keys())} ===", flush=True)
        for seg_k, sdir in sorted(dirs.items()):
            tta = 4 if sname in ("G1", "A", "B", "P1", "P4a", "F0", "F3", "P6") else 1
            insts = decode_stream_dict([(seg_k, sdir, 1.0, tta)], stems, thr=0.60, conf_floor=0.25)
            # Quick conf sweep at 0.25/0.30/0.35
            best_pq, best_c, best_rep = -1.0, 0.25, None
            for ct in (0.25, 0.30, 0.35, 0.40, 0.50):
                cnt = eh.score_rle_strs_dict(insts_to_rles(insts, ct), stems=stems)
                m = eh.score_from_counts(cnt, stems)
                if m["pq"] > best_pq:
                    best_pq, best_c, best_rep = m["pq"], ct, m
            res = score_and_profile(insts_to_rles(insts, best_c), stems, f"{sname}:{seg_k}@c>={best_c}")
            results[f"{sname}:{seg_k}"] = {"best_conf": best_c, "summary": res["summary"], "bins": res["profile"]["bins"]}
    return results


def run_stage2_cv(stream_name: str, stems: list[str], split: str) -> dict:
    """5-fold CV over Stage-2 continuous crop weights BEFORE thresholding (grouped by competition folds)."""
    dirs = resolve_stream_seg_dirs(stream_name, split)
    if len(dirs) < 2:
        print(f"[stage2_cv] {stream_name} only has {list(dirs.keys())}, skipping.", flush=True)
        return {}
    fold_of = data.folds()
    avail = sorted(dirs.keys())
    print(f"\n=== 5-Fold Stage-2 Blend CV on {stream_name} ({len(stems)} stems) segs={avail} ===", flush=True)

    # Build clean simplex candidates dynamically from available segmenters
    candidates = [("e4_only", {"e4": 1.0})] if "e4" in dirs else []
    if "e4_g1" in dirs:
        candidates.append(("e4_g1", {"e4_g1": 1.0}))
    if "e4x" in dirs:
        candidates.append(("e4x_only", {"e4x": 1.0}))
    if "e4x_g2" in dirs:
        candidates.append(("e4x_g2_only", {"e4x_g2": 1.0}))
    for w_name, base_k in (("base_e4x_g2", "e4x_g2"), ("base_e4_g1", "e4_g1"), ("base_e4", "e4")):
        if base_k not in dirs:
            continue
        if "s1e" in dirs and "sam2" in dirs:
            for wb, ws1, wsm in ((0.50, 0.25, 0.25), (0.40, 0.30, 0.30), (0.60, 0.20, 0.20)):
                candidates.append((f"{base_k}+{wb:.2f}_s1e+{ws1:.2f}_sam2+{wsm:.2f}", {base_k: wb, "s1e": ws1, "sam2": wsm}))
        if "s1e_g2" in dirs and "sam2_g2" in dirs and base_k == "e4x_g2":
            for wb, ws1, wsm in ((0.50, 0.25, 0.25), (0.40, 0.30, 0.30), (0.45, 0.35, 0.20)):
                candidates.append((f"g2_{wb:.2f}_s1e+{ws1:.2f}_sam2+{wsm:.2f}", {"e4x_g2": wb, "s1e_g2": ws1, "sam2_g2": wsm}))
        if "cnxtopo" in dirs and "s1e" in dirs and "sam2" in dirs:
            for wc in (0.15, 0.25, 0.35):
                rem = 1.0 - wc
                candidates.append((f"{base_k}+s1e+sam2+cnx{wc:.2f}", {base_k: 0.50 * rem, "s1e": 0.25 * rem, "sam2": 0.25 * rem, "cnxtopo": wc}))

    per_cand_counts = {}
    for cname, wmap in candidates:
        for thr in (0.55, 0.60):
            specs = [(k, dirs[k], float(w), 1) for k, w in wmap.items() if k in dirs and w > 0]
            # Preserve grown box ordering by placing e4x_g2/e4_g1 first
            specs.sort(key=lambda t: (0 if "g" in t[0] else 1, -t[2]))
            insts = decode_stream_dict(specs, stems, thr=thr, conf_floor=0.25)
            for ct in (0.28, 0.32, 0.36):
                tag = f"{cname}|thr={thr}|cf={ct}"
                cnt = eh.score_rle_strs_dict(insts_to_rles(insts, ct), stems=stems)
                per_cand_counts[tag] = cnt

    # Nested 5-fold CV evaluation (4 folds choose best tag -> score held-out fold k)
    pooled_cv_counts = {}
    chosen_tags = {}
    for fk in range(5):
        val_s = [s for s in stems if fold_of.get(s) == fk]
        tr_s = [s for s in stems if fold_of.get(s) != fk]
        if not val_s:
            continue
        best_tag = max(per_cand_counts, key=lambda t: eh.score_from_counts(per_cand_counts[t], tr_s)["pq"])
        chosen_tags[fk] = best_tag
        for s in val_s:
            pooled_cv_counts[s] = per_cand_counts[best_tag][s]
    cv_rep = eh.score_from_counts(pooled_cv_counts, stems)
    top5 = sorted(
        ((eh.score_from_counts(cnt, stems)["pq"], k) for k, cnt in per_cand_counts.items()),
        reverse=True,
    )[:5]
    print(f"[{stream_name}] 5-Fold Held-Out Stage-2 CV: PQ={cv_rep['pq']:.5f} SQ={cv_rep['sq']:.5f} RQ={cv_rep['rq']:.5f} chosen={chosen_tags}", flush=True)
    for pq_val, tag in top5:
        print(f"  top: {pq_val:.5f}  {tag}", flush=True)
    return {"cv": cv_rep, "chosen_by_fold": chosen_tags, "top5": top5}


def _default_blended_specs(dirs: dict[str, Path], pref_weights: dict[str, float] | None = None, tta: int = 4) -> list[tuple[str, Path, float, int]]:
    if pref_weights:
        specs = [(k, dirs[k], float(w), tta) for k, w in pref_weights.items() if k in dirs and w > 0]
        if specs:
            specs.sort(key=lambda t: (0 if "g" in t[0] else 1, -t[2]))
            return specs
    if "e4x_g2" in dirs and "s1e_g2" in dirs and "sam2_g2" in dirs:
        specs = [("e4x_g2", dirs["e4x_g2"], 0.45, tta), ("s1e_g2", dirs["s1e_g2"], 0.30, tta), ("sam2_g2", dirs["sam2_g2"], 0.25, tta)]
        if "cnxtopo" in dirs:
            specs.append(("cnxtopo", dirs["cnxtopo"], 0.20, tta))
        return specs
    if "e4_g1" in dirs and "s1e" in dirs and "sam2" in dirs:
        return [("e4_g1", dirs["e4_g1"], 0.45, tta), ("s1e", dirs["s1e"], 0.30, tta), ("sam2", dirs["sam2"], 0.25, tta)]
    if "e4" in dirs and "s1e" in dirs and "sam2" in dirs:
        specs = [("e4", dirs["e4"], 0.45, tta), ("s1e", dirs["s1e"], 0.30, tta), ("sam2", dirs["sam2"], 0.25, tta)]
        if "cnxtopo" in dirs:
            specs.append(("cnxtopo", dirs["cnxtopo"], 0.20, tta))
        return specs
    # Single fallback
    k = next(iter(sorted(dirs.keys())))
    return [(k, dirs[k], 1.0, tta)]


def run_track_a_predict(
    stems: list[str],
    split: str,
    sahi_scale: float = 1.0,
    f5f_scale: float = 0.90,
    vote_thr: float = 0.45,
    c_floor: float = 0.30,
    rescue_min_score: float = 0.58,
) -> dict[str, list[str]]:
    """Shared Track-A 5-Fold Full-Train cross-family predictor for both OOF ('oof') and 'test'."""
    decoded = {}
    weights = {"ms": 1.25, "s": 1.0, "main": 0.90, "SAHI": 1.0, "F5F": 1.10}
    for sname in ("ms", "s", "main", "SAHI", "F5F"):
        dirs = resolve_stream_seg_dirs(sname, split)
        if not dirs:
            continue
        meta = eh.STREAM_META.get(sname, {})
        tta = int(meta.get("tta", 4))
        thr, off, cf, drp, ma = meta.get("cell", (0.60, 0, 0.25, 0.70, 0))
        specs = _default_blended_specs(dirs, tta=tta)
        sc = sahi_scale if sname == "SAHI" else (f5f_scale if sname == "F5F" else 1.0)
        decoded[sname] = decode_stream_dict(specs, stems, thr=thr, offset=off, conf_floor=cf, drop=drp, min_area=ma, conf_scale=sc)
        print(f"[TrackA decode:{split}] {sname}: segs={[sp[0] for sp in specs]} sc={sc}", flush=True)

    rles = {}
    for st in stems:
        st_list = [(k, weights[k], decoded[k][st]) for k in ("ms", "s", "main", "SAHI", "F5F") if k in decoded]
        rles[st] = bgs.weighted_consensus_stem(st_list, match_iou=0.40, vote_thr=vote_thr, c_floor=c_floor)
    return rles


def run_track_a_707(stems: list[str], split: str = "oof") -> dict:
    """Evaluate True 5-Fold Full-Train Multi-Family Pipeline (ms/s/main + SAHI@1024 + FRCNN-5F)."""
    decoded = {}
    for sname in ("ms", "s", "main", "SAHI", "F5F"):
        dirs = resolve_stream_seg_dirs(sname, split)
        if not dirs:
            continue
        meta = eh.STREAM_META.get(sname, {})
        tta = int(meta.get("tta", 4))
        thr, off, cf, drp, ma = meta.get("cell", (0.60, 0, 0.25, 0.70, 0))
        specs = _default_blended_specs(dirs, tta=tta)
        decoded[sname] = decode_stream_dict(specs, stems, thr=thr, offset=off, conf_floor=cf, drop=drp, min_area=ma)
        print(f"[TrackA decode] {sname}: segs={[sp[0] for sp in specs]}", flush=True)

    fold_of = data.folds()
    results = {}
    if all(k in decoded for k in ("ms", "s", "main")):
        rles_base = {
            st: bgs.weighted_consensus_stem([("ms", 1.25, decoded["ms"][st]), ("s", 1.0, decoded["s"][st]), ("main", 0.9, decoded["main"][st])], vote_thr=0.45, c_floor=0.30)
            for st in stems
        }
        results["ms+s+main"] = score_and_profile(rles_base, stems, "TrackA ms+s+main")["summary"]

    if "SAHI" in decoded or "F5F" in decoded:
        grid = []
        sahi_scales = (0.85, 1.0, 1.15) if "SAHI" in decoded else (1.0,)
        f5f_scales = (0.75, 0.90, 1.05) if "F5F" in decoded else (1.0,)
        for s_sc in sahi_scales:
            for f_sc in f5f_scales:
                for cf in (0.28, 0.31, 0.34):
                    tag = f"sahi={s_sc}_f5f={f_sc}_cf={cf}"
                    rles = {}
                    for st in stems:
                        st_list = [("ms", 1.25, decoded["ms"][st]), ("s", 1.0, decoded["s"][st]), ("main", 0.9, decoded["main"][st])]
                        if "SAHI" in decoded:
                            st_list.append(("SAHI", 1.0, [(min(0.99, c * s_sc), b, m) for c, b, m in decoded["SAHI"][st]]))
                        if "F5F" in decoded:
                            st_list.append(("F5F", 1.1, [(min(0.99, c * f_sc), b, m) for c, b, m in decoded["F5F"][st]]))
                        rles[st] = bgs.weighted_consensus_stem(st_list, match_iou=0.40, vote_thr=0.45, c_floor=cf)
                    grid.append((tag, eh.score_rle_strs_dict(rles, stems=stems)))
        pooled = {}
        for fk in range(5):
            val_s = [s for s in stems if fold_of.get(s) == fk]
            tr_s = [s for s in stems if fold_of.get(s) != fk]
            if not val_s:
                continue
            b_tag, _ = max(grid, key=lambda t: eh.score_from_counts(t[1], tr_s)["pq"])
            b_cnt = dict(grid)[b_tag]
            for s in val_s:
                pooled[s] = b_cnt[s]
        results["TrackA_CV"] = eh.score_from_counts(pooled, stems)
        print(f"[TrackA 5F-CV Full] PQ={results['TrackA_CV']['pq']:.5f} TP={results['TrackA_CV']['tp']} FP={results['TrackA_CV']['fp']} FN={results['TrackA_CV']['fn']}", flush=True)
    return results


def run_track_b_predict(stems: list[str], split: str) -> dict[str, list[str]]:
    """Shared Clean Super-Ensemble (G17_no_F1 + upgraded Wave-2) for both 'val' (Fold-0) and 'test'."""
    yolo_p = bss.FP_ROOT / ("yolo_val_raw.json" if split == "val" else "yolo_test_raw.json")
    with open(yolo_p) as f:
        yolo_raw = json.load(f)

    core_keys = ("G1", "A", "B", "P1", "P4a", "F0")
    cores = {k: {s: bss.paint_ts_cropped(bss.load_raw_npz(eh.STREAM_META[k][split] / f"{s}.npz", tta=eh.STREAM_META[k]["tta"]), *eh.STREAM_META[k]["cell"]) for s in stems} for k in core_keys}
    yolo_s = {s: bss.load_yolo_cropped(yolo_raw, s, 0.2, -1, 0.7) for s in stems}

    donors = {}
    has_ms_upgrade = len(resolve_stream_seg_dirs("ms", split)) > 1
    for k in ("ms", "s", "main", "F3", "P6", "P4a", "SAHI", "F5F"):
        dirs = resolve_stream_seg_dirs(k, split)
        if not dirs:
            continue
        meta = eh.STREAM_META.get(k, {})
        tta = int(meta.get("tta", 4))
        thr, off, cf, drp, ma = meta.get("cell", (0.60, 0, 0.25, 0.70, 0))
        specs = _default_blended_specs(dirs, tta=tta)
        donors[k] = decode_stream_dict(specs, stems, thr=thr, offset=off, conf_floor=cf, drop=drp, min_area=ma)
        print(f"[TrackB decode:{split}] {k}: segs={[sp[0] for sp in specs]} cf={cf} tta={tta}", flush=True)

    rles = {}
    for s in stems:
        c_list = [cores[k][s] for k in core_keys] + [yolo_s[s]]
        if "ms" in donors and has_ms_upgrade:
            c_list.append(donors["ms"][s])
        if "SAHI" in donors:
            c_list.append(donors["SAHI"][s])
        if "F5F" in donors:
            c_list.append(donors["F5F"][s])
        has_w2 = ("SAHI" in donors or "F5F" in donors or has_ms_upgrade)
        ex = {k: donors[k][s] for k in ("ms", "s", "main", "F3") if k in donors}
        ex_w = {"ms": 0.22 if has_w2 else 0.18, "s": 0.20 if has_w2 else 0.18, "main": 0.12, "F3": 0.08 if has_w2 else 0.06}
        rescue = []
        if "SAHI" in donors:
            rescue.append(("SAHI_sm", 1.0, [(c, b, m) for c, b, m in donors["SAHI"][s] if int(m.sum()) < 580]))
        if "F5F" in donors:
            rescue.append(("F5F", 0.85, donors["F5F"][s]))
        rles[s] = bgs.super_ensemble_upgraded_stem(c_list, ex, extra_weights=ex_w, rescue_streams=rescue or None, rescue_min_score=0.58)
    return rles


def run_track_b_fold0() -> dict:
    stems = list(eh.FOLD0_STEMS)
    rles = run_track_b_predict(stems, "val")
    rep = bgs.evaluate_rles_with_nested_cv(rles, stems)
    print(f"[TrackB Clean Fold-0] {json.dumps(rep)}", flush=True)
    return rep


def build_submission(track: str, out_csv: Path, metrics_json: Path | None = None) -> dict:
    test_stems = list(eh.TEST_STEMS)
    if track == "track_b":
        val_rep = run_track_b_fold0()
        test_rles_str = run_track_b_predict(test_stems, "test")
    else:
        val_rep = run_track_a_707(sorted(eh.GT.keys()), "oof")
        test_rles_str = run_track_a_predict(test_stems, "test")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rle_dicts = {s: [{"size": [2048, 2048], "counts": str(c)} for c in strs] for s, strs in test_rles_str.items()}
    submission.write_csv(out_csv, rle_dicts)
    submission.validate(out_csv, test_stems)
    shift = test_shift_audit.audit_csv(out_csv, REPO / "results" / "submission.csv")
    if metrics_json is not None:
        metrics_json.parent.mkdir(parents=True, exist_ok=True)
        metrics_json.write_text(json.dumps({"val": val_rep, "shift": shift}, indent=2))
    print(f"[build_sub:{track}] wrote {out_csv}\nShift: {json.dumps(shift, indent=2)}", flush=True)
    return {"val": val_rep, "shift": shift}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["unpack", "profile", "stage2_cv", "track_a", "track_b", "build_sub", "all"], default="all")
    ap.add_argument("--stream", default="ms")
    ap.add_argument("--split", choices=["oof", "val"], default="oof")
    ap.add_argument("--track", choices=["track_a", "track_b"], default="track_b")
    ap.add_argument("--out-csv", type=Path, default=REPO / "gap_research" / "submissions" / "wave2_candidate.csv")
    ap.add_argument("--metrics-json", type=Path, default=REPO / "gap_research" / "submissions" / "wave2_candidate_metrics.json")
    args = ap.parse_args()

    if args.mode in ("unpack", "all"):
        print("Unpacked:", unpack_archives(), flush=True)
    stems = sorted(eh.GT.keys()) if args.split == "oof" else list(eh.FOLD0_STEMS)
    if args.mode in ("profile", "all"):
        run_profile(stems, args.split)
    if args.mode in ("stage2_cv", "all"):
        run_stage2_cv(args.stream, stems, args.split)
    if args.mode in ("track_a", "all"):
        run_track_a_707(sorted(eh.GT.keys()), "oof")
    if args.mode in ("track_b", "all"):
        run_track_b_fold0()
    if args.mode == "build_sub":
        build_submission(args.track, args.out_csv, args.metrics_json)


if __name__ == "__main__":
    main()
