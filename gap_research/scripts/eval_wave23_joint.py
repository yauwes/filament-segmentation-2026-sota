"""Unified Wave-2 & Wave-3 Unpacker, Stream Auto-Router, and Joint Post-Proc Optimizer.

Unpacks Kaggle/Colab tarballs (including `a100_out` routing + `out_cnxtopo -> cnxtopo` symlink),
discovers all clean 5-fold Stage-1/2 streams, evaluates both Track-A (707 OOF) and Track-B (145 Fold-0
Nested Date-CV vs E19=0.48563) + <580px bin PQ using `small_px=120` (`t68_d90_sm20_v52_c28`),
and emits submission CSV + metrics JSON ready for `auto_submit.py`.
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "gap_research" / "scripts"))

import build_gap_submission as bgs  # noqa: E402
import build_super_submission as bss  # noqa: E402
import eval_harness as eh  # noqa: E402
import eval_wave2 as ew2  # noqa: E402
import test_shift_audit  # noqa: E402
from filseg import metric, submission  # noqa: E402


WAVE2_ROOT = Path("/private/tmp/gap_wave2_cache")


def route_and_unpack(root: Path = WAVE2_ROOT) -> dict[str, list[str]]:
    """Route a100_out tarballs cleanly by prefix and unpack all Wave-2/3 archives."""
    a100_dir = root / "a100_out"
    if a100_dir.exists():
        for tgz in sorted(a100_dir.glob("*.tar.gz")):
            if tgz.name.startswith("exp008a_"):
                dst = root / "exp008a"
            elif tgz.name.startswith("exp008b_"):
                dst = root / "exp008b"
            elif tgz.name.startswith("exp009a_"):
                dst = root / "exp009a"
            elif tgz.name.startswith("exp009b_"):
                dst = root / "exp009b"
            else:
                continue
            dst.mkdir(parents=True, exist_ok=True)
            link = dst / tgz.name
            if not link.exists():
                link.symlink_to(tgz)

    unpacked: dict[str, list[str]] = {}
    for sub in ("exp006", "exp007", "exp008a", "exp008b", "exp009a", "exp009b"):
        sdir = root / sub
        if not sdir.exists():
            continue
        names = []
        for tgz in sorted(sdir.glob("*.tar.gz")):
            if tgz.name.startswith("."):
                continue
            names.append(tgz.name)
            marker = sdir / f"{tgz.name}.unpacked"
            if not marker.exists():
                with tarfile.open(tgz, "r:gz") as tf:
                    tf.extractall(sdir)
                marker.write_text("ok\n")
        if sub == "exp006" and (sdir / "e4x_root").exists() and not (sdir / "e4x").exists():
            (sdir / "e4x").symlink_to(sdir / "e4x_root")
        if sub == "exp008b" and (sdir / "out_cnxtopo").exists() and not (sdir / "cnxtopo").exists():
            (sdir / "cnxtopo").symlink_to(sdir / "out_cnxtopo")
        if names:
            unpacked[sub] = names
    return unpacked


def bin_pq_summary(rles_by_stem: dict[str, list[str]], stems: list[str], recs: dict | None = None) -> dict:
    if recs is None:
        recs = metric.load_records(stems=stems)
    bins = {b: dict(tp=0, fp=0, fn=0, iou=0.0) for b in ("lt580", "b580_2k", "ge2k")}
    for s in stems:
        pr = [{"size": [2048, 2048], "counts": c.encode()} for c in rles_by_stem[s]]
        pa = mu.area(pr) if pr else []
        for b, lo, hi in (("lt580", 0, 580), ("b580_2k", 580, 2000), ("ge2k", 2000, 10**9)):
            ps = [p for p, a in zip(pr, pa) if lo <= a < hi]
            rs = [[g for g in gt if lo <= mu.area(g) < hi] for gt in recs[s]]
            c = metric.counts(ps, rs)
            for k in bins[b]:
                bins[b][k] += c[k]
    out = {}
    for b, c in bins.items():
        denom = c["tp"] + 0.5 * (c["fp"] + c["fn"])
        out[b] = dict(pq=round(c["iou"] / denom, 5) if denom else 0.0, tp=c["tp"], fp=c["fp"], fn=c["fn"])
    return out


def discover_clean_streams(split: str, root: Path = WAVE2_ROOT) -> list[tuple[str, float, dict[str, Path]]]:
    """Discover all available clean 5-fold Stage-1 families and their blended Stage-2 dirs."""
    specs = []
    for sname, wt in (("ms", 1.25), ("s", 1.00), ("main", 0.90), ("SAHI", 1.15), ("F5F", 0.95)):
        dirs = ew2.resolve_stream_seg_dirs(sname, split, wave2_root=root)
        if dirs:
            specs.append((sname, wt, dirs))
    # Wave-3 EXP-009a/009b extensions
    enh_dirs = {}
    e9a, e9b = root / "exp009a", root / "exp009b"
    if e9a.exists():
        for tag in ("enh_e4_g1", "enh_e4", "enh_s1e", "enh_sam2"):
            d = e9a / tag / split
            if d.exists():
                enh_dirs[tag] = d
    if enh_dirs:
        specs.append(("ENH", 1.10, enh_dirs))
    # Inject EXP-009b SegFormer masks into ms/s/main/F3/P6/P4a where present
    if e9b.exists():
        sf_base = e9b / "segformer"
        for sname, _, dirs in specs:
            for sp_alt in (split, "val" if split == "oof" else split):
                cand = sf_base / sname / sp_alt
                if cand.exists() and any(cand.glob("*.npz")):
                    dirs["segformer"] = cand
                    break
    return specs


DEFAULT_SEG_SPECS = (
    ("e4", 0.45, 4),
    ("e4_g1", 0.40, 4),
    ("e4x", 0.50, 4),
    ("e4x_g2", 0.40, 4),
    ("enh_e4", 0.45, 4),
    ("enh_e4_g1", 0.40, 4),
    ("s1e", 0.30, 4),
    ("enh_s1e", 0.30, 4),
    ("sam2", 0.25, 1),
    ("enh_sam2", 0.25, 1),
    ("cnxtopo", 0.30, 4),
    ("segformer", 0.30, 4),
)


def build_blended_specs(dirs: dict[str, Path]) -> list[tuple[str, Path, float, int]]:
    out = []
    seen = set()
    for key, wt, tta in DEFAULT_SEG_SPECS:
        if key in dirs:
            out.append((key, dirs[key], wt, tta))
            seen.add(key)
    for key in sorted(dirs):
        if key not in seen:
            out.append((key, dirs[key], 0.25, 4))
    return out


def load_blended_clamped(
    seg_dirs: list[tuple[str, Path, float, int]],
    stem: str,
    thr: float = 0.68,
    conf_floor: float = 0.25,
    drop: float = 0.90,
    min_area: int = 100,
    small_scale: float = 1.20,
    small_px: int = 120,
) -> list[tuple[float, tuple[int, int, int, int], np.ndarray]]:
    """Blend Stage-2 probabilities while clamping _g2 masks strictly to primary box0."""
    loaded = []
    for sname, sdir, w, tta in seg_dirs:
        if w <= 0:
            continue
        p = sdir / f"{stem}.npz"
        if not p.exists():
            continue
        conf, box, probs = bgs._unpack_npz(p, tta_prefer=tta)
        loaded.append((sname, w, conf, box, probs))
    if not loaded:
        return []
    _, _, conf0, box0, _ = loaded[0]
    if len(conf0) == 0:
        return []

    occ = np.zeros((2048, 2048), dtype=bool)
    t_u8 = int(thr * 255.0 + 0.5)
    out = []
    for idx in np.argsort(-conf0):
        raw_c = float(conf0[idx])
        bx0, by0, bx1, by1 = [int(v) for v in box0[idx]]
        eff_c = raw_c * (small_scale if max(bx1 - bx0, by1 - by0) < small_px else 1.0)
        if eff_c < conf_floor or bx1 <= bx0 or by1 <= by0:
            continue
        uh, uw = by1 - by0, bx1 - bx0
        acc = np.zeros((uh, uw), dtype=np.float32)
        wsum = np.zeros((uh, uw), dtype=np.float32)
        for _, w, _, b_arr, p_list in loaded:
            if idx >= len(b_arr):
                continue
            sx1, sy1, sx2, sy2 = [int(v) for v in b_arr[idx]]
            if sx2 <= sx1 or sy2 <= sy1:
                continue
            pr = p_list[idx]
            if pr.shape != (sy2 - sy1, sx2 - sx1):
                pr = cv2.resize(pr, (sx2 - sx1, sy2 - sy1), interpolation=cv2.INTER_LINEAR)
            ix1, iy1, ix2, iy2 = max(bx0, sx1), max(by0, sy1), min(bx1, sx2), min(by1, sy2)
            if ix2 > ix1 and iy2 > iy1:
                acc[iy1 - by0 : iy2 - by0, ix1 - bx0 : ix2 - bx0] += (w / 255.0) * pr[iy1 - sy1 : iy2 - sy1, ix1 - sx1 : ix2 - sx1]
                wsum[iy1 - by0 : iy2 - by0, ix1 - bx0 : ix2 - bx0] += w
        if not (wsum > 0).any():
            continue
        prob_u8 = np.where(wsum > 0, (acc / np.maximum(wsum, 1e-6)) * 255.0 + 0.5, 0.0).astype(np.uint8)
        m = prob_u8 > t_u8
        v = bss.trim(eff_c, (bx0, by0, bx1, by1), m)
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


def _append_disjoint_rescue(
    rles_by_stem: dict[str, list[str]],
    rescue_by_stem: dict[str, list[tuple[float, tuple[int, int, int, int], np.ndarray]]],
    stems: list[str],
    min_conf: float = 0.22,
    max_area: int = 400,
    min_area: int = 100,
    drop: float = 0.90,
) -> tuple[dict[str, list[str]], int]:
    out = {}
    added = 0
    for s in stems:
        bm = np.zeros((2048, 2048), dtype=bool)
        for r in rles_by_stem[s]:
            bm |= mu.decode({"size": [2048, 2048], "counts": r.encode()}).astype(bool)
        cur = list(rles_by_stem[s])
        for c, (x1, y1, x2, y2), m in rescue_by_stem[s]:
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
                    added += 1
        out[s] = cur
    return out, added


def build_track_a_rles(
    stems: list[str],
    split: str,
    thr: float = 0.68,
    drop: float = 0.90,
    small_scale: float = 1.20,
    small_px: int = 120,
    vote_thr: float = 0.52,
    c_floor: float = 0.28,
    match_iou: float = 0.40,
    root: Path = WAVE2_ROOT,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    e6 = root / "exp006"
    sp_dir = "oof" if split in ("oof", "val") else "test"
    specs_map = {
        "ms": (1.35, [("e4", REPO / "work" / "cache" / "ms" / sp_dir, 0.10, 4), ("sam2", e6 / "sam2" / "ms" / sp_dir, 0.55, 4), ("s1e_g2", e6 / "s1e_g2" / "ms" / sp_dir, 0.25, 4), ("e4x", e6 / "e4x" / "e4x" / "ms" / sp_dir, 0.10, 4)]),
        "s": (1.05, [("e4", REPO / "work" / "cache" / "s" / sp_dir, 0.30, 4), ("s1e_g2", e6 / "s1e_g2" / "s" / sp_dir, 0.50, 4), ("e4x", e6 / "e4x" / "e4x" / "s" / sp_dir, 0.20, 4)]),
        "main": (0.95, [("e4", REPO / "work" / "cache" / "main" / sp_dir, 0.30, 4), ("s1e_g2", e6 / "s1e_g2" / "main" / sp_dir, 0.50, 4), ("e4x", e6 / "e4x" / "e4x" / "main" / sp_dir, 0.20, 4)]),
    }
    loaded = {}
    summary = {}
    for sname, (wt, seg_specs) in specs_map.items():
        seg_avail = [(k, p, w, tta) for k, p, w, tta in seg_specs if p.exists()]
        summary[sname] = [d[0] for d in seg_avail]
        cf = 0.25 if sname == "main" else 0.28
        loaded[sname] = (
            wt,
            {
                s: load_blended_clamped(
                    seg_avail, s, thr=thr, conf_floor=cf, drop=drop,
                    min_area=100, small_scale=small_scale, small_px=small_px,
                )
                for s in stems
            },
        )
    rles = {
        s: bgs.weighted_consensus_stem(
            [(sname, wt, dmap[s]) for sname, (wt, dmap) in loaded.items()],
            match_iou=match_iou,
            vote_thr=vote_thr,
            c_floor=c_floor,
            min_area=100,
            drop=drop,
        )
        for s in stems
    }
    sahi_dirs = ew2.resolve_stream_seg_dirs("SAHI", split, wave2_root=root)
    if sahi_dirs:
        sahi_specs = [(k, sahi_dirs[k], 0.25, 4) for k in ("e4", "e4_g1", "s1e", "sam2") if k in sahi_dirs]
        sahi_dec = {s: bgs.load_blended_stream(sahi_specs, s, thr=thr, conf_floor=0.20, drop=drop, min_area=100) for s in stems}
        rles, added = _append_disjoint_rescue(rles, sahi_dec, stems, min_conf=0.22, max_area=400, min_area=100, drop=drop)
        summary["SAHI_rescue"] = [f"added={added}"]
    return rles, summary


def build_track_b_rles(
    stems: list[str],
    split: str,
    root: Path = WAVE2_ROOT,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Clean 12-stream EXP-006 + EXP-007 Track-B (zero F1; Fold-0 PQ=0.48637)."""
    assert split in ("val", "test")
    yolo_p = bss.FP_ROOT / ("yolo_val_raw.json" if split == "val" else "yolo_test_raw.json")
    with open(yolo_p) as f:
        yolo_raw = json.load(f)

    cores_68 = {}
    for k in ("G1", "A", "B", "P1", "F0"):
        m = eh.STREAM_META[k]
        _, off, cf, _, ma = m["cell"]
        cores_68[k] = {
            s: bss.paint_ts_cropped(bss.load_raw_npz(m[split] / f"{s}.npz", tta=m["tta"]), 0.68, off, cf, 0.90, max(100, ma))
            for s in stems
        }
    yolo_90 = {s: bss.load_yolo_cropped(yolo_raw, s, 0.2, -1, 0.90) for s in stems}

    up_cfgs = {
        "ms": (["sam2", "s1e", "e4"], [0.50, 0.35, 0.15], 0.25, 1.20),
        "s": (["sam2", "s1e"], [0.50, 0.50], 0.25, 1.20),
        "main": (["sam2", "s1e", "e4"], [0.45, 0.35, 0.20], 0.25, 1.20),
        "F3": (["sam2", "e4x", "s1e"], [0.50, 0.25, 0.25], 0.75, 1.00),
        "P6": (["sam2", "e4x", "s1e"], [0.50, 0.25, 0.25], 0.25, 1.00),
        "P4a": (["sam2", "s1e"], [0.55, 0.45], 0.30, 1.00),
    }
    up = {}
    summary = {}
    for k, (segs, ws, cf, sm) in up_cfgs.items():
        dirs = ew2.resolve_stream_seg_dirs(k, split, wave2_root=root)
        specs = [(sn, dirs[sn], w, 4) for sn, w in zip(segs, ws) if sn in dirs]
        summary[k] = [d[0] for d in specs]
        up[k] = {
            s: bgs.load_blended_stream(
                specs, s, thr=0.68, conf_floor=cf, drop=0.90, min_area=100, small_scale=sm, small_px=120,
            )
            for s in stems
        }

    sahi_dirs = ew2.resolve_stream_seg_dirs("SAHI", split, wave2_root=root)
    sahi_specs = [(k, sahi_dirs[k], 0.25, 4) for k in ("e4", "e4_g1", "s1e", "sam2") if k in sahi_dirs]
    sahi_dec = {s: bgs.load_blended_stream(sahi_specs, s, thr=0.68, conf_floor=0.22, drop=0.90, min_area=100) for s in stems}
    summary["SAHI"] = [d[0] for d in sahi_specs]

    ex_w = {"ms": 0.25, "s": 0.22, "main": 0.14, "F3": 0.12, "P6": 0.12, "SAHI": 0.12}
    clist = [cores_68[k] for k in ("G1", "A", "B", "P1", "F0")] + [up["P4a"], up["F3"], up["P6"], yolo_90]
    rles = {}
    for s in stems:
        ex = {k: up[k][s] for k in ("ms", "s", "main", "F3", "P6")}
        ex["SAHI"] = sahi_dec[s]
        resc = [("SAHI_sm", 1.0, [(c, b, m) for c, b, m in sahi_dec[s] if int(m.sum()) <= 580])]
        rles[s] = bgs.super_ensemble_upgraded_stem(
            [c[s] for c in clist],
            ex,
            extra_weights=ex_w,
            vr=0.50,
            c_floor=0.315,
            boost_both=0.06,
            drop=0.70,
            min_area=100,
            rescue_streams=resc,
            rescue_min_score=0.28,
        )
    return rles, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["track_a", "track_b"], default="track_b")
    ap.add_argument("--split", choices=["oof", "val", "test"], default="val")
    ap.add_argument("--dry-check", action="store_true")
    ap.add_argument("--thr", type=float, default=0.68)
    ap.add_argument("--drop", type=float, default=0.90)
    ap.add_argument("--small-scale", type=float, default=1.20)
    ap.add_argument("--vote-thr", type=float, default=0.52)
    ap.add_argument("--c-floor", type=float, default=0.28)
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--metrics-json", type=Path, default=None)
    args = ap.parse_args()

    t0 = time.time()
    unpacked = route_and_unpack()
    print(f"Unpacked archives: {unpacked}", flush=True)

    if args.track == "track_b":
        val_stems = list(eh.FOLD0_STEMS)
        rles_val, seg_summary = build_track_b_rles(val_stems, "val")
        f0 = bgs.evaluate_rles_with_nested_cv(rles_val, val_stems)
        bins = bin_pq_summary(rles_val, val_stems)
        print(f"[Track-B Fold-0 Clean] ({seg_summary}) in {time.time()-t0:.1f}s -> {json.dumps(f0)}", flush=True)
        print("Bins:", json.dumps(bins), flush=True)
        val_rep = f0
        if args.split == "test" and args.out_csv:
            t1 = time.time()
            test_stems = list(eh.TEST_STEMS)
            rles_test, _ = build_track_b_rles(test_stems, "test")
            print(f"Built {len(rles_test)} test stems in {time.time()-t1:.1f}s", flush=True)
            args.out_csv.parent.mkdir(parents=True, exist_ok=True)
            rle_dicts = {s: [{"size": [2048, 2048], "counts": str(c)} for c in strs] for s, strs in rles_test.items()}
            submission.write_csv(args.out_csv, rle_dicts)
            probs = submission.validate(args.out_csv, test_stems)
            assert not probs, f"Submission validation failed: {probs}"
            audit = test_shift_audit.audit_csv(args.out_csv, REPO / "results" / "submission.csv")
            print("Test shift audit:", json.dumps(audit, indent=2), flush=True)
            if args.metrics_json:
                args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
                args.metrics_json.write_text(json.dumps({"val": val_rep, "bins": bins, "shift": audit}, indent=2))
        elif args.metrics_json:
            args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_json.write_text(json.dumps({"val": val_rep, "bins": bins}, indent=2))
        return

    stems = list(eh.TEST_STEMS) if args.split == "test" else (list(eh.FOLD0_STEMS) if args.dry_check else sorted(eh.GT.keys()))
    rles, seg_summary = build_track_a_rles(
        stems,
        args.split,
        thr=args.thr,
        drop=args.drop,
        small_scale=args.small_scale,
        vote_thr=args.vote_thr,
        c_floor=args.c_floor,
    )
    print(f"Built Track-A {len(rles)} stems ({seg_summary}) in {time.time()-t0:.1f}s", flush=True)

    if args.split in ("oof", "val"):
        cnt = eh.score_rle_strs_dict(rles, stems=stems)
        rep = eh.score_from_counts(cnt, stems=stems)
        bins = bin_pq_summary(rles, stems)
        print(f"[Track-A {len(stems)}stems] PQ={rep['pq']:.5f} SQ={rep['sq']:.5f} RQ={rep['rq']:.5f} TP={rep['tp']} FP={rep['fp']} FN={rep['fn']}", flush=True)
        print("Bins:", json.dumps(bins), flush=True)
        if len(stems) == 145:
            f0 = bgs.evaluate_rles_with_nested_cv(rles, stems)
            print("Fold0 Nested-CV:", json.dumps(f0), flush=True)
        if args.metrics_json:
            args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_json.write_text(json.dumps({"val": rep, "bins": bins}, indent=2))
    elif args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        rle_dicts = {s: [{"size": [2048, 2048], "counts": str(c)} for c in strs] for s, strs in rles.items()}
        submission.write_csv(args.out_csv, rle_dicts)
        probs = submission.validate(args.out_csv, list(eh.TEST_STEMS))
        assert not probs, f"Submission validation failed: {probs}"
        audit = test_shift_audit.audit_csv(args.out_csv, REPO / "results" / "submission.csv")
        print("Test shift audit:", json.dumps(audit, indent=2), flush=True)
        if args.metrics_json:
            args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_json.write_text(json.dumps({"shift": audit}, indent=2))


if __name__ == "__main__":
    main()
