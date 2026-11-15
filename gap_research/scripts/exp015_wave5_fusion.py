"""EXP-015: Wave-5 4-Family Solar Filament Fusion + PQ-Optimal Disjoint MWIS Rescorer.

Integrates cleanly (100% leak-free, date-grouped 5-fold OOF <-> 180 Test):
  1. Track-A++ continuous multi-stream clusters (ms, s, main, SAHI) + exact Track-A Clean anchors.
  2. EXP-012 Full-Frame Dense Physics (mit_b3 + cnx_upp): foreground p_dfg + boundary ridge p_dedge + cc_proposals.
  3. EXP-013a Stage-1 SOTA Hybrid/Transformer Detectors (YOLO26x-seg + RT-DETR-X w/ GONG SSL): boxes + s1masks.
  4. EXP-013c Stage-2 High-Res 5-ch Crop Refiners (UNet-MiT-B4 + ConvNeXt-Small EdgeGuided MHSA on SolarRestormer).
  5. Literature-verified PQ geometry:
       - Exclusive greedy dense-map tail claiming (fixes rectangular crop truncation without merging).
       - Non-inflating gap bridging `bridge_and_label(gap=6)` (fixes fragmented GT without mask dilation).
       - 1-px erosion refinement candidates on small/medium filaments (<1400 px, fixes 2.96x over-dilation).
       - Dual-threshold boundary-aware hysteresis discovery on p_dfg * (1 - 0.45 * p_dedge) (fixes 1,663 zero-ov FNs).

All candidate variants (Track-A + Wave-5) receive 86 leak-free features and 4x48x48 multimodal patches,
scored by Date-Grouped 5-Fold GBDT Hurdle + VisionCropTransformer, and resolved via exact MWIS PQ search.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import concurrent.futures
import json
import math
import pickle
import shutil
import sys
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

cv2.setNumThreads(1)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "gap_research" / "scripts"))

import build_super_submission as bss  # noqa: E402
import exp011b_extract_candidates as e11x  # noqa: E402
import exp011b_hurdle_antichain as e11b  # noqa: E402
import exp011c_transformer_rescorer as e11c  # noqa: E402
import test_shift_audit as tsa  # noqa: E402
from filseg import data, metric, submission  # noqa: E402

FULL = 2048
HALF = 1024
WAVE5_DEFAULT = Path("/private/tmp/gap_wave5_cache")
POOL_DEFAULT = Path("/private/tmp/exp015_wave5_pool.pkl")

KIND_LIST_W5 = e11c.KIND_LIST + ("w5_b48", "w5_b56", "w5_claim", "w5_link", "w5_erode1", "w5_disc")
W5_EXTRA_FEATURES = [
    "yolo26_conf",
    "rtdetr_conf",
    "cc_conf",
    "n_w5_det",
    "s2_mit_mean",
    "s2_mit_iou",
    "s2_cnx_mean",
    "s2_cnx_iou",
    "y26m_iou",
    "dfg_mean",
    "dfg_q75",
    "dfg_core",
    "dfg_rim",
    "dedge_mean",
    "dedge_rim",
    "dfg_iou50",
    "is_w5_cand",
    "m2f_conf",
    "m2f_inst_iou",
    "m2f_fg_mean",
    "m2f_fg_iou",
]
FEATURE_NAMES_W5 = e11x.FEATURE_NAMES + W5_EXTRA_FEATURES

_KERNEL_CROSS3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
_KERNEL_BRIDGE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
_KERNEL_CLAIM = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
_W5_INDEX_GLOBAL: dict = {}


# ---------------------------------------------------------------------------
# Wave-5 Archive Unpacking & Stream Discovery
# ---------------------------------------------------------------------------

def unpack_wave5_archives(w5_root: Path) -> list[str]:
    """Extract each Wave-5 tarball into an isolated dir so identically-named roots never clobber one another."""
    if not w5_root.exists():
        return []
    extracted: list[str] = []
    for tgz in sorted(w5_root.rglob("*.tar.gz")):
        if tgz.name.startswith("."):
            continue
        stem = tgz.name[:-7] if tgz.name.endswith(".tar.gz") else tgz.stem
        out_sub = tgz.parent / f"_unpacked_{stem}"
        marker = out_sub / ".unpacked_ok"
        if not marker.exists():
            if out_sub.exists():
                shutil.rmtree(out_sub, ignore_errors=True)
            out_sub.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tgz, "r:gz") as tf:
                tf.extractall(out_sub)
            marker.write_text("ok\n")
            extracted.append(str(tgz.relative_to(w5_root)))
    return extracted


def _collect_npz_tree(
    w5_root: Path,
    patterns: str | list[str],
    is_branched: bool = False,
    deprioritize_substrs: tuple[str, ...] = (),
) -> dict:
    """Index {pattern}/{oof,test}/{stem}.npz (or branched) with explicit fallback ordering."""
    out: dict = {}
    if not w5_root.exists():
        return out
    pats = [patterns] if isinstance(patterns, str) else list(patterns)
    seen_dirs: set[Path] = set()
    candidates: list[Path] = []
    for pat in pats:
        for p in w5_root.rglob(pat):
            if p.is_dir():
                rp = p.resolve()
                if rp not in seen_dirs:
                    seen_dirs.add(rp)
                    candidates.append(p)
    candidates.sort(key=lambda p: (any(sub in str(p) for sub in deprioritize_substrs), len(str(p)), str(p)))
    for base in candidates:
        if is_branched:
            for br in sorted(base.iterdir()):
                if not br.is_dir():
                    continue
                for sp_raw in ("oof", "val", "test"):
                    sub = br / sp_raw
                    if not sub.exists():
                        continue
                    sp = "oof" if sp_raw in ("oof", "val") else "test"
                    bmap = out.setdefault(sp, {}).setdefault(br.name, {})
                    for f in sorted(sub.glob("*.npz")):
                        bmap.setdefault(f.stem, str(f))
        else:
            for sp_raw in ("oof", "val", "test"):
                sub = base / sp_raw
                if not sub.exists():
                    continue
                sp = "oof" if sp_raw in ("oof", "val") else "test"
                smap = out.setdefault(sp, {})
                for f in sorted(sub.glob("*.npz")):
                    smap.setdefault(f.stem, str(f))
    return out


def discover_wave5_index(w5_root: Path) -> dict:
    """Build O(1) lookup index across all Wave-5 streams under w5_root."""
    return {
        "dense_probs": _collect_npz_tree(w5_root, "dense_probs", is_branched=False, deprioritize_substrs=("mitb3_",)),
        "cc_proposals": _collect_npz_tree(w5_root, "cc_proposals", is_branched=False, deprioritize_substrs=("mitb3_",)),
        "yolo26x_boxes": _collect_npz_tree(w5_root, ["yolo26x_boxes", "boxes_yolo26x-seg*", "boxes_yolo26x_seg*"], is_branched=False, deprioritize_substrs=("baseline",)),
        "yolo26x_s1masks": _collect_npz_tree(w5_root, ["yolo26x_s1masks", "s1_inst_yolo26x-seg*", "s1_inst_yolo26x_seg*"], is_branched=False, deprioritize_substrs=("baseline",)),
        "rtdetrx_boxes": _collect_npz_tree(w5_root, ["rtdetrx_boxes", "boxes_rtdetr-x*", "boxes_rtdetr_x*"], is_branched=False, deprioritize_substrs=("baseline",)),
        "m2f_probs": _collect_npz_tree(w5_root, "m2f_probs", is_branched=False, deprioritize_substrs=("baseline",)),
        "m2f_inst": _collect_npz_tree(w5_root, "m2f_inst", is_branched=False, deprioritize_substrs=("baseline",)),
        "s2_mitb4_dl": _collect_npz_tree(w5_root, ["s2_mitb4_dl", "exp013c_out_unet_mitb4*", "exp013c_out_unet-mitb4*"], is_branched=True, deprioritize_substrs=("baseline",)),
        "s2_cnxedge_dl": _collect_npz_tree(w5_root, ["s2_cnxedge_dl", "exp013c_out_convnext_edge*", "exp013c_out_convnext-edge*"], is_branched=True, deprioritize_substrs=("baseline",)),
    }


def summarize_wave5_coverage(idx: dict, n_oof: int, n_test: int) -> dict:
    rep: dict[str, dict] = {}
    for name, val in idx.items():
        if name in ("s2_mitb4_dl", "s2_cnxedge_dl"):
            rep[name] = {
                sp: {br: len(st_map) for br, st_map in val.get(sp, {}).items()}
                for sp in ("oof", "test")
            }
        else:
            rep[name] = {
                "oof": len(val.get("oof", {})),
                "test": len(val.get("test", {})),
                "oof_pct": round(100.0 * len(val.get("oof", {})) / max(1, n_oof), 1),
                "test_pct": round(100.0 * len(val.get("test", {})) / max(1, n_test), 1),
            }
    return rep


# ---------------------------------------------------------------------------
# Literature-Proven PQ Mask Refinement Primitives
# ---------------------------------------------------------------------------

def _bridge_and_label(mask: np.ndarray, kernel: np.ndarray = _KERNEL_BRIDGE) -> np.ndarray:
    """Non-inflating gap bridging (Agnuxo1): dilate ONLY to link fragments, keep original mask pixels."""
    if int(mask.sum()) < 90:
        return mask
    dil = cv2.dilate(mask.astype(np.uint8), kernel)
    n_cc, lbl, stats, _ = cv2.connectedComponentsWithStats(dil, connectivity=8)
    if n_cc <= 1:
        return mask
    keep = np.zeros_like(mask, dtype=bool)
    for cid in range(1, n_cc):
        comp = mask & (lbl == cid)
        if int(comp.sum()) >= 75:
            keep |= comp
    return keep if int(keep.sum()) >= 90 else mask


def _greedy_dense_claim(
    box: tuple[int, int, int, int],
    core_mask: np.ndarray,
    full_guide: np.ndarray,
    full_edge: np.ndarray,
) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    """Exclusive tail/limb claiming inside padded crop using full-frame dense probability & ridge barrier."""
    x1, y1, x2, y2 = box
    px1, py1 = max(0, x1 - 32), max(0, y1 - 32)
    px2, py2 = min(FULL, x2 + 32), min(FULL, y2 + 32)
    core_pad = np.zeros((py2 - py1, px2 - px1), dtype=np.uint8)
    core_pad[y1 - py1 : y2 - py1, x1 - px1 : x2 - px1] = core_mask.astype(np.uint8)
    guide = full_guide[py1:py2, px1:px2] >= 0.52
    barrier = full_edge[py1:py2, px1:px2] >= 0.50
    valid = guide & ~barrier
    cur = core_pad > 0
    for _ in range(2):
        halo = cv2.dilate(cur.astype(np.uint8), _KERNEL_CLAIM) > 0
        cur = cur | (halo & valid)
    n_cc, lbl = cv2.connectedComponents(cur.astype(np.uint8), connectivity=8)
    seed_ids = set(np.unique(lbl[core_pad > 0]).tolist()) - {0}
    if not seed_ids:
        return None
    out = np.isin(lbl, list(seed_ids))
    return (px1, py1, px2, py2), out


def _load_crop_stream(npz_path: str | None) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    if not npz_path or not Path(npz_path).exists():
        return np.zeros(0, dtype=np.float32), np.zeros((0, 4), dtype=np.int32), []
    with np.load(npz_path) as z:
        conf = np.asarray(z["conf"], np.float32).ravel()
        box = np.asarray(z["box"], np.int32).reshape(-1, 4)
        flat = np.asarray(z["p"], np.uint8) if "p" in z else (np.asarray(z["p4"], np.uint8) if "p4" in z else np.zeros(0, np.uint8))
    probs: list[np.ndarray] = []
    if len(flat) == 0:
        return conf, box, probs
    off = 0
    for x1, y1, x2, y2 in box:
        w, h = max(0, int(x2 - x1)), max(0, int(y2 - y1))
        sz = w * h
        probs.append(flat[off : off + sz].reshape(h, w) if (sz > 0 and off + sz <= len(flat)) else np.zeros((h, w), np.uint8))
        off += sz
    return conf, box, probs


def _load_box_stream(npz_path: str | None) -> tuple[np.ndarray, np.ndarray]:
    if not npz_path or not Path(npz_path).exists():
        return np.zeros(0, dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
    with np.load(npz_path) as z:
        conf = np.asarray(z["conf"], np.float32).ravel()
        box = np.asarray(z["box"], np.float32).reshape(-1, 4)
    return conf, box


def _paint_envelope(items: tuple[np.ndarray, np.ndarray, list[np.ndarray]], buf: np.ndarray) -> list[dict]:
    """Paint max continuous probability envelope from (conf, box, probs) onto (2048, 2048) float32 buffer."""
    buf.fill(0.0)
    confs, boxes, probs = items
    crops: list[dict] = []
    for c, (x1, y1, x2, y2), pm in zip(confs, boxes, probs):
        cx1, cy1, cx2, cy2 = max(0, int(x1)), max(0, int(y1)), min(FULL, int(x2)), min(FULL, int(y2))
        if cx2 <= cx1 or cy2 <= cy1 or pm.size == 0:
            continue
        p_sub = pm[: cy2 - cy1, : cx2 - cx1]
        if p_sub.shape != (cy2 - cy1, cx2 - cx1):
            p_sub = cv2.resize(pm, (cx2 - cx1, cy2 - cy1), interpolation=cv2.INTER_LINEAR)
        pf = p_sub.astype(np.float32) / 255.0
        buf[cy1:cy2, cx1:cx2] = np.maximum(buf[cy1:cy2, cx1:cx2], pf)
        crops.append({"c": float(c), "box": (cx1, cy1, cx2, cy2), "p": p_sub})
    return crops


def _best_box_conf_iou(
    box: tuple[int, int, int, int],
    msk: np.ndarray,
    det_confs: np.ndarray,
    det_boxes: np.ndarray,
    det_masks: list[np.ndarray] | None = None,
) -> tuple[float, float]:
    """Return best overlapping detector confidence and mask IoU (if det_masks provided, else box IoU)."""
    if len(det_confs) == 0:
        return 0.0, 0.0
    x1, y1, x2, y2 = box
    best_c, best_iou = 0.0, 0.0
    b_area = float(max(1, (x2 - x1) * (y2 - y1)))
    for k, (dc, (dx1, dy1, dx2, dy2)) in enumerate(zip(det_confs, det_boxes)):
        ix1, iy1 = max(x1, int(dx1)), max(y1, int(dy1))
        ix2, iy2 = min(x2, int(dx2)), min(y2, int(dy2))
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        if det_masks is not None and k < len(det_masks) and det_masks[k].size > 0:
            dm = det_masks[k] > 127
            iou, _ = e11x._crop_iou(box, msk, (int(dx1), int(dy1), int(dx2), int(dy2)), dm)
            thr = 0.20
        else:
            inter = float((ix2 - ix1) * (iy2 - iy1))
            iou = inter / max(1.0, b_area + max(1.0, (dx2 - dx1) * (dy2 - dy1)) - inter)
            thr = 0.25
        if iou >= thr:
            best_c = max(best_c, float(dc))
            best_iou = max(best_iou, float(iou))
    return best_c, best_iou


# ---------------------------------------------------------------------------
# Wave-5 Candidate + 86-Feature + 4x48x48 Patch Extractor
# ---------------------------------------------------------------------------

def _extract_stem_wave5(task: tuple[str, str, int, list]) -> tuple[str, str, dict]:
    stem, split, fold, stem_gt = task
    idx = _W5_INDEX_GLOBAL
    sp_dir = "oof" if split in ("oof", "val") else "test"
    specs = e11x._get_split_specs(split)

    # 1. Exact Track-A Clean anchors + continuous multi-stream clusters
    raw_crops: dict[str, tuple[float, list[dict]]] = {}
    ta_stream_insts = []
    for sname in ("ms", "s", "main"):
        sw, cf_ta, sspecs = specs["ta_streams"][sname]
        crops = e11x._load_blended_crops_once(sspecs, stem, min_raw_floor=0.14, track_sam_cnn=(sname == "ms"), clamp_to_primary=True)
        raw_crops[sname] = (sw, crops)
        insts = e11x._paint_exact_from_crops(crops, thr=0.68, conf_floor=cf_ta, drop=0.90, min_area=100, use_raw_c=False)
        ta_stream_insts.append((sname, sw, insts))
    sahi_wt, _, sahi_specs = specs["sahi"]
    sahi_crops = e11x._load_blended_crops_once(sahi_specs, stem, min_raw_floor=0.14, track_sam_cnn=True)
    raw_crops["SAHI"] = (sahi_wt, sahi_crops)

    ta_base_rles = e11x.bgs.weighted_consensus_stem(ta_stream_insts, match_iou=0.40, vote_thr=0.52, c_floor=0.28, min_area=100, drop=0.90)
    ta_bm = np.zeros((FULL, FULL), dtype=bool)
    ta_base_items = []
    for rstr in ta_base_rles:
        mf = mu.decode({"size": [FULL, FULL], "counts": rstr.encode("ascii")}).astype(bool)
        ys, xs = np.nonzero(mf)
        if not len(ys):
            continue
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        cm = mf[y1:y2, x1:x2]
        ta_bm[y1:y2, x1:x2] |= cm
        ta_base_items.append(((x1, y1, x2, y2), cm, rstr))

    sahi_dec = e11x._paint_exact_from_crops(sahi_crops, thr=0.68, conf_floor=0.20, drop=0.90, min_area=100, use_raw_c=True)
    ta_rescue_items = []
    for c, (x1, y1, x2, y2), m in sahi_dec:
        ma = int(m.sum())
        if c < 0.22 or ma < 100 or ma > 400:
            continue
        if int((m & ta_bm[y1:y2, x1:x2]).sum()) / ma <= 0.15:
            keep = m & ~ta_bm[y1:y2, x1:x2]
            ka = int(keep.sum())
            if ka >= 100 and ka >= 0.90 * ma:
                ta_bm[y1:y2, x1:x2] |= keep
                rstr = bss.encode_crop_ascii((x1, y1, x2, y2), keep)
                trimmed = e11x._trim_cand((x1, y1, x2, y2), keep, min_area=100)
                if trimmed is not None:
                    ta_rescue_items.append((trimmed[0], trimmed[1], rstr, float(c)))

    total_w = sum(w for w, _ in raw_crops.values())
    cont_streams = {s: e11x._paint_continuous_from_crops(crops, conf_floor=0.14, drop=0.88, min_area=80) for s, (_, crops) in raw_crops.items()}
    flat_items = sorted(((s, sw, it) for s, (sw, _) in raw_crops.items() for it in cont_streams[s]), key=lambda z: -(z[2]["c"] * min(1.5, z[1])))

    clusters: list[dict] = []
    for sname, sw, it in flat_items:
        box, m = it["box"], it["m62"]
        best_j, best_iou = -1, 0.0
        for j, cl in enumerate(clusters):
            if sname in cl["seen"]:
                continue
            bx1, by1, bx2, by2 = box
            cx1, cy1, cx2, cy2 = cl["ubox"]
            if min(bx2, cx2) <= max(bx1, cx1) or min(by2, cy2) <= max(by1, cy1):
                continue
            iou, _ = e11x._crop_iou(box, m, cl["rbox"], cl["rmask"])
            if iou >= 0.40 and iou > best_iou:
                best_iou, best_j = iou, j
        if best_j < 0:
            clusters.append({"seen": {sname}, "items": [(sname, sw, it)], "rbox": box, "rmask": m, "ubox": list(box)})
        else:
            cl = clusters[best_j]
            cl["seen"].add(sname)
            cl["items"].append((sname, sw, it))
            ub = cl["ubox"]
            ub[0], ub[1], ub[2], ub[3] = min(ub[0], box[0]), min(ub[1], box[1]), max(ub[2], box[2]), max(ub[3], box[3])

    full_ta = np.zeros((FULL, FULL), dtype=np.float32)
    for cl in clusters:
        gx1, gy1, gx2, gy2 = cl["ubox"]
        uh, uw = gy2 - gy1, gx2 - gx1
        pacc = np.zeros((uh, uw), dtype=np.float32)
        pwsum = np.zeros((uh, uw), dtype=np.float32)
        vacc68 = np.zeros((uh, uw), dtype=np.float32)
        vacc62 = np.zeros((uh, uw), dtype=np.float32)
        w_present = sum_wc = 0.0
        sconf = {"ms": 0.0, "s": 0.0, "main": 0.0, "SAHI": 0.0}
        sc_vals, sm_masks = [], {}
        for sname, sw, it in cl["items"]:
            bx1, by1, bx2, by2 = it["box"]
            ys_s, xs_s = slice(by1 - gy1, by2 - gy1), slice(bx1 - gx1, bx2 - gx1)
            pacc[ys_s, xs_s] += sw * (it["p"].astype(np.float32) / 255.0)
            pwsum[ys_s, xs_s] += sw
            vacc68[ys_s, xs_s] += sw * it["m68"].astype(np.float32)
            vacc62[ys_s, xs_s] += sw * it["m62"].astype(np.float32)
            w_present += sw
            sum_wc += sw * it["c"]
            sconf[sname] = max(sconf[sname], it["c"])
            if it["sam_cnn"] > 0:
                sc_vals.append(it["sam_cnn"])
            sm_masks[sname] = (it["box"], it["m62"])
        pmap = np.where(pwsum > 0, pacc / np.maximum(pwsum, 1e-6), 0.0).astype(np.float32)
        full_ta[gy1:gy2, gx1:gx2] = np.maximum(full_ta[gy1:gy2, gx1:gx2], pmap)
        confs_nz = [v for v in sconf.values() if v > 0]
        ms_s_iou = e11x._crop_iou(sm_masks["ms"][0], sm_masks["ms"][1], sm_masks["s"][0], sm_masks["s"][1])[0] if ("ms" in sm_masks and "s" in sm_masks) else 0.0
        skeys = list(sm_masks.keys())
        pair_ious = [e11x._crop_iou(sm_masks[skeys[a]][0], sm_masks[skeys[a]][1], sm_masks[skeys[b]][0], sm_masks[skeys[b]][1])[0] for a in range(len(skeys)) for b in range(a + 1, len(skeys))]
        cl["meta"] = {
            "box": (gx1, gy1, gx2, gy2),
            "pmap": pmap,
            "vmap68": vacc68 / max(w_present, 1e-6),
            "vmap62": vacc62 / max(w_present, 1e-6),
            "c_tracka": sum_wc / total_w,
            "w_vote": w_present / total_w,
            "n_streams": len(cl["seen"]),
            "sconf": sconf,
            "max_conf": max(confs_nz) if confs_nz else 0.0,
            "min_conf_nz": min(confs_nz) if confs_nz else 0.0,
            "std_conf_nz": float(np.std(confs_nz)) if len(confs_nz) > 1 else 0.0,
            "sahi_only": 1.0 if (sconf["SAHI"] > 0 and max(sconf["ms"], sconf["s"], sconf["main"]) == 0) else 0.0,
            "ms_only": 1.0 if (sconf["ms"] > 0 and max(sconf["s"], sconf["main"], sconf["SAHI"]) == 0) else 0.0,
            "sam_cnn_iou": float(np.mean(sc_vals)) if sc_vals else 0.0,
            "ms_s_iou": float(ms_s_iou),
            "stream_iou_mean": float(np.mean(pair_ious)) if pair_ious else 0.0,
        }

    # 2. Load Wave-5 Streams (EXP-012 dense + cc, EXP-013a yolo26x/rtdetrx, EXP-013c s2_mitb4/s2_cnxedge, opt m2f)
    dp_path = idx.get("dense_probs", {}).get(sp_dir, {}).get(stem)
    if dp_path and Path(dp_path).exists():
        with np.load(dp_path) as zd:
            p_raw = np.asarray(zd["p"], np.float32) / 255.0
        dfg = cv2.resize(p_raw[0], (FULL, FULL), interpolation=cv2.INTER_LINEAR)
        dedge = cv2.resize(p_raw[1], (FULL, FULL), interpolation=cv2.INTER_LINEAR)
        has_dense = True
    else:
        dfg = np.zeros((FULL, FULL), dtype=np.float32)
        dedge = np.zeros((FULL, FULL), dtype=np.float32)
        has_dense = False

    m2f_path = idx.get("m2f_probs", {}).get(sp_dir, {}).get(stem)
    if m2f_path and Path(m2f_path).exists():
        with np.load(m2f_path) as zm:
            mp = np.asarray(zm["prob"], np.float32) / 255.0
        m2f_fg = cv2.resize(mp, (FULL, FULL), interpolation=cv2.INTER_LINEAR) if mp.shape != (FULL, FULL) else mp
        has_m2f = True
    else:
        m2f_fg = np.zeros((FULL, FULL), dtype=np.float32)
        has_m2f = False

    cc_c, cc_b, cc_p = _load_crop_stream(idx.get("cc_proposals", {}).get(sp_dir, {}).get(stem))
    y26_c, y26_b = _load_box_stream(idx.get("yolo26x_boxes", {}).get(sp_dir, {}).get(stem))
    y26m_c, y26m_b, y26m_p = _load_crop_stream(idx.get("yolo26x_s1masks", {}).get(sp_dir, {}).get(stem))
    rtd_c, rtd_b = _load_box_stream(idx.get("rtdetrx_boxes", {}).get(sp_dir, {}).get(stem))
    m2fi_c, m2fi_b, m2fi_p = _load_crop_stream(idx.get("m2f_inst", {}).get(sp_dir, {}).get(stem))

    def _merge_s2_branches(stream_key: str) -> tuple[list[dict], np.ndarray]:
        env = np.zeros((FULL, FULL), dtype=np.float32)
        crops_all: list[dict] = []
        bmap = idx.get(stream_key, {}).get(sp_dir, {})
        tmp = np.zeros((FULL, FULL), dtype=np.float32)
        for br_name in sorted(bmap):
            p = bmap[br_name].get(stem)
            if not p:
                continue
            items = _load_crop_stream(p)
            cs = _paint_envelope(items, tmp)
            env = np.maximum(env, tmp)
            for it in cs:
                it["branch"] = br_name
                crops_all.append(it)
        return crops_all, env

    s2_mit_crops, full_s2_mit = _merge_s2_branches("s2_mitb4_dl")
    s2_cnx_crops, full_s2_cnx = _merge_s2_branches("s2_cnxedge_dl")

    # Blended full-frame Wave-5 map normalized cleanly over available maps
    w_tot = np.full((FULL, FULL), 0.35, dtype=np.float32)
    p_acc = 0.35 * full_ta
    if full_s2_mit.any():
        p_acc += 0.30 * full_s2_mit
        w_tot += 0.30
    if full_s2_cnx.any():
        p_acc += 0.25 * full_s2_cnx
        w_tot += 0.25
    if has_dense:
        p_acc += 0.15 * dfg
        w_tot += 0.15
    if has_m2f:
        p_acc += 0.20 * m2f_fg
        w_tot += 0.20
    full_w5 = np.clip(p_acc / w_tot, 0.0, 1.0)
    full_w5_u8 = np.clip(full_w5 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    has_any_w5 = has_dense or has_m2f or full_s2_mit.any() or full_s2_cnx.any() or bool(len(y26_c) or len(rtd_c) or len(y26m_c) or len(m2fi_c))

    # 3. Emit candidates
    cands: list[dict] = []

    def _match_cluster(box: tuple[int, int, int, int], m: np.ndarray) -> dict | None:
        best_cl, best_iou = None, 0.15
        for cl in clusters:
            cb = cl["meta"]["box"]
            if min(box[2], cb[2]) <= max(box[0], cb[0]) or min(box[3], cb[3]) <= max(box[1], cb[1]):
                continue
            iou, _ = e11x._crop_iou(box, m, cl["rbox"], cl["rmask"])
            if iou > best_iou:
                best_iou, best_cl = iou, cl
        return best_cl

    def _default_meta(c_val: float, sname: str = "SAHI") -> dict:
        sc = {"ms": 0.0, "s": 0.0, "main": 0.0, "SAHI": 0.0}
        if sname in sc:
            sc[sname] = c_val
        return {
            "c_tracka": c_val,
            "w_vote": 0.15,
            "n_streams": 0 if sname == "w5" else 1,
            "sconf": sc,
            "max_conf": c_val,
            "min_conf_nz": c_val,
            "std_conf_nz": 0.0,
            "sahi_only": 1.0 if sname == "SAHI" else 0.0,
            "ms_only": 1.0 if sname == "ms" else 0.0,
            "sam_cnn_iou": 0.0,
            "ms_s_iou": 0.0,
            "stream_iou_mean": 0.0,
        }

    def _add_cand(box: tuple[int, int, int, int], m: np.ndarray, kind: str, meta: dict, rle_override: str | None = None, dedup_thr: float = 0.94) -> bool:
        trimmed = e11x._trim_cand(box, m, min_area=90)
        if trimmed is None:
            return False
        tb, tm = trimmed
        if dedup_thr <= 1.0:
            for ex in cands:
                eb = ex["box"]
                if min(tb[2], eb[2]) <= max(tb[0], eb[0]) or min(tb[3], eb[3]) <= max(tb[1], eb[1]):
                    continue
                iou, _ = e11x._crop_iou(tb, tm, eb, ex["mask"])
                if iou >= dedup_thr:
                    return False
        rstr = rle_override if rle_override is not None else bss.encode_crop_ascii(tb, tm)
        cands.append({"box": tb, "mask": tm, "rle": rstr, "kind": kind, "meta": meta})
        return True

    # 3a. Exact Track-A Base & Rescue + Track-A++ variants
    for tb, tm, rstr in ta_base_items:
        cl = _match_cluster(tb, tm)
        meta = cl["meta"] if cl is not None else _default_meta(0.28, "ms")
        _add_cand(tb, tm, "ta_base", meta, rle_override=rstr, dedup_thr=1.01)

    for tb, tm, rstr, c_raw in ta_rescue_items:
        cl = _match_cluster(tb, tm)
        meta = cl["meta"] if cl is not None else _default_meta(float(c_raw) * 1.15 / total_w, "SAHI")
        _add_cand(tb, tm, "ta_rescue", meta, rle_override=rstr, dedup_thr=1.01)

    cons_for_merge = []
    for cl in clusters:
        cm = cl["meta"]
        gx1, gy1, gx2, gy2 = cm["box"]
        pmap, v68, v62 = cm["pmap"], cm["vmap68"], cm["vmap62"]
        m68 = v68 >= 0.52
        if int(m68.sum()) >= 90:
            _add_cand((gx1, gy1, gx2, gy2), m68, "cons68", cm, dedup_thr=0.95)
            t68 = e11x._trim_cand((gx1, gy1, gx2, gy2), m68, min_area=90)
            if t68 is not None and cm["c_tracka"] >= 0.15:
                cons_for_merge.append((t68[0], t68[1], cm))
        m62 = (pmap >= 0.56) | (v62 >= 0.52)
        if int(m62.sum()) >= 90:
            _add_cand((gx1, gy1, gx2, gy2), m62, "cons62", cm, dedup_thr=0.94)
        m74 = pmap >= 0.72
        if int(m74.sum()) >= 90:
            _add_cand((gx1, gy1, gx2, gy2), m74, "cons74", cm, dedup_thr=0.94)
            # 1-px cross erosion for small/medium over-dilated filaments
            if int(m74.sum()) < 1400:
                m_e1 = cv2.erode(m74.astype(np.uint8), _KERNEL_CROSS3) > 0
                if int(m_e1.sum()) >= 75:
                    _add_cand((gx1, gy1, gx2, gy2), m_e1, "w5_erode1", cm, dedup_thr=0.92)

        base_m = m68 if int(m68.sum()) >= 180 else m62
        if int(base_m.sum()) >= 180:
            core = (pmap > 0.76) & base_m
            n_cc, lbl, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), connectivity=8)
            big_ids = [i for i in range(1, n_cc) if stats[i, cv2.CC_STAT_AREA] >= 55]
            if 2 <= len(big_ids) <= 4:
                dists = [cv2.distanceTransform((lbl != i).astype(np.uint8), cv2.DIST_L2, 3) for i in big_ids]
                assign = np.argmin(np.stack(dists, axis=0), axis=0)
                for k_idx in range(len(big_ids)):
                    child = base_m & (assign == k_idx)
                    if int(child.sum()) >= 90:
                        _add_cand((gx1, gy1, gx2, gy2), child, "split", cm, dedup_thr=0.93)

        # Wave-5 Stage-2 + Dense blend variants inside cluster crop
        if has_any_w5:
            p_loc_w5 = full_w5[gy1:gy2, gx1:gx2]
            m_b48 = p_loc_w5 >= 0.48
            m_b56 = p_loc_w5 >= 0.56
            if int(m_b48.sum()) >= 90:
                _add_cand((gx1, gy1, gx2, gy2), m_b48, "w5_b48", cm, dedup_thr=0.93)
                m_link = _bridge_and_label(m_b48)
                if int(m_link.sum()) >= 90:
                    _add_cand((gx1, gy1, gx2, gy2), m_link, "w5_link", cm, dedup_thr=0.93)
            if int(m_b56.sum()) >= 90:
                _add_cand((gx1, gy1, gx2, gy2), m_b56, "w5_b56", cm, dedup_thr=0.93)
                if has_dense:
                    clm = _greedy_dense_claim((gx1, gy1, gx2, gy2), m_b56, np.maximum(dfg, full_w5), dedge)
                    if clm is not None:
                        _add_cand(clm[0], clm[1], "w5_claim", cm, dedup_thr=0.92)

    for sname in raw_crops:
        for it in cont_streams[sname]:
            if it["c"] < 0.14:
                continue
            m = it["p"] > int(0.66 * 255 + 0.5)
            if int(m.sum()) < 90:
                continue
            cl = _match_cluster(it["box"], m)
            meta = cl["meta"] if cl is not None else _default_meta(it["c"] * raw_crops[sname][0] / total_w, sname)
            _add_cand(it["box"], m, "solo", meta, dedup_thr=0.90)

    # Collinear near-gap Track-A merge
    merge_pool = cons_for_merge[:24]
    for i in range(len(merge_pool)):
        ba, ma, cma = merge_pool[i]
        ax1, ay1, ax2, ay2 = ba
        for j in range(i + 1, len(merge_pool)):
            bb, mb, cmb = merge_pool[j]
            bx1, by1, bx2, by2 = bb
            if max(max(0, max(ax1, bx1) - min(ax2, bx2)), max(0, max(ay1, by1) - min(ay2, by2))) > 16:
                continue
            ux1, uy1, ux2, uy2 = min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)
            if max(ux2 - ux1, uy2 - uy1) > 650:
                continue
            um = np.zeros((uy2 - uy1, ux2 - ux1), dtype=bool)
            um[ay1 - uy1 : ay2 - uy1, ax1 - ux1 : ax2 - ux1] |= ma
            um[by1 - uy1 : by2 - uy1, bx1 - ux1 : bx2 - ux1] |= mb
            dil = cv2.dilate(um.astype(np.uint8), e11x._KERNEL_7) > 0
            if cv2.connectedComponents(dil.astype(np.uint8))[0] < cv2.connectedComponents(um.astype(np.uint8))[0]:
                um = um | (dil & (full_w5_u8[uy1:uy2, ux1:ux2] > 90))
            _add_cand((ux1, uy1, ux2, uy2), um, "merge", cma if cma["c_tracka"] >= cmb["c_tracka"] else cmb, dedup_thr=0.92)

    # 3b. Wave-5 Discovery Candidates (EXP-013c crops + EXP-012 CC + Dense Hysteresis + YOLO26x s1masks)
    for src_crops in (s2_mit_crops, s2_cnx_crops):
        for it in src_crops:
            if it["c"] < 0.16:
                continue
            bx1, by1, bx2, by2 = it["box"]
            pl = 0.55 * (it["p"].astype(np.float32) / 255.0) + 0.45 * full_w5[by1:by2, bx1:bx2]
            m = pl >= 0.50
            if int(m.sum()) < 90:
                continue
            cl = _match_cluster(it["box"], m)
            meta = cl["meta"] if cl is not None else _default_meta(0.0, "w5")
            _add_cand(it["box"], m, "w5_b48" if cl is not None else "w5_disc", meta, dedup_thr=0.90)

    for c, (bx1, by1, bx2, by2), pm in list(zip(cc_c, cc_b, cc_p)) + list(zip(y26m_c, y26m_b, y26m_p)) + list(zip(m2fi_c, m2fi_b, m2fi_p)):
        if float(c) < 0.18:
            continue
        cx1, cy1, cx2, cy2 = max(0, int(bx1)), max(0, int(by1)), min(FULL, int(bx2)), min(FULL, int(by2))
        if cx2 <= cx1 or cy2 <= cy1 or pm.size == 0:
            continue
        ps = cv2.resize(pm, (cx2 - cx1, cy2 - cy1), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        pl = 0.55 * ps + 0.45 * full_w5[cy1:cy2, cx1:cx2]
        m = pl >= 0.48
        if int(m.sum()) >= 90:
            cl = _match_cluster((cx1, cy1, cx2, cy2), m)
            meta = cl["meta"] if cl is not None else _default_meta(0.0, "w5")
            _add_cand((cx1, cy1, cx2, cy2), m, "w5_disc", meta, dedup_thr=0.88)

    if has_dense:
        # Full-frame dual-threshold ridge-separated hysteresis on p_dfg * (1 - 0.45 * dedge)
        seed_map = dfg * np.clip(1.0 - 0.45 * dedge, 0.25, 1.0)
        bw_lo = (seed_map >= 0.34).astype(np.uint8)
        n_cc, lbl, stats, _ = cv2.connectedComponentsWithStats(bw_lo, connectivity=8)
        for cid in range(1, n_cc):
            ar = int(stats[cid, cv2.CC_STAT_AREA])
            if ar < 90 or ar > 60000:
                continue
            hx1, hy1 = int(stats[cid, cv2.CC_STAT_LEFT]), int(stats[cid, cv2.CC_STAT_TOP])
            hx2, hy2 = hx1 + int(stats[cid, cv2.CC_STAT_WIDTH]), hy1 + int(stats[cid, cv2.CC_STAT_HEIGHT])
            sub_m = lbl[hy1:hy2, hx1:hx2] == cid
            vals = dfg[hy1:hy2, hx1:hx2][sub_m]
            if float(vals.max()) < 0.70 or float(vals.mean()) < 0.46:
                continue
            cl = _match_cluster((hx1, hy1, hx2, hy2), sub_m)
            meta = cl["meta"] if cl is not None else _default_meta(0.0, "w5")
            _add_cand((hx1, hy1, hx2, hy2), sub_m, "w5_disc", meta, dedup_thr=0.86)

    # 4. Exact Physics + 86-dim Features + 4x48x48 Patches + Edges
    if not cands:
        return split, stem, {
            "stem": stem,
            "fold": fold,
            "obs": stem[-2:],
            "r_count": len(stem_gt),
            "rles": [],
            "kinds": [],
            "boxes": np.zeros((0, 4), dtype=np.int32),
            "areas": np.zeros(0, dtype=np.int32),
            "X": np.zeros((0, len(FEATURE_NAMES_W5)), dtype=np.float32),
            "h": np.zeros(0, dtype=np.float32),
            "m": np.zeros(0, dtype=np.float32),
            "u": np.zeros(0, dtype=np.float32),
            "edges": [],
            "patches": np.zeros((0, 4, 48, 48), dtype=np.uint8),
        }

    phys = e11x._compute_img_physics(stem)
    ta_cands = [c for c in cands if c["kind"] in ("ta_base", "ta_rescue")]
    n = len(cands)
    n_cands_img = float(n)
    sum_c_img = float(sum(c["meta"]["c_tracka"] for c in cands))
    c_arr = np.array([c["meta"]["c_tracka"] for c in cands], dtype=np.float32)
    ranks = np.argsort(np.argsort(-c_arr)).astype(np.float32)

    ov_cnt = np.zeros(n, dtype=np.float32)
    ov_max = np.zeros(n, dtype=np.float32)
    edges = []
    areas_arr = np.array([int(c["mask"].sum()) for c in cands], dtype=np.int32)
    for i in range(n):
        bi, mi = cands[i]["box"], cands[i]["mask"]
        for j in range(i + 1, n):
            bj, mj = cands[j]["box"], cands[j]["mask"]
            if min(bi[2], bj[2]) <= max(bi[0], bj[0]) or min(bi[3], bj[3]) <= max(bi[1], bj[1]):
                continue
            iou, inter = e11x._crop_iou(bi, mi, bj, mj)
            if inter > 0:
                if iou > 0.05:
                    ov_cnt[i] += 1.0
                    ov_cnt[j] += 1.0
                    ov_max[i] = max(ov_max[i], float(iou))
                    ov_max[j] = max(ov_max[j], float(iou))
                inter_frac = inter / float(max(1, min(int(areas_arr[i]), int(areas_arr[j]))))
                edges.append((i, j, float(inter_frac), float(iou)))

    kind_map = {k: i for i, k in enumerate(KIND_LIST_W5)}
    obs_code = e11x.OBS_MAP.get(stem[-2:], -1.0)
    X = np.zeros((n, len(FEATURE_NAMES_W5)), dtype=np.float32)
    patches = np.zeros((n, 4, 48, 48), dtype=np.uint8)
    for i, cd in enumerate(cands):
        x1, y1, x2, y2 = cd["box"]
        msk = cd["mask"]
        area = float(areas_arr[i])
        meta = cd["meta"]
        bw, bh = float(x2 - x1), float(y2 - y1)

        pcrop = full_w5[y1:y2, x1:x2]
        pv = pcrop[msk] if msk.any() else np.array([0.5], dtype=np.float32)
        pq = np.percentile(pv, [10, 25, 50, 75, 90])
        ero = cv2.erode(msk.astype(np.uint8), e11x._KERNEL_3) > 0
        rim = msk & ~ero
        p_core = float(pcrop[ero].mean()) if ero.any() else float(pv.mean())
        p_rim = float(pcrop[rim].mean()) if rim.any() else float(pv.mean())
        n_cc76_raw, _, st76, _ = cv2.connectedComponentsWithStats(((pcrop > 0.76) & msk).astype(np.uint8), connectivity=8)
        n_cc_core76 = float(sum(1 for kid in range(1, n_cc76_raw) if st76[kid, cv2.CC_STAT_AREA] >= 40))

        m_u8 = msk.astype(np.uint8)
        n_cc_m = float(cv2.connectedComponents(m_u8, connectivity=8)[0] - 1)
        cnts, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            pts = np.vstack(cnts)
            (_, (rw, rh), _) = cv2.minAreaRect(pts)
            rot_len, rot_wid = float(max(rw, rh, 1.0)), float(max(1.0, min(rw, rh)))
            hull_area = float(max(area, cv2.contourArea(cv2.convexHull(pts))))
            perim = float(sum(cv2.arcLength(ct, True) for ct in cnts))
        else:
            rot_len, rot_wid, hull_area, perim = max(bw, bh), min(bw, bh), area, 2.0 * (bw + bh)
        solidity = area / max(1.0, hull_area)
        compact = (perim * perim) / max(1.0, 4.0 * math.pi * area)
        mean_width = (2.0 * area) / max(1.0, perim)

        hx1, hy1 = max(0, x1 // 2), max(0, y1 // 2)
        hx2, hy2 = min(HALF, max(hx1 + 1, (x2 + 1) // 2)), min(HALF, max(hy1 + 1, (y2 + 1) // 2))
        mh = cv2.resize(m_u8, (hx2 - hx1, hy2 - hy1), interpolation=cv2.INTER_NEAREST) > 0
        if not mh.any():
            mh[0, 0] = True
        skirt_h = (cv2.dilate(mh.astype(np.uint8), e11x._KERNEL_7) > 0) & ~mh
        fz, bz, rz = phys["flat_z"][hy1:hy2, hx1:hx2], phys["bth_z"][hy1:hy2, hx1:hx2], phys["ridge"][hy1:hy2, hx1:hx2]
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
        rho_c = float(math.hypot(cx_p - phys["cx"], cy_p - phys["cy"]) / max(1.0, phys["R"]))
        rho_max = float(max(math.hypot(cx - phys["cx"], cy - phys["cy"]) / max(1.0, phys["R"]) for cx in (x1, x2) for cy in (y1, y2)))

        iou_ta, c_ta, near_d = 0.0, 0.0, 999.0
        if cd["kind"] in ("ta_base", "ta_rescue"):
            iou_ta, c_ta, near_d = 1.0, float(meta["c_tracka"]), 0.0
        elif ta_cands:
            for tac in ta_cands:
                tb = tac["box"]
                d = float(math.hypot(max(0, max(x1, tb[0]) - min(x2, tb[2])), max(0, max(y1, tb[1]) - min(y2, tb[3]))))
                if d < near_d:
                    near_d = d
                if d == 0.0:
                    tiou, _ = e11x._crop_iou(cd["box"], msk, tb, tac["mask"])
                    if tiou > iou_ta:
                        iou_ta, c_ta = float(tiou), float(tac["meta"]["c_tracka"])

        # Wave-5 cross-family features
        y26_cf, _ = _best_box_conf_iou(cd["box"], msk, y26_c, y26_b, None)
        rtd_cf, _ = _best_box_conf_iou(cd["box"], msk, rtd_c, rtd_b, None)
        cc_cf, _ = _best_box_conf_iou(cd["box"], msk, cc_c, cc_b, cc_p)
        _, y26m_i = _best_box_conf_iou(cd["box"], msk, y26m_c, y26m_b, y26m_p)
        m2f_cf, m2fi_iou = _best_box_conf_iou(cd["box"], msk, m2fi_c, m2fi_b, m2fi_p)
        n_w5_det = float((y26_cf > 0.15) + (rtd_cf > 0.18) + (cc_cf > 0.18) + (m2f_cf > 0.20))

        s2m_crop = full_s2_mit[y1:y2, x1:x2]
        s2c_crop = full_s2_cnx[y1:y2, x1:x2]
        s2m_bin = s2m_crop >= 0.50
        s2c_bin = s2c_crop >= 0.50
        s2m_iou = float((s2m_bin & msk).sum() / max(1, (s2m_bin | msk).sum())) if full_s2_mit.any() else 0.0
        s2c_iou = float((s2c_bin & msk).sum() / max(1, (s2c_bin | msk).sum())) if full_s2_cnx.any() else 0.0

        m2fg_c = m2f_fg[y1:y2, x1:x2]
        m2fg_bin = m2fg_c >= 0.50
        m2fg_iou = float((m2fg_bin & msk).sum() / max(1, (m2fg_bin | msk).sum())) if has_m2f else 0.0

        dfg_c = dfg[y1:y2, x1:x2]
        ded_c = dedge[y1:y2, x1:x2]
        dfg_v = dfg_c[msk] if msk.any() else np.zeros(1, dtype=np.float32)
        dfg_bin = dfg_c >= 0.50
        dfg_iou50 = float((dfg_bin & msk).sum() / max(1, (dfg_bin | msk).sum())) if has_dense else 0.0

        sconf = meta["sconf"]
        is_w5_c = 1.0 if cd["kind"].startswith("w5_") else 0.0
        X[i] = np.array([
            float(meta["c_tracka"]), float(meta["w_vote"]), float(meta["n_streams"]),
            float(sconf["ms"]), float(sconf["s"]), float(sconf["main"]), float(sconf["SAHI"]),
            float(meta["max_conf"]), float(meta["min_conf_nz"]), float(meta["std_conf_nz"]),
            float(meta["sahi_only"]), float(meta["ms_only"]),
            float(meta["sam_cnn_iou"]), float(meta["ms_s_iou"]), float(meta["stream_iou_mean"]),
            float(pv.mean()), float(pq[0]), float(pq[1]), float(pq[2]), float(pq[3]), float(pq[4]), float(pv.std()),
            p_core, p_rim, p_core - p_rim,
            float((pv > 0.65).mean()), float((pv > 0.75).mean()), float((pv > 0.85).mean()), n_cc_core76,
            area, math.log1p(area), bw, bh, max(bw, bh),
            rot_len, rot_wid, rot_len / max(1.0, rot_wid),
            area / max(1.0, bw * bh), solidity, min(compact, 200.0), mean_width, n_cc_m,
            rho_c, rho_max, float(q_loc - np.percentile(f_m, 25)), float(q_loc - np.percentile(f_m, 10)),
            float(f_m.mean()), float(np.percentile(f_m, 10)), float(b_m.mean()), float(np.percentile(b_m, 10)),
            bth_cs_diff, bright_ring_z, float((f_m < -1.0).mean()),
            float(r_m.mean()), float(np.percentile(r_m, 75)), ring_std,
            phys["mad"], obs_code,
            float(kind_map.get(cd["kind"], 0)),
            1.0 if cd["kind"] == "ta_base" else 0.0,
            1.0 if cd["kind"] == "ta_rescue" else 0.0,
            iou_ta, float(meta["c_tracka"]) - c_ta,
            n_cands_img, sum_c_img, float(ranks[i]), ov_cnt[i], ov_max[i], near_d,
            # Wave-5 17 extra features
            y26_cf, rtd_cf, cc_cf, n_w5_det,
            float(s2m_crop[msk].mean()) if msk.any() else 0.0, s2m_iou,
            float(s2c_crop[msk].mean()) if msk.any() else 0.0, s2c_iou, y26m_i,
            float(dfg_v.mean()), float(np.percentile(dfg_v, 75)),
            float(dfg_c[ero].mean()) if ero.any() else float(dfg_v.mean()),
            float(dfg_c[rim].mean()) if rim.any() else float(dfg_v.mean()),
            float(ded_c[msk].mean()) if msk.any() else 0.0,
            float(ded_c[rim].mean()) if rim.any() else 0.0,
            dfg_iou50, is_w5_c,
            m2f_cf, m2fi_iou,
            float(m2fg_c[msk].mean()) if (has_m2f and msk.any()) else 0.0,
            m2fg_iou,
        ], dtype=np.float32)

        # 4x48x48 aspect-preserving patch: [flat_z, dense_fg_or_bth_z, w5_pmap, mask]
        use_dch1 = bool(idx.get("use_dense_ch1", False)) and has_dense
        side = max(32, int(round(max(bw, bh) * 1.25)) + 16)
        px1 = int(round(0.5 * (x1 + x2))) - side // 2
        py1 = int(round(0.5 * (y1 + y2))) - side // 2
        cx1, cy1, cx2, cy2 = max(0, px1), max(0, py1), min(FULL, px1 + side), min(FULL, py1 + side)
        interp = cv2.INTER_AREA if side > 48 else cv2.INTER_LINEAR
        fl_sq = np.full((side, side), 153, dtype=np.uint8)
        d2_sq = np.full((side, side), 0 if use_dch1 else 153, dtype=np.uint8)
        pm_sq = np.zeros((side, side), dtype=np.uint8)
        mk_sq = np.zeros((side, side), dtype=np.uint8)
        if cx2 > cx1 and cy2 > cy1:
            phx1, phy1 = max(0, cx1 // 2), max(0, cy1 // 2)
            phx2, phy2 = min(HALF, max(phx1 + 1, (cx2 + 1) // 2)), min(HALF, max(phy1 + 1, (cy2 + 1) // 2))
            fz_u8 = np.clip((phys["flat_z"][phy1:phy2, phx1:phx2] + 6.0) * (255.0 / 10.0), 0.0, 255.0).astype(np.uint8)
            sy0, sy1, sx0, sx1 = cy1 - py1, cy2 - py1, cx1 - px1, cx2 - px1
            fl_sq[sy0:sy1, sx0:sx1] = cv2.resize(fz_u8, (cx2 - cx1, cy2 - cy1), interpolation=cv2.INTER_LINEAR)
            if use_dch1:
                d2_sq[sy0:sy1, sx0:sx1] = np.clip(dfg[cy1:cy2, cx1:cx2] * 255.0 + 0.5, 0, 255).astype(np.uint8)
            else:
                bz_u8 = np.clip((phys["bth_z"][phy1:phy2, phx1:phx2] + 6.0) * (255.0 / 10.0), 0.0, 255.0).astype(np.uint8)
                d2_sq[sy0:sy1, sx0:sx1] = cv2.resize(bz_u8, (cx2 - cx1, cy2 - cy1), interpolation=cv2.INTER_LINEAR)
            pm_sq[sy0:sy1, sx0:sx1] = full_w5_u8[cy1:cy2, cx1:cx2]
        mx1, my1, mx2, my2 = max(x1, px1), max(y1, py1), min(x2, px1 + side), min(y2, py1 + side)
        if mx2 > mx1 and my2 > my1:
            mk_sq[my1 - py1 : my2 - py1, mx1 - px1 : mx2 - px1] = m_u8[my1 - y1 : my2 - y1, mx1 - x1 : mx2 - x1] * 255
        patches[i, 0] = cv2.resize(fl_sq, (48, 48), interpolation=interp)
        patches[i, 1] = cv2.resize(d2_sq, (48, 48), interpolation=interp)
        patches[i, 2] = cv2.resize(pm_sq, (48, 48), interpolation=interp)
        patches[i, 3] = cv2.resize(mk_sq, (48, 48), interpolation=cv2.INTER_AREA if side > 48 else cv2.INTER_NEAREST)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    r_count = len(stem_gt)
    h_tgt = np.zeros(n, dtype=np.float32)
    m_tgt = np.zeros(n, dtype=np.float32)
    u_tgt = np.zeros(n, dtype=np.float32)
    if r_count > 0:
        pred_rles = [{"size": [FULL, FULL], "counts": c["rle"].encode("ascii")} for c in cands]
        hit_cnt = np.zeros(n, dtype=np.float32)
        iou_sum = np.zeros(n, dtype=np.float32)
        for gr in stem_gt:
            if not gr:
                continue
            best = mu.iou(pred_rles, gr, [0] * len(gr)).max(axis=1)
            hit = best > 0.5
            hit_cnt += hit.astype(np.float32)
            iou_sum += np.where(hit, best, 0.0).astype(np.float32)
        h_tgt = hit_cnt / float(r_count)
        m_tgt = np.where(hit_cnt > 0, iou_sum / np.maximum(hit_cnt, 1.0), 0.0).astype(np.float32)
        u_tgt = iou_sum / float(r_count)

    return split, stem, {
        "stem": stem,
        "fold": fold,
        "obs": stem[-2:],
        "r_count": r_count,
        "rles": [c["rle"] for c in cands],
        "kinds": [c["kind"] for c in cands],
        "boxes": np.array([c["box"] for c in cands], dtype=np.int32),
        "areas": areas_arr,
        "X": X,
        "h": h_tgt,
        "m": m_tgt,
        "u": u_tgt,
        "edges": edges,
        "patches": patches,
    }


def _init_w5_worker(idx: dict) -> None:
    global _W5_INDEX_GLOBAL
    _W5_INDEX_GLOBAL = idx


def build_wave5_pool(w5_root: Path, out_pool: Path, workers: int = 7, smoke: int = 0) -> dict:
    t0 = time.time()
    unpacked = unpack_wave5_archives(w5_root)
    idx = discover_wave5_index(w5_root)
    folds = data.folds()
    oof_stems = sorted(folds)
    test_stems = sorted(data.test_stems())
    if smoke > 0:
        oof_stems = oof_stems[:smoke]
        test_stems = test_stems[: min(smoke, len(test_stems))]

    cov = summarize_wave5_coverage(idx, len(oof_stems), len(test_stems))
    idx["use_dense_ch1"] = (cov["dense_probs"]["oof_pct"] >= 95.0) and (cov["dense_probs"]["test_pct"] >= 95.0)
    print(f"Wave-5 unpacked={unpacked} | use_dense_ch1={idx['use_dense_ch1']}\nCoverage: {json.dumps(cov, indent=2)}", flush=True)

    gt_recs = metric.load_records(oof_stems)
    tasks = [(s, "oof", int(folds[s]), gt_recs[s]) for s in oof_stems] + [(s, "test", -1, []) for s in test_stems]
    oof_out, test_out = {}, {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=_init_w5_worker, initargs=(idx,)) as ex:
        for done, (sp, st, res) in enumerate(ex.map(_extract_stem_wave5, tasks, chunksize=8), 1):
            (oof_out if sp == "oof" else test_out)[st] = res
            if done % 100 == 0 or done == len(tasks):
                print(f"  [EXP-015 extract] {done}/{len(tasks)} ({time.time()-t0:.1f}s)", flush=True)

    patches_oof_p = out_pool.with_name(out_pool.stem + "_patches_oof.npy")
    patches_test_p = out_pool.with_name(out_pool.stem + "_patches_test.npy")
    for sp, ddict, ppath in (("oof", oof_out, patches_oof_p), ("test", test_out, patches_test_p)):
        arrs, off = [], 0
        for s in sorted(ddict):
            ps = ddict[s].pop("patches")
            arrs.append(ps)
            ddict[s]["patch_slice"] = (off, off + len(ps))
            off += len(ps)
        np.save(ppath, np.concatenate(arrs, axis=0) if arrs else np.zeros((0, 4, 48, 48), dtype=np.uint8))

    out_pool.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "oof": oof_out,
        "test": test_out,
        "feature_names": FEATURE_NAMES_W5,
        "kind_list": KIND_LIST_W5,
        "patches_oof_path": str(patches_oof_p),
        "patches_test_path": str(patches_test_p),
        "coverage": cov,
    }
    with open(out_pool, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tot_o = sum(len(v["X"]) for v in oof_out.values())
    tot_t = sum(len(v["X"]) for v in test_out.values())
    print(f"Saved {out_pool} (OOF cands={tot_o}, Test cands={tot_t}, feat={len(FEATURE_NAMES_W5)}) in {time.time()-t0:.1f}s", flush=True)
    return payload


# ---------------------------------------------------------------------------
# Wave-5 Rescoring (GBDT + VisionCropTransformer) & Nested MWIS Optimization
# ---------------------------------------------------------------------------

def _node_weights_w5(entry: dict, u_score: np.ndarray, params: tuple[float, float, float, float, float, bool]) -> np.ndarray:
    tk, ta, ab, tb, w5_bonus, allow_solo = params
    kinds = entry["kinds"]
    n = len(kinds)
    w = np.full(n, -1.0, dtype=np.float32)
    if n == 0:
        return w
    iou_ta = entry["X"][:, 61]
    for i in range(n):
        ui = float(u_score[i])
        k = kinds[i]
        if k in ("ta_base", "ta_rescue"):
            if ui >= tk:
                w[i] = max(1e-4 + 1e-3 * ui, (ui - tb) + ab)
        elif iou_ta[i] >= 0.25:
            if ab < 5.0 and ui >= tb:
                wb = w5_bonus if k.startswith("w5_") else 0.0
                w[i] = ui - tb + wb
        else:
            if k == "solo" and not allow_solo:
                continue
            if ui >= ta:
                wb = w5_bonus if k.startswith("w5_") else 0.0
                w[i] = max(1e-4 + 1e-3 * ui, ui - tb + wb)
    return w


def solve_wave5(
    pool_path: Path,
    out_csv: Path,
    summary_json: Path,
    vit_epochs: int = 16,
    smoke: bool = False,
) -> dict:
    t0 = time.time()
    e11c.KIND_LIST = KIND_LIST_W5
    e11c.KIND_TO_IDX = {k: i for i, k in enumerate(KIND_LIST_W5)}

    pool = pickle.load(open(pool_path, "rb"))
    oof, test = pool["oof"], pool["test"]
    patches_oof = np.load(pool["patches_oof_path"], mmap_mode="r")
    patches_test = np.load(pool["patches_test_path"], mmap_mode="r")

    folds = data.folds()
    all_stems = sorted(oof)
    test_stems = sorted(test)
    f0_stems = [s for s in all_stems if folds[s] == 0]
    gt_recs = metric.load_records(all_stems)
    adj_oof = {s: e11b._build_adj(oof[s]) for s in all_stems}
    adj_test = {s: e11b._build_adj(test[s]) for s in test_stems}

    # Exact Track-A Clean sanity inside Wave-5 pool
    ta_u = {s: np.array([1.0 if k in ("ta_base", "ta_rescue") else 0.0 for k in oof[s]["kinds"]], dtype=np.float32) for s in all_stems}
    sm_ta, rows_ta, _ = e11b._eval_exact_stems(oof, adj_oof, all_stems, gt_recs, ta_u, {s: (0.5, 9.9, 9.9, 0.24) for s in all_stems})
    print(f"[Track-A Clean Baseline]: PQ={sm_ta['pq']:.5f} SQ={sm_ta['sq']:.5f} RQ={sm_ta['rq']:.5f} TP={sm_ta['tp']} FP={sm_ta['fp']} FN={sm_ta['fn']}", flush=True)

    device = e11c._get_device()
    epochs_run, seeds = (3, (0,)) if smoke else (vit_epochs, (0, 42))
    preds = {k: {"oof_h": {}, "oof_m": {}, "oof_u": {}, "test_h": [], "test_m": [], "test_u": []} for k in ("gbdt", "vit")}
    cache_path = Path(os.environ["W5_PREDS_CACHE"]) if os.environ.get("W5_PREDS_CACHE") else summary_json.with_suffix(".preds.pkl")

    def _valid_branch(br_d: dict | None) -> bool:
        if not isinstance(br_d, dict) or not br_d.get("oof_h") or not br_d.get("test_h"):
            return False
        return all(
            s in br_d["oof_h"] and len(br_d["oof_h"][s]) == len(oof[s]["X"])
            for s in all_stems
        ) and all(
            s in br_d["test_h"][0] and len(br_d["test_h"][0][s]) == len(test[s]["X"])
            for s in test_stems
        )

    have_gbdt, have_vit = False, False
    if not smoke and cache_path.exists():
        try:
            loaded = pickle.load(open(cache_path, "rb"))
            if _valid_branch(loaded.get("gbdt")):
                preds["gbdt"] = loaded["gbdt"]
                have_gbdt = True
            if _valid_branch(loaded.get("vit")):
                preds["vit"] = loaded["vit"]
                have_vit = True
            if have_gbdt or have_vit:
                print(f"Loaded Wave-5 preds cache {cache_path}: gbdt={have_gbdt} vit={have_vit}", flush=True)
        except Exception as exc:
            print(f"Warn: failed loading preds cache {cache_path}: {exc}", flush=True)

    if not (have_gbdt and have_vit):
        for fk in range(5):
            tr_s = [s for s in all_stems if folds[s] != fk and len(oof[s]["X"]) > 0]
            va_s = [s for s in all_stems if folds[s] == fk]
            if not tr_s or not va_s:
                continue
            if not have_gbdt:
                X_tr = np.vstack([oof[s]["X"] for s in tr_s])
                y_h = np.concatenate([oof[s]["h"] for s in tr_s])
                y_m = np.concatenate([oof[s]["m"] for s in tr_s])
                y_u = np.concatenate([oof[s]["u"] for s in tr_s])
                w_r = np.concatenate([np.full(len(oof[s]["X"]), float(oof[s]["r_count"]), dtype=np.float32) for s in tr_s])

                md_gb = e11c._fit_fold_gbdt_fast(X_tr, y_h, y_m, y_u, w_r, seeds=seeds)
                va_gb = e11c._predict_gbdt_batch(md_gb, oof, va_s)
                te_gb = e11c._predict_gbdt_batch(md_gb, test, test_stems)
                for s in va_s:
                    preds["gbdt"]["oof_h"][s], preds["gbdt"]["oof_m"][s], preds["gbdt"]["oof_u"][s] = va_gb[s]
                preds["gbdt"]["test_h"].append({s: te_gb[s][0] for s in test_stems})
                preds["gbdt"]["test_m"].append({s: te_gb[s][1] for s in test_stems})
                preds["gbdt"]["test_u"].append({s: te_gb[s][2] for s in test_stems})

            if not have_vit:
                vh, vm, vu, te_vit = e11c._fit_predict_vit_fold(
                    patches_oof, patches_test, oof, test, tr_s, va_s, test_stems, device=device, epochs=epochs_run, seeds=seeds
                )
                preds["vit"]["oof_h"].update(vh)
                preds["vit"]["oof_m"].update(vm)
                preds["vit"]["oof_u"].update(vu)
                preds["vit"]["test_h"].append({s: te_vit[s][0] for s in test_stems})
                preds["vit"]["test_m"].append({s: te_vit[s][1] for s in test_stems})
                preds["vit"]["test_u"].append({s: te_vit[s][2] for s in test_stems})
            print(f"  Fold {fk} trained (gbdt={not have_gbdt} vit={not have_vit} on {device}) elapsed={time.time()-t0:.1f}s", flush=True)

        if not smoke:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(preds, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved Wave-5 preds cache {cache_path}", flush=True)

    test_avg = {
        br: {
            s: (
                np.mean([d[s] for d in preds[br]["test_h"]], axis=0),
                np.mean([d[s] for d in preds[br]["test_m"]], axis=0),
                np.mean([d[s] for d in preds[br]["test_u"]], axis=0),
            )
            for s in test_stems
        }
        for br in ("gbdt", "vit")
    }

    def _logit(p: np.ndarray) -> np.ndarray:
        p_c = np.clip(p, 1e-4, 1.0 - 1e-4)
        return np.log(p_c / (1.0 - p_c))

    def _sig(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

    def _blend_u(hg: np.ndarray, mg: np.ndarray, ug: np.ndarray, hv: np.ndarray, mv: np.ndarray, uv: np.ndarray, spec: tuple[str, float, float], alp: float) -> np.ndarray:
        mode, wg, wv = spec
        if mode == "logit_h":
            hf = _sig(wg * _logit(hg) + wv * _logit(hv))
            mf = wg * mg + wv * mv
            uh = hf * mf
        else:
            uh = wg * (hg * mg) + wv * (hv * mv)
        ud = wg * ug + wv * uv
        return (alp * uh + (1.0 - alp) * ud).astype(np.float32)

    stems_by_fold = {fk: [s for s in all_stems if folds[s] == fk] for fk in range(5)}
    eval_stems = [s for fk in range(5) for s in stems_by_fold[fk] if s in preds["gbdt"]["oof_h"]]
    bws = (
        [("lin", 1.0, 0.0), ("logit_h", 0.70, 0.30), ("logit_h", 0.65, 0.35), ("logit_h", 0.60, 0.40)]
        if not smoke else [("lin", 1.0, 0.0), ("logit_h", 0.60, 0.40)]
    )
    alphas = (0.55, 0.65, 0.78) if not smoke else (0.70,)
    grid = [
        (tk, ta, ab, tb, w5b, allow)
        for tk in (0.195, 0.21, 0.225)
        for ta in (0.24, 0.265, 0.29)
        for ab in (0.02, 0.06, 9.9)
        for tb in (0.21, 0.225, 0.24)
        for w5b in (0.0, 0.015)
        for allow in (True, False)
    ] if not smoke else [(0.20, 0.24, 0.04, 0.22, 0.015, True)]

    u_cache = {}
    for spec in bws:
        for alp in alphas:
            um = {}
            for s in eval_stems:
                if len(oof[s]["X"]) == 0:
                    um[s] = np.zeros(0, dtype=np.float32)
                    continue
                um[s] = _blend_u(
                    preds["gbdt"]["oof_h"][s], preds["gbdt"]["oof_m"][s], preds["gbdt"]["oof_u"][s],
                    preds["vit"]["oof_h"][s], preds["vit"]["oof_m"][s], preds["vit"]["oof_u"][s],
                    spec, alp,
                )
            u_cache[(spec, alp)] = um

    exact_cache = {}

    def _run_nested(allowed_bws: list[tuple[str, float, float]], label: str) -> dict:
        fast_res = {}
        for bw in allowed_bws:
            for alp in alphas:
                um = u_cache[(bw, alp)]
                for p in grid:
                    per_s = {}
                    for s in eval_stems:
                        ent = oof[s]
                        if len(ent["X"]) == 0:
                            per_s[s] = (0.0, 0.0)
                            continue
                        wt = _node_weights_w5(ent, um[s], p)
                        sel = e11b._solve_mwis_exact(wt, adj_oof[s])
                        rc = float(max(1, ent["r_count"]))
                        per_s[s] = (float(ent["u"][sel].sum()) * rc, float(len(sel)) * rc)
                    fast_res[(bw, alp, p)] = per_s

        chosen, u_n, p_n = [], {}, {}
        for fk in range(5):
            va_k = stems_by_fold[fk]
            if not any(s in preds["gbdt"]["oof_h"] for s in va_k):
                continue
            in_k = [s for s in eval_stems if folds[s] != fk]
            if not in_k:
                in_k = eval_stems
            cands = sorted(
                ((sum(ps[s][0] - 0.235 * ps[s][1] for s in in_k), key) for key, ps in fast_res.items()),
                key=lambda z: -z[0],
            )
            best_pq, best_key = -1.0, cands[0][1]
            for _, key in cands[:32]:
                if key not in exact_cache:
                    bw, alp, p = key
                    um = u_cache[(bw, alp)]
                    rmap = {}
                    for s in eval_stems:
                        ent = oof[s]
                        wt = _node_weights_w5(ent, um[s], p)
                        sel = e11b._solve_mwis_exact(wt, adj_oof[s])
                        rstrs = e11b._decode_rles_disjoint(ent, sel, wt)
                        rmap[s] = metric.counts([{"size": [FULL, FULL], "counts": r.encode("ascii")} for r in rstrs], gt_recs[s])
                    exact_cache[key] = rmap
                pq_in = metric.summarize([exact_cache[key][s] for s in in_k])["pq"]
                if pq_in > best_pq:
                    best_pq, best_key = pq_in, key
            chosen.append(best_key)
            for s in va_k:
                if s in preds["gbdt"]["oof_h"]:
                    u_n[s] = u_cache[(best_key[0], best_key[1])][s]
                    p_n[s] = best_key[2]

        rows_f, rles_f = {}, {}
        for s in eval_stems:
            ent = oof[s]
            wt = _node_weights_w5(ent, u_n[s], p_n[s])
            sel = e11b._solve_mwis_exact(wt, adj_oof[s])
            rstrs = e11b._decode_rles_disjoint(ent, sel, wt)
            rles_f[s] = rstrs
            rows_f[s] = metric.counts([{"size": [FULL, FULL], "counts": r.encode("ascii")} for r in rstrs], gt_recs[s])
        sm = metric.summarize(list(rows_f.values()))
        s80_m, s80_s = e11b._sim_r80(eval_stems, rles_f, gt_recs, trials=10)
        one_r = [s for s in eval_stems if oof[s]["r_count"] == 1]
        sm_1r = metric.summarize([rows_f[s] for s in one_r]) if one_r else {"pq": 0.0}
        f0_k = [s for s in eval_stems if s in f0_stems]
        sm_f0 = metric.summarize([rows_f[s] for s in f0_k]) if f0_k else {"pq": 0.0}
        boot = e11b._paired_date_bootstrap(rows_f, rows_ta, eval_stems, n_boot=500 if smoke else 1000, seed=42)
        out = {
            "label": label,
            "pq": round(sm["pq"], 5),
            "sq": round(sm["sq"], 5),
            "rq": round(sm["rq"], 5),
            "tp": sm["tp"],
            "fp": sm["fp"],
            "fn": sm["fn"],
            "fold0_pq": round(sm_f0["pq"], 5),
            "single_rec_pq": round(sm_1r["pq"], 5),
            "sim_r80_mean": round(s80_m, 5),
            "sim_r80_std": round(s80_s, 5),
            "vs_ta_gain": round(boot["mean_gain"], 5),
            "vs_ta_ci95": [round(x, 5) for x in boot["ci95"]],
            "vs_ta_p_win": round(boot["p_win"], 4),
            "chosen": chosen,
            "_rles_f": rles_f,
        }
        print(f"[{label:18s}] OOF={out['pq']:.5f} SQ={out['sq']:.5f} RQ={out['rq']:.5f} TP={out['tp']} FP={out['fp']} FN={out['fn']} | 1rec={out['single_rec_pq']:.5f} Sim80={out['sim_r80_mean']:.5f} | boot={out['vs_ta_gain']:+.5f} p={out['vs_ta_p_win']:.4f}", flush=True)
        return out

    res_gb = _run_nested([("lin", 1.0, 0.0)], "W5_GBDT")
    res_gv = _run_nested(bws, "W5_GBDT_ViT")
    winner = res_gv if res_gv["pq"] >= res_gb["pq"] else res_gb

    oof_rles = winner.get("_rles_f", {})
    for d in (res_gb, res_gv):
        d.pop("_rles_f", None)
    if smoke:
        print("EXP-015 SOLVE SMOKE COMPLETE OK", flush=True)
        return {"winner": winner, "gbdt": res_gb, "joint": res_gv}
    with open(Path("/private/tmp/exp015_winner_oof_rles.pkl"), "wb") as f:
        pickle.dump(oof_rles, f, protocol=pickle.HIGHEST_PROTOCOL)

    chosen_all = winner["chosen"]
    modes_cnt = {}
    for c in chosen_all:
        modes_cnt[c[0][0]] = modes_cnt.get(c[0][0], 0) + 1
    cons_mode = max(modes_cnt, key=modes_cnt.get)
    wg = float(np.mean([c[0][1] for c in chosen_all]))
    wv = float(np.mean([c[0][2] for c in chosen_all]))
    s_bw = max(1e-6, wg + wv)
    cons_bw = (cons_mode, wg / s_bw, wv / s_bw)
    cons_alp = float(np.median([c[1] for c in chosen_all]))
    cons_p = (
        float(np.median([c[2][0] for c in chosen_all])),
        float(np.median([c[2][1] for c in chosen_all])),
        float(np.median([c[2][2] for c in chosen_all])),
        float(np.median([c[2][3] for c in chosen_all])),
        float(np.median([c[2][4] for c in chosen_all])),
        bool(np.mean([1.0 if c[2][5] else 0.0 for c in chosen_all]) >= 0.5),
    )

    rles_test_csv = {}
    for s in test_stems:
        if len(test[s]["X"]) == 0:
            rles_test_csv[s] = []
            continue
        um = _blend_u(
            test_avg["gbdt"][s][0], test_avg["gbdt"][s][1], test_avg["gbdt"][s][2],
            test_avg["vit"][s][0], test_avg["vit"][s][1], test_avg["vit"][s][2],
            cons_bw, cons_alp,
        )
        wt = _node_weights_w5(test[s], um, cons_p)
        sel = e11b._solve_mwis_exact(wt, adj_test[s])
        rstrs = e11b._decode_rles_disjoint(test[s], sel, wt)
        rles_test_csv[s] = [{"size": [FULL, FULL], "counts": r.encode("ascii")} for r in rstrs]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.write_csv(out_csv, rles_test_csv)
    probs = submission.validate(out_csv, data.test_stems())
    assert probs == [], f"Validation failed: {probs}"
    audit = tsa.audit_csv(out_csv, REPO / "results" / "submission.csv")
    report = {"tracka": sm_ta, "gbdt": res_gb, "joint": res_gv, "winner": winner, "cons_bw": cons_bw, "cons_alp": cons_alp, "cons_p": cons_p, "shift": audit}
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(report, indent=2, default=str))
    print(f"Saved {out_csv} (Validate=[] OK) & {summary_json}", flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["extract", "solve", "all"], default="all")
    ap.add_argument("--w5-root", type=Path, default=WAVE5_DEFAULT)
    ap.add_argument("--pool", type=Path, default=POOL_DEFAULT)
    ap.add_argument("--out-csv", type=Path, default=REPO / "gap_research" / "submissions" / "exp015_wave5_clean.csv")
    ap.add_argument("--summary-json", type=Path, default=REPO / "gap_research" / "scripts" / "exp015_summary.json")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--vit-epochs", type=int, default=16)
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()

    if args.mode in ("extract", "all"):
        build_wave5_pool(args.w5_root, args.pool, workers=args.workers, smoke=args.smoke)
    if args.mode in ("solve", "all"):
        solve_wave5(args.pool, args.out_csv, args.summary_json, vit_epochs=args.vit_epochs, smoke=(args.smoke > 0))


if __name__ == "__main__":
    main()
