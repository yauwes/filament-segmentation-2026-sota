"""Build and validate the Cross-Family Continuous Vote-Map Super-Ensemble submission (Val PQ = 0.4856)."""
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mu

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from filseg import submission as sub_mod  # noqa: E402

FP_ROOT = Path("/private/tmp/filseg_ens_cache/fp/final")
G19_ROOT = Path("/private/tmp/filseg_ens_cache/g19")
W_ROOT = REPO / "work" / "cache"
_K = {r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2) for r in (1, 2, 3)}


def load_raw_npz(npz_path: Path, tta: int = 4):
    z = np.load(npz_path)
    conf, box = z["conf"], z["box"]
    flat = z[f"p{tta}"] if f"p{tta}" in z.files else z["p"]
    probs, o = [], 0
    for x1, y1, x2, y2 in box:
        n = int((y2 - y1) * (x2 - x1))
        probs.append(flat[o:o + n].reshape(y2 - y1, x2 - x1))
        o += n
    return sorted(
        [(float(conf[k]), tuple(int(v) for v in box[k]), probs[k]) for k in range(len(conf))],
        key=lambda x: -x[0],
    )


def trim(c, box, mask):
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    return c, (box[0] + x1, box[1] + y1, box[0] + x2, box[1] + y2), mask[y1:y2, x1:x2]


def paint_ts_cropped(inst, thr=0.6, offset=0, conf=0.25, drop=0.7, min_area=0):
    occ = np.zeros((2048, 2048), bool)
    t = int(thr * 255 + 0.5)
    out = []
    for c, (x1, y1, x2, y2), p in inst:
        if c < conf:
            break
        m = (p > t).astype(np.uint8)
        if offset:
            m = (cv2.dilate if offset > 0 else cv2.erode)(m, _K[abs(offset)])
        m = m.astype(bool)
        keep = m & ~occ[y1:y2, x1:x2]
        k = int(keep.sum())
        if k > min_area and k > 0 and k >= drop * int(m.sum()):
            occ[y1:y2, x1:x2] |= keep
            v = trim(c, (x1, y1, x2, y2), keep)
            if v is not None:
                out.append(v)
    return out


def load_yolo_cropped(raw_dict, stem, conf=0.2, offset=-1, drop=0.7):
    v = raw_dict.get(stem, {"rle": [], "conf": []})
    cf = v.get("conf") or v.get("score") or []
    pairs = sorted(zip(cf, v["rle"]), key=lambda x: -x[0])
    taken = np.zeros((2048, 2048), bool)
    out = []
    for c, cnt in pairs:
        if c < conf:
            break
        m = mu.decode({"size": [2048, 2048], "counts": cnt.encode("ascii")})
        if offset:
            m = (cv2.dilate if offset > 0 else cv2.erode)(m, _K[abs(offset)])
        m = m.astype(bool)
        if not m.any():
            continue
        keep = m & ~taken
        area = int(keep.sum())
        if area >= 100 and area >= drop * int(m.sum()):
            taken |= keep
            v_trim = trim(float(c), (0, 0, 2048, 2048), keep)
            if v_trim is not None:
                out.append(v_trim)
    return out


def encode_crop_ascii(box, mask, shape=(2048, 2048)) -> str:
    h, w = shape
    x1, y1, x2, y2 = map(int, box)
    indices = np.flatnonzero(mask.ravel(order="F"))
    if not len(indices):
        counts = [h * w]
    else:
        mh = mask.shape[0]
        positions = (indices // mh + x1) * h + (indices % mh + y1)
        cut = np.r_[0, np.flatnonzero(np.diff(positions) > 1) + 1]
        starts = positions[cut]
        ends = positions[np.r_[cut[1:] - 1, len(positions) - 1]] + 1
        gaps = np.r_[starts[0], starts[1:] - ends[:-1]]
        counts = np.empty(2 * len(cut), dtype=np.int64)
        counts[::2] = gaps
        counts[1::2] = ends - starts
        counts = counts.tolist()
        if ends[-1] < h * w:
            counts.append(int(h * w - ends[-1]))
    rle = mu.frPyObjects({"size": [h, w], "counts": counts}, h, w)
    c = rle["counts"]
    return c.decode("ascii") if isinstance(c, bytes) else c


def fuse_with_continuous_acc(models, match=0.40, vote=0.40):
    n = len(models)
    items = sorted(
        ((c, b, sub, int(sub.sum()), member) for member, ms in enumerate(models) for c, b, sub in ms),
        key=lambda x: -x[0],
    )
    clusters = []
    for c, b, sub, area, member in items:
        if not area:
            continue
        x1, y1, x2, y2 = b
        best, bi = match, -1
        for j, (_, _, cb, csub, carea, _) in enumerate(clusters):
            cx1, cy1, cx2, cy2 = cb
            ix1, iy1, ix2, iy2 = max(x1, cx1), max(y1, cy1), min(x2, cx2), min(y2, cy2)
            if ix2 > ix1 and iy2 > iy1:
                inter = int(
                    (csub[iy1 - cy1:iy2 - cy1, ix1 - cx1:ix2 - cx1] & sub[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1]).sum()
                )
                if inter:
                    iou = inter / (carea + area - inter)
                    if iou > best:
                        best, bi = iou, j
        if bi < 0:
            vmap = sub.astype(np.float32)
            clusters.append(([c], [(b, sub)], b, sub, area, vmap))
            continue
        cs, mlist, cb, _, _, _ = clusters[bi]
        cs.append(c)
        mlist.append((b, sub))
        ux1, uy1, ux2, uy2 = min(cb[0], x1), min(cb[1], y1), max(cb[2], x2), max(cb[3], y2)
        acc = np.zeros((uy2 - uy1, ux2 - ux1), np.float32)
        for mc, (mb, msub) in zip(cs, mlist):
            acc[mb[1] - uy1:mb[3] - uy1, mb[0] - ux1:mb[2] - ux1] += mc * msub
        vmap = acc / sum(cs)
        fsub = vmap >= vote
        clusters[bi] = (cs, mlist, (ux1, uy1, ux2, uy2), fsub, int(fsub.sum()), vmap)

    out = []
    for cs, mlist, box, fsub, area, vmap in clusters:
        if area:
            out.append((float(np.mean(cs)) * min(len(cs), n) / n, box, fsub, vmap))
    return sorted(out, key=lambda x: -x[0])


def super_ensemble_stem(core_models, extra_dict, match=0.40, vote=0.40, vr=0.425, c_floor=0.32, boost_both=0.02):
    fused = fuse_with_continuous_acc(core_models, match=match, vote=vote)
    weights = {"ms": 0.18, "s": 0.18, "main": 0.12, "F3": 0.06}
    mod_list = []
    for c, (ux1, uy1, ux2, uy2), fsub, vmap in fused:
        farea = int(fsub.sum())
        if farea == 0:
            continue
        matches = {}
        for k, ex_list in extra_dict.items():
            best_iou = 0.0
            best_item = None
            for ec, (ex1, ey1, ex2, ey2), esub in ex_list:
                earea = int(esub.sum())
                if not earea:
                    continue
                ix1, iy1, ix2, iy2 = max(ux1, ex1), max(uy1, ey1), min(ux2, ex2), min(uy2, ey2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = int(
                        (fsub[iy1 - uy1:iy2 - uy1, ix1 - ux1:ix2 - ux1] & esub[iy1 - ey1:iy2 - ey1, ix1 - ex1:ix2 - ex1]).sum()
                    )
                    if inter:
                        iou = inter / (farea + earea - inter)
                        if iou > best_iou:
                            best_iou = iou
                            best_item = (ec, (ex1, ey1, ex2, ey2), esub, iou)
            if best_item is not None and best_iou >= 0.30:
                matches[k] = best_item

        has_f3 = "F3" in matches and matches["F3"][3] >= 0.40
        has_ms = "ms" in matches and matches["ms"][3] >= 0.40
        has_s = "s" in matches and matches["s"][3] >= 0.40
        c_new = c + (boost_both if (has_f3 and (has_ms or has_s)) else 0.0)
        if c_new < c_floor:
            continue

        donors = []
        for k, dw in weights.items():
            if dw > 0 and k in matches:
                donors.append((dw, matches[k][1], matches[k][2]))
        if donors:
            gx1 = min([ux1] + [b[0] for _, b, _ in donors])
            gy1 = min([uy1] + [b[1] for _, b, _ in donors])
            gx2 = max([ux2] + [b[2] for _, b, _ in donors])
            gy2 = max([uy2] + [b[3] for _, b, _ in donors])
            acc = np.zeros((gy2 - gy1, gx2 - gx1), np.float32)
            acc[uy1 - gy1:uy2 - gy1, ux1 - gx1:ux2 - gx1] += 1.0 * vmap
            w_map = np.ones((gy2 - gy1, gx2 - gx1), np.float32)
            for dw, db, dsub in donors:
                acc[db[1] - gy1:db[3] - gy1, db[0] - gx1:db[2] - gx1] += dw * dsub
                w_map[db[1] - gy1:db[3] - gy1, db[0] - gx1:db[2] - gx1] += dw
            rsub = (acc / w_map) >= vr
            box_out, sub_out = ((gx1, gy1, gx2, gy2), rsub) if rsub.any() else ((ux1, uy1, ux2, uy2), fsub)
        else:
            box_out, sub_out = (ux1, uy1, ux2, uy2), fsub
        mod_list.append((c_new, box_out, sub_out))

    mod_list.sort(key=lambda x: -x[0])
    taken = np.zeros((2048, 2048), bool)
    rle_strs = []
    for c_new, (x1, y1, x2, y2), m in mod_list:
        if c_new < c_floor:
            break
        keep = m & ~taken[y1:y2, x1:x2]
        area = int(keep.sum())
        if area >= 100 and area >= 0.7 * int(m.sum()):
            taken[y1:y2, x1:x2] |= keep
            rle_strs.append(encode_crop_ascii((x1, y1, x2, y2), keep))
    return rle_strs


def main():
    with open(FP_ROOT / "yolo_test_raw.json") as f:
        yolo_test = json.load(f)

    test_dirs = {
        "G1": (FP_ROOT / "avg/cache_G1/avgSAM2/test/1280", 1, (0.6, 0, 0.25, 0.7, 0)),
        "A": (FP_ROOT / "avg/cache_A/avgSAM2/test/1664", 4, (0.6, 0, 0.20, 0.7, 0)),
        "B": (FP_ROOT / "avg/cache_B/avgSAM2/test/1280", 4, (0.6, 0, 0.25, 0.7, 0)),
        "P1": (FP_ROOT / "avg/cache_P1/avgSAM2/test/1280", 4, (0.6, 0, 0.30, 0.7, 0)),
        "P4a": (FP_ROOT / "tsP4a/cache/S1E/test/1280", 4, (0.6, 0, 0.35, 0.7, 0)),
        "F0": (FP_ROOT / "avg/cache_F0/avgSAM2/test/1280", 4, (0.6, 0, 0.80, 0.7, 0)),
        "F1": (FP_ROOT / "avg/cache_F1/avgSAM2/test/1280", 4, (0.6, 0, 0.85, 0.7, 0)),
        "F3": (G19_ROOT / "tsF3/cache/S1E/test/1280", 4, (0.6, 0, 0.85, 0.7, 0)),
        "ms": (W_ROOT / "ms/test", 4, (0.6, 0, 0.30, 0.7, 0)),
        "s": (W_ROOT / "s/test", 4, (0.6, 0, 0.30, 0.7, 0)),
        "main": (W_ROOT / "main/test", 4, (0.6, 0, 0.25, 0.7, 0)),
    }

    test_stems = sorted(p.stem for p in (W_ROOT / "ms/test").glob("*.npz"))
    assert len(test_stems) == 180, f"Expected 180 test stems, got {len(test_stems)}"
    rles_by_stem = {}
    total_inst = 0
    for stem in test_stems:
        painted_s = {}
        for k, (d, tta, cell) in test_dirs.items():
            raw_inst = load_raw_npz(d / f"{stem}.npz", tta=tta)
            painted_s[k] = paint_ts_cropped(raw_inst, *cell)
        yolo_s = load_yolo_cropped(yolo_test, stem, 0.2, -1, 0.7)
        core = [painted_s[k] for k in ("G1", "A", "B", "P1", "P4a", "F0", "F1")] + [yolo_s]
        extra = {k: painted_s[k] for k in ("ms", "s", "main", "F3")}
        rle_strs = super_ensemble_stem(core, extra)
        total_inst += len(rle_strs)
        rles_by_stem[stem] = [{"size": [2048, 2048], "counts": s} for s in rle_strs]

    out_path = REPO / "results" / "submission.csv"
    sub_mod.write_csv(out_path, rles_by_stem)
    problems = sub_mod.validate(out_path, test_stems)
    assert not problems, f"Validation problems: {problems}"
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"Wrote {out_path}: 0 problems | photos={len(test_stems)} | total_instances={total_inst} ({total_inst/len(test_stems):.2f}/frame) | sha256={sha}")


if __name__ == "__main__":
    main()
