"""Build compact trimmed uint8 continuous-probability Fold-0 cache across all 12 two-stage streams + YOLOv8s-seg."""
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gap_research" / "scripts"))
import eval_harness as eh

OUT_PKL = Path("/private/tmp/gap_soft_fold0.pkl")
TRIM_U8 = 15  # keep p >= ~0.06 prob pixels so soft wings & calibration never truncate prematurely


def trim_soft_prob(c, box, p_u8, min_u8=TRIM_U8):
    nz = p_u8 > min_u8
    ys, xs = np.nonzero(nz)
    if not len(ys):
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    tb = (box[0] + x1, box[1] + y1, box[0] + x2, box[1] + y2)
    return float(c), tb, np.ascontiguousarray(p_u8[y1:y2, x1:x2], dtype=np.uint8)


def load_stream_soft(npz_path, tta=4, conf_min=0.05):
    z = np.load(npz_path)
    conf, box = z["conf"], z["box"]
    flat = z[f"p{tta}"] if f"p{tta}" in z.files else z["p"]
    probs, o = [], 0
    for x1, y1, x2, y2 in box:
        n = int((y2 - y1) * (x2 - x1))
        probs.append(flat[o:o + n].reshape(y2 - y1, x2 - x1))
        o += n
    order = np.argsort(-conf)
    out = []
    for k in order:
        c = float(conf[k])
        if c < conf_min:
            continue
        t = trim_soft_prob(c, tuple(int(v) for v in box[k]), probs[k])
        if t is not None:
            out.append(t)
    return out


def load_yolo_soft(yolo_dict, stem, conf_min=0.05):
    """YOLOv8s-seg is stored as binary RLEs; use soft tapered mask (eroded vs raw) to preserve edge uncertainty."""
    v = yolo_dict.get(stem, {"rle": [], "conf": []})
    cf = v.get("conf") or v.get("score") or []
    pairs = sorted(zip(cf, v["rle"]), key=lambda x: -x[0])
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out = []
    for c, cnt in pairs:
        c = float(c)
        if c < conf_min:
            continue
        m = mu.decode({"size": [2048, 2048], "counts": cnt.encode("ascii")}).astype(np.uint8)
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        sub = m[y1:y2, x1:x2]
        er = cv2.erode(sub, k1)
        # Interior core = 230 (~0.90), outer 1-px fringe = 115 (~0.45) to reflect offset=-1 preference cleanly!
        soft = np.where(er > 0, 230, np.where(sub > 0, 115, 0)).astype(np.uint8)
        out.append((c, (x1, y1, x2, y2), soft))
    return out


def main():
    t0 = time.time()
    with open(eh.FP_ROOT / "yolo_val_raw.json") as f:
        yolo_val = json.load(f)

    data = {k: {} for k in list(eh.STREAM_META) + ["YOLO"]}
    total_bytes = 0
    total_inst = 0
    for i, s in enumerate(eh.FOLD0_STEMS):
        for k, meta in eh.STREAM_META.items():
            # Keep down to 0.05 (0.01 for 5fold test alignment checks)
            cmin = 0.01 if k in ("ms", "s", "main") else 0.05
            lst = load_stream_soft(meta["val"] / f"{s}.npz", tta=meta["tta"], conf_min=cmin)
            data[k][s] = lst
            total_inst += len(lst)
            total_bytes += sum(arr.nbytes for _, _, arr in lst)
        yl = load_yolo_soft(yolo_val, s, conf_min=0.05)
        data["YOLO"][s] = yl
        total_inst += len(yl)
        total_bytes += sum(arr.nbytes for _, _, arr in yl)

    with open(OUT_PKL, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Built {OUT_PKL} in {time.time()-t0:.2f}s | instances={total_inst} | arrays={total_bytes/1e6:.1f} MB | file={OUT_PKL.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
