"""From scored crop probabilities to disjoint filament instances.

`paint` is the panoptic merge of Kirillov et al. (2019): instances are laid down in confidence order and a pixel
belongs to the first instance that claims it, so the output can never overlap. Five knobs, all chosen on
out-of-fold predictions (scripts/tune.py), decide what is emitted:

    thr       probability threshold of the crop mask
    offset    erode (-) / dilate (+) the mask by this many pixels (annotators draw slightly looser outlines)
    conf      confidence floor: a box below it is not emitted (a false positive costs as much as a miss)
    drop      discard an instance if less than this share of its mask survives the instances painted before it
    min_area  discard instances smaller than this many pixels
"""
import cv2
import numpy as np
from pycocotools import mask as mu

FULL = 2048
_K = {r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2) for r in (1, 2, 3)}


def paint(inst, thr=0.5, offset=0, conf=0.2, drop=0.7, min_area=0):
    """inst: [(conf, (x1, y1, x2, y2), uint8 prob at window size)] sorted by confidence.
    -> [(conf, window, bool mask at window size)], pixel-exclusive."""
    occ, out = np.zeros((FULL, FULL), bool), []
    t = int(thr * 255 + 0.5)
    for c, (x1, y1, x2, y2), p in inst:
        if c < conf:
            break
        m = (p > t).astype(np.uint8)
        if offset:
            m = (cv2.dilate if offset > 0 else cv2.erode)(m, _K[abs(offset)])
        m = m.astype(bool)
        keep = m & ~occ[y1:y2, x1:x2]
        k = int(keep.sum())
        if k > min_area and k >= drop * int(m.sum()):
            occ[y1:y2, x1:x2] |= keep
            out.append((c, (x1, y1, x2, y2), keep))
    return out


def full_mask(window, local):
    x1, y1, x2, y2 = window
    m = np.zeros((FULL, FULL), np.uint8)
    m[y1:y2, x1:x2] = local
    return m


def to_rles(painted):
    """Painted instances -> COCO RLE dicts (size [2048, 2048], compressed counts as bytes)."""
    return [mu.encode(np.asfortranarray(full_mask(w, m))) for _, w, m in painted]


def save_cache(path, conf, windows, probs):
    """One photo's proposals: confidences, crop windows and the crop probabilities (uint8, flattened)."""
    flat = np.concatenate([p.ravel() for p in probs]) if len(probs) else np.zeros(0, np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, conf=np.asarray(conf, np.float32), box=np.asarray(windows, np.int32).reshape(-1, 4), p=flat)


def load_cache(path):
    """-> [(conf, window, prob)] sorted by confidence (the input of `paint`)."""
    z = np.load(path)
    conf, box, flat = z["conf"], z["box"], z["p"]
    inst, o = [], 0
    for c, (x1, y1, x2, y2) in zip(conf, box):
        n = int(y2 - y1) * int(x2 - x1)
        inst.append((float(c), (int(x1), int(y1), int(x2), int(y2)), flat[o:o + n].reshape(y2 - y1, x2 - x1)))
        o += n
    return sorted(inst, key=lambda x: -x[0])
