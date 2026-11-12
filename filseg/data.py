"""Paths, the MAGFiLO annotation index and the locked date-grouped 5-fold split.

One property of the annotation file drives the design: the same photo appears under several ``image_id`` values,
one per human annotator ("record"), and the records disagree (one annotator scored against another reaches only
PQ 0.33). A photo therefore owns a *list* of records; training uses every record and validation pools the
official counts over all of them.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

FULL = 2048                                                  # every GONG H-alpha frame is 2048 x 2048
REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("MAGFILO_ROOT", REPO / "data" / "MAGFiLO_1.0_Kaggle_2026"))
WORK = Path(os.environ.get("FILSEG_WORK", REPO / "work"))   # caches, runs, predictions (never committed)
WEIGHTS = Path(os.environ.get("FILSEG_WEIGHTS", WORK / "weights"))   # set to <repo>/weights to use the released models
TRAIN_IMAGES = ROOT / "train" / "train_images"
TEST_IMAGES = ROOT / "test" / "test_images"
TRAIN_JSON = ROOT / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
N_FOLDS = 5
SMOKE = int(os.environ.get("FILSEG_SMOKE", 0))               # >0: use only this many photos per split (code check)


def coco_index():
    """-> (stem -> [record id, ...] in file order, record id -> [annotation, ...])."""
    d = json.load(open(TRAIN_JSON))
    recs, anns = {}, {}
    for im in d["images"]:
        recs.setdefault(Path(im["file_name"]).stem, []).append(im["id"])
    for a in d["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    return recs, anns


def ann_rle(a):
    """Polygon -> RLE with pycocotools, the rasterisation the official metric scores against."""
    return mu.merge(mu.frPyObjects([p for p in a["segmentation"] if len(p) >= 6], FULL, FULL))


def ann_mask(a):
    return mu.decode(ann_rle(a))


def folds():
    """stem -> fold. Photos of one calendar date share a fold (same filaments seen by several GONG sites)."""
    return json.load(open(REPO / "folds" / "folds.json"))


def split(fold):
    """-> (train stems, validation stems) for one fold; fold < 0 trains on everything and validates on fold 0."""
    f = folds()
    tr, va = sorted(s for s, k in f.items() if k != fold), sorted(s for s, k in f.items() if k == max(fold, 0))
    return (tr[:SMOKE], va[:SMOKE]) if SMOKE else (tr, va)


def test_stems():
    stems = sorted(p.stem for p in TEST_IMAGES.glob("*.jpeg"))
    return stems[:SMOKE] if SMOKE else stems


def image_path(stem):
    p = TRAIN_IMAGES / f"{stem}.jpeg"
    return p if p.exists() else TEST_IMAGES / f"{stem}.jpeg"


def read_gray(stem):
    return cv2.imread(str(image_path(stem)), cv2.IMREAD_GRAYSCALE)


def make_folds(seed=0):
    """How folds/folds.json was built: greedy assignment of whole observation dates, largest first, balancing photos
    and filaments per fold. Kept for documentation; the committed JSON is the split every reported number uses."""
    recs, anns = coco_index()
    groups = {}
    for s in recs:
        groups.setdefault(s[:8], []).append(s)
    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda k: -len(groups[k]))
    load, fil, out = np.zeros(N_FOLDS), np.zeros(N_FOLDS), {}
    for k in keys:
        cost = load / max(load.sum(), 1) + fil / max(fil.sum(), 1)
        f = int(np.argmin(cost + 1e-9 * rng.random(N_FOLDS)))
        for s in groups[k]:
            out[s] = f
        load[f] += len(groups[k])
        fil[f] += sum(len(anns.get(r, [])) for s in groups[k] for r in recs[s])
    return out
