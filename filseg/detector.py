"""Stage 1 -- where are the filaments? A YOLO11 box detector on the full 2048-px frame (resized to 1280).

The detector only proposes boxes and a confidence; it never draws a mask. Its confidence is the instance score used
later to decide which filaments are emitted and which one owns a contested pixel.
"""
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from . import data

FULL = data.FULL


def build_dataset(root, train, val, labels="records", extra=None):
    """YOLO-format dataset. labels="records": every annotator record of a training photo is one sample (the photo
    with that annotator's boxes), so a filament drawn by two of three annotators is a positive in two of three
    copies and the confidence learns how likely an annotator is to mark it. Validation uses the first record.
    extra: directory of pseudo-labelled archive frames (images/*.jpg, labels/*.txt from `pseudo_label`) added to
    the training split."""
    recs, anns = data.coco_index()
    for name, stems in (("train", train), ("val", val)):
        for sub in ("images", "labels"):
            shutil.rmtree(root / sub / name, ignore_errors=True)
            (root / sub / name).mkdir(parents=True)
        for s in stems:
            rids = recs[s] if (labels == "records" and name == "train") else recs[s][:1]
            for rid in rids:
                stem = s if len(rids) == 1 else f"{s}__{rid}"
                os.symlink(data.TRAIN_IMAGES / f"{s}.jpeg", root / "images" / name / f"{stem}.jpg")
                (root / "labels" / name / f"{stem}.txt").write_text("\n".join(
                    f"0 {(x + w / 2) / FULL:.6f} {(y + h / 2) / FULL:.6f} {w / FULL:.6f} {h / FULL:.6f}"
                    for x, y, w, h in (a["bbox"] for a in anns.get(rid, []))))
    for lab in (sorted(Path(extra, "labels").glob("*.txt")) if extra else []):
        os.symlink(Path(extra, "images", f"{lab.stem}.jpg").resolve(), root / "images" / "train" / f"pseudo__{lab.stem}.jpg")
        (root / "labels" / "train" / f"pseudo__{lab.stem}.txt").write_text(lab.read_text())
    (root / "data.yaml").write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n")
    return root / "data.yaml"


def train(fold, name, weights="yolo11m.pt", imgsz=1280, epochs=50, patience=10, batch=8, labels="records",
          lr0=8.7e-4, device=0, workers=8, cache="ram", extra=None):
    """Train on the four other folds, early-stop on fold `fold` (box mAP), keep best.pt -> WORK/weights.
    extra: name of the teacher detector whose pseudo-labelled archive frames (WORK/pseudo/<extra>_f<fold>) join."""
    from ultralytics import YOLO
    tr, va = data.split(fold)
    yaml = build_dataset(data.WORK / "det_data" / f"{name}_f{fold}", tr, va, labels,
                         data.WORK / "pseudo" / f"{extra}_f{fold}" if extra else None)
    r = YOLO(weights).train(data=str(yaml), optimizer="AdamW", lr0=lr0, warmup_epochs=5, epochs=epochs, imgsz=imgsz,
                            batch=batch, patience=patience, pretrained=True, device=device, workers=workers,
                            cache=cache, plots=False, project=str(data.WORK / "det" / name), name=f"fold{fold}",
                            exist_ok=True, verbose=False, seed=fold)
    out = data.WEIGHTS / f"det_{name}_f{fold}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(r.save_dir) / "weights" / "best.pt", out)
    return out


def load(weights):
    from ultralytics import YOLO
    return YOLO(str(weights))


def _iou(a, b):
    iw = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter + 1e-9)


def wbf(views, thr=0.55):
    """Weighted Boxes Fusion (Solovyev et al. 2021). views: list of (n, 5) arrays [x1, y1, x2, y2, conf], one per
    model/flip. Boxes sorted by confidence join the fused box they overlap most (IoU > thr); coordinates are the
    confidence-weighted mean and the confidence is the cluster mean scaled by the share of views that saw it, so
    a box found in one of three views keeps a third of its confidence. -> [(xyxy, conf)], most confident first."""
    n = len(views)
    items = sorted(((np.asarray(b[:4], float), float(b[4])) for v in views for b in v), key=lambda x: -x[1])
    clusters, fused = [], []
    for box, c in items:
        j = -1
        if fused:
            ious = _iou(box, np.array([f[0] for f in fused]))
            if ious.max() > thr:
                j = int(ious.argmax())
        if j < 0:
            clusters.append([(box, c)])
            fused.append((box, c))
            continue
        clusters[j].append((box, c))
        w = np.array([x[1] for x in clusters[j]])
        fused[j] = ((np.array([x[0] for x in clusters[j]]) * w[:, None]).sum(0) / w.sum(), float(w.mean()))
    out = [(f[0], f[1] * min(len(cl), n) / n) for f, cl in zip(fused, clusters)]
    return sorted(out, key=lambda x: -x[1])


def predict(models, img, imgsz=1280, conf=0.05, max_det=60, tta=True, device=0):
    """Boxes of one photo: every model sees the photo and (tta) its horizontal and vertical flips; all views are
    fused by WBF. -> (boxes (n, 4) float xyxy, conf (n,)), most confident first, at most `max_det`."""
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    views = []
    for m in models:
        for ax in ((None, 1, 0) if tta else (None,)):
            x = bgr if ax is None else np.ascontiguousarray(np.flip(bgr, ax))
            r = m.predict(x, imgsz=imgsz, conf=conf, max_det=max_det, verbose=False, device=device)[0]
            b = r.boxes.xyxy.cpu().numpy().astype(np.float64).copy()
            if ax == 1:
                b[:, [0, 2]] = FULL - b[:, [2, 0]]
            elif ax == 0:
                b[:, [1, 3]] = FULL - b[:, [3, 1]]
            views.append(np.c_[b, r.boxes.conf.cpu().numpy()])
    fused = wbf(views)[:max_det] if len(views) > 1 else [(v[:4], float(v[4])) for v in views[0]]
    boxes = np.array([f[0] for f in fused], np.float64).reshape(-1, 4)
    return boxes, np.array([f[1] for f in fused], np.float32)


# ------------------------------------------------------------------ self-training on unlabeled GONG archive frames
def clean_archive(img):
    """GONG archive JPEG -> competition-style frame. The public archive stamps a logo and a caption outside the
    disk; beyond 1.1 solar radii, pixels that deviate from the median of their radius are that overlay and are
    replaced by the median (the on-disk image is untouched)."""
    from scipy import ndimage
    ys, xs = np.nonzero(img > 0.25 * np.percentile(img, 99))
    cx, cy, r = xs.mean(), ys.mean(), np.sqrt(len(xs) / np.pi)
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    rad = np.hypot(yy - cy, xx - cx)
    far = rad > 1.1 * r
    bins = (rad // 2).astype(np.int32)
    idx = np.unique(bins[far])
    med = np.zeros(bins.max() + 1, np.float32)
    med[idx] = ndimage.median(img, labels=np.where(far, bins, -1), index=idx)
    prof = med[bins]
    bad = far & (np.abs(img.astype(np.float32) - prof) > 12)
    bad = cv2.dilate(bad.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & far
    out = img.copy()
    out[bad] = prof[bad].astype(np.uint8)
    return out


def _clean_one(args):
    src, dst = args
    img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if img is not None and img.shape == (FULL, FULL) and not dst.exists():
        cv2.imwrite(str(dst), clean_archive(img), [cv2.IMWRITE_JPEG_QUALITY, 95])


def pseudo_label(fold, name, src, conf=0.25, imgsz=1280, device=0, procs=8):
    """Self-training set of one fold (Lee 2013; Xie et al. 2020): the fold's own detector `name`, which never saw
    the fold's validation annotations, labels unlabeled archive frames (photo + flips, WBF); boxes at or above
    `conf` become YOLO labels. Only images and the model's own boxes are used. -> WORK/pseudo/<name>_f<fold>"""
    import multiprocessing as mp
    known = set(data.folds()) | {p.stem for p in data.TEST_IMAGES.glob("*.jpeg")}
    shared = data.WORK / "pseudo" / "images"
    shared.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in Path(src).rglob("*.jpg") if f.stem not in known)       # never a competition photo
    with mp.get_context("fork").Pool(procs) as pool:
        pool.map(_clean_one, [(f, shared / f.name) for f in files], chunksize=8)
    out = data.WORK / "pseudo" / f"{name}_f{fold}"
    shutil.rmtree(out, ignore_errors=True)
    (out / "labels").mkdir(parents=True)
    os.symlink(shared, out / "images")
    model, n = load(data.WEIGHTS / f"det_{name}_f{fold}.pt"), 0
    for i, f in enumerate(sorted(shared.glob("*.jpg")), 1):
        boxes, cf = predict([model], cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), imgsz, device=device)
        keep = boxes[cf >= conf]
        (out / "labels" / f"{f.stem}.txt").write_text("\n".join(
            f"0 {(b[0] + b[2]) / 2 / FULL:.6f} {(b[1] + b[3]) / 2 / FULL:.6f} {(b[2] - b[0]) / FULL:.6f} {(b[3] - b[1]) / FULL:.6f}"
            for b in keep))
        n += len(keep)
        if i % 500 == 0:
            print(f"pseudo {name} fold {fold}: {i} frames, {n} boxes", flush=True)
    print(f"PSEUDO DONE {name} fold {fold}: {i} frames, {n} boxes ({n / i:.2f} per frame)", flush=True)
    return out
