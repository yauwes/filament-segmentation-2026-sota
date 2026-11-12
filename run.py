"""Command line of the pipeline. Every reported number and the submission come from these steps:

    python run.py planes                                   # cache the enhanced input planes of all photos
    python run.py det  --fold K --name m1280               # stage 1: box detector, for K = 0..4
    python run.py seg  --fold K --name e4                  # stage 2: crop segmenter, for K = 0..4
    python run.py pseudo --fold K --det m1280 --src DIR    # optional self-training: label unlabeled GONG frames
    python run.py det  --fold K --name m1280s --extra m1280 --epochs 20    # ... and retrain the detector with them
    python run.py predict --run main --det m1280 --seg e4 --fold K     # out-of-fold proposals (K = 0..4)
    python run.py tune --run main                          # choose the emission rule on out-of-fold PQ
    python run.py predict --run main --det m1280 --seg e4 --test       # test proposals, 5-fold ensemble
    python run.py submit --run main --out submission.csv

`--det` / `--seg` accept comma-separated names: detectors are fused by Weighted Boxes Fusion, segmenters by averaging
their probabilities.
"""
import argparse
import itertools
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from filseg import data, detector, enhance, metric, postprocess, segmenter, submission

GRID = dict(thr=(0.5, 0.6, 0.7, 0.8), offset=(-1, 0, 1), conf=tuple(round(0.10 + 0.025 * i, 3) for i in range(13)),
            drop=(0.5, 0.7, 0.85, 0.95), min_area=(0,))
CELL = ("thr", "offset", "conf", "drop", "min_area")


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def cmd_planes(a):
    stems = sorted(data.folds()) + data.test_stems()
    with mp.get_context("fork").Pool(a.procs) as pool:
        for i, _ in enumerate(pool.imap_unordered(_plane, stems, chunksize=4), 1):
            if i % 100 == 0 or i == len(stems):
                print(f"planes {i}/{len(stems)}", flush=True)


def _plane(stem):
    enhance.planes(stem)


def cmd_det(a):
    out = data.WEIGHTS / f"det_{a.name}_f{a.fold}.pt"
    if out.exists():                                         # resumable: a finished fold is never retrained
        return print("DET EXISTS", out, flush=True)
    print("DET DONE", detector.train(a.fold, a.name, a.weights, a.imgsz, a.epochs, a.patience, a.batch, a.labels,
                                     device=0 if _device() == "cuda" else _device(), workers=a.workers,
                                     cache=a.cache if a.cache != "none" else False, extra=a.extra), flush=True)


def cmd_pseudo(a):
    detector.pseudo_label(a.fold, a.det, a.src, a.conf, device=0 if _device() == "cuda" else _device(), procs=a.procs)


def cmd_seg(a):
    out = data.WEIGHTS / f"seg_{a.name}_f{a.fold}.pth"
    if out.exists() and not out.with_suffix(".history.json").exists():      # restored final weights: keep them
        return print("SEG EXISTS", out, flush=True)
    print("SEG DONE", segmenter.train(a.fold, a.name, a.epochs, a.batch, lr=a.lr, device=_device(), workers=a.workers,
                                     init=a.init, trunc=a.trunc, pad_hi=a.pad_hi), flush=True)


def cmd_predict(a):
    dev = _device()
    ks = range(data.N_FOLDS) if a.test else [a.fold]
    w = data.WEIGHTS
    dets = [detector.load(w / f"det_{n}_f{k}.pt") for n in a.det.split(",") for k in ks]
    segs = [segmenter.load(w / f"seg_{n}_f{k}.pth", dev) for n in a.seg.split(",") for k in ks]
    stems = data.test_stems() if a.test else data.split(a.fold)[1]
    out = data.WORK / "cache" / a.run / ("test" if a.test else "oof")
    for i, s in enumerate(stems[::-1] if a.reverse else stems, 1):
        f = out / f"{s}.npz"
        if f.exists():
            continue
        boxes, conf = detector.predict(dets, data.read_gray(s), a.imgsz, tta=not a.no_det_tta, device=0 if dev == "cuda" else dev)
        windows = [segmenter.crop_box(x1, y1, x2 - x1, y2 - y1, a.pad) for x1, y1, x2, y2 in boxes]
        planes = enhance.planes(s, cache=False)
        seg = lambda ws: segmenter.predict(segs, planes, ws, batch=a.seg_batch, tta=not a.no_seg_tta, device=dev)
        probs = seg(windows)
        for _ in range(a.grow):                              # filaments that leave their crop: enlarge and redo
            windows, changed = segmenter.grow_windows(windows, probs, pad=a.pad)
            if not changed:
                break
            for i, p in zip(changed, seg([windows[i] for i in changed])):
                probs[i] = p
        postprocess.save_cache(f, conf, windows, probs)
        if i % 20 == 0 or i == len(stems):
            print(f"predict {a.run} {'test' if a.test else f'fold {a.fold}'} {i}/{len(stems)}", flush=True)


# ------------------------------------------------------------------ tuning on out-of-fold predictions
_CTX = {}


def _gt_windows(records):
    """[[RLE]] -> [[(window, bool mask at window size, area)]]: annotations as box-local masks for fast overlaps."""
    from pycocotools import mask as mu
    out = []
    for rec in records:
        row = []
        for r in rec:
            x, y, w, h = (int(v) for v in mu.toBbox(r))
            if w and h:
                row.append(((x, y, x + w, y + h), mu.decode(r)[y:y + h, x:x + w].astype(bool), int(mu.area(r))))
        out.append(row)
    return out


def _counts(painted, gts):
    """Same counts as metric.counts, on box-local masks (no full-frame arrays)."""
    c = dict(tp=0, fp=0, fn=0, iou=0.0)
    for rec in gts:
        hit_g = np.zeros(len(rec), bool)
        for _, (x1, y1, x2, y2), m in painted:
            k, hit = int(m.sum()), False
            for j, ((gx1, gy1, gx2, gy2), g, ga) in enumerate(rec):
                ix1, iy1, ix2, iy2 = max(x1, gx1), max(y1, gy1), min(x2, gx2), min(y2, gy2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter = int((m[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1] & g[iy1 - gy1:iy2 - gy1, ix1 - gx1:ix2 - gx1]).sum())
                if inter and inter / (k + ga - inter) > 0.5:
                    c["tp"] += 1; c["iou"] += inter / (k + ga - inter); hit_g[j] = True; hit = True
            c["fp"] += not hit
        c["fn"] += int((~hit_g).sum())
    return c


def _sweep(stem):
    gts = _gt_windows(_CTX["records"][stem])
    inst = postprocess.load_cache(_CTX["root"] / f"{stem}.npz")
    rows = {}
    for cell in itertools.product(*(GRID[k] for k in CELL)):
        rows[cell] = _counts(postprocess.paint(inst, *cell), gts)
    return stem, rows


def cmd_tune(a):
    root = data.WORK / "cache" / a.run / "oof"
    stems = sorted(p.stem for p in root.glob("*.npz"))
    _CTX.update(root=root, records=metric.load_records(stems))
    with mp.get_context("fork").Pool(a.procs) as pool:
        per = dict(pool.imap_unordered(_sweep, stems, chunksize=4))
    cells = list(per[stems[0]])
    score = {c: metric.summarize([per[s][c] for s in stems]) for c in cells}
    ranked = sorted(cells, key=lambda c: -score[c]["pq"])
    best = ranked[0]
    # the chosen cell, re-scored through the RLE path the submission takes (must agree with the fast counts)
    rows = [metric.counts(postprocess.to_rles(postprocess.paint(postprocess.load_cache(root / f"{s}.npz"), *best)),
                          _CTX["records"][s]) for s in stems]
    final = metric.summarize(rows)
    assert abs(final["pq"] - score[best]["pq"]) < 1e-9, (final, score[best])
    fold = data.folds()
    res = dict(run=a.run, photos=len(stems), cell=dict(zip(CELL, best)), **final,
               ci95=metric.bootstrap_ci(rows, [s[:8] for s in stems]),
               per_fold={k: metric.summarize([r for r, s in zip(rows, stems) if fold[s] == k])["pq"] for k in range(data.N_FOLDS)},
               top=[dict(zip(CELL, c), pq=score[c]["pq"]) for c in ranked[:10]],
               by_knob={k: {str(v): max(score[c]["pq"] for c in cells if c[i] == v) for v in GRID[k]} for i, k in enumerate(CELL)})
    json.dump(res, open(data.WORK / "cache" / a.run / "tune.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("photos", "cell", "pq", "sq", "rq", "tp", "fp", "fn", "ci95", "per_fold")}, indent=1))


def cmd_submit(a):
    cell = json.load(open(data.WORK / "cache" / a.run / "tune.json"))["cell"]
    if a.cell:
        cell.update(json.loads(a.cell))
    root, stems = data.WORK / "cache" / a.run / "test", data.test_stems()
    rles = {s: postprocess.to_rles(postprocess.paint(postprocess.load_cache(root / f"{s}.npz"), **cell)) for s in stems}
    submission.write_csv(a.out, rles)
    problems = submission.validate(a.out, stems)
    n = sum(len(v) for v in rles.values())
    print(f"{a.out}: {n} filaments on {len(stems)} photos ({n / len(stems):.2f} per photo), cell {cell}, "
          f"{'VALID' if not problems else problems[:5]}")
    assert not problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("planes"); p.add_argument("--procs", type=int, default=8)
    p = sub.add_parser("det")
    p.add_argument("--fold", type=int, required=True); p.add_argument("--name", required=True)
    p.add_argument("--weights", default="yolo11m.pt"); p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--epochs", type=int, default=50); p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch", type=int, default=8); p.add_argument("--labels", default="records", choices=["records", "first"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--extra", help="teacher detector name: add its pseudo-labelled archive frames (run.py pseudo)")
    p.add_argument("--cache", default="ram", choices=["ram", "none"], help="keep decoded training images in RAM")
    p = sub.add_parser("pseudo")
    p.add_argument("--fold", type=int, required=True); p.add_argument("--det", required=True)
    p.add_argument("--src", required=True, help="directory of unlabeled GONG archive JPEGs")
    p.add_argument("--conf", type=float, default=0.25); p.add_argument("--procs", type=int, default=8)
    p = sub.add_parser("seg")
    p.add_argument("--fold", type=int, required=True); p.add_argument("--name", required=True)
    p.add_argument("--epochs", type=int, default=15); p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=8); p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--init", help="fine-tune from this segmenter of the same fold")
    p.add_argument("--trunc", type=float, default=0.0, help="partial-box augmentation: smallest kept share per axis")
    p.add_argument("--pad-hi", type=float, help="with --trunc: padding drawn from [0.2, PAD_HI]")
    p = sub.add_parser("predict")
    p.add_argument("--run", required=True); p.add_argument("--det", required=True); p.add_argument("--seg", required=True)
    p.add_argument("--fold", type=int); p.add_argument("--test", action="store_true")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--reverse", action="store_true", help="walk the photos backwards (second GPU on the same cache)")
    p.add_argument("--pad", type=float, default=0.2, help="crop padding around a box, as a share of its size")
    p.add_argument("--grow", type=int, default=0, help="rounds of crop enlargement for masks that touch their crop border")
    p.add_argument("--seg-batch", type=int, default=32)
    p.add_argument("--no-det-tta", action="store_true", help="ablation: detector without flips")
    p.add_argument("--no-seg-tta", action="store_true", help="ablation: segmenter without flips")
    p = sub.add_parser("tune"); p.add_argument("--run", required=True); p.add_argument("--procs", type=int, default=8)
    p = sub.add_parser("submit")
    p.add_argument("--run", required=True); p.add_argument("--out", default="submission.csv")
    p.add_argument("--cell", help='JSON overriding tuned knobs, e.g. \'{"conf": 0.25}\'')
    a = ap.parse_args()
    if a.cmd == "predict" and (a.fold is None) == (not a.test):
        ap.error("predict needs exactly one of --fold K or --test")
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
