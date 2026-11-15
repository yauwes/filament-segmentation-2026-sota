"""EXP-007 (Wave 2 GPU Kernel #2): 5-Fold Native-Resolution SAHI@1024 YOLO11m + Multi-Segmenter Pipeline.

Addresses the verified small/thin filament recall gap (-7.0% <580px shortfall vs GT):
  1. Tiles 2048x2048 GONG frames into 3x3 overlapping 1024x1024 native-resolution (1:1 px, zero downscale) tiles
     (starts [0, 512, 1024], stride 512).
  2. Fine-tunes 5 folds of YOLO11m initialized from det_m1280_f0..4.pt on clipped tile annotations (vis_frac >= 0.40)
     plus negative background tiles, strictly OOF across folds 0..4 on 2 x T4.
  3. Performs batched 3x3 tile + flip TTA inference with visibility-normalized cross-tile WBF back into 2048x2048.
  4. Runs 3-family Stage-2 segmentation (seg_e4, S1E morphology UNet++, SAM2.1 Hiera-B+) on recovered SAHI boxes
     and archives compact bundles to /kaggle/working/.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

FULL = 2048
TILE_SIZE = 1024
STARTS = (0, 512, 1024)
TILES = [(tx, ty, tx + TILE_SIZE, ty + TILE_SIZE) for ty in STARTS for tx in STARTS]


def _find_comp_root() -> Path:
    for p in Path("/kaggle/input").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"):
        if "magfilo-extra" not in str(p).lower():
            return p.parents[1]
    raise FileNotFoundError("MAGFiLO_1.0_Kaggle_2026 not found under /kaggle/input")


def _crop_all_tiles(stem_paths: dict[str, Path], tiles_dir: Path) -> None:
    tiles_dir.mkdir(parents=True, exist_ok=True)
    missing = [
        (st, p) for st, p in sorted(stem_paths.items())
        if not (tiles_dir / f"{st}_t8.jpg").exists()
    ]
    if not missing:
        return

    def _one(item):
        st, p = item
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(str(p))
        for ti, (tx1, ty1, tx2, ty2) in enumerate(TILES):
            out_p = tiles_dir / f"{st}_t{ti}.jpg"
            if not out_p.exists():
                cv2.imwrite(str(out_p), img[ty1:ty2, tx1:tx2], [cv2.IMWRITE_JPEG_QUALITY, 95])

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(_one, missing))


def _project_boxes_to_tile(boxes_xywh: list[list[float]], tx1: int, ty1: int, min_vis: float = 0.40, min_side: int = 6) -> list[str]:
    tx2, ty2 = tx1 + TILE_SIZE, ty1 + TILE_SIZE
    lines = []
    for x, y, w, h in boxes_xywh:
        x1, y1, x2, y2 = x, y, x + w, y + h
        ix1, iy1 = max(x1, tx1), max(y1, ty1)
        ix2, iy2 = min(x2, tx2), min(y2, ty2)
        iw, ih = ix2 - ix1, iy2 - iy1
        if iw < min_side or ih < min_side:
            continue
        vis = (iw * ih) / max(1.0, w * h)
        if vis < min_vis:
            continue
        cx = ((ix1 + ix2) * 0.5 - tx1) / TILE_SIZE
        cy = ((iy1 + iy2) * 0.5 - ty1) / TILE_SIZE
        nw = iw / TILE_SIZE
        nh = ih / TILE_SIZE
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def _build_tile_dataset(repo_dir: Path, data_root: Path, tiles_dir: Path, ds_root: Path, fold: int) -> Path:
    sys.path.insert(0, str(repo_dir))
    os.environ["MAGFILO_ROOT"] = str(data_root)
    from filseg import data

    recs, anns = data.coco_index()
    tr, va = data.split(fold)
    for split, stems in (("train", tr), ("val", va)):
        for sub in ("images", "labels"):
            shutil.rmtree(ds_root / sub / split, ignore_errors=True)
            (ds_root / sub / split).mkdir(parents=True)
        for idx_s, s in enumerate(stems):
            rids = recs[s] if split == "train" else recs[s][:1]
            for idx_r, rid in enumerate(rids):
                boxes = [a["bbox"] for a in anns.get(rid, [])]
                pos_tiles = []
                neg_tiles = []
                for ti, (tx1, ty1, _, _) in enumerate(TILES):
                    lines = _project_boxes_to_tile(boxes, tx1, ty1)
                    if split == "val":
                        pos_tiles.append((ti, lines))
                    elif lines:
                        pos_tiles.append((ti, lines))
                    else:
                        neg_tiles.append((ti, []))
                chosen = pos_tiles
                if split == "train" and neg_tiles:
                    # Keep up to 2 deterministic empty solar tiles per record to calibrate negatives quickly
                    off = (idx_s * 7 + idx_r * 3 + fold) % len(neg_tiles)
                    neg_pick = [neg_tiles[(off + j) % len(neg_tiles)] for j in range(min(2, len(neg_tiles)))]
                    chosen = pos_tiles + neg_pick
                for ti, lines in chosen:
                    tag = f"{s}__{rid}_t{ti}" if len(rids) > 1 else f"{s}_t{ti}"
                    img_link = ds_root / "images" / split / f"{tag}.jpg"
                    os.symlink(tiles_dir / f"{s}_t{ti}.jpg", img_link)
                    (ds_root / "labels" / split / f"{tag}.txt").write_text("\n".join(lines))

    yaml_p = ds_root / "data.yaml"
    yaml_p.write_text(f"path: {ds_root}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n")
    return yaml_p


def _box_iou_ios(box: np.ndarray, others: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xx1 = np.maximum(box[0], others[:, 0])
    yy1 = np.maximum(box[1], others[:, 1])
    xx2 = np.minimum(box[2], others[:, 2])
    yy2 = np.minimum(box[3], others[:, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    a1 = max(1.0, float((box[2] - box[0]) * (box[3] - box[1])))
    a2 = np.maximum(1.0, (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1]))
    iou = inter / np.maximum(1.0, a1 + a2 - inter)
    ios = inter / np.minimum(a1, a2)
    return iou, ios


def _visibility_wbf(
    hits: list[tuple[np.ndarray, float, int]],
    n_models_times_flips: int,
    iou_thr: float = 0.50,
    ios_thr: float = 0.78,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-tile WBF whose expected view count adapts to how many of the 9 tiles cover the box."""
    if not hits:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)

    hits_sorted = sorted(hits, key=lambda h: -h[1])
    clusters: list[list[tuple[np.ndarray, float, int]]] = []
    fused_boxes: list[np.ndarray] = []

    for box, conf, vid in hits_sorted:
        j = -1
        if fused_boxes:
            ious, ioss = _box_iou_ios(box, np.asarray(fused_boxes))
            score_m = np.where((ious > iou_thr) | (ioss > ios_thr), np.maximum(ious, 0.9 * ioss), -1.0)
            best_i = int(np.argmax(score_m))
            if score_m[best_i] > 0:
                j = best_i
        if j < 0:
            clusters.append([(box, conf, vid)])
            fused_boxes.append(box.copy())
        else:
            clusters[j].append((box, conf, vid))
            ws = np.array([h[1] for h in clusters[j]], np.float32)
            bs = np.stack([h[0] for h in clusters[j]])
            fused_boxes[j] = (bs * ws[:, None]).sum(0) / ws.sum()

    out_boxes, out_confs = [], []
    tile_arr = np.asarray(TILES, np.float32)
    for b, cl in zip(fused_boxes, clusters):
        # How many of the 9 overlapping tiles cover >=55% of fused box `b`?
        ix1 = np.maximum(b[0], tile_arr[:, 0])
        iy1 = np.maximum(b[1], tile_arr[:, 1])
        ix2 = np.minimum(b[2], tile_arr[:, 2])
        iy2 = np.minimum(b[3], tile_arr[:, 3])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        b_area = max(1.0, float((b[2] - b[0]) * (b[3] - b[1])))
        cov_tiles = int((inter / b_area >= 0.55).sum())
        if cov_tiles == 0:
            # Large box spanning multiple tiles: expect at least tiles capturing >=25%
            cov_tiles = max(1, int((inter / b_area >= 0.25).sum()))
        expected_views = max(1, cov_tiles) * n_models_times_flips
        uniq_views = len({h[2] for h in cl})
        # Mean over distinct views; if a cluster received > expected_views due to boundary partial hits, cap factor at 1.0
        mean_c = float(np.mean([h[1] for h in cl]))
        cal_c = mean_c * min(uniq_views, expected_views) / expected_views
        out_boxes.append(b)
        out_confs.append(cal_c)

    order = np.argsort(-np.asarray(out_confs))[:60]
    return np.asarray(out_boxes, np.float32)[order], np.asarray(out_confs, np.float32)[order]


def worker_train_fold(args) -> None:
    from ultralytics import YOLO

    t0 = time.time()
    repo_dir = Path(args.repo)
    data_root = Path(args.data_root)
    work_dir = Path(args.work_dir)
    fold = args.fold

    ds_root = work_dir / "sahi_ds" / f"f{fold}"
    yaml_p = _build_tile_dataset(repo_dir, data_root, Path(args.tiles_dir), ds_root, fold)
    init_w = Path(args.init_weights)
    assert init_w.exists(), f"Missing init weights: {init_w}"

    r = YOLO(str(init_w)).train(
        data=str(yaml_p),
        optimizer="AdamW",
        lr0=3e-4,
        lrf=0.05,
        warmup_epochs=2,
        epochs=args.epochs,
        patience=5,
        imgsz=TILE_SIZE,
        batch=12,
        close_mosaic=4,
        mosaic=0.5,
        pretrained=True,
        device=0,
        workers=2,
        cache=False,
        plots=False,
        project=str(work_dir / "det_sahi"),
        name=f"fold{fold}",
        exist_ok=True,
        verbose=False,
        seed=fold,
    )
    out = work_dir / "weights" / f"det_sahi1024_f{fold}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(r.save_dir) / "weights" / "best.pt", out)
    shutil.rmtree(ds_root, ignore_errors=True)
    print(f"[SAHI train fold {fold}] saved {out} in {(time.time()-t0)/60:.1f}m", flush=True)


def worker_sahi_boxes(args) -> None:
    from ultralytics import YOLO

    t0 = time.time()
    repo_dir = Path(args.repo)
    data_root = Path(args.data_root)
    work_dir = Path(args.work_dir)
    sys.path.insert(0, str(repo_dir))
    os.environ["MAGFILO_ROOT"] = str(data_root)
    from filseg import data

    w_dir = work_dir / "weights"
    dets = {k: YOLO(str(w_dir / f"det_sahi1024_f{k}.pt")) for k in range(data.N_FOLDS)}

    stem_tasks: list[tuple[str, str, int | None, Path]] = []
    splits = set(args.splits.split(","))
    if "oof" in splits:
        for k in range(data.N_FOLDS):
            for s in data.split(k)[1]:
                stem_tasks.append(("oof", s, k, data.TRAIN_IMAGES / f"{s}.jpeg"))
    if "test" in splits:
        for s in data.test_stems():
            stem_tasks.append(("test", s, None, data.TEST_IMAGES / f"{s}.jpeg"))

    if args.num_shards > 1:
        stem_tasks = [
            t for t in stem_tasks
            if int(hashlib.md5(f"{t[0]}_{t[1]}".encode()).hexdigest()[:8], 16) % args.num_shards == args.shard_id
        ]

    out_dir = Path(args.boxes_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_done = 0
    for sp, stem, fk, img_path in stem_tasks:
        dst = out_dir / sp / f"{stem}.npz"
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        tiles_id = [np.ascontiguousarray(img_bgr[ty1:ty2, tx1:tx2]) for tx1, ty1, tx2, ty2 in TILES]
        tiles_hf = [np.ascontiguousarray(np.flip(t, 1)) for t in tiles_id]

        models = [dets[fk]] if fk is not None else [dets[k] for k in range(data.N_FOLDS)]
        hits: list[tuple[np.ndarray, float, int]] = []
        view_counter = 0
        for model in models:
            for is_hf, batch_imgs in ((False, tiles_id), (True, tiles_hf)):
                res = model.predict(batch_imgs, imgsz=TILE_SIZE, conf=0.05, max_det=40, batch=9, verbose=False, device=0)
                for ti, r in enumerate(res):
                    vid = view_counter * 9 + ti
                    tx1, ty1, tx2, ty2 = TILES[ti]
                    bs = r.boxes.xyxy.cpu().numpy()
                    cs = r.boxes.conf.cpu().numpy()
                    if is_hf and len(bs):
                        bs = bs.copy()
                        bs[:, [0, 2]] = TILE_SIZE - bs[:, [2, 0]]
                    for (bx1, by1, bx2, by2), c in zip(bs, cs):
                        # Filter thin slivers truncated at internal overlapping tile boundaries
                        touches_internal_x = (tx1 > 0 and bx1 <= 2) or (tx2 < FULL and bx2 >= TILE_SIZE - 2)
                        touches_internal_y = (ty1 > 0 and by1 <= 2) or (ty2 < FULL and by2 >= TILE_SIZE - 2)
                        if (touches_internal_x and (bx2 - bx1) < 12) or (touches_internal_y and (by2 - by1) < 12):
                            continue
                        gbox = np.array([bx1 + tx1, by1 + ty1, bx2 + tx1, by2 + ty1], np.float32)
                        hits.append((gbox, float(c), vid))
                view_counter += 1

        boxes, confs = _visibility_wbf(hits, n_models_times_flips=len(models) * 2)
        keep = confs >= 0.05
        boxes, confs = boxes[keep], confs[keep]
        np.savez_compressed(dst, box=boxes.astype(np.float32), conf=confs.astype(np.float32))
        n_done += 1
        if n_done % 50 == 0 or n_done == len(stem_tasks):
            print(f"[sahi_boxes shard {args.shard_id}] {n_done}/{len(stem_tasks)} ({(time.time()-t0)/60:.1f}m)", flush=True)


def worker_seg_stage2(args) -> None:
    import torch

    t0 = time.time()
    repo_dir = Path(args.repo)
    data_root = Path(args.data_root)
    work_dir = Path(args.work_dir)
    boxes_dir = Path(args.boxes_dir)
    sys.path.insert(0, str(repo_dir))
    os.environ["MAGFILO_ROOT"] = str(data_root)
    os.environ["FILSEG_WORK"] = str(work_dir)
    from filseg import data, enhance, postprocess, segmenter

    stem_to_fold = {s: k for k in range(data.N_FOLDS) for s in data.split(k)[1]}
    stem_paths = {p.stem: p for p in data.TRAIN_IMAGES.glob("*.jpeg")}
    stem_paths.update({p.stem: p for p in data.TEST_IMAGES.glob("*.jpeg")})

    items: list[tuple[str, str, int | None, Path]] = []
    for sp in ("oof", "test"):
        for f in sorted((boxes_dir / sp).glob("*.npz")):
            fk = stem_to_fold[f.stem] if sp == "oof" else None
            if sp == "oof" and args.fold0_only and fk != 0:
                continue
            items.append((sp, f.stem, fk, f))

    if args.num_shards > 1:
        items = [
            t for t in items
            if int(hashlib.md5(f"{t[0]}_{t[1]}".encode()).hexdigest()[:8], 16) % args.num_shards == args.shard_id
        ]

    out_dir = Path(args.out_dir)
    if args.seg_family == "e4":
        dev = "cuda"
        models = {k: segmenter.load(data.WEIGHTS / f"seg_e4_f{k}.pth", dev) for k in range(data.N_FOLDS)}
        out_orig = out_dir / "sahi_e4"
        out_g1 = out_dir / "sahi_e4_g1"
        for i, (sp, stem, fk, bf) in enumerate(items, 1):
            p_o = out_orig / sp / f"{stem}.npz"
            p_g = out_g1 / sp / f"{stem}.npz"
            if p_o.exists() and p_g.exists():
                continue
            with np.load(bf) as bd:
                boxes, conf = bd["box"], bd["conf"]
            ws = [segmenter.crop_box(float(x1), float(y1), float(x2 - x1), float(y2 - y1), 0.2) for x1, y1, x2, y2 in boxes]
            if not ws:
                postprocess.save_cache(p_o, conf, ws, [])
                postprocess.save_cache(p_g, conf, ws, [])
                continue
            planes = enhance.planes(stem, cache=True)
            segs = [models[fk]] if fk is not None else [models[k] for k in range(data.N_FOLDS)]
            probs = segmenter.predict(segs, planes, ws, batch=32, tta=True, device=dev)
            postprocess.save_cache(p_o, conf, ws, probs)

            ws_g, ch = segmenter.grow_windows(ws, probs, pad=0.2)
            probs_g = list(probs)
            if ch:
                np_g = segmenter.predict(segs, planes, [ws_g[c] for c in ch], batch=32, tta=True, device=dev)
                for ci, pa in zip(ch, np_g):
                    probs_g[ci] = pa
            postprocess.save_cache(p_g, conf, ws_g, probs_g)
            if i % 80 == 0 or i == len(items):
                print(f"[seg e4 shard {args.shard_id}] {i}/{len(items)} ({(time.time()-t0)/60:.1f}m)", flush=True)
        return

    # TPU segmenters (S1E or SAM2) using cached pad=0.2 windows
    tpu_dir = Path(args.tpu_code)
    for k in list(sys.modules):
        if k == "filseg" or k.startswith("filseg."):
            del sys.modules[k]
    if str(repo_dir) in sys.path:
        sys.path.remove(str(repo_dir))
    sys.path.insert(0, str(tpu_dir / "tpu"))
    sys.path.insert(0, str(tpu_dir / "src"))
    import sam_crop
    import twostage as ts

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.seg_family == "sam2":
        model = sam_crop.SamCrop("sam2", 512).to(dev)
        model.load_state_dict(torch.load(args.seg_weights, map_location="cpu"))
        model.eval()
        pre, bs = "raw", 12
    else:
        cfg_p = Path(args.seg_weights).with_name("S1E_config.json")
        cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {"pre": "bth7+flat+ridge", "arch": "unetpp"}
        pre, bs = cfg["pre"], 32
        model = ts.seg_model(pre, cfg["arch"], cfg.get("encoder", "efficientnet-b4"))
        model.load_state_dict(torch.load(args.seg_weights, map_location="cpu"))
        model.to(dev).eval()

    # Order Val(fold0) + Test before OOF folds 1..4 so critical splits finish first
    items.sort(key=lambda t: (0 if (t[2] == 0 or t[0] == "test") else 1, t[0], t[1]))
    out_seg = out_dir / f"sahi_{args.seg_family}"
    for i, (sp, stem, _, bf) in enumerate(items, 1):
        dst = out_seg / sp / f"{stem}.npz"
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with np.load(bf) as bd:
            boxes, conf = bd["box"], bd["conf"]
        # Use standard pad=0.2 crop_box identical to e4 & SamCrop PAD=0.2
        ws = np.array([ts.crop_box(float(x1), float(y1), float(x2 - x1), float(y2 - y1), 0.2) for x1, y1, x2, y2 in boxes], np.int32).reshape(-1, 4)
        img = cv2.imread(str(stem_paths[stem]), cv2.IMREAD_GRAYSCALE)
        planes = ts.channels(img, pre)
        crops = [(ts.crop_planes(planes, b, 512) - 0.5) / 0.5 for b in ws]
        probs_u8 = []
        j, cur_bs = 0, bs
        while j < len(crops):
            chunk = crops[j : j + cur_bs]
            try:
                x = torch.from_numpy(np.stack(chunk).astype(np.float32)).to(dev)
                acc = 0
                with torch.no_grad():
                    for k in range(4):
                        xx = x.flip(3) if k in (1, 3) else x
                        xx = xx.flip(2) if k in (2, 3) else xx
                        with torch.autocast("cuda", dtype=torch.float16):
                            y = model(xx).float().sigmoid()
                        y = y.flip(2) if k in (2, 3) else y
                        acc = acc + (y.flip(3) if k in (1, 3) else y)
                for p, b in zip((acc / 4.0).cpu().numpy()[:, 0], ws[j : j + len(chunk)]):
                    probs_u8.append((cv2.resize(p, (max(1, int(b[2] - b[0])), max(1, int(b[3] - b[1])))) * 255.0 + 0.5).astype(np.uint8))
                j += len(chunk)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and cur_bs > 2:
                    torch.cuda.empty_cache()
                    cur_bs = max(2, cur_bs // 2)
                else:
                    raise
        flat = np.concatenate([p.ravel() for p in probs_u8]) if probs_u8 else np.zeros(0, np.uint8)
        np.savez_compressed(dst, conf=conf.astype(np.float32), box=ws, p=flat)
        if i % 80 == 0 or i == len(items):
            print(f"[seg {args.seg_family} shard {args.shard_id}] {i}/{len(items)} ({(time.time()-t0)/60:.1f}m)", flush=True)


def _tar_dir(src_dir: Path, tar_path: Path) -> float:
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(src_dir, arcname=src_dir.name)
    return tar_path.stat().st_size / 1e6


def orchestrate() -> None:
    t0 = time.time()
    print(
        subprocess.run(
            "pip install -q ultralytics==8.4.160 segmentation-models-pytorch==0.5.0 pycocotools scikit-image 'transformers==5.14.1' 2>&1 | tail -2",
            shell=True,
            capture_output=True,
            text=True,
        ).stdout,
        flush=True,
    )
    subprocess.run("nvidia-smi -L; nproc", shell=True)

    data_root = _find_comp_root()
    repo_dir = next(p for p in Path("/kaggle/input").rglob("run.py") if (p.parent / "filseg").is_dir()).parent
    tpu_dir = next(p for p in Path("/kaggle/input").rglob("sam_crop.py") if p.parent.name == "tpu").parents[1]
    a100_weights = next(Path("/kaggle/input").rglob("det_m1280_f0.pt")).parent
    s1e_weights = next(Path("/kaggle/input").rglob("S1E_best.pth"))
    sam2_weights = next(p for p in Path("/kaggle/input").rglob("best.pth") if p.parent.name == "SAM2")

    work_dir = Path("/kaggle/working/work")
    planes_tmp = Path("/tmp/planes")
    tiles_dir = Path("/tmp/sahi_tiles")
    boxes_dir = Path("/tmp/sahi_boxes")
    out_root = Path("/tmp/out")
    weights_dir = work_dir / "weights"
    for d in (work_dir, planes_tmp, tiles_dir, boxes_dir, out_root, weights_dir):
        d.mkdir(parents=True, exist_ok=True)
    planes_link = work_dir / "planes"
    if not planes_link.exists():
        planes_link.symlink_to(planes_tmp)

    for k in range(5):
        shutil.copy2(a100_weights / f"det_m1280_f{k}.pt", weights_dir / f"det_m1280_f{k}.pt")
        shutil.copy2(a100_weights / f"seg_e4_f{k}.pth", weights_dir / f"seg_e4_f{k}.pth")

    # Pre-warm HF Sam2Model + materialize 3x3 tiles in parallel on CPU
    sys.path.insert(0, str(repo_dir))
    os.environ["MAGFILO_ROOT"] = str(data_root)
    os.environ["FILSEG_WORK"] = str(work_dir)
    from filseg import data

    stem_paths = {p.stem: p for p in data.TRAIN_IMAGES.glob("*.jpeg")}
    stem_paths.update({p.stem: p for p in data.TEST_IMAGES.glob("*.jpeg")})
    subprocess.run(
        [sys.executable, "-c", "from transformers import Sam2Model; Sam2Model.from_pretrained('facebook/sam2.1-hiera-base-plus')"],
        check=True,
    )
    print("Cropping 3x3 native 1024x1024 tiles for all 887 photos...", flush=True)
    _crop_all_tiles(stem_paths, tiles_dir)
    print(f"Tiles ready in {(time.time()-t0)/60:.1f}m", flush=True)

    py = sys.executable
    self_path = str(Path(__file__).resolve())
    common_env = dict(
        os.environ,
        MAGFILO_ROOT=str(data_root),
        FILSEG_WORK=str(work_dir),
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(repo_dir),
    )

    # Background CPU planes while 2x T4 train 5-fold SAHI YOLO11m
    p_planes = subprocess.Popen(
        [py, str(repo_dir / "run.py"), "planes", "--procs", "2"],
        env=common_env,
        stdout=open("/tmp/log_planes.txt", "w"),
        stderr=subprocess.STDOUT,
    )

    err_box: list[BaseException] = []

    def _train_list(gpu: int, folds: tuple[int, ...]):
        try:
            for fk in folds:
                rc = subprocess.call(
                    [
                        py, self_path, "--worker", "train_fold", "--fold", str(fk), "--epochs", "15",
                        "--repo", str(repo_dir), "--data-root", str(data_root), "--work-dir", str(work_dir),
                        "--tiles-dir", str(tiles_dir), "--init-weights", str(weights_dir / f"det_m1280_f{fk}.pt"),
                    ],
                    env=dict(common_env, CUDA_VISIBLE_DEVICES=str(gpu)),
                )
                assert rc == 0, f"SAHI fold {fk} failed"
        except BaseException as exc:
            err_box.append(exc)

    t0_g = threading.Thread(target=_train_list, args=(0, (0, 2, 4)))
    t1_g = threading.Thread(target=_train_list, args=(1, (1, 3)))
    t0_g.start(); t1_g.start()
    t0_g.join(); t1_g.join()
    assert p_planes.wait() == 0, "planes failed"
    if err_box:
        raise err_box[0]

    shutil.rmtree(tiles_dir, ignore_errors=True)
    print(f"=== Training + planes done in {(time.time()-t0)/60:.1f}m -> SAHI 2-GPU Box Inference ===", flush=True)

    # 2-GPU Sharded SAHI Box Inference (707 OOF + 180 Test)
    ps = [
        subprocess.Popen(
            [
                py, self_path, "--worker", "sahi_boxes", "--repo", str(repo_dir), "--data-root", str(data_root),
                "--work-dir", str(work_dir), "--boxes-dir", str(boxes_dir),
                "--splits", "oof,test", "--shard-id", str(g), "--num-shards", "2",
            ],
            env=dict(common_env, CUDA_VISIBLE_DEVICES=str(g)),
        )
        for g in (0, 1)
    ]
    assert all(p.wait() == 0 for p in ps), "sahi_boxes failed"

    # Stage 2: e4 + e4_g1 (sharded across GPU0 & GPU1)
    print(f"=== Stage 2a: seg_e4 & e4_g1 at {(time.time()-t0)/60:.1f}m ===", flush=True)
    ps_e4 = [
        subprocess.Popen(
            [
                py, self_path, "--worker", "seg_stage2", "--seg-family", "e4",
                "--repo", str(repo_dir), "--data-root", str(data_root), "--work-dir", str(work_dir),
                "--boxes-dir", str(boxes_dir), "--out-dir", str(out_root), "--shard-id", str(g), "--num-shards", "2",
            ],
            env=dict(common_env, CUDA_VISIBLE_DEVICES=str(g)),
        )
        for g in (0, 1)
    ]
    assert all(p.wait() == 0 for p in ps_e4), "seg_e4 Stage 2 failed"

    # Stage 2b: GPU0 SAM2 + GPU1 S1E on recovered SAHI boxes
    print(f"=== Stage 2b: GPU0 SAM2 + GPU1 S1E at {(time.time()-t0)/60:.1f}m ===", flush=True)
    p_sam2 = subprocess.Popen(
        [
            py, self_path, "--worker", "seg_stage2", "--seg-family", "sam2",
            "--repo", str(repo_dir), "--data-root", str(data_root), "--tpu-code", str(tpu_dir),
            "--work-dir", str(work_dir), "--boxes-dir", str(boxes_dir), "--out-dir", str(out_root),
            "--seg-weights", str(sam2_weights),
        ],
        env=dict(common_env, CUDA_VISIBLE_DEVICES="0", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"),
    )
    p_s1e = subprocess.Popen(
        [
            py, self_path, "--worker", "seg_stage2", "--seg-family", "s1e",
            "--repo", str(repo_dir), "--data-root", str(data_root), "--tpu-code", str(tpu_dir),
            "--work-dir", str(work_dir), "--boxes-dir", str(boxes_dir), "--out-dir", str(out_root),
            "--seg-weights", str(s1e_weights),
        ],
        env=dict(common_env, CUDA_VISIBLE_DEVICES="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"),
    )
    assert p_sam2.wait() == 0 and p_s1e.wait() == 0, "SAM2 or S1E Stage 2 failed"

    det_w_dir = Path("/tmp/det_sahi_weights")
    det_w_dir.mkdir(parents=True, exist_ok=True)
    for k in range(5):
        shutil.copy2(weights_dir / f"det_sahi1024_f{k}.pt", det_w_dir / f"det_sahi1024_f{k}.pt")

    summary = {"minutes": round((time.time() - t0) / 60.0, 2), "archives": {}, "counts": {}}
    for name, sdir, tpath in (
        ("sahi_boxes", boxes_dir, "/kaggle/working/exp007_sahi_boxes.tar.gz"),
        ("sahi_e4", out_root / "sahi_e4", "/kaggle/working/exp007_sahi_e4.tar.gz"),
        ("sahi_e4_g1", out_root / "sahi_e4_g1", "/kaggle/working/exp007_sahi_e4_g1.tar.gz"),
        ("sahi_s1e", out_root / "sahi_s1e", "/kaggle/working/exp007_sahi_s1e.tar.gz"),
        ("sahi_sam2", out_root / "sahi_sam2", "/kaggle/working/exp007_sahi_sam2.tar.gz"),
        ("det_weights", det_w_dir, "/kaggle/working/exp007_det_weights.tar.gz"),
    ):
        if sdir.exists() and any(sdir.rglob("*")):
            sz = _tar_dir(sdir, Path(tpath))
            summary["archives"][Path(tpath).name] = round(sz, 2)
            summary["counts"][name] = sum(1 for _ in sdir.rglob("*"))

    Path("/kaggle/working/exp007_summary.json").write_text(json.dumps(summary, indent=2))
    print("EXP007 SUMMARY:", json.dumps(summary), flush=True)
    shutil.rmtree(work_dir, ignore_errors=True)
    shutil.rmtree(planes_tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default="orchestrate", choices=["orchestrate", "train_fold", "sahi_boxes", "seg_stage2"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--repo", default="")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--tpu-code", default="")
    ap.add_argument("--work-dir", default="")
    ap.add_argument("--tiles-dir", default="")
    ap.add_argument("--boxes-dir", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--init-weights", default="")
    ap.add_argument("--seg-weights", default="")
    ap.add_argument("--seg-family", default="e4")
    ap.add_argument("--splits", default="oof,test")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--fold0-only", action="store_true")
    args = ap.parse_args()
    if args.worker == "orchestrate":
        orchestrate()
    elif args.worker == "train_fold":
        worker_train_fold(args)
    elif args.worker == "sahi_boxes":
        worker_sahi_boxes(args)
    elif args.worker == "seg_stage2":
        worker_seg_stage2(args)


if __name__ == "__main__":
    main()
