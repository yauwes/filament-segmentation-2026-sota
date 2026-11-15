"""EXP-006 (Wave 2 GPU Kernel #1): Multi-Architecture Stage-2 & Border-Grown Segmenter Upgrade on 2 x T4.

Upgrades existing non-transductive detector box streams (ms, s, main, F3, P4a, P6) with:
  1. SAM 2.1 Hiera-B+ crop segmenter (g15/SAM2/best.pth, 512px, 4-flip TTA)
  2. S1E morphology-enhanced UNet++ segmenter (S1E_best.pth, bth7+flat+ridge, 4-flip TTA) on ms/s/main
  3. Newly trained 5-fold seg_e4x (UNet++ initialized from e4, trunc=0.4, pad_hi=0.6, lr=1e-4, 6 epochs/fold)
  4. Recursive crop enlargement (grow_windows up to 2 rounds -> *_g2) with SAM2/S1E patch-up on grown crops.

All outputs are tightly archived into /kaggle/working/exp006_{sam2,s1e,e4x,e4x_weights}.tar.gz + summary JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np


def _find_comp_root() -> Path:
    for p in Path("/kaggle/input").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"):
        if "magfilo-extra" not in str(p).lower():
            return p.parents[1]
    raise FileNotFoundError("MAGFiLO_1.0_Kaggle_2026 not found under /kaggle/input")


def _find_input_dir(slug: str) -> Path:
    candidates = [
        Path(f"/kaggle/input/datasets/ajikajadeh/{slug}"),
        Path(f"/kaggle/input/notebooks/ajikajadeh/{slug}"),
        Path(f"/kaggle/input/{slug}"),
    ]
    for c in candidates:
        if c.exists():
            return c
    for p in Path("/kaggle/input").glob(f"**/{slug}"):
        if p.is_dir():
            return p
    raise FileNotFoundError(f"Input mount not found for {slug}")


def _extract_tgz(tgz_path: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz_path, "r:gz") as tf:
        tf.extractall(dst_dir)


def _index_npz(root: Path, split_names: tuple[str, ...]) -> dict[str, dict[str, Path]]:
    out: dict[str, dict[str, Path]] = {}
    for sp in split_names:
        sub = root / sp
        if not sub.exists():
            continue
        files = sorted(sub.rglob("*.npz"))
        norm_sp = "val" if sp == "val" else ("oof" if sp == "oof" else "test")
        out.setdefault(norm_sp, {})
        for f in files:
            out[norm_sp][f.stem] = f
    return out


def _build_stream_index(base_cache: Path, g19_dir: Path, final_dir: Path) -> dict[str, dict[str, dict[str, Path]]]:
    idx: dict[str, dict[str, dict[str, Path]]] = {}
    for sname in ("ms", "s", "main"):
        idx[sname] = _index_npz(base_cache / "cache" / sname, ("oof", "test"))
    # Handle both nested tsF3/cache/S1E and fallback rglob
    for sname, parent, rel in (
        ("F3", g19_dir, "tsF3/cache/S1E"),
        ("P6", g19_dir, "tsP6/cache/S1E"),
        ("P4a", final_dir, "final/tsP4a/cache/S1E"),
    ):
        cand = parent / rel
        if not cand.exists():
            matches = [p for p in parent.rglob(sname if sname == "S1E" else f"*{sname}*") if p.is_dir()]
            for m in matches:
                if (m / "cache" / "S1E").is_dir():
                    cand = m / "cache" / "S1E"
                    break
        idx[sname] = _index_npz(cand, ("val", "test"))
    return idx


def _load_folds(repo_dir: Path, data_root: Path) -> tuple[dict[str, int], list[str], dict[str, Path]]:
    sys.path.insert(0, str(repo_dir))
    os.environ["MAGFILO_ROOT"] = str(data_root)
    from filseg import data

    stem_to_fold: dict[str, int] = {}
    for k in range(data.N_FOLDS):
        _, val_stems = data.split(k)
        for s in val_stems:
            stem_to_fold[s] = k
    test_stems = list(data.test_stems())
    stem_paths: dict[str, Path] = {}
    for p in data.TRAIN_IMAGES.glob("*.jpeg"):
        stem_paths[p.stem] = p
    for p in data.TEST_IMAGES.glob("*.jpeg"):
        stem_paths[p.stem] = p
    assert len(stem_paths) >= 880, f"Expected >=880 GONG images, got {len(stem_paths)} under {data.ROOT}"
    return stem_to_fold, test_stems, stem_paths


def _unpack_conf_box(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path) as d:
        conf = np.asarray(d["conf"], np.float32)
        box = np.asarray(d["box"], np.int32).reshape(-1, 4)
    return conf, box


def _save_npz(path: Path, conf: np.ndarray, box: np.ndarray, probs: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = np.concatenate([p.ravel() for p in probs]) if len(probs) else np.zeros(0, np.uint8)
    np.savez_compressed(path, conf=conf.astype(np.float32), box=box.astype(np.int32).reshape(-1, 4), p=flat)


def _infer_tpu_crop_model(
    model,
    pre: str,
    planes_fn,
    crop_fn,
    img_path: Path,
    boxes: np.ndarray,
    dev,
    bs: int,
    crop_size: int = 512,
) -> list[np.ndarray]:
    import torch

    if len(boxes) == 0:
        return []
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(img_path))
    planes = planes_fn(img, pre)
    crops = [(crop_fn(planes, b, crop_size) - 0.5) / 0.5 for b in boxes]
    out: list[np.ndarray] = []
    cur_bs = bs
    i = 0
    while i < len(crops):
        chunk = crops[i : i + cur_bs]
        try:
            x = torch.from_numpy(np.stack(chunk).astype(np.float32)).to(dev)
            acc = 0
            with torch.no_grad():
                for k in range(4):
                    xx = x.flip(3) if k in (1, 3) else x
                    xx = xx.flip(2) if k in (2, 3) else xx
                    with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                        y = model(xx).float().sigmoid()
                    y = y.flip(2) if k in (2, 3) else y
                    acc = acc + (y.flip(3) if k in (1, 3) else y)
            probs = (acc / 4.0).cpu().numpy()[:, 0]
            for p, b in zip(probs, boxes[i : i + len(chunk)]):
                w, h = max(1, int(b[2] - b[0])), max(1, int(b[3] - b[1]))
                out.append((cv2.resize(p, (w, h)) * 255.0 + 0.5).astype(np.uint8))
            i += len(chunk)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and cur_bs > 2:
                torch.cuda.empty_cache()
                cur_bs = max(2, cur_bs // 2)
                print(f"[OOM retry] reducing batch to {cur_bs}", flush=True)
            else:
                raise
    return out


def worker_tpu_seg(args) -> None:
    import torch

    t0 = time.time()
    repo_dir = Path(args.repo)
    data_root = Path(args.data_root)
    tpu_dir = Path(args.tpu_code)
    stem_to_fold, test_stems, stem_paths = _load_folds(repo_dir, data_root)
    fold0_stems = {s for s, k in stem_to_fold.items() if k == 0}

    # Switch cleanly to tpu/ + tpu/src to avoid filseg package collision
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
    idx = json.loads(Path(args.index_json).read_text())
    out_root = Path(args.out_dir)

    if args.mode == "sam2":
        model = sam_crop.SamCrop("sam2", 512).to(dev)
        ckpt = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(ckpt)
        model.eval()
        pre = "raw"
        bs = 12
        # Priority queue so highest-value Fold-0 + Test + ms OOF always finish first
        queue: list[tuple[str, str, list[str]]] = [
            ("ms", "oof_f0", sorted(s for s in idx["ms"].get("oof", {}) if s in fold0_stems)),
            ("ms", "test", sorted(idx["ms"].get("test", {}))),
            ("F3", "val", sorted(idx["F3"].get("val", {}))),
            ("F3", "test", sorted(idx["F3"].get("test", {}))),
            ("P6", "val", sorted(idx["P6"].get("val", {}))),
            ("P6", "test", sorted(idx["P6"].get("test", {}))),
            ("P4a", "val", sorted(idx["P4a"].get("val", {}))),
            ("P4a", "test", sorted(idx["P4a"].get("test", {}))),
            ("ms", "oof_rest", sorted(s for s in idx["ms"].get("oof", {}) if s not in fold0_stems)),
            ("s", "oof_f0", sorted(s for s in idx["s"].get("oof", {}) if s in fold0_stems)),
            ("s", "test", sorted(idx["s"].get("test", {}))),
            ("main", "oof_f0", sorted(s for s in idx["main"].get("oof", {}) if s in fold0_stems)),
            ("main", "test", sorted(idx["main"].get("test", {}))),
            ("s", "oof_rest", sorted(s for s in idx["s"].get("oof", {}) if s not in fold0_stems)),
            ("main", "oof_rest", sorted(s for s in idx["main"].get("oof", {}) if s not in fold0_stems)),
        ]
    elif args.mode == "s1e":
        cfg_path = Path(args.weights).with_name("S1E_config.json")
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {"pre": "bth7+flat+ridge", "arch": "unetpp"}
        pre = cfg["pre"]
        model = ts.seg_model(pre, cfg["arch"], cfg.get("encoder", "efficientnet-b4"))
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
        model.to(dev).eval()
        bs = 32
        queue = [
            ("ms", "oof_f0", sorted(s for s in idx["ms"].get("oof", {}) if s in fold0_stems)),
            ("ms", "test", sorted(idx["ms"].get("test", {}))),
            ("s", "oof_f0", sorted(s for s in idx["s"].get("oof", {}) if s in fold0_stems)),
            ("s", "test", sorted(idx["s"].get("test", {}))),
            ("main", "oof_f0", sorted(s for s in idx["main"].get("oof", {}) if s in fold0_stems)),
            ("main", "test", sorted(idx["main"].get("test", {}))),
            ("ms", "oof_rest", sorted(s for s in idx["ms"].get("oof", {}) if s not in fold0_stems)),
            ("s", "oof_rest", sorted(s for s in idx["s"].get("oof", {}) if s not in fold0_stems)),
            ("main", "oof_rest", sorted(s for s in idx["main"].get("oof", {}) if s not in fold0_stems)),
        ]
    else:
        raise ValueError(args.mode)

    total_done = 0
    for sname, tag, stems in queue:
        src_sp = "val" if tag == "val" else ("test" if tag == "test" else "oof")
        dst_sp = "test" if tag == "test" else ("val" if sname in ("F3", "P6", "P4a") else "oof")
        st_t0 = time.time()
        n_run = 0
        for stem in stems:
            elapsed_min = (time.time() - t0) / 60.0
            if args.max_minutes > 0 and tag == "oof_rest" and elapsed_min >= args.max_minutes:
                print(f"[{args.mode}] budget cap {args.max_minutes:.1f}m reached at {sname}/{tag} ({n_run}/{len(stems)})", flush=True)
                return
            out_path = out_root / sname / dst_sp / f"{stem}.npz"
            if out_path.exists():
                continue
            src_npz = Path(idx[sname][src_sp][stem])
            conf, boxes = _unpack_conf_box(src_npz)
            probs = _infer_tpu_crop_model(model, pre, ts.channels, ts.crop_planes, stem_paths[stem], boxes, dev, bs)
            _save_npz(out_path, conf, boxes, probs)
            n_run += 1
            total_done += 1
        print(f"[{args.mode}] done {sname}/{tag}: +{n_run} (total={total_done}) in {time.time() - st_t0:.1f}s (elapsed {(time.time() - t0)/60:.1f}m)", flush=True)


def worker_e4x_infer(args) -> None:
    t0 = time.time()
    repo_dir = Path(args.repo)
    data_root = Path(args.data_root)
    sys.path.insert(0, str(repo_dir))
    os.environ["MAGFILO_ROOT"] = str(data_root)
    os.environ["FILSEG_WORK"] = args.work_dir
    from filseg import data, enhance, segmenter

    dev = "cuda" if shutil.which("nvidia-smi") else "cpu"
    w_dir = data.WEIGHTS
    splits = [s for s in args.splits.split(",") if s]
    if "test" not in splits and args.folds:
        load_folds = sorted(int(x) for x in args.folds.split(","))
    else:
        load_folds = list(range(data.N_FOLDS))
    models = {k: segmenter.load(w_dir / f"seg_e4x_f{k}.pth", dev) for k in load_folds
              if (w_dir / f"seg_e4x_f{k}.pth").exists()}
    assert len(models) > 0, f"No seg_e4x weights found in {w_dir} for folds {load_folds}"

    idx = json.loads(Path(args.index_json).read_text())
    out_orig = Path(args.out_dir) / "e4x"
    out_g2 = Path(args.out_dir) / "e4x_g2"

    stem_to_fold: dict[str, int] = {}
    for k in range(data.N_FOLDS):
        _, val_stems = data.split(k)
        for s in val_stems:
            stem_to_fold[s] = k

    tasks: list[tuple[str, str, str, int | None]] = []
    streams = [s for s in args.streams.split(",") if s]
    for sname in streams:
        if "oof" in splits and "oof" in idx[sname]:
            for stem in sorted(idx[sname]["oof"]):
                fk = stem_to_fold[stem]
                if args.folds and fk not in {int(x) for x in args.folds.split(",")}:
                    continue
                tasks.append((sname, "oof", stem, fk))
        if "val" in splits and "val" in idx[sname]:
            for stem in sorted(idx[sname]["val"]):
                fk = stem_to_fold[stem]
                if args.folds and fk not in {int(x) for x in args.folds.split(",")}:
                    continue
                tasks.append((sname, "val", stem, fk))
        if "test" in splits and "test" in idx[sname]:
            for stem in sorted(idx[sname]["test"]):
                tasks.append((sname, "test", stem, None))

    # Deterministic 2-GPU sharding by (split, stem) so planes are read sequentially per GPU
    if args.num_shards > 1:
        tasks = [
            t for t in tasks
            if int(hashlib.md5(f"{t[1]}_{t[2]}".encode()).hexdigest()[:8], 16) % args.num_shards == args.shard_id
        ]
    # Group by (sp, stem) so enhance.planes is loaded ONLY ONCE per photo across all streams
    by_photo: dict[tuple[str, str, int | None], list[str]] = {}
    for sname, sp, stem, fk in tasks:
        by_photo.setdefault((sp, stem, fk), []).append(sname)

    print(f"[e4x_infer shard {args.shard_id}/{args.num_shards}] photos={len(by_photo)} tasks={len(tasks)}", flush=True)
    n_photos = 0
    n_grown_boxes = 0
    n_total_boxes = 0
    for (sp, stem, fk), photo_streams in sorted(by_photo.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        need_any = any(
            not (out_orig / sname / sp / f"{stem}.npz").exists()
            or not (out_g2 / sname / sp / f"{stem}.npz").exists()
            for sname in photo_streams
        )
        if not need_any:
            continue
        planes = enhance.planes(stem, cache=True)
        seg_list = [models[fk]] if fk is not None else [models[k] for k in sorted(models)]
        for sname in photo_streams:
            conf, boxes = _unpack_conf_box(Path(idx[sname][sp][stem]))
            ws = [tuple(int(v) for v in b) for b in boxes]
            p_orig_path = out_orig / sname / sp / f"{stem}.npz"
            p_g2_path = out_g2 / sname / sp / f"{stem}.npz"
            if len(ws) == 0:
                _save_npz(p_orig_path, conf, boxes, [])
                _save_npz(p_g2_path, conf, boxes, [])
                continue
            probs = segmenter.predict(seg_list, planes, ws, batch=32, tta=True, device=dev)
            _save_npz(p_orig_path, conf, boxes, probs)

            # Recursive border growth (up to 2 rounds) reusing unchanged crops
            ws_g, probs_g = list(ws), list(probs)
            changed_mask = np.zeros(len(ws), dtype=bool)
            for _ in range(2):
                ws_next, ch = segmenter.grow_windows(ws_g, probs_g, pad=0.2)
                if not ch:
                    break
                new_p = segmenter.predict(seg_list, planes, [ws_next[i] for i in ch], batch=32, tta=True, device=dev)
                for ci, np_arr in zip(ch, new_p):
                    probs_g[ci] = np_arr
                    changed_mask[ci] = True
                ws_g = ws_next
            n_total_boxes += len(ws)
            n_grown_boxes += int(changed_mask.sum())
            _save_npz(p_g2_path, conf, np.asarray(ws_g, np.int32), probs_g)
        n_photos += 1
        if n_photos % 50 == 0 or n_photos == len(by_photo):
            print(
                f"[e4x_infer shard {args.shard_id}] {n_photos}/{len(by_photo)} photos | "
                f"grown={n_grown_boxes}/{n_total_boxes} ({(time.time()-t0)/60:.1f}m)",
                flush=True,
            )


def worker_patch_g2(args) -> None:
    """Apply SAM2 and S1E only to crops whose windows changed during grow_windows (fast delta pass)."""
    import torch

    t0 = time.time()
    repo_dir = Path(args.repo)
    data_root = Path(args.data_root)
    tpu_dir = Path(args.tpu_code)
    _, _, stem_paths = _load_folds(repo_dir, data_root)
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
    e4x_g2_root = Path(args.e4x_g2_dir)
    base_seg_root = Path(args.base_seg_dir)
    out_g2_root = Path(args.out_dir)
    idx = json.loads(Path(args.index_json).read_text()) if args.index_json else {}

    if args.mode == "sam2":
        model = sam_crop.SamCrop("sam2", 512).to(dev)
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
        model.eval()
        pre, bs = "raw", 12
    else:
        cfg_path = Path(args.weights).with_name("S1E_config.json")
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {"pre": "bth7+flat+ridge", "arch": "unetpp"}
        pre, bs = cfg["pre"], 32
        model = ts.seg_model(pre, cfg["arch"], cfg.get("encoder", "efficientnet-b4"))
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
        model.to(dev).eval()

    n_patched = 0
    n_crops_recomputed = 0
    for sname in sorted(p.name for p in e4x_g2_root.iterdir() if p.is_dir()):
        for sp_dir in sorted(p for p in (e4x_g2_root / sname).iterdir() if p.is_dir()):
            sp = sp_dir.name
            for g2_npz in sorted(sp_dir.glob("*.npz")):
                stem = g2_npz.stem
                base_npz = base_seg_root / sname / sp / f"{stem}.npz"
                if not base_npz.exists() and args.mode == "s1e":
                    alt = idx.get(sname, {}).get(sp, {}).get(stem)
                    if alt:
                        base_npz = Path(alt)
                if not base_npz.exists():
                    continue
                dst_npz = out_g2_root / sname / sp / f"{stem}.npz"
                if dst_npz.exists():
                    continue
                conf_g, box_g = _unpack_conf_box(g2_npz)
                with np.load(base_npz) as bd:
                    box_b = np.asarray(bd["box"], np.int32).reshape(-1, 4)
                    pkey = "p" if "p" in bd else ("p4" if "p4" in bd else "p1")
                    p_flat = np.asarray(bd[pkey], np.uint8)
                if len(box_g) != len(box_b):
                    continue
                # Extract original crops
                orig_crops = []
                off = 0
                for b in box_b:
                    w, h = max(0, int(b[2] - b[0])), max(0, int(b[3] - b[1]))
                    sz = w * h
                    orig_crops.append(p_flat[off : off + sz].reshape(h, w))
                    off += sz
                diff_idx = [i for i in range(len(box_g)) if not np.array_equal(box_g[i], box_b[i])]
                if diff_idx:
                    new_crops = _infer_tpu_crop_model(
                        model, pre, ts.channels, ts.crop_planes, stem_paths[stem], box_g[diff_idx], dev, bs
                    )
                    for i_d, nc in zip(diff_idx, new_crops):
                        orig_crops[i_d] = nc
                    n_crops_recomputed += len(diff_idx)
                    n_patched += 1
                _save_npz(dst_npz, conf_g, box_g, orig_crops)
    print(f"[patch_g2 {args.mode}] files_changed={n_patched}, crops_recomputed={n_crops_recomputed} in {time.time()-t0:.1f}s", flush=True)


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
    a100_weights = next(Path("/kaggle/input").rglob("seg_e4_f0.pth")).parent
    s1e_weights = next(Path("/kaggle/input").rglob("S1E_best.pth"))
    sam2_weights = next(p for p in Path("/kaggle/input").rglob("best.pth") if p.parent.name == "SAM2")

    infer_ms = _find_input_dir("filseg-infer-ms")
    infer_s = _find_input_dir("filseg-infer-s")
    infer_main = _find_input_dir("filseg-infer-main")
    g19_dir = _find_input_dir("filseg-twostage-g19")
    final_dir = _find_input_dir("filseg-final-pipeline")

    work_dir = Path("/tmp/work")
    planes_tmp = Path("/tmp/planes")
    weights_dir = work_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    planes_tmp.mkdir(parents=True, exist_ok=True)
    planes_link = work_dir / "planes"
    if not planes_link.exists():
        planes_link.symlink_to(planes_tmp)

    for k in range(5):
        shutil.copy2(a100_weights / f"seg_e4_f{k}.pth", weights_dir / f"seg_e4_f{k}.pth")

    base_cache = Path("/tmp/base_caches")
    for sname, inp in (("ms", infer_ms), ("s", infer_s), ("main", infer_main)):
        tgz = inp / f"cache_{sname}.tgz"
        if tgz.exists():
            _extract_tgz(tgz, base_cache)
        elif (inp / "work" / "cache" / sname).exists():
            shutil.copytree(inp / "work" / "cache" / sname, base_cache / "cache" / sname, dirs_exist_ok=True)
        else:
            raise FileNotFoundError(f"cache_{sname} missing under {inp}")

    stream_idx = _build_stream_index(base_cache, g19_dir, final_dir)
    idx_serial = {k: {sp: {st: str(p) for st, p in d.items()} for sp, d in splits.items()} for k, splits in stream_idx.items()}
    index_json = Path("/tmp/stream_index.json")
    index_json.write_text(json.dumps(idx_serial))
    print("Stream index counts:", {k: {sp: len(d) for sp, d in splits.items()} for k, splits in idx_serial.items()}, flush=True)

    # Pre-cache HF + timm backbone weights cleanly before launching parallel GPU workers
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from transformers import Sam2Model; import segmentation_models_pytorch as smp; "
            "Sam2Model.from_pretrained('facebook/sam2.1-hiera-base-plus'); "
            "smp.UnetPlusPlus('efficientnet-b4', encoder_weights='imagenet', in_channels=3, classes=1); print('Pretrained caches warm')",
        ],
        check=True,
    )

    py = sys.executable
    self_path = str(Path(__file__).resolve())
    common_env = dict(
        os.environ,
        MAGFILO_ROOT=str(data_root),
        FILSEG_WORK=str(work_dir),
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    run_env = dict(common_env, PYTHONPATH=str(repo_dir))

    out_sam2 = Path("/tmp/out/sam2")
    out_s1e = Path("/tmp/out/s1e")
    out_e4x_root = Path("/tmp/out/e4x_root")
    for d in (out_sam2, out_s1e, out_e4x_root):
        d.mkdir(parents=True, exist_ok=True)

    # Phase 1: Parallel CPU planes + GPU0 SAM2 + GPU1 S1E
    print("=== Phase 1: planes (CPU) + SAM2 (GPU0) + S1E (GPU1) ===", flush=True)
    p_planes = subprocess.Popen(
        [py, str(repo_dir / "run.py"), "planes", "--procs", "3"],
        env=run_env,
        stdout=open("/tmp/log_planes.txt", "w"),
        stderr=subprocess.STDOUT,
    )
    p_sam2 = subprocess.Popen(
        [
            py, self_path, "--worker", "tpu_seg", "--mode", "sam2",
            "--repo", str(repo_dir), "--data-root", str(data_root), "--tpu-code", str(tpu_dir),
            "--index-json", str(index_json), "--weights", str(sam2_weights),
            "--out-dir", str(out_sam2), "--max-minutes", "75",
        ],
        env=dict(common_env, CUDA_VISIBLE_DEVICES="0", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"),
        stdout=open("/tmp/log_sam2.txt", "w"),
        stderr=subprocess.STDOUT,
    )
    p_s1e = subprocess.Popen(
        [
            py, self_path, "--worker", "tpu_seg", "--mode", "s1e",
            "--repo", str(repo_dir), "--data-root", str(data_root), "--tpu-code", str(tpu_dir),
            "--index-json", str(index_json), "--weights", str(s1e_weights),
            "--out-dir", str(out_s1e), "--max-minutes", "45",
        ],
        env=dict(common_env, CUDA_VISIBLE_DEVICES="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"),
        stdout=open("/tmp/log_s1e.txt", "w"),
        stderr=subprocess.STDOUT,
    )

    rc_planes = p_planes.wait()
    rc_s1e = p_s1e.wait()
    print(f"planes rc={rc_planes}, s1e rc={rc_s1e} at {(time.time()-t0)/60:.1f}m", flush=True)
    print(Path("/tmp/log_s1e.txt").read_text()[-2500:], flush=True)
    assert rc_planes == 0 and rc_s1e == 0, "Phase 1 planes or s1e failed"

    # GPU1 starts seg_e4x Folds 1, 3 IMMEDIATELY once planes + S1E finish
    def _gpu1_chain():
        for fk in (1, 3):
            rc = subprocess.call(
                [py, str(repo_dir / "run.py"), "seg", "--fold", str(fk), "--name", "e4x",
                 "--init", "e4", "--trunc", "0.4", "--pad-hi", "0.6", "--lr", "1e-4", "--epochs", "6", "--workers", "2"],
                env=dict(run_env, CUDA_VISIBLE_DEVICES="1"),
            )
            assert rc == 0, f"seg e4x fold {fk} failed"
        # Immediately infer OOF folds 1, 3 on ms + s + main while GPU0 finishes fold 4
        rc = subprocess.call(
            [
                py, self_path, "--worker", "e4x_infer",
                "--repo", str(repo_dir), "--data-root", str(data_root), "--work-dir", str(work_dir),
                "--index-json", str(index_json), "--out-dir", str(out_e4x_root),
                "--streams", "ms,s,main", "--splits", "oof", "--folds", "1,3",
            ],
            env=dict(common_env, CUDA_VISIBLE_DEVICES="1"),
        )
        assert rc == 0, "e4x_infer oof 1,3 failed"

    import threading
    err_box: list[BaseException] = []

    def _run_gpu1():
        try:
            _gpu1_chain()
        except BaseException as exc:
            err_box.append(exc)

    def _run_gpu0():
        try:
            rc_sam2 = p_sam2.wait()
            print(f"sam2 rc={rc_sam2} at {(time.time()-t0)/60:.1f}m", flush=True)
            print(Path("/tmp/log_sam2.txt").read_text()[-3000:], flush=True)
            assert rc_sam2 == 0, "sam2 worker failed"
            for fk in (0, 2, 4):
                rc = subprocess.call(
                    [py, str(repo_dir / "run.py"), "seg", "--fold", str(fk), "--name", "e4x",
                     "--init", "e4", "--trunc", "0.4", "--pad-hi", "0.6", "--lr", "1e-4", "--epochs", "6", "--workers", "2"],
                    env=dict(run_env, CUDA_VISIBLE_DEVICES="0"),
                )
                assert rc == 0, f"seg e4x fold {fk} failed"
        except BaseException as exc:
            err_box.append(exc)

    th0 = threading.Thread(target=_run_gpu0)
    th1 = threading.Thread(target=_run_gpu1)
    th0.start(); th1.start()
    th0.join(); th1.join()
    if err_box:
        raise err_box[0]

    print(f"=== Phase 2 training done at {(time.time()-t0)/60:.1f}m -> Phase 3 2-GPU e4x + g2 inference ===", flush=True)
    # Infer Fold 0 / OOF (0,2,4) / Val (F3, P6, P4a) + Test (180 across all 6 streams) sharded across both GPUs
    p_inf0 = subprocess.Popen(
        [
            py, self_path, "--worker", "e4x_infer",
            "--repo", str(repo_dir), "--data-root", str(data_root), "--work-dir", str(work_dir),
            "--index-json", str(index_json), "--out-dir", str(out_e4x_root),
            "--streams", "ms,F3,P6,P4a,s,main", "--splits", "oof,val,test", "--folds", "0,2,4",
            "--shard-id", "0", "--num-shards", "2",
        ],
        env=dict(common_env, CUDA_VISIBLE_DEVICES="0"),
        stdout=open("/tmp/log_inf0.txt", "w"),
        stderr=subprocess.STDOUT,
    )
    p_inf1 = subprocess.Popen(
        [
            py, self_path, "--worker", "e4x_infer",
            "--repo", str(repo_dir), "--data-root", str(data_root), "--work-dir", str(work_dir),
            "--index-json", str(index_json), "--out-dir", str(out_e4x_root),
            "--streams", "ms,F3,P6,P4a,s,main", "--splits", "oof,val,test", "--folds", "0,2,4",
            "--shard-id", "1", "--num-shards", "2",
        ],
        env=dict(common_env, CUDA_VISIBLE_DEVICES="1"),
        stdout=open("/tmp/log_inf1.txt", "w"),
        stderr=subprocess.STDOUT,
    )
    assert p_inf0.wait() == 0 and p_inf1.wait() == 0, "Phase 3 e4x inference failed"
    print(Path("/tmp/log_inf0.txt").read_text()[-1500:], flush=True)
    print(Path("/tmp/log_inf1.txt").read_text()[-1500:], flush=True)

    # Phase 4: Fast delta patch of SAM2 & S1E onto grown windows (~5% of crops) if budget allows
    out_sam2_g2 = Path("/tmp/out/sam2_g2")
    out_s1e_g2 = Path("/tmp/out/s1e_g2")
    if (time.time() - t0) / 60.0 < 470:
        print(f"=== Phase 4: patch_g2 for SAM2 + S1E at {(time.time()-t0)/60:.1f}m ===", flush=True)
        p_pg0 = subprocess.Popen(
            [
                py, self_path, "--worker", "patch_g2", "--mode", "sam2",
                "--repo", str(repo_dir), "--data-root", str(data_root), "--tpu-code", str(tpu_dir),
                "--weights", str(sam2_weights), "--e4x-g2-dir", str(out_e4x_root / "e4x_g2"),
                "--base-seg-dir", str(out_sam2), "--out-dir", str(out_sam2_g2),
            ],
            env=dict(common_env, CUDA_VISIBLE_DEVICES="0"),
            stdout=open("/tmp/log_pg0.txt", "w"),
            stderr=subprocess.STDOUT,
        )
        p_pg1 = subprocess.Popen(
            [
                py, self_path, "--worker", "patch_g2", "--mode", "s1e",
                "--repo", str(repo_dir), "--data-root", str(data_root), "--tpu-code", str(tpu_dir),
                "--index-json", str(index_json),
                "--weights", str(s1e_weights), "--e4x-g2-dir", str(out_e4x_root / "e4x_g2"),
                "--base-seg-dir", str(out_s1e), "--out-dir", str(out_s1e_g2),
            ],
            env=dict(common_env, CUDA_VISIBLE_DEVICES="1"),
            stdout=open("/tmp/log_pg1.txt", "w"),
            stderr=subprocess.STDOUT,
        )
        p_pg0.wait(); p_pg1.wait()
        print(Path("/tmp/log_pg0.txt").read_text(), flush=True)
        print(Path("/tmp/log_pg1.txt").read_text(), flush=True)

    # Package cleanly into compact archives in /kaggle/working and delete raw scratch
    e4x_w_dir = Path("/tmp/out/e4x_weights")
    e4x_w_dir.mkdir(parents=True, exist_ok=True)
    for k in range(5):
        p = weights_dir / f"seg_e4x_f{k}.pth"
        if p.exists():
            shutil.copy2(p, e4x_w_dir / p.name)
        hp = weights_dir / f"seg_e4x_f{k}.history.json"
        if hp.exists():
            shutil.copy2(hp, e4x_w_dir / hp.name)

    summary = {"minutes": round((time.time() - t0) / 60.0, 2), "archives": {}, "counts": {}}
    bundles = [
        ("sam2", out_sam2, "/kaggle/working/exp006_sam2.tar.gz"),
        ("s1e", out_s1e, "/kaggle/working/exp006_s1e.tar.gz"),
        ("e4x", out_e4x_root, "/kaggle/working/exp006_e4x.tar.gz"),
        ("sam2_g2", out_sam2_g2, "/kaggle/working/exp006_sam2_g2.tar.gz"),
        ("s1e_g2", out_s1e_g2, "/kaggle/working/exp006_s1e_g2.tar.gz"),
        ("e4x_weights", e4x_w_dir, "/kaggle/working/exp006_e4x_weights.tar.gz"),
    ]
    for name, sdir, tpath in bundles:
        if sdir.exists() and any(sdir.rglob("*")):
            sz = _tar_dir(sdir, Path(tpath))
            n_npz = sum(1 for _ in sdir.rglob("*.npz"))
            summary["archives"][Path(tpath).name] = round(sz, 2)
            summary["counts"][name] = n_npz

    Path("/kaggle/working/exp006_summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY:", json.dumps(summary), flush=True)
    shutil.rmtree(work_dir, ignore_errors=True)
    shutil.rmtree(planes_tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default="orchestrate", choices=["orchestrate", "tpu_seg", "e4x_infer", "patch_g2"])
    ap.add_argument("--mode", default="sam2")
    ap.add_argument("--repo", default="")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--tpu-code", default="")
    ap.add_argument("--work-dir", default="")
    ap.add_argument("--index-json", default="")
    ap.add_argument("--weights", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--e4x-g2-dir", default="")
    ap.add_argument("--base-seg-dir", default="")
    ap.add_argument("--streams", default="ms,F3,P6,P4a,s,main")
    ap.add_argument("--splits", default="oof,val,test")
    ap.add_argument("--folds", default="")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-minutes", type=float, default=60.0)
    args = ap.parse_args()
    if args.worker == "orchestrate":
        orchestrate()
    elif args.worker == "tpu_seg":
        worker_tpu_seg(args)
    elif args.worker == "e4x_infer":
        worker_e4x_infer(args)
    elif args.worker == "patch_g2":
        worker_patch_g2(args)


if __name__ == "__main__":
    main()
