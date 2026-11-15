"""EXP-009a (Wave 3 GPU Kernel): Physics-Enhanced Stage-1 Detector YOLO11l on (raw_norm, flat, bth7) + Multi-Segmenter.

Research Gap Addressed:
  ALL prior Stage-1 detectors (ms, s, main, F3, P6, P4a, SAHI, FRCNN-5F) feed raw grayscale H-alpha JPEGs
  with strong center-to-limb darkening and exposure/seeing shifts across the 6 GONG sites — causing 32% of all
  False Negatives on <580px faint filaments.
  1. Precomputes disk-normalized 3-channel physical planes [raw_norm, flat, bth7] using `filseg.enhance.enhance`
     and disables Ultralytics hue/saturation mixing (`hsv_h=0.0, hsv_s=0.0`) so channel semantics stay intact.
  2. Trains strictly 5-fold `yolo11l.pt` @ imgsz=1280 (`labels="records"`) across all 5 competition folds.
  3. Runs multi-scale (1280+1536) dihedral TTA + WBF inference strictly OOF on 707 train + 5-model on 180 test frames.
  4. Segments all detected boxes with strictly OOF Stage-2 E4 (+ grow=1), S1E, and SAM2.1 Hiera-B+.
"""
from __future__ import annotations

import argparse
import concurrent.futures
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
ENH_SZ = 1536


def _find_comp_root() -> Path:
    for p in Path("/kaggle/input").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"):
        if "magfilo-extra" not in str(p).lower():
            return p.parents[1]
    raise FileNotFoundError("MAGFiLO_1.0_Kaggle_2026 not found under /kaggle/input")


def _bind_filseg(repo_dir: Path, data_root: Path, work_dir: Path):
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    os.environ["MAGFILO_ROOT"] = str(data_root)
    os.environ["FILSEG_WORK"] = str(work_dir)
    from filseg import data, detector, enhance, postprocess, segmenter

    data.ROOT = data_root
    data.TRAIN_JSON = data_root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    data.TRAIN_IMAGES = data_root / "train" / "train_images"
    data.TEST_IMAGES = data_root / "test" / "test_images"
    data.FOLDS = repo_dir / "folds" / "folds.json"
    data.WORK = work_dir
    data.WEIGHTS = work_dir / "weights"
    return data, detector, enhance, postprocess, segmenter


def _make_enh_one(args: tuple[str, Path, Path]) -> str:
    stem, src_path, dst_path = args
    if dst_path.exists():
        return stem
    from filseg import enhance

    img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(src_path)
    p = enhance.enhance(img, ("raw", "flat", "bth7"))
    bgr = np.ascontiguousarray(p[[2, 1, 0]].transpose(1, 2, 0))
    if bgr.shape[0] != ENH_SZ or bgr.shape[1] != ENH_SZ:
        bgr = cv2.resize(bgr, (ENH_SZ, ENH_SZ), interpolation=cv2.INTER_AREA)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_path.with_suffix(".tmp.jpg")
    cv2.imwrite(str(tmp), bgr, [cv2.IMWRITE_JPEG_QUALITY, 98])
    tmp.replace(dst_path)
    return stem


def worker_prepare_enh(repo_dir: Path, data_root: Path, work_dir: Path, enh_dir: Path, procs: int) -> None:
    data, _, _, _, _ = _bind_filseg(repo_dir, data_root, work_dir)
    tasks = []
    for s in sorted(data.folds()):
        tasks.append((s, data.TRAIN_IMAGES / f"{s}.jpeg", enh_dir / "train" / f"{s}.jpg"))
    for s in data.test_stems():
        tasks.append((s, data.TEST_IMAGES / f"{s}.jpeg", enh_dir / "test" / f"{s}.jpg"))
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=procs) as ex:
        list(ex.map(_make_enh_one, tasks, chunksize=16))
    print(f"[enh_imgs] Prepared {len(tasks)} frames at {ENH_SZ}px in {(time.time() - t0)/60:.1f}m", flush=True)


def _build_yolo_dataset(data, enh_train_dir: Path, root: Path, tr_stems: list[str], va_stems: list[str]) -> Path:
    recs, anns = data.coco_index()
    for sp, stems in (("train", tr_stems), ("val", va_stems)):
        for sub in ("images", "labels"):
            shutil.rmtree(root / sub / sp, ignore_errors=True)
            (root / sub / sp).mkdir(parents=True, exist_ok=True)
        for s in stems:
            rids = recs[s] if sp == "train" else recs[s][:1]
            src_img = enh_train_dir / f"{s}.jpg"
            for rid in rids:
                stem = s if len(rids) == 1 else f"{s}__{rid}"
                os.symlink(src_img, root / "images" / sp / f"{stem}.jpg")
                lines = []
                for a in anns.get(rid, []):
                    x, y, w, h = a["bbox"]
                    if w >= 2 and h >= 2:
                        lines.append(f"0 {(x + w / 2) / FULL:.6f} {(y + h / 2) / FULL:.6f} {w / FULL:.6f} {h / FULL:.6f}")
                (root / "labels" / sp / f"{stem}.txt").write_text("\n".join(lines))
    yaml_p = root / "data.yaml"
    yaml_p.write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n")
    return yaml_p


def worker_train_fold(repo_dir: Path, data_root: Path, work_dir: Path, enh_dir: Path, fold: int, epochs: int) -> None:
    import torch
    from ultralytics import YOLO

    data, _, _, _, _ = _bind_filseg(repo_dir, data_root, work_dir)
    tr, va = data.split(fold)
    yaml_p = _build_yolo_dataset(data, enh_dir / "train", work_dir / "det_data" / f"enh11l_f{fold}", tr, va)

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 16.0
    batch = 12 if vram_gb >= 32 else 6
    out_w = work_dir / "weights" / f"det_enh11l_f{fold}.pt"
    out_w.parent.mkdir(parents=True, exist_ok=True)

    for b_try in (batch, max(2, batch // 2)):
        try:
            r = YOLO("yolo11l.pt").train(
                data=str(yaml_p),
                optimizer="AdamW",
                lr0=7.5e-4,
                warmup_epochs=4,
                epochs=epochs,
                patience=12,
                imgsz=1280,
                batch=b_try,
                pretrained=True,
                device=0,
                workers=4,
                cache="disk",
                hsv_h=0.0,
                hsv_s=0.0,
                hsv_v=0.15,
                fliplr=0.5,
                flipud=0.5,
                degrees=0.0,
                mosaic=0.4,
                scale=0.35,
                mixup=0.0,
                copy_paste=0.0,
                plots=False,
                project=str(work_dir / "det" / "enh11l"),
                name=f"fold{fold}",
                exist_ok=True,
                verbose=False,
                seed=fold,
            )
            shutil.copy2(Path(r.save_dir) / "weights" / "best.pt", out_w)
            print(f"[YOLO-Enh f{fold}] Saved {out_w} (batch={b_try})", flush=True)
            return
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and b_try > 2:
                torch.cuda.empty_cache()
                print(f"[YOLO-Enh f{fold}] OOM at batch={b_try}, retrying batch={max(2, b_try // 2)}", flush=True)
                continue
            raise


def _predict_enh(models: list, detector, bgr: np.ndarray, scale_views: list[tuple[int, int | None]], conf: float = 0.04, max_det: int = 65):
    views = []
    for m in models:
        for imgsz, ax in scale_views:
            x = bgr if ax is None else np.ascontiguousarray(np.flip(bgr, ax))
            r = m.predict(x, imgsz=imgsz, conf=conf, max_det=max_det, verbose=False, device=0)[0]
            b = r.boxes.xyxyn.cpu().numpy().astype(np.float64) * FULL
            if ax == 1:
                b[:, [0, 2]] = FULL - b[:, [2, 0]]
            elif ax == 0:
                b[:, [1, 3]] = FULL - b[:, [3, 1]]
            c = r.boxes.conf.cpu().numpy()
            views.append(np.c_[b, c])
    fused = detector.wbf(views, thr=0.55)[:max_det] if len(views) > 1 else [(v[:4], float(v[4])) for v in views[0]]
    boxes = np.array([f[0] for f in fused], dtype=np.float32).reshape(-1, 4)
    confs = np.array([f[1] for f in fused], dtype=np.float32)
    return boxes, confs


def worker_infer_boxes(repo_dir: Path, data_root: Path, work_dir: Path, enh_dir: Path, boxes_dir: Path, part: int) -> None:
    from ultralytics import YOLO

    data, detector, _, _, _ = _bind_filseg(repo_dir, data_root, work_dir)
    folds_map = data.folds()
    models = {k: YOLO(str(work_dir / "weights" / f"det_enh11l_f{k}.pt")) for k in range(5)}

    oof_views = [(1280, None), (1280, 1), (1280, 0), (1536, None), (1536, 1)]
    test_views = [(1280, None), (1280, 1), (1280, 0), (1536, None)]

    oof_dir = boxes_dir / "oof"
    test_dir = boxes_dir / "test"
    oof_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_stems = sorted(folds_map)[part::2]
    t0 = time.time()
    for i, s in enumerate(train_stems):
        bgr = cv2.imread(str(enh_dir / "train" / f"{s}.jpg"))
        boxes, confs = _predict_enh([models[folds_map[s]]], detector, bgr, oof_views)
        np.savez(oof_dir / f"{s}.npz", boxes=boxes, conf=confs)
        if (i + 1) % 100 == 0:
            print(f"[Boxes OOF p{part}] {i+1}/{len(train_stems)} ({(time.time()-t0)/60:.1f}m)", flush=True)

    test_stems = sorted(data.test_stems())[part::2]
    t1 = time.time()
    all_m = [models[k] for k in range(5)]
    for i, s in enumerate(test_stems):
        bgr = cv2.imread(str(enh_dir / "test" / f"{s}.jpg"))
        boxes, confs = _predict_enh(all_m, detector, bgr, test_views)
        np.savez(test_dir / f"{s}.npz", boxes=boxes, conf=confs)
    print(f"[Boxes Test p{part}] {len(test_stems)} done ({(time.time()-t1)/60:.1f}m)", flush=True)


def _load_s1e(ckpt_path: Path, device: str):
    import segmentation_models_pytorch as smp
    import torch

    m = smp.UnetPlusPlus("efficientnet-b4", encoder_weights=None, in_channels=3, classes=1)
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("model.", ""): (v.float() if hasattr(v, "is_floating_point") and v.is_floating_point() else v) for k, v in sd.items()}
    m.load_state_dict(sd, strict=True)
    return m.to(device).eval()


def worker_seg(
    repo_dir: Path,
    tpu_dir: Path,
    data_root: Path,
    work_dir: Path,
    boxes_dir: Path,
    out_dir: Path,
    s1e_ckpt: Path,
    sam2_ckpt: Path,
    part: int,
) -> None:
    if str(tpu_dir) not in sys.path:
        sys.path.insert(0, str(tpu_dir))
    data, _, enhance, postprocess, segmenter = _bind_filseg(repo_dir, data_root, work_dir)
    from tpu import sam_crop

    folds_map = data.folds()
    dev = "cuda:0"
    all_stems = [("oof", s) for s in sorted(folds_map)] + [("test", s) for s in sorted(data.test_stems())]
    my_items = all_stems[part::2]

    e4_models = {k: segmenter.load(data.WEIGHTS / f"seg_e4_f{k}.pth", dev) for k in range(5)}
    out_o, out_g = out_dir / "enh_e4", out_dir / "enh_e4_g1"
    t0 = time.time()
    for i, (sp, s) in enumerate(my_items):
        d = np.load(boxes_dir / sp / f"{s}.npz")
        boxes, confs = d["boxes"], d["conf"]
        segs = [e4_models[folds_map[s]]] if sp == "oof" else [e4_models[k] for k in range(5)]
        planes = enhance.planes(s)
        ws = [segmenter.crop_box(float(x1), float(y1), float(x2 - x1), float(y2 - y1)) for x1, y1, x2, y2 in boxes]
        probs = segmenter.predict(segs, planes, ws, batch=32, tta=True, device=dev) if ws else []
        postprocess.save_cache(out_o / sp / f"{s}.npz", confs, ws, probs)
        if ws:
            wg, ch = segmenter.grow_windows(ws, probs)
            if ch:
                pg = segmenter.predict(segs, planes, [ws[idx] for idx in ch], batch=32, tta=True, device=dev)
                probs_g = list(probs)
                for idx, val in zip(ch, pg):
                    probs_g[idx] = val
                ws, probs = wg, probs_g
        postprocess.save_cache(out_g / sp / f"{s}.npz", confs, ws, probs)
        if (i + 1) % 150 == 0:
            print(f"[Seg E4 p{part}] {i+1}/{len(my_items)} ({(time.time()-t0)/60:.1f}m)", flush=True)
    del e4_models

    s1e = _load_s1e(s1e_ckpt, dev)
    out_s1e = out_dir / "enh_s1e"
    for sp, s in my_items:
        d = np.load(boxes_dir / sp / f"{s}.npz")
        boxes, confs = d["boxes"], d["conf"]
        planes = enhance.planes(s)
        ws = [segmenter.crop_box(float(x1), float(y1), float(x2 - x1), float(y2 - y1)) for x1, y1, x2, y2 in boxes]
        probs = segmenter.predict([s1e], planes, ws, batch=32, tta=True, device=dev) if ws else []
        postprocess.save_cache(out_s1e / sp / f"{s}.npz", confs, ws, probs)
    del s1e

    sam2 = sam_crop.load(sam2_ckpt, sam_crop.SAM2_CFG, dev)
    out_sam2 = out_dir / "enh_sam2"
    t1 = time.time()
    for i, (sp, s) in enumerate(my_items):
        d = np.load(boxes_dir / sp / f"{s}.npz")
        boxes, confs = d["boxes"], d["conf"]
        img = data.read_gray(s)
        ws, probs = sam_crop.predict(sam2, img, boxes, batch=8, tta=False, device=dev) if len(boxes) else ([], [])
        postprocess.save_cache(out_sam2 / sp / f"{s}.npz", confs, ws, probs)
        if (i + 1) % 150 == 0:
            print(f"[Seg SAM2 p{part}] {i+1}/{len(my_items)} ({(time.time()-t1)/60:.1f}m)", flush=True)


def _pack(src_dir: Path, dst_tgz: Path) -> None:
    tmp = Path("/tmp") / f".tmp_{dst_tgz.name}"
    with tarfile.open(tmp, "w:gz") as tf:
        tf.add(src_dir, arcname=src_dir.name)
    dst_tgz.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp), str(dst_tgz))
    print(f"Packed {dst_tgz.name}: {dst_tgz.stat().st_size/1e6:.2f} MB", flush=True)


def run_driver() -> None:
    t0 = time.time()
    subprocess.run(
        "pip install -q ultralytics==8.4.160 segmentation-models-pytorch==0.5.0 pycocotools scikit-image 'transformers==5.14.1' 2>&1 | tail -2",
        shell=True,
        check=True,
    )

    import torch

    n_gpu = torch.cuda.device_count()
    data_root = _find_comp_root()
    repo_dir = next(p for p in Path("/kaggle/input").rglob("run.py") if (p.parent / "filseg").is_dir()).parent
    tpu_dir = next(p for p in Path("/kaggle/input").rglob("sam_crop.py") if p.parent.name == "tpu").parents[1]
    a100_weights = next(Path("/kaggle/input").rglob("seg_e4_f0.pth")).parent
    s1e_ckpt = next(Path("/kaggle/input").rglob("S1E_best.pth"))
    sam2_ckpt = next(p for p in Path("/kaggle/input").rglob("best.pth") if p.parent.name == "SAM2")

    work_dir = Path("/tmp/work")
    planes_tmp = Path("/tmp/planes")
    enh_dir = Path("/tmp/enh_imgs")
    boxes_dir = Path("/tmp/enh_boxes")
    out_root = Path("/tmp/out_009a")
    weights_dir = work_dir / "weights"
    for d in (work_dir, planes_tmp, enh_dir, boxes_dir, out_root, weights_dir):
        d.mkdir(parents=True, exist_ok=True)
    planes_link = work_dir / "planes"
    if not planes_link.exists():
        planes_link.symlink_to(planes_tmp)

    for k in range(5):
        shutil.copy2(a100_weights / f"seg_e4_f{k}.pth", weights_dir / f"seg_e4_f{k}.pth")

    subprocess.run(
        [
            sys.executable,
            "-c",
            "from ultralytics import YOLO; from transformers import Sam2Model; "
            "YOLO('yolo11l.pt'); Sam2Model.from_pretrained('facebook/sam2.1-hiera-base-plus'); print('Models ready')",
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
        PYTHONPATH=str(repo_dir),
    )

    p_planes = subprocess.Popen(
        [py, str(repo_dir / "run.py"), "planes", "--procs", "2"],
        env=common_env,
        stdout=open("/tmp/log_planes.txt", "w"),
        stderr=subprocess.STDOUT,
    )
    subprocess.run(
        [py, self_path, "--worker", "prepare_enh", "--repo", str(repo_dir), "--data-root", str(data_root), "--work-dir", str(work_dir), "--enh-dir", str(enh_dir)],
        env=common_env,
        check=True,
    )

    fold_groups = [(0, (0, 2, 4)), (min(1, max(0, n_gpu - 1)), (1, 3))] if n_gpu >= 2 else [(0, (0, 1, 2, 3, 4))]
    errs: list[BaseException] = []

    def _run_folds(gpu: int, fks: tuple[int, ...]):
        try:
            for fk in fks:
                rc = subprocess.call(
                    [
                        py, self_path, "--worker", "train_fold", "--fold", str(fk), "--epochs", "45",
                        "--repo", str(repo_dir), "--data-root", str(data_root), "--work-dir", str(work_dir), "--enh-dir", str(enh_dir),
                    ],
                    env=dict(common_env, CUDA_VISIBLE_DEVICES=str(gpu)),
                )
                assert rc == 0, f"YOLO-Enh fold {fk} failed"
        except BaseException as exc:
            errs.append(exc)

    threads = [threading.Thread(target=_run_folds, args=(g, fks)) for g, fks in fold_groups]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errs, errs

    def _run_parts(worker_name: str, extra: list[str]):
        p_errs: list[BaseException] = []

        def _p(part: int):
            try:
                gpu = part if n_gpu >= 2 else 0
                rc = subprocess.call(
                    [py, self_path, "--worker", worker_name, "--part", str(part)] + extra,
                    env=dict(common_env, CUDA_VISIBLE_DEVICES=str(gpu)),
                )
                assert rc == 0, f"{worker_name} part {part} failed"
            except BaseException as exc:
                p_errs.append(exc)

        ts = [threading.Thread(target=_p, args=(0,)), threading.Thread(target=_p, args=(1,))]
        for th in ts:
            th.start()
        for th in ts:
            th.join()
        assert not p_errs, p_errs

    _run_parts(
        "infer_boxes",
        ["--repo", str(repo_dir), "--data-root", str(data_root), "--work-dir", str(work_dir), "--enh-dir", str(enh_dir), "--boxes-dir", str(boxes_dir)],
    )
    assert p_planes.wait() == 0, "Planes generation failed"
    _run_parts(
        "seg_worker",
        [
            "--repo", str(repo_dir), "--tpu-dir", str(tpu_dir), "--data-root", str(data_root),
            "--work-dir", str(work_dir), "--boxes-dir", str(boxes_dir), "--out-dir", str(out_root),
            "--s1e", str(s1e_ckpt), "--sam2", str(sam2_ckpt),
        ],
    )

    dw_dir = Path("/tmp/enh_det_weights")
    dw_dir.mkdir(parents=True, exist_ok=True)
    for k in range(5):
        shutil.copy2(weights_dir / f"det_enh11l_f{k}.pt", dw_dir / f"det_enh11l_f{k}.pt")

    for sdir, tgz in (
        (boxes_dir, "/kaggle/working/exp009a_enh_boxes.tar.gz"),
        (out_root / "enh_e4", "/kaggle/working/exp009a_enh_e4.tar.gz"),
        (out_root / "enh_e4_g1", "/kaggle/working/exp009a_enh_e4_g1.tar.gz"),
        (out_root / "enh_s1e", "/kaggle/working/exp009a_enh_s1e.tar.gz"),
        (out_root / "enh_sam2", "/kaggle/working/exp009a_enh_sam2.tar.gz"),
        (dw_dir, "/kaggle/working/exp009a_weights.tar.gz"),
    ):
        if sdir.exists() and any(sdir.rglob("*")):
            _pack(sdir, Path(tgz))

    summary = {"elapsed_min": round((time.time() - t0) / 60, 2), "enh_sz": ENH_SZ}
    Path("/kaggle/working/exp009a_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"EXP-009a COMPLETE in {summary['elapsed_min']}m", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["prepare_enh", "train_fold", "infer_boxes", "seg_worker"])
    ap.add_argument("--repo", type=Path)
    ap.add_argument("--tpu-dir", type=Path)
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--work-dir", type=Path)
    ap.add_argument("--enh-dir", type=Path)
    ap.add_argument("--boxes-dir", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--s1e", type=Path)
    ap.add_argument("--sam2", type=Path)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()

    if args.worker is None:
        run_driver()
    elif args.worker == "prepare_enh":
        worker_prepare_enh(args.repo, args.data_root, args.work_dir, args.enh_dir, args.procs)
    elif args.worker == "train_fold":
        worker_train_fold(args.repo, args.data_root, args.work_dir, args.enh_dir, args.fold, args.epochs)
    elif args.worker == "infer_boxes":
        worker_infer_boxes(args.repo, args.data_root, args.work_dir, args.enh_dir, args.boxes_dir, args.part)
    elif args.worker == "seg_worker":
        worker_seg(args.repo, args.tpu_dir, args.data_root, args.work_dir, args.boxes_dir, args.out_dir, args.s1e, args.sam2, args.part)


if __name__ == "__main__":
    main()
