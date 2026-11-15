"""EXP-013a (Wave 4 GPU Kernel): Stage-1 SOTA Transformer (RT-DETR-X) & Hybrid Attention (YOLO12-X)

Strictly leak-free 5-fold pipeline on physical [raw_clahe/restored, flat_z, bth7_z] inputs:
- Validation (`va`, `folds[s] == fold`): ALWAYS 100% untouched original competition photos & labels.
- Train (`tr`, `folds[s] != fold`): optional fold-only SolarRestormer DL enhancement + object-aware
  small-filament Copy-Paste (`--scenario dl_enh_copypaste`) benchmarked on untouched real `va`.
- Multi-scale + Dihedral Flip TTA + WBF inference -> 707 OOF + 180 Test boxes (.npz), plus
  optional Stage-2 segmentation with Seg_E4 + SAM2.1 (+ SegFormer-B3 if present).
"""
from __future__ import annotations

import argparse
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

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

FULL = 2048


def _find_comp_root() -> Path:
    for p in Path("/kaggle/input").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"):
        if "magfilo-extra" not in str(p).lower():
            return p.parents[1]
    raise FileNotFoundError("MAGFiLO_1.0_Kaggle_2026 not found under /kaggle/input")


def _find_input_dir(slug: str) -> Path | None:
    for p in (
        Path(f"/kaggle/input/datasets/ajikajadeh/{slug}"),
        Path(f"/kaggle/input/notebooks/ajikajadeh/{slug}"),
        Path(f"/kaggle/input/{slug}"),
    ):
        if p.exists():
            return p
    for p in Path("/kaggle/input").glob(f"**/{slug}"):
        if p.is_dir():
            return p
    return None


def _bind_filseg(repo_dir: Path, data_root: Path, work_dir: Path):
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    os.environ["MAGFILO_ROOT"] = str(data_root)
    os.environ["FILSEG_WORK"] = str(work_dir)
    from filseg import data, detector, enhance, postprocess, segmenter  # noqa: E402

    data.ROOT = data_root
    data.TRAIN_JSON = data_root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    data.TRAIN_IMAGES = data_root / "train" / "train_images"
    data.TEST_IMAGES = data_root / "test" / "test_images"
    data.FOLDS = repo_dir / "folds" / "folds.json"
    data.WORK = work_dir
    data.WEIGHTS = work_dir / "weights"
    return data, detector, enhance, segmenter, postprocess


def _prepare_fold_images(
    data,
    tr_stems: list[str],
    va_stems: list[str],
    test_stems: list[str],
    cache_dir: Path,
    scenario: str,
    fold: int,
    va_use_restormer: bool = False,
    gong_dir: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Build fold-specific enhanced/physics images.

    Guarantees zero validation leakage:
      - `SolarRestormerNet` trains strictly on sharp `tr_stems` (`folds != fold`) + optional legal GONG SSL (excluding test +-1h/dates & va dates).
      - `va_stems` use untouched clean real single-image physics (`va_use_restormer=False`)
        OR purely inductive `tr`-only Restormer restoration (`va_use_restormer=True`),
        with original real competition labels in all scenarios.
    """
    import solar_dl_enhance as sde  # noqa: E402

    tr_dir = cache_dir / f"f{fold}" / "tr"
    va_dir = cache_dir / f"f{fold}" / ("va_rest" if va_use_restormer else "va_orig")
    te_dir = cache_dir / f"f{fold}" / "te"
    for d in (tr_dir, va_dir, te_dir):
        d.mkdir(parents=True, exist_ok=True)

    model = None
    if scenario == "dl_enh_copypaste":
        w_rest = cache_dir / f"restormer_f{fold}.pth"
        tr_paths = [data.image_path(s) for s in tr_stems]
        legal_gong = None
        if gong_dir is not None and gong_dir.exists():
            raw_gong = list(gong_dir.rglob("*.jpg"))
            te_all = sorted(data.test_stems())
            legal_gong = sde.filter_legal_gong_paths(raw_gong, te_all, va_stems, min_gap_sec=3600)
        model = sde.train_fold_restormer(tr_paths, w_rest, device="cuda", epochs=20, patience=5, gong_paths=legal_gong)

    def _render(stems: list[str], out_d: Path, use_dl: bool) -> None:
        shared_env = os.environ.get("EXP013_SHARED_PHYS_DIR", "")
        shared_dir = Path(shared_env).resolve() if shared_env else None
        for s in stems:
            dst = out_d / f"{s}.png"
            if dst.exists():
                continue
            if (not use_dl or model is None) and shared_dir is not None:
                src_shared = shared_dir / f"{s}.png"
                if src_shared.exists():
                    try:
                        os.link(src_shared, dst)
                    except OSError:
                        try:
                            os.symlink(src_shared, dst)
                        except OSError:
                            shutil.copy2(src_shared, dst)
                    continue
            im = cv2.imread(str(data.image_path(s)), cv2.IMREAD_GRAYSCALE)
            bgr = sde.enhance_full_image_with_restormer(model, im, device="cuda") if (use_dl and model is not None) else sde.physics_3ch_from_gray(im)
            cv2.imwrite(str(dst), bgr)

    _render(tr_stems, tr_dir, use_dl=(model is not None))
    _render(va_stems, va_dir, use_dl=(va_use_restormer and model is not None))
    _render(test_stems, te_dir, use_dl=(va_use_restormer and model is not None))
    if model is not None:
        del model
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return tr_dir, va_dir, te_dir


def _poly_to_yolo_seg_line(pts_flat: list[float], full: int = FULL) -> str | None:
    if len(pts_flat) < 6:
        return None
    coords = [f"{max(0.0, min(1.0, float(v) / full)):.6f}" for v in pts_flat]
    return "0 " + " ".join(coords)


def _ann_to_yolo_seg_line(a: dict, sde, full: int = FULL) -> str | None:
    seg = a.get("segmentation")
    if isinstance(seg, list):
        polys = [p for p in seg if isinstance(p, (list, tuple)) and len(p) >= 6]
        if polys:
            best = max(polys, key=len)
            return _poly_to_yolo_seg_line(list(best), full)
    m = sde.decode_ann_mask(a, full)
    if not m.any():
        return None
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    eps = max(1.0, 0.004 * cv2.arcLength(cnt, True))
    approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    return _poly_to_yolo_seg_line(approx.reshape(-1).tolist(), full)


def _build_yolo_dataset_fold(
    data,
    tr_dir: Path,
    va_dir: Path,
    root: Path,
    tr: list[str],
    va: list[str],
    scenario: str,
    fold: int,
    is_seg: bool = False,
) -> Path:
    import solar_dl_enhance as sde  # noqa: E402

    recs, anns = data.coco_index()
    for sp, stems, src_dir in (("train", tr, tr_dir), ("val", va, va_dir)):
        for sub in ("images", "labels"):
            shutil.rmtree(root / sub / sp, ignore_errors=True)
            (root / sub / sp).mkdir(parents=True)
        for s in stems:
            src_img = src_dir / f"{s}.png"
            # Validation strictly uses untouched clean real photo & first canonical record (no augmentation/leakage)
            rids = recs[s] if sp == "train" else recs[s][:1]
            for rid in rids:
                stem = s if len(rids) == 1 else f"{s}__{rid}"
                os.symlink(src_img, root / "images" / sp / f"{stem}.png")
                lines = []
                for a in anns.get(rid, []):
                    x, y, w, h = a["bbox"]
                    if w < 2 or h < 2:
                        continue
                    if is_seg:
                        ln = _ann_to_yolo_seg_line(a, sde, FULL)
                        if ln is not None:
                            lines.append(ln)
                    else:
                        lines.append(f"0 {(x + w / 2) / FULL:.6f} {(y + h / 2) / FULL:.6f} {w / FULL:.6f} {h / FULL:.6f}")
                (root / "labels" / sp / f"{stem}.txt").write_text("\n".join(lines))

    # Add strictly train-only object-aware small-filament Copy-Paste samples (`va` stays 100% original!)
    if scenario == "dl_enh_copypaste":
        rng = np.random.default_rng(100 + fold)
        bank = sde.build_tr_copypaste_bank(tr, tr_dir, recs, anns, max_bank=420, seed=fold)
        n_cp = min(len(tr), 240)
        for idx, s in enumerate(rng.choice(tr, size=n_cp, replace=False)):
            bgr = cv2.imread(str(tr_dir / f"{s}.png"))
            rid = recs[s][int(rng.integers(len(recs[s])))]
            ann_list = [a for a in anns.get(rid, []) if a["bbox"][2] >= 2 and a["bbox"][3] >= 2]
            boxes = [list(a["bbox"]) for a in ann_list]
            if is_seg:
                aug_bgr, _aug_boxes, pasted_polys = sde.apply_copypaste_sample(bgr, boxes, bank, rng, n_paste=3, return_polys=True)
                lines = []
                for a in ann_list:
                    ln = _ann_to_yolo_seg_line(a, sde, FULL)
                    if ln is not None:
                        lines.append(ln)
                for pflat in pasted_polys:
                    ln = _poly_to_yolo_seg_line(pflat, FULL)
                    if ln is not None:
                        lines.append(ln)
            else:
                aug_bgr, aug_boxes = sde.apply_copypaste_sample(bgr, boxes, bank, rng, n_paste=3)
                lines = [f"0 {(x + w / 2) / FULL:.6f} {(y + h / 2) / FULL:.6f} {w / FULL:.6f} {h / FULL:.6f}" for x, y, w, h in aug_boxes]
            cp_stem = f"cp_{idx:03d}_{s}"
            cv2.imwrite(str(root / "images" / "train" / f"{cp_stem}.png"), aug_bgr)
            (root / "labels" / "train" / f"{cp_stem}.txt").write_text("\n".join(lines))

    yaml_p = root / "data.yaml"
    yaml_p.write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n")
    return yaml_p


def _load_det_model(arch: str, weights_path: str | None = None):
    from ultralytics import RTDETR, YOLO

    if weights_path is not None:
        return RTDETR(weights_path) if arch.startswith("rtdetr") else YOLO(weights_path)
    candidates = {
        "rtdetr-x": ("rtdetr-x.pt", "rtdetr-l.pt"),
        "rtdetr-l": ("rtdetr-l.pt",),
        "yolo26x-seg": ("yolo26x-seg.pt", "yolo26l-seg.pt", "yolo11x-seg.pt"),
        "yolo26l-seg": ("yolo26l-seg.pt", "yolo11x-seg.pt"),
        "yolo26x": ("yolo26x.pt", "yolo26l.pt", "yolo12x.pt"),
        "yolo12x": ("yolo12x.pt", "yolo12l.pt", "yolo11x.pt"),
        "yolo12l": ("yolo12l.pt", "yolo11l.pt"),
    }.get(arch, (f"{arch}.pt",))
    for w in candidates:
        try:
            return RTDETR(w) if arch.startswith("rtdetr") else YOLO(w)
        except Exception as exc:
            print(f"Fallback from {w}: {exc}", flush=True)
    raise RuntimeError(f"Could not instantiate {arch}")


def worker_train_fold(
    repo_dir: Path,
    data_root: Path,
    work_dir: Path,
    arch: str,
    scenario: str,
    fold: int,
    epochs: int,
    patience: int = 5,
    gong_dir: Path | None = None,
    pseudo_conf: float = 0.0,
    init_weights_dir: Path | None = None,
) -> None:
    import torch
    import solar_dl_enhance as sde  # noqa: E402

    out_w = work_dir / "weights" / f"det_{arch}_{scenario}_f{fold}.pt"
    done_p = work_dir / "weights" / f"det_{arch}_{scenario}_f{fold}.done.json"
    out_w.parent.mkdir(parents=True, exist_ok=True)
    if out_w.exists() and done_p.exists():
        print(f"[{arch} | {scenario} f{fold}] already completed -> skipping", flush=True)
        return

    data, _, _, _, _ = _bind_filseg(repo_dir, data_root, work_dir)
    tr, va = data.split(fold)
    te = sorted(data.test_stems()) if scenario == "dl_enh_copypaste" else []

    is_seg = arch.endswith("-seg")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 16.0
    imgsz = (1344 if vram_gb < 20 else 1536) if arch.startswith("yolo26") else 1280
    cache_dir = Path(f"/tmp/exp013_prep_{scenario}")
    tr_dir, va_dir, _ = _prepare_fold_images(data, tr, va, te, cache_dir, scenario, fold, va_use_restormer=False, gong_dir=gong_dir)
    ds_root = work_dir / "det_data" / f"{arch}_{scenario}_f{fold}"
    yaml_p = _build_yolo_dataset_fold(data, tr_dir, va_dir, ds_root, tr, va, scenario, fold, is_seg=is_seg)

    if is_seg:
        batch = 6 if vram_gb >= 32 else (3 if vram_gb >= 22 else 2)
    else:
        batch = 8 if (arch.startswith("rtdetr") and vram_gb >= 32) else (2 if arch.startswith("rtdetr") else (10 if vram_gb >= 32 else 4))
    lr0 = 1.5e-4 if arch.startswith("rtdetr") else 6.0e-4

    best_map = -1.0
    val_rec: dict = {}
    used_bs = batch
    init_pt = (init_weights_dir / f"det_{arch}_{scenario}_f{fold}.pt") if init_weights_dir is not None else None
    init_val_p = (init_weights_dir / f"det_{arch}_{scenario}_f{fold}_val.json") if init_weights_dir is not None else None
    if init_pt is not None and init_pt.exists():
        shutil.copy2(init_pt, out_w)
        if init_val_p is not None and init_val_p.exists():
            val_rec = json.loads(init_val_p.read_text())
            best_map = float(val_rec.get("va_orig_phys", {}).get("metrics/mAP50(B)", -1.0))
        else:
            teacher_chk = _load_det_model(arch, str(out_w))
            rm_t = teacher_chk.val(data=str(yaml_p), imgsz=imgsz, batch=used_bs, device=0, verbose=False, plots=False)
            val_m = getattr(rm_t, "results_dict", {})
            val_rec = {"va_orig_phys": {str(k): float(v) for k, v in val_m.items()}}
            best_map = float(val_m.get("metrics/mAP50(B)", -1.0))
    else:
        for b_try in (batch, max(2, batch // 2), 2, 1):
            try:
                model = _load_det_model(arch)
                r = model.train(
                    data=str(yaml_p),
                    optimizer="AdamW",
                    lr0=lr0,
                    warmup_epochs=4,
                    epochs=epochs,
                    patience=patience,
                    imgsz=imgsz,
                    batch=b_try,
                    pretrained=True,
                    device=0,
                    workers=4,
                    cache=False,
                    hsv_h=0.0,
                    hsv_s=0.0,
                    hsv_v=0.12,
                    fliplr=0.5,
                    flipud=0.5,
                    degrees=180.0 if is_seg else 0.0,
                    mosaic=0.0 if arch.startswith("rtdetr") else 0.35,
                    scale=0.30,
                    mixup=0.0,
                    overlap_mask=True,
                    mask_ratio=2 if is_seg else 4,
                    plots=False,
                    project=str(work_dir / "det" / f"{arch}_{scenario}"),
                    name=f"fold{fold}",
                    exist_ok=True,
                    verbose=False,
                    seed=fold,
                )
                shutil.copy2(Path(r.save_dir) / "weights" / "best.pt", out_w)
                val_m = getattr(r, "results_dict", {})
                val_rec = {"va_orig_phys": {str(k): float(v) for k, v in val_m.items()}}
                best_map = float(val_m.get("metrics/mAP50(B)", -1.0))
                used_bs = b_try
                del model, r
                torch.cuda.empty_cache()
                break
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and b_try > 1:
                    torch.cuda.empty_cache()
                    continue
                raise

    # Optional legal external GONG pseudo-label self-training (strictly excluding test +-1h/dates & fold va dates)
    if pseudo_conf > 0.0 and gong_dir is not None and gong_dir.exists():
        legal_gong = sde.filter_legal_gong_paths(list(gong_dir.rglob("*.jpg")), sorted(data.test_stems()), va, min_gap_sec=3600)
        if legal_gong:
            rng_g = np.random.default_rng(2026 + fold)
            pick_g = [legal_gong[int(j)] for j in rng_g.choice(len(legal_gong), size=min(len(legal_gong), 320), replace=False)]
            teacher = _load_det_model(arch, str(out_w))
            pseudo_img_dir = ds_root / "images" / "train"
            pseudo_lbl_dir = ds_root / "labels" / "train"
            n_pseudo = 0
            for gp in pick_g:
                im_g = cv2.imread(str(gp), cv2.IMREAD_GRAYSCALE)
                if im_g is None or min(im_g.shape[:2]) < 512:
                    continue
                if im_g.shape != (FULL, FULL):
                    im_g = cv2.resize(im_g, (FULL, FULL), interpolation=cv2.INTER_AREA)
                bgr_g = sde.physics_3ch_from_gray(im_g)
                pred_r = teacher.predict(bgr_g, imgsz=imgsz, conf=pseudo_conf, max_det=45, verbose=False, device=0)[0]
                n_det = len(pred_r.boxes) if pred_r.boxes is not None else 0
                if n_det < 1 or n_det > 45:
                    continue
                p_lines = []
                if is_seg and pred_r.masks is not None and len(pred_r.masks.xy) == n_det:
                    for poly in pred_r.masks.xy:
                        if len(poly) >= 3:
                            p_lines.append(_poly_to_yolo_seg_line(poly.reshape(-1).tolist(), FULL))
                else:
                    for x1, y1, x2, y2 in pred_r.boxes.xyxy.cpu().numpy():
                        w_b, h_b = max(2.0, x2 - x1), max(2.0, y2 - y1)
                        p_lines.append(f"0 {((x1 + x2) * 0.5) / FULL:.6f} {((y1 + y2) * 0.5) / FULL:.6f} {w_b / FULL:.6f} {h_b / FULL:.6f}")
                p_lines = [ln for ln in p_lines if ln]
                if p_lines:
                    ps = f"gong_{gp.stem}"
                    cv2.imwrite(str(pseudo_img_dir / f"{ps}.png"), bgr_g)
                    (pseudo_lbl_dir / f"{ps}.txt").write_text("\n".join(p_lines))
                    n_pseudo += 1
            del teacher
            torch.cuda.empty_cache()
            if n_pseudo >= 20:
                stu = _load_det_model(arch, str(out_w))
                rs = stu.train(
                    data=str(yaml_p),
                    optimizer="AdamW",
                    lr0=lr0 * 0.45,
                    warmup_epochs=2,
                    epochs=max(16, epochs // 2),
                    patience=patience,
                    imgsz=imgsz,
                    batch=used_bs,
                    pretrained=True,
                    device=0,
                    workers=4,
                    cache=False,
                    hsv_h=0.0,
                    hsv_s=0.0,
                    hsv_v=0.10,
                    fliplr=0.5,
                    flipud=0.5,
                    degrees=180.0 if is_seg else 0.0,
                    mosaic=0.0,
                    scale=0.25,
                    overlap_mask=True,
                    mask_ratio=2 if is_seg else 4,
                    plots=False,
                    project=str(work_dir / "det" / f"{arch}_{scenario}_ssl"),
                    name=f"fold{fold}",
                    exist_ok=True,
                    verbose=False,
                    seed=100 + fold,
                )
                stu_m = getattr(rs, "results_dict", {})
                stu_map = float(stu_m.get("metrics/mAP50(B)", -1.0))
                val_rec["gong_ssl_pseudo"] = {"n_pseudo": n_pseudo, "va_mAP50": stu_map, "promoted": bool(stu_map > best_map)}
                if stu_map > best_map:
                    best_map = stu_map
                    shutil.copy2(Path(rs.save_dir) / "weights" / "best.pt", out_w)
                del stu, rs
                torch.cuda.empty_cache()

    # If DL enhancer scenario, benchmark inductive tr-only Restormer on untouched real va GT labels
    if scenario == "dl_enh_copypaste":
        _, va_rest_dir, _ = _prepare_fold_images(data, tr, va, [], cache_dir, scenario, fold, va_use_restormer=True)
        yaml_r = _build_yolo_dataset_fold(data, tr_dir, va_rest_dir, work_dir / "det_data" / f"{arch}_{scenario}_f{fold}_restva", tr, va, "baseline", fold, is_seg=is_seg)
        best_m = _load_det_model(arch, str(out_w))
        rm = best_m.val(data=str(yaml_r), imgsz=imgsz, batch=used_bs, device=0, verbose=False, plots=False)
        val_rec["va_tr_restormer"] = {str(k): float(v) for k, v in getattr(rm, "results_dict", {}).items()}

    metrics_p = work_dir / "weights" / f"det_{arch}_{scenario}_f{fold}_val.json"
    metrics_p.write_text(json.dumps(val_rec, indent=2))
    done_p.write_text(json.dumps({"arch": arch, "fold": fold, "best_map50": best_map, "is_seg": is_seg}))
    print(f"[{arch} | {scenario} f{fold}] va_orig mAP50={best_map:.4f} | va_rest={val_rec.get('va_tr_restormer', {}).get('metrics/mAP50(B)', -1):.4f}", flush=True)


def _unflip_mask_crop(mask_hw: np.ndarray, ax: int | None, full: int, x0: int = 0, y0: int = 0, tw: int = FULL) -> np.ndarray:
    """Resize raw instance mask tile/canvas cleanly and undo flip transform into crop [0..tw]."""
    mh, mw = mask_hw.shape[:2]
    m_rs = cv2.resize(mask_hw.astype(np.float32), (tw, tw), interpolation=cv2.INTER_LINEAR)
    if ax == 1:
        m_rs = m_rs[:, ::-1]
    elif ax == 0:
        m_rs = m_rs[::-1, :]
    return m_rs


def _predict_boxes(models: list, detector, bgr: np.ndarray, arch: str, conf: float = 0.05, max_det: int = 85):
    import torch

    is_seg = arch.endswith("-seg")
    views = []
    seg_records: list[tuple[np.ndarray, float, np.ndarray]] = []  # (xyxy_full, cf, crop_u8)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 16.0
    sz_y = 1344 if vram_gb < 24 else 1536

    if is_seg or arch.startswith("yolo26"):
        scales = [(sz_y, None), (sz_y, 1), (sz_y, 0)]
    else:
        scales = [(1280, None), (1280, 1), (1280, 0)]
        if not arch.startswith("rtdetr"):
            scales.append((sz_y, None))

    for m in models:
        for imgsz, ax in scales:
            x = bgr if ax is None else np.ascontiguousarray(np.flip(bgr, ax))
            r = m.predict(x, imgsz=imgsz, conf=conf, max_det=max_det, retina_masks=is_seg, verbose=False, device=0)[0]
            if len(r.boxes) == 0:
                continue
            b = r.boxes.xyxyn.cpu().numpy().astype(np.float64) * FULL
            confs_v = r.boxes.conf.cpu().numpy().astype(np.float64)
            if ax == 1:
                b[:, [0, 2]] = FULL - b[:, [2, 0]]
            elif ax == 0:
                b[:, [1, 3]] = FULL - b[:, [3, 1]]
            views.append(np.c_[b, confs_v])

            if is_seg and r.masks is not None and len(r.masks.data) == len(b):
                m_data = r.masks.data.cpu().numpy()
                for j, (bx1, by1, bx2, by2) in enumerate(b):
                    cf = float(confs_v[j])
                    ix1, iy1 = max(0, int(math.floor(bx1)) - 2), max(0, int(math.floor(by1)) - 2)
                    ix2, iy2 = min(FULL, int(math.ceil(bx2)) + 2), min(FULL, int(math.ceil(by2)) + 2)
                    if ix2 <= ix1 or iy2 <= iy1:
                        continue
                    # Transform FULL crop coords into raw mask space without resizing entire 2048x2048 tensor
                    mh, mw = m_data[j].shape
                    rx1, rx2 = (FULL - ix2, FULL - ix1) if ax == 1 else (ix1, ix2)
                    ry1, ry2 = (FULL - iy2, FULL - iy1) if ax == 0 else (iy1, iy2)
                    mx1 = max(0, int(math.floor(rx1 * mw / FULL)))
                    mx2 = min(mw, max(mx1 + 1, int(math.ceil(rx2 * mw / FULL))))
                    my1 = max(0, int(math.floor(ry1 * mh / FULL)))
                    my2 = min(mh, max(my1 + 1, int(math.ceil(ry2 * mh / FULL))))
                    sub_m = cv2.resize(m_data[j, my1:my2, mx1:mx2].astype(np.float32), (ix2 - ix1, iy2 - iy1), interpolation=cv2.INTER_LINEAR)
                    if ax == 1:
                        sub_m = sub_m[:, ::-1]
                    elif ax == 0:
                        sub_m = sub_m[::-1, :]
                    seg_records.append((np.array([ix1, iy1, ix2, iy2], dtype=np.float32), cf, np.clip(sub_m * 255.0 + 0.5, 0, 255).astype(np.uint8)))

        # Native 1.0x resolution 2x2 overlapping 1280x1280 SAHI tiles recover tiny (<580 px) filaments
        tile = 1280
        sahi_det = []
        for y0 in (0, FULL - tile):
            for x0 in (0, FULL - tile):
                sub = np.ascontiguousarray(bgr[y0 : y0 + tile, x0 : x0 + tile])
                r = m.predict(sub, imgsz=1280, conf=conf, max_det=40, retina_masks=is_seg, verbose=False, device=0)[0]
                if len(r.boxes) == 0:
                    continue
                tb = r.boxes.xyxyn.cpu().numpy().astype(np.float64) * tile
                tc = r.boxes.conf.cpu().numpy().astype(np.float64)
                m_tile = r.masks.data.cpu().numpy() if (is_seg and r.masks is not None and len(r.masks.data) == len(tb)) else None
                for j, ((bx1, by1, bx2, by2), cf) in enumerate(zip(tb, tc)):
                    if (x0 > 0 and bx1 <= 4) or (x0 == 0 and bx2 >= tile - 4):
                        continue
                    if (y0 > 0 and by1 <= 4) or (y0 == 0 and by2 >= tile - 4):
                        continue
                    fx1, fy1, fx2, fy2 = x0 + bx1, y0 + by1, x0 + bx2, y0 + by2
                    sahi_det.append([fx1, fy1, fx2, fy2, cf * 0.90])
                    if m_tile is not None:
                        ix1, iy1 = max(0, int(math.floor(fx1)) - 2), max(0, int(math.floor(fy1)) - 2)
                        ix2, iy2 = min(FULL, int(math.ceil(fx2)) + 2), min(FULL, int(math.ceil(fy2)) + 2)
                        if ix2 > ix1 and iy2 > iy1:
                            mh, mw = m_tile[j].shape
                            lx1 = max(0, int(math.floor((ix1 - x0) * mw / tile)))
                            lx2 = min(mw, max(lx1 + 1, int(math.ceil((ix2 - x0) * mw / tile))))
                            ly1 = max(0, int(math.floor((iy1 - y0) * mh / tile)))
                            ly2 = min(mh, max(ly1 + 1, int(math.ceil((iy2 - y0) * mh / tile))))
                            sub_m = cv2.resize(m_tile[j, ly1:ly2, lx1:lx2].astype(np.float32), (ix2 - ix1, iy2 - iy1), interpolation=cv2.INTER_LINEAR)
                            seg_records.append((np.array([ix1, iy1, ix2, iy2], dtype=np.float32), float(cf * 0.90), np.clip(sub_m * 255.0 + 0.5, 0, 255).astype(np.uint8)))
        if sahi_det:
            views.append(np.array(sahi_det, dtype=np.float64))

    if not views:
        empty_b = np.zeros((0, 4), dtype=np.float32)
        empty_c = np.zeros((0,), dtype=np.float32)
        return (empty_b, empty_c, [], []) if is_seg else (empty_b, empty_c)

    fused = detector.wbf(views, thr=0.55)[:max_det] if len(views) > 1 else [(v[:4], float(v[4])) for v in views[0][:max_det]]
    boxes = np.array([f[0] for f in fused], dtype=np.float32).reshape(-1, 4)
    confs = np.array([f[1] for f in fused], dtype=np.float32)
    if not is_seg:
        return boxes, confs

    ws_out: list[tuple[int, int, int, int]] = []
    probs_out: list[np.ndarray] = []
    for bx1, by1, bx2, by2 in boxes:
        cx1, cy1 = max(0, int(math.floor(bx1)) - 4), max(0, int(math.floor(by1)) - 4)
        cx2, cy2 = min(FULL, int(math.ceil(bx2)) + 4), min(FULL, int(math.ceil(by2)) + 4)
        cw, ch = max(1, cx2 - cx1), max(1, cy2 - cy1)
        acc = np.zeros((ch, cw), dtype=np.float32)
        wsum = 0.0
        b_area = max(1.0, (bx2 - bx1) * (by2 - by1))
        for rbox, rcf, rcrop in seg_records:
            rx1, ry1, rx2, ry2 = float(rbox[0]), float(rbox[1]), float(rbox[2]), float(rbox[3])
            ix1, iy1 = max(bx1, rx1), max(by1, ry1)
            ix2, iy2 = min(bx2, rx2), min(by2, ry2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            union = b_area + max(1.0, (rx2 - rx1) * (ry2 - ry1)) - inter
            if inter / max(union, 1e-6) < 0.42:
                continue
            ox1, oy1 = max(cx1, int(rbox[0])), max(cy1, int(rbox[1]))
            ox2, oy2 = min(cx2, int(rbox[2])), min(cy2, int(rbox[3]))
            if ox2 <= ox1 or oy2 <= oy1:
                continue
            acc[oy1 - cy1 : oy2 - cy1, ox1 - cx1 : ox2 - cx1] += rcf * (
                rcrop[oy1 - int(rbox[1]) : oy2 - int(rbox[1]), ox1 - int(rbox[0]) : ox2 - int(rbox[0])].astype(np.float32) / 255.0
            )
            wsum += rcf
        mean_m = (acc / max(wsum, 1e-6)) if wsum > 0 else np.zeros((ch, cw), dtype=np.float32)
        ws_out.append((cx1, cy1, cx2, cy2))
        probs_out.append(np.clip(mean_m * 255.0 + 0.5, 0, 255).astype(np.uint8))
    return boxes, confs, ws_out, probs_out


def worker_infer_boxes(repo_dir: Path, data_root: Path, work_dir: Path, arch: str, scenario: str, boxes_dir: Path, part: int, s1_inst_dir: Path | None = None) -> None:
    import solar_dl_enhance as sde  # noqa: E402

    data, detector, _, _, postprocess = _bind_filseg(repo_dir, data_root, work_dir)
    folds_map = data.folds()
    models = {k: _load_det_model(arch, str(work_dir / "weights" / f"det_{arch}_{scenario}_f{k}.pt")) for k in range(5)}
    is_seg = arch.endswith("-seg")

    oof_dir, test_dir = boxes_dir / "oof", boxes_dir / "test"
    oof_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    def _get_bgr(stem: str, fk: int | None) -> np.ndarray:
        p = Path(f"/tmp/exp013_prep_{scenario}") / f"f{fk}" / "va_orig" / f"{stem}.png" if fk is not None else None
        if p is not None and p.exists():
            return cv2.imread(str(p))
        im = cv2.imread(str(data.image_path(stem)), cv2.IMREAD_GRAYSCALE)
        return sde.physics_3ch_from_gray(im)

    oof_stems = sorted(folds_map)[part::2]
    for s in oof_stems:
        fk = int(folds_map[s])
        bgr = _get_bgr(s, fk)
        out = _predict_boxes([models[fk]], detector, bgr, arch)
        boxes, confs = out[0], out[1]
        np.savez_compressed(oof_dir / f"{s}.npz", box=boxes, conf=confs)
        if is_seg and s1_inst_dir is not None:
            postprocess.save_cache(s1_inst_dir / "oof" / f"{s}.npz", confs, out[2], out[3])

    test_stems = sorted(data.test_stems())[part::2]
    test_list = list(models.values())
    for s in test_stems:
        bgr = _get_bgr(s, None)
        out = _predict_boxes(test_list, detector, bgr, arch)
        boxes, confs = out[0], out[1]
        np.savez_compressed(test_dir / f"{s}.npz", box=boxes, conf=confs)
        if is_seg and s1_inst_dir is not None:
            postprocess.save_cache(s1_inst_dir / "test" / f"{s}.npz", confs, out[2], out[3])
    print(f"[{arch} infer p{part}] Done ({len(oof_stems)} oof, {len(test_stems)} test)", flush=True)


def run_driver(arch: str, scenario: str, epochs: int, patience: int, smoke_fold0_ab: bool, gong_dir: Path | None, pseudo_conf: float, out_tag: str = "", init_weights_dir: Path | None = None) -> None:
    import torch

    t0 = time.time()
    data_root = _find_comp_root()
    code_root = _find_input_dir("filseg-gap-wave4-code") or _find_input_dir("filseg-gap-wave2-code") or Path("/kaggle/working/filseg_repo")
    repo_dir = Path("/tmp/filseg_repo_w4")
    if not (repo_dir / "filseg").exists():
        if (code_root / "filseg").exists():
            shutil.copytree(code_root, repo_dir, dirs_exist_ok=True)
        else:
            repo_dir = Path(__file__).resolve().parents[3]

    tag_sfx = f"_{out_tag}" if out_tag else ""
    work_dir = Path(f"/tmp/exp013_{arch}_{scenario}{tag_sfx}")
    work_dir.mkdir(parents=True, exist_ok=True)
    common_env = {**os.environ, "PYTHONPATH": f"{repo_dir}:{Path(__file__).resolve().parent}"}
    py = sys.executable

    gong_args = ["--gong-dir", str(gong_dir)] if gong_dir is not None else []
    pconf_args = ["--pseudo-conf", str(pseudo_conf)] if pseudo_conf > 0 else []
    init_args = ["--init-weights-dir", str(init_weights_dir)] if init_weights_dir is not None else []

    if smoke_fold0_ab:
        # Clean untouched Fold-0 validation comparison: Baseline Physics vs DL-Enhancer+CopyPaste
        for sc in ("baseline", "dl_enh_copypaste"):
            rc = subprocess.call(
                [py, __file__, "--worker", "train_fold", "--repo", str(repo_dir), "--data-root", str(data_root),
                 "--work-dir", str(work_dir), "--arch", arch, "--scenario", sc, "--fold", "0",
                 "--epochs", str(epochs), "--patience", str(patience)] + gong_args,
                env=dict(common_env, CUDA_VISIBLE_DEVICES="0"),
            )
            assert rc == 0, f"Fold-0 {sc} failed"
        return

    n_gpu = torch.cuda.device_count()
    fold_groups = [(0, [0, 1, 2]), (1, [3, 4])] if n_gpu >= 2 else [(0, [0, 1, 2, 3, 4])]
    errs: list[BaseException] = []

    def _run_group(gpu: int, fks: list[int]):
        try:
            for fk in fks:
                rc = subprocess.call(
                    [py, __file__, "--worker", "train_fold", "--repo", str(repo_dir), "--data-root", str(data_root),
                     "--work-dir", str(work_dir), "--arch", arch, "--scenario", scenario, "--fold", str(fk),
                     "--epochs", str(epochs), "--patience", str(patience)] + gong_args + pconf_args + init_args,
                    env=dict(common_env, CUDA_VISIBLE_DEVICES=str(gpu)),
                )
                assert rc == 0, f"{arch} fold {fk} failed"
        except BaseException as exc:
            errs.append(exc)

    ts = [threading.Thread(target=_run_group, args=(g, fks)) for g, fks in fold_groups]
    for th in ts:
        th.start()
    for th in ts:
        th.join()
    assert not errs, errs

    boxes_dir = work_dir / f"boxes_{arch}_{scenario}{tag_sfx}"
    s1_inst_dir = (work_dir / f"s1_inst_{arch}_{scenario}{tag_sfx}") if arch.endswith("-seg") else None
    p_errs: list[BaseException] = []

    def _run_inf(part: int):
        try:
            gpu = part if n_gpu >= 2 else 0
            cmd = [py, __file__, "--worker", "infer", "--repo", str(repo_dir), "--data-root", str(data_root),
                   "--work-dir", str(work_dir), "--arch", arch, "--scenario", scenario, "--boxes-dir", str(boxes_dir), "--part", str(part)]
            if s1_inst_dir is not None:
                cmd += ["--s1-inst-dir", str(s1_inst_dir)]
            rc = subprocess.call(cmd, env=dict(common_env, CUDA_VISIBLE_DEVICES=str(gpu)))
            assert rc == 0, f"{arch} infer p{part} failed"
        except BaseException as exc:
            p_errs.append(exc)

    if n_gpu >= 2:
        tis = [threading.Thread(target=_run_inf, args=(0,)), threading.Thread(target=_run_inf, args=(1,))]
        for th in tis:
            th.start()
        for th in tis:
            th.join()
    else:
        for p_idx in (0, 1):
            _run_inf(p_idx)
    assert not p_errs, p_errs

    safe_arch = arch.replace("-", "_")
    base_name = f"exp013_{safe_arch}_{scenario}{tag_sfx}"
    out_tar = Path(f"/kaggle/working/{base_name}_boxes.tar.gz")
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tar, "w:gz") as tf:
        tf.add(boxes_dir, arcname=boxes_dir.name)
    if s1_inst_dir is not None and s1_inst_dir.exists():
        mask_tar = Path(f"/kaggle/working/{base_name}_s1masks.tar.gz")
        with tarfile.open(mask_tar, "w:gz") as tf:
            tf.add(s1_inst_dir, arcname=s1_inst_dir.name)
    print(f"EXP-013a {arch}/{scenario}{tag_sfx} COMPLETE in {(time.time()-t0)/60:.1f}m -> {out_tar}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["train_fold", "infer"])
    ap.add_argument("--repo", type=Path)
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--work-dir", type=Path)
    ap.add_argument("--boxes-dir", type=Path)
    ap.add_argument("--s1-inst-dir", type=Path)
    ap.add_argument("--gong-dir", type=Path)
    ap.add_argument("--init-weights-dir", type=Path)
    ap.add_argument("--out-tag", default="")
    ap.add_argument("--pseudo-conf", type=float, default=0.0)
    ap.add_argument("--arch", default="yolo26x-seg")
    ap.add_argument("--scenario", default="dl_enh_copypaste", choices=["baseline", "dl_enh_copypaste"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--smoke-fold0-ab", action="store_true")
    args = ap.parse_args()

    if args.worker is None:
        run_driver(args.arch, args.scenario, args.epochs, args.patience, args.smoke_fold0_ab, args.gong_dir, args.pseudo_conf, args.out_tag, args.init_weights_dir)
    elif args.worker == "train_fold":
        worker_train_fold(args.repo, args.data_root, args.work_dir, args.arch, args.scenario, args.fold, args.epochs, args.patience, args.gong_dir, args.pseudo_conf, args.init_weights_dir)
    elif args.worker == "infer":
        worker_infer_boxes(args.repo, args.data_root, args.work_dir, args.arch, args.scenario, args.boxes_dir, args.part, args.s1_inst_dir)


if __name__ == "__main__":
    main()
