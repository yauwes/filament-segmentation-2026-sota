"""EXP-013b (Wave 4 GPU Kernel): 5-Fold Mask2Former (Swin-Base) Box-Free Transformer Instance & Dense Segmentation.

Why Mask2Former Swin-Base directly targets our gap:
1. Box-free Masked-Attention Transformer + Deformable Multi-Scale Pixel Decoder (retains 1/4 res
   spatial details for thin 2-5px filaments without Stage-1 bbox clipping).
2. Global Swin-Base context separates dark magnetic filaments from sunspots, plages & limb haze.
3. Trained strictly leak-free on 5-fold CV over 1152x1152 overlapping solar tiles (5-tile 2048x2048
   coverage + 256px seam overlap so equatorial filaments stay intact).
4. Scenario support (`baseline` vs `dl_enh_copypaste` via fold-only SolarRestormer + tr-only
   Copy-Paste) while ALWAYS validating on untouched clean original competition images & labels.
5. Exports BOTH:
   - `m2f_inst`: Instance-level box-free proposals (.npz compatible with EXP-011c pool).
   - `m2f_probs`: Full 2048x2048 dense Transformer filament probability maps (uint8 .npz).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np

FULL = 2048
TILE = 1152
TILE_COORDS = [(0, 0), (0, FULL - TILE), (FULL - TILE, 0), (FULL - TILE, FULL - TILE), ((FULL - TILE) // 2, (FULL - TILE) // 2)]
IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


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
    from filseg import data, postprocess, segmenter  # noqa: E402

    data.ROOT = data_root
    data.TRAIN_JSON = data_root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    data.TRAIN_IMAGES = data_root / "train" / "train_images"
    data.TEST_IMAGES = data_root / "test" / "test_images"
    data.FOLDS = repo_dir / "folds" / "folds.json"
    data.WORK = work_dir
    data.WEIGHTS = work_dir / "weights"
    return data, postprocess, segmenter


def _normalize_bgr(bgr: np.ndarray) -> np.ndarray:
    # Convert uint8 BGR -> normalized float32 RGB (3, H, W)
    rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
    return (rgb.transpose(2, 0, 1) - IM_MEAN) / IM_STD


def _build_m2f_model(model_name: str = "facebook/mask2former-swin-base-coco-instance", ckpt_dir: str | None = None):
    from transformers import Mask2FormerForUniversalSegmentation

    src = ckpt_dir or model_name
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        src,
        num_labels=1,
        id2label={0: "filament"},
        label2id={"filament": 0},
        ignore_mismatched_sizes=True,
    )
    return model


def _decode_ann_instances(anns_list: list[dict]) -> list[tuple[np.ndarray, list[float]]]:
    import solar_dl_enhance as sde  # noqa: E402

    out = []
    for a in anns_list:
        m = sde.decode_ann_mask(a, FULL)
        if int(m.sum()) >= 9:
            out.append((m, list(a["bbox"])))
    return out


def _decode_tile_instances(anns_list: list[dict], x0: int, y0: int, tile_sz: int = TILE) -> list[np.ndarray]:
    import solar_dl_enhance as sde  # noqa: E402

    crop_masks = []
    x1, y1 = x0 + tile_sz, y0 + tile_sz
    for a in anns_list:
        bx, by, bw, bh = a.get("bbox", [0, 0, FULL, FULL])
        if bx + bw <= x0 or bx >= x1 or by + bh <= y0 or by >= y1:
            continue
        seg = a.get("segmentation")
        if isinstance(seg, list):
            cm = np.zeros((tile_sz, tile_sz), dtype=np.uint8)
            for poly in seg:
                if len(poly) >= 6:
                    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                    pts[:, 0] -= x0
                    pts[:, 1] -= y0
                    cv2.fillPoly(cm, [np.round(pts).astype(np.int32)], 1)
        else:
            full_m = sde.decode_ann_mask(a, FULL)
            cm = full_m[y0:y1, x0:x1].astype(np.uint8)
        if int(cm.sum()) >= 9:
            crop_masks.append(cm)
    return crop_masks


class SolarMask2FormerDataset:
    def __init__(
        self,
        stems: list[str],
        recs: dict,
        anns: dict,
        img_dir: Path,
        cp_bank: list | None = None,
        train: bool = True,
        seed: int = 0,
    ):
        self.stems = list(stems)
        self.recs = recs
        self.anns = anns
        self.img_dir = img_dir
        self.cp_bank = cp_bank
        self.train = train
        self.rng = np.random.default_rng(seed)
        self._bgr_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.stems) * (4 if self.train else 1)

    def __getitem__(self, idx: int):
        import torch
        import solar_dl_enhance as sde  # noqa: E402

        s = self.stems[idx % len(self.stems)]
        q_idx = idx // len(self.stems)
        bgr = self._bgr_cache.get(s)
        if bgr is None:
            bgr = cv2.imread(str(self.img_dir / f"{s}.png"))
        rids = self.recs[s]
        rid = rids[int(self.rng.integers(len(rids)))] if self.train else rids[0]

        if self.train:
            base_y, base_x = TILE_COORDS[q_idx % 4]
            y0 = int(np.clip(base_y + self.rng.integers(-160, 161), 0, FULL - TILE))
            x0 = int(np.clip(base_x + self.rng.integers(-160, 161), 0, FULL - TILE))
        else:
            y0, x0 = (FULL - TILE) // 2, (FULL - TILE) // 2

        if not (self.train and self.cp_bank):
            img_t = bgr[y0 : y0 + TILE, x0 : x0 + TILE].copy()
            crop_masks = _decode_tile_instances(self.anns.get(rid, []), x0, y0, TILE)
        else:
            raw_insts = _decode_ann_instances(self.anns.get(rid, []))
            masks_full = [m for m, _ in raw_insts]
            boxes_full = [b for _, b in raw_insts]
            if self.rng.random() < 0.65:
                bgr_new = bgr.copy()
                h_img, w_img = bgr_new.shape[:2]
                cx, cy, rad, _ = sde.disk_geometry(bgr_new[:, :, 0])
                occ = np.zeros((h_img, w_img), dtype=bool)
                for bx, by, bw, bh in boxes_full:
                    occ[max(0, int(by)) : min(h_img, int(by + bh)), max(0, int(bx)) : min(w_img, int(bx + bw))] = True
                for _ in range(3):
                    patch, pmask = self.cp_bank[int(self.rng.integers(len(self.cp_bank)))]
                    if self.rng.random() < 0.5:
                        patch, pmask = patch[:, ::-1], pmask[:, ::-1]
                    ph, pw = patch.shape[:2]
                    for _try in range(10):
                        ang = float(self.rng.uniform(0, 2 * math.pi))
                        rr = float(self.rng.uniform(0.08, 0.78)) * rad
                        px = int(cx + rr * math.cos(ang) - pw * 0.5)
                        py = int(cy + rr * math.sin(ang) - ph * 0.5)
                        if px < 4 or py < 4 or px + pw >= w_img - 4 or py + ph >= h_img - 4:
                            continue
                        if occ[py : py + ph, px : px + pw].any():
                            continue
                        alpha = np.clip(cv2.GaussianBlur(pmask.astype(np.float32), (5, 5), 1.0)[..., None], 0.0, 1.0)
                        roi = bgr_new[py : py + ph, px : px + pw].astype(np.float32)
                        p_adj = patch.astype(np.float32)
                        bgr_new[py : py + ph, px : px + pw] = np.clip(alpha * p_adj + (1.0 - alpha) * roi, 0, 255).astype(np.uint8)
                        m_full = np.zeros((h_img, w_img), dtype=np.uint8)
                        m_full[py : py + ph, px : px + pw] = (pmask > 0).astype(np.uint8)
                        masks_full.append(m_full)
                        occ[py : py + ph, px : px + pw] = True
                        break
                bgr = bgr_new
            img_t = bgr[y0 : y0 + TILE, x0 : x0 + TILE].copy()
            crop_masks = []
            for m in masks_full:
                cm = m[y0 : y0 + TILE, x0 : x0 + TILE]
                if int(cm.sum()) >= 9:
                    crop_masks.append(cm)

        if self.train:
            if self.rng.random() < 0.5:
                img_t = np.ascontiguousarray(img_t[:, ::-1])
                crop_masks = [np.ascontiguousarray(m[:, ::-1]) for m in crop_masks]
            if self.rng.random() < 0.5:
                img_t = np.ascontiguousarray(img_t[::-1, :])
                crop_masks = [np.ascontiguousarray(m[::-1, :]) for m in crop_masks]
            krot = int(self.rng.integers(4))
            if krot:
                img_t = np.ascontiguousarray(np.rot90(img_t, krot))
                crop_masks = [np.ascontiguousarray(np.rot90(m, krot)) for m in crop_masks]

        pix = torch.from_numpy(_normalize_bgr(img_t))
        if crop_masks:
            ms = torch.from_numpy(np.stack(crop_masks).astype(np.float32))
            cls = torch.zeros((len(crop_masks),), dtype=torch.long)
        else:
            ms = torch.zeros((0, TILE, TILE), dtype=torch.float32)
            cls = torch.zeros((0,), dtype=torch.long)
        return pix, ms, cls


def _collate_m2f(batch):
    import torch

    pix = torch.stack([b[0] for b in batch], dim=0)
    ms = [b[1] for b in batch]
    cls = [b[2] for b in batch]
    return pix, ms, cls


def _prepare_images(data, tr_stems: list[str], va_stems: list[str], te_stems: list[str], cache_dir: Path, scenario: str, fold: int):
    import solar_dl_enhance as sde  # noqa: E402

    tr_dir = cache_dir / f"f{fold}" / "tr"
    va_dir = cache_dir / f"f{fold}" / "va_orig"
    te_dir = cache_dir / f"f{fold}" / "te"
    for d in (tr_dir, va_dir, te_dir):
        d.mkdir(parents=True, exist_ok=True)

    model = None
    if scenario == "dl_enh_copypaste":
        w_rest = cache_dir / f"restormer_f{fold}.pth"
        model = sde.train_fold_restormer([data.image_path(s) for s in tr_stems], w_rest, device="cuda", epochs=10)

    shared_phys = Path("/tmp/shared_phys")
    shared_phys.mkdir(parents=True, exist_ok=True)

    def _render(stems: list[str], out_d: Path, use_dl: bool):
        for s in stems:
            dst = out_d / f"{s}.png"
            if dst.exists():
                continue
            if not use_dl:
                sp_png = shared_phys / f"{s}.png"
                if not sp_png.exists():
                    im = cv2.imread(str(data.image_path(s)), cv2.IMREAD_GRAYSCALE)
                    cv2.imwrite(str(sp_png), sde.physics_3ch_from_gray(im))
                try:
                    os.link(sp_png, dst)
                except OSError:
                    shutil.copy2(sp_png, dst)
                continue
            im = cv2.imread(str(data.image_path(s)), cv2.IMREAD_GRAYSCALE)
            bgr = sde.enhance_full_image_with_restormer(model, im, device="cuda") if (use_dl and model is not None) else sde.physics_3ch_from_gray(im)
            cv2.imwrite(str(dst), bgr)

    _render(tr_stems, tr_dir, use_dl=(model is not None))
    # Validation ALWAYS keeps untouched real original physical rendering (zero leakage)
    _render(va_stems, va_dir, use_dl=False)
    _render(te_stems, te_dir, use_dl=False)
    return tr_dir, va_dir, te_dir


def worker_train_fold(repo_dir: Path, data_root: Path, work_dir: Path, scenario: str, fold: int, epochs: int, patience: int = 5) -> None:
    import torch
    from torch.utils.data import DataLoader

    ckpt_out = work_dir / "weights" / f"m2f_{scenario}_f{fold}"
    done_p = ckpt_out.with_suffix(".done.json")
    if ckpt_out.exists() and done_p.exists():
        print(f"[Mask2Former {scenario} f{fold}] already completed -> skipping", flush=True)
        return

    t0 = time.time()
    data, _, _ = _bind_filseg(repo_dir, data_root, work_dir)
    import solar_dl_enhance as sde  # noqa: E402

    torch.manual_seed(fold)
    np.random.seed(fold)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and (torch.cuda.get_device_capability(0)[0] >= 8) and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    recs, anns = data.coco_index()
    tr, va = data.split(fold)
    cache_dir = Path(f"/tmp/exp013_prep_{scenario}")
    tr_dir, va_dir, _ = _prepare_images(data, tr, va, [], cache_dir, scenario, fold)

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 16.0
    bs_m2f = 8 if vram_gb >= 48 else 4
    cp_bank = sde.build_tr_copypaste_bank(tr, tr_dir, recs, anns, max_bank=320, seed=fold) if scenario == "dl_enh_copypaste" else None
    ds_tr = SolarMask2FormerDataset(tr, recs, anns, tr_dir, cp_bank=cp_bank, train=True, seed=fold)
    ds_va = SolarMask2FormerDataset(va, recs, anns, va_dir, cp_bank=None, train=False, seed=fold)

    dl_tr = DataLoader(ds_tr, batch_size=bs_m2f, shuffle=True, num_workers=4, collate_fn=_collate_m2f, drop_last=True, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=bs_m2f, shuffle=False, num_workers=2, collate_fn=_collate_m2f, pin_memory=True)

    model = _build_m2f_model().to(dev)
    backbone_p, head_p = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "pixel_level_module.encoder" in n:
            backbone_p.append(p)
        else:
            head_p.append(p)
    opt = torch.optim.AdamW([{"params": backbone_p, "lr": 4e-5}, {"params": head_p, "lr": 1.2e-4}], weight_decay=5e-2)
    total_steps = epochs * len(dl_tr)
    warm = max(1, len(dl_tr))

    def _lr_f(st: int) -> float:
        if st < warm:
            return (st + 1) / warm
        prog = (st - warm) / max(1, total_steps - warm)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_f)
    scaler = torch.amp.GradScaler("cuda", enabled=(not use_bf16 and torch.cuda.is_available()))
    best_dice = -1.0
    best_ep = 0
    no_imp = 0
    min_delta = 1e-4
    ckpt_out = work_dir / "weights" / f"m2f_{scenario}_f{fold}"

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for pix, ms, cls in dl_tr:
            pix = pix.to(dev, non_blocking=True)
            ms_d = [m.to(dev, non_blocking=True) for m in ms]
            cls_d = [c.to(dev, non_blocking=True) for c in cls]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                out = model(pixel_values=pix, mask_labels=ms_d, class_labels=cls_d)
                loss = out.loss
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            sched.step()
            losses.append(float(loss.item()))

        # Semantic center-tile Dice check on untouched original va (early stopping with patience=5)
        model.eval()
        inter, union = 0.0, 0.0
        with torch.no_grad():
            for pix, ms, _ in dl_va:
                pix = pix.to(dev, non_blocking=True)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                    out = model(pixel_values=pix)
                    cq = torch.softmax(out.class_queries_logits.float(), dim=-1)[..., 0]
                    mq = torch.sigmoid(out.masks_queries_logits.float())
                    mq_up = torch.nn.functional.interpolate(mq, size=(TILE, TILE), mode="bilinear", align_corners=False)
                    fg_prob = (cq[:, :, None, None] * mq_up).amax(dim=1) > 0.35
                for bi, m_gt in enumerate(ms):
                    gt_u = (m_gt.to(dev).sum(dim=0) > 0).float() if len(m_gt) else torch.zeros((TILE, TILE), device=dev)
                    pr_u = fg_prob[bi].float()
                    inter += float((pr_u * gt_u).sum().item())
                    union += float((pr_u + gt_u).sum().item())
        dice = (2.0 * inter + 1e-6) / (union + 1e-6)
        if dice > best_dice + min_delta:
            best_dice = dice
            best_ep = ep
            no_imp = 0
            model.save_pretrained(ckpt_out)
        else:
            if dice > best_dice:
                best_dice = dice
                best_ep = ep
                model.save_pretrained(ckpt_out)
            no_imp += 1
        print(f"[Mask2Former {scenario} f{fold}] ep={ep}/{epochs} loss={np.mean(losses):.4f} va_orig_dice={dice:.4f} best={best_dice:.4f} no_imp={no_imp}/{patience} ({(time.time()-t0)/60:.1f}m)", flush=True)
        if no_imp >= max(1, int(patience)):
            print(f"[Mask2Former {scenario} f{fold}] early-stop at ep={ep} (best_ep={best_ep} best_va_dice={best_dice:.4f})", flush=True)
            break

    done_p = ckpt_out.with_suffix(".done.json")
    done_p.write_text(json.dumps({"scenario": scenario, "fold": fold, "best_dice": best_dice, "best_ep": best_ep, "epochs_run": ep}))
    print(f"[Mask2Former {scenario} f{fold}] DONE best_va_dice={best_dice:.4f}", flush=True)


def _infer_single_photo(models: list, proc, bgr: np.ndarray, dev, use_bf16: bool, min_area: int = 35):
    """5-tile native 1152x1152 Mask2Former inference -> dense 2048x2048 prob + multi-fold consensus instances."""
    import torch
    import torch.nn.functional as F

    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    dense_acc = np.zeros((FULL, FULL), dtype=np.float32)
    dense_cnt = np.zeros((FULL, FULL), dtype=np.float32)
    raw_insts: list[tuple[np.ndarray, float, tuple[int, int, int, int], int]] = []

    tiles = [bgr[y0 : y0 + TILE, x0 : x0 + TILE] for y0, x0 in TILE_COORDS]
    pix = torch.from_numpy(np.stack([_normalize_bgr(t) for t in tiles])).to(dev)

    with torch.no_grad():
        per_model_cq = []
        for m_idx, m in enumerate(models):
            with torch.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                out = m(pixel_values=pix)
            cq = torch.softmax(out.class_queries_logits.float(), dim=-1)[..., 0]
            mq = F.interpolate(torch.sigmoid(out.masks_queries_logits.float()), size=(TILE, TILE), mode="bilinear", align_corners=False)
            p_tile = (cq[:, :, None, None] * mq).amax(dim=1).cpu().numpy()
            per_model_cq.append(p_tile)

            res = proc.post_process_instance_segmentation(
                out,
                threshold=0.28,
                mask_threshold=0.45,
                overlap_mask_area_threshold=0.5,
                target_sizes=[(TILE, TILE)] * len(TILE_COORDS),
                return_binary_maps=True,
            )
            mq_np = mq.cpu().numpy()
            cq_np = cq.cpu().numpy()

            for ti, (y0, x0) in enumerate(TILE_COORDS):
                seg_map_t = res[ti]["segmentation"]
                seg_map = seg_map_t.cpu().numpy() if hasattr(seg_map_t, "cpu") else np.asarray(seg_map_t)
                for seg_info in res[ti].get("segments_info", []):
                    sc = float(seg_info["score"])
                    sid = int(seg_info["id"])
                    bm = (seg_map[sid] > 0) if seg_map.ndim == 3 else (seg_map == sid)
                    if int(bm.sum()) < min_area:
                        continue
                    ys, xs = np.nonzero(bm)
                    by1, bx1, by2, bx2 = int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1
                    if ti < 4 and ((x0 > 0 and bx1 <= 2) or (x0 == 0 and bx2 >= TILE - 2) or (y0 > 0 and by1 <= 2) or (y0 == 0 and by2 >= TILE - 2)):
                        continue
                    sub_bm = bm[by1:by2, bx1:bx2]
                    best_q, best_ov = -1, -1.0
                    for q in range(cq_np.shape[1]):
                        if cq_np[ti, q] < 0.15:
                            continue
                        pm_q = mq_np[ti, q, by1:by2, bx1:bx2] > 0.45
                        ov = float((pm_q & sub_bm).sum()) / float((pm_q | sub_bm).sum() + 1e-6)
                        if ov > best_ov:
                            best_ov, best_q = ov, q
                    soft_tile = mq_np[ti, best_q] if best_q >= 0 else bm.astype(np.float32)
                    pad_b = 12
                    lx1, ly1 = max(0, bx1 - pad_b), max(0, by1 - pad_b)
                    lx2, ly2 = min(TILE, bx2 + pad_b), min(TILE, by2 + pad_b)
                    if lx2 <= lx1 or ly2 <= ly1:
                        continue
                    cx1, cy1 = max(0, x0 + lx1), max(0, y0 + ly1)
                    cx2, cy2 = min(FULL, x0 + lx2), min(FULL, y0 + ly2)
                    lx1, ly1 = cx1 - x0, cy1 - y0
                    lx2, ly2 = cx2 - x0, cy2 - y0
                    sub_p = np.clip(soft_tile[ly1:ly2, lx1:lx2] * 255.0 + 0.5, 0, 255).astype(np.uint8)
                    if sub_p.shape != (cy2 - cy1, cx2 - cx1):
                        continue
                    raw_insts.append((sub_p, sc, (cx1, cy1, cx2, cy2), m_idx))

        mean_dense_tiles = np.mean(per_model_cq, axis=0)
        for ti, (y0, x0) in enumerate(TILE_COORDS):
            dense_acc[y0 : y0 + TILE, x0 : x0 + TILE] += mean_dense_tiles[ti]
            dense_cnt[y0 : y0 + TILE, x0 : x0 + TILE] += 1.0

    dense_u8 = np.clip((dense_acc / np.maximum(dense_cnt, 1.0)) * 255.0 + 0.5, 0, 255).astype(np.uint8)

    # Cluster overlapping cross-fold/cross-tile instances via weighted mask consensus instead of keeping only fold-0
    raw_insts.sort(key=lambda t: t[1], reverse=True)
    clusters: list[dict] = []
    n_models = max(1, len(models))
    for sub_p, sc, (x1, y1, x2, y2), m_idx in raw_insts:
        m = sub_p >= 128
        if int(m.sum()) < min_area:
            continue
        matched_cl = None
        for cl in clusters:
            kx1, ky1, kx2, ky2 = cl["box"]
            km = cl["mask"]
            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            if ix2 > ix1 and iy2 > iy1:
                inter = int((m[iy1 - y1 : iy2 - y1, ix1 - x1 : ix2 - x1] & km[iy1 - ky1 : iy2 - ky1, ix1 - kx1 : ix2 - kx1]).sum())
                if inter / float(min(int(m.sum()), int(km.sum())) + 1e-6) >= 0.38:
                    matched_cl = cl
                    break
        if matched_cl is None:
            if len(clusters) >= 85:
                continue
            clusters.append({
                "box": (x1, y1, x2, y2),
                "mask": m,
                "acc": sub_p.astype(np.float32) * sc,
                "wsum": sc,
                "scores": [sc],
                "models": {m_idx},
            })
        elif m_idx not in matched_cl["models"]:
            kx1, ky1, kx2, ky2 = matched_cl["box"]
            ox1, oy1 = max(x1, kx1), max(y1, ky1)
            ox2, oy2 = min(x2, kx2), min(y2, ky2)
            if ox2 > ox1 and oy2 > oy1:
                matched_cl["acc"][oy1 - ky1 : oy2 - ky1, ox1 - kx1 : ox2 - kx1] += (
                    sub_p[oy1 - y1 : oy2 - y1, ox1 - x1 : ox2 - x1].astype(np.float32) * sc
                )
                matched_cl["wsum"] += sc
                matched_cl["scores"].append(sc)
                matched_cl["models"].add(m_idx)

    kept_ws, kept_confs, kept_probs = [], [], []
    for cl in clusters:
        mean_p = np.clip(cl["acc"] / max(cl["wsum"], 1e-6), 0.0, 255.0).astype(np.uint8)
        base_sc = float(np.mean(cl["scores"]))
        if n_models > 1:
            # Penalize single-fold FPs while rewarding multi-fold consensus
            consensus = min(1.0, 0.68 + 0.08 * len(cl["models"]))
            base_sc *= consensus
        kept_ws.append(cl["box"])
        kept_confs.append(base_sc)
        kept_probs.append(mean_p)
    return dense_u8, np.asarray(kept_confs, dtype=np.float32), kept_ws, kept_probs


def worker_infer(repo_dir: Path, data_root: Path, work_dir: Path, scenario: str, out_dir: Path, part: int) -> None:
    import torch
    from transformers import Mask2FormerImageProcessor

    data, postprocess, _ = _bind_filseg(repo_dir, data_root, work_dir)
    import solar_dl_enhance as sde  # noqa: E402

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and (torch.cuda.get_device_capability(0)[0] >= 8) and torch.cuda.is_bf16_supported()
    folds_map = data.folds()
    models = {k: _build_m2f_model(ckpt_dir=str(work_dir / "weights" / f"m2f_{scenario}_f{k}")).to(dev).eval() for k in range(5)}
    proc = Mask2FormerImageProcessor(do_resize=False, do_rescale=False, do_normalize=False)

    inst_dir = out_dir / "m2f_inst"
    dense_dir = out_dir / "m2f_probs"
    (dense_dir / "oof").mkdir(parents=True, exist_ok=True)
    (dense_dir / "test").mkdir(parents=True, exist_ok=True)

    shared_phys = Path("/tmp/shared_phys")
    for sp, stems in (("oof", sorted(folds_map)[part::2]), ("test", sorted(data.test_stems())[part::2])):
        for s in stems:
            sp_png = shared_phys / f"{s}.png"
            bgr = cv2.imread(str(sp_png)) if sp_png.exists() else None
            if bgr is None:
                im = cv2.imread(str(data.image_path(s)), cv2.IMREAD_GRAYSCALE)
                bgr = sde.physics_3ch_from_gray(im)
            ml = [models[folds_map[s]]] if sp == "oof" else [models[k] for k in range(5)]
            dense_u8, conf, ws, probs = _infer_single_photo(ml, proc, bgr, dev, use_bf16)
            postprocess.save_cache(inst_dir / sp / f"{s}.npz", conf, ws, probs)
            np.savez_compressed(dense_dir / sp / f"{s}.npz", prob=dense_u8)
    print(f"[Mask2Former infer p{part}] Complete", flush=True)


def run_driver(scenario: str, epochs: int, patience: int = 5) -> None:
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

    work_dir = Path(f"/tmp/exp013b_m2f_{scenario}")
    out_dir = Path(f"/tmp/exp013b_out_{scenario}")
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    common_env = {**os.environ, "PYTHONPATH": f"{repo_dir}:{Path(__file__).resolve().parent}"}
    py = sys.executable

    for fk in range(5):
        rc = subprocess.call(
            [py, __file__, "--worker", "train_fold", "--repo", str(repo_dir), "--data-root", str(data_root),
             "--work-dir", str(work_dir), "--scenario", scenario, "--fold", str(fk),
             "--epochs", str(epochs), "--patience", str(patience)],
            env=common_env,
        )
        assert rc == 0, f"Mask2Former fold {fk} failed"

    n_gpu = torch.cuda.device_count()
    if n_gpu >= 2:
        procs = [
            subprocess.Popen(
                [py, __file__, "--worker", "infer", "--repo", str(repo_dir), "--data-root", str(data_root),
                 "--work-dir", str(work_dir), "--scenario", scenario, "--out-dir", str(out_dir), "--part", str(p)],
                env=dict(common_env, CUDA_VISIBLE_DEVICES=str(p)),
            )
            for p in (0, 1)
        ]
        assert all(p.wait() == 0 for p in procs), "Mask2Former infer failed"
    else:
        for p in (0, 1):
            rc = subprocess.call(
                [py, __file__, "--worker", "infer", "--repo", str(repo_dir), "--data-root", str(data_root),
                 "--work-dir", str(work_dir), "--scenario", scenario, "--out-dir", str(out_dir), "--part", str(p)],
                env=dict(common_env, CUDA_VISIBLE_DEVICES="0"),
            )
            assert rc == 0, f"Mask2Former infer part {p} failed"

    for sub in ("m2f_inst", "m2f_probs"):
        tgz = Path(f"/kaggle/working/exp013b_{scenario}_{sub}.tar.gz")
        with tarfile.open(tgz, "w:gz") as tf:
            tf.add(out_dir / sub, arcname=sub)
        print(f"Packed {tgz.name} ({tgz.stat().st_size/1e6:.2f} MB)", flush=True)
    print(f"EXP-013b {scenario} COMPLETE in {(time.time()-t0)/60:.1f}m", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["train_fold", "infer"])
    ap.add_argument("--repo", type=Path)
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--work-dir", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--scenario", default="baseline", choices=["baseline", "dl_enh_copypaste"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()

    if args.worker is None:
        run_driver(args.scenario, args.epochs, args.patience)
    elif args.worker == "train_fold":
        worker_train_fold(args.repo, args.data_root, args.work_dir, args.scenario, args.fold, args.epochs, args.patience)
    elif args.worker == "infer":
        worker_infer(args.repo, args.data_root, args.work_dir, args.scenario, args.out_dir, args.part)


if __name__ == "__main__":
    main()
