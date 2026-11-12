"""Stage 2 -- what is the shape of the filament inside a box? A U-Net++ / EfficientNet-B4 on 512-px crops.

Each sample is one annotated filament: the box plus 20% padding, resized to 512 x 512, with *that* filament's mask
as the target (neighbouring filaments in the crop are background). The network therefore learns to outline the
object a box points at, which is what keeps touching filaments apart. Boxes are jittered during training to mimic
the detector's localisation error.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import data, enhance

FULL = data.FULL


def crop_box(x, y, w, h, pad=0.2, jitter=0.0, rng=None):
    """xywh box -> padded integer crop window (x1, y1, x2, y2) clipped to the frame."""
    if jitter:
        x += rng.uniform(-jitter, jitter) * w
        y += rng.uniform(-jitter, jitter) * h
        w *= 1 + rng.uniform(-jitter, jitter)
        h *= 1 + rng.uniform(-jitter, jitter)
    px, py = w * pad, h * pad
    x1, y1, x2, y2 = max(0, int(x - px)), max(0, int(y - py)), min(FULL, int(x + w + px)), min(FULL, int(y + h + py))
    return (0, 0, FULL, FULL) if (x2 <= x1 + 1 or y2 <= y1 + 1) else (x1, y1, x2, y2)


def crop_input(planes, b, size):
    """uint8 planes (C, H, W) + crop window -> float32 (C, size, size) in [0, 255]."""
    return np.stack([cv2.resize(np.ascontiguousarray(p[b[1]:b[3], b[0]:b[2]]), (size, size)) for p in planes]).astype(np.float32)


def partial_box(mask, bbox, share, rng):
    """A box around only part of a filament, as a detector produces when it sees the dark core but not the faint
    continuation: the annotation's pixels inside a window of `share` of the box (per axis) around a random pixel
    of the filament. -> xywh"""
    x, y, w, h = (int(round(v)) for v in bbox)
    sub = mask[y:y + h, x:x + w]
    ys, xs = np.nonzero(sub)
    if len(ys) < 20:
        return bbox
    i = int(rng.integers(len(ys)))
    hw, hh = max(4, share[0] * w / 2), max(4, share[1] * h / 2)
    keep = (np.abs(xs - xs[i]) <= hw) & (np.abs(ys - ys[i]) <= hh)
    return (x + xs[keep].min(), y + ys[keep].min(), xs[keep].max() - xs[keep].min() + 1, ys[keep].max() - ys[keep].min() + 1)


class CropDataset:
    """One sample per annotated filament of every annotator record.
    trunc > 0 (the "x" segmenter): half of the samples get a box around only part of the filament (each axis keeps
    a share in [trunc, 1]) and every sample a padding drawn from [pad, pad_hi]; the target stays the whole
    annotation, so the network learns to follow a filament beyond a box that covers only its core."""

    def __init__(self, stems, size=512, pad=0.2, jitter=0.0, train=False, seed=0, trunc=0.0, pad_hi=None):
        recs, anns = data.coco_index()
        self.items = [(s, a) for s in stems for rid in recs.get(s, []) for a in anns.get(rid, [])]
        self.size, self.pad, self.jitter, self.train, self.seed = size, pad, jitter, train, seed
        self.trunc, self.pad_hi = trunc, pad_hi
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torch
        s, a = self.items[i]
        full = data.ann_mask(a)
        bbox, pad = a["bbox"], self.pad
        if self.trunc:                                       # validation draws are fixed per sample
            r = self.rng if self.train else np.random.default_rng(i)
            if r.random() < 0.5:
                bbox = partial_box(full, bbox, r.uniform(self.trunc, 1, 2), r)
            pad = r.uniform(self.pad, self.pad_hi or self.pad)
        b = crop_box(*bbox, pad, self.jitter if self.train else 0.0, self.rng)
        c = crop_input(enhance.planes(s), b, self.size)
        m = cv2.resize(full[b[1]:b[3], b[0]:b[2]], (self.size, self.size), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        if self.train:                                       # flips, quarter turns, brightness / contrast
            r = self.rng
            if r.random() < 0.5:
                c, m = c[:, :, ::-1], m[:, ::-1]
            if r.random() < 0.5:
                c, m = c[:, ::-1], m[::-1]
            if r.random() < 0.5:
                k = int(r.integers(1, 4))
                c, m = np.rot90(c, k, axes=(1, 2)), np.rot90(m, k)
            if r.random() < 0.5:
                c = np.clip(c * (1 + r.uniform(-0.5, 0.5)) + r.uniform(-0.5, 0.5) * 255, 0, 255)
        c = (c / 255.0 - 0.5) / 0.5
        return torch.from_numpy(np.ascontiguousarray(c, dtype=np.float32)), torch.from_numpy(np.ascontiguousarray(m)[None])


def _reseed(worker_id):
    """DataLoader workers would otherwise replay the same augmentation stream."""
    import torch
    ds = torch.utils.data.get_worker_info().dataset
    ds.rng = np.random.default_rng([ds.seed, worker_id])


def build_model(encoder="efficientnet-b4", pretrained=True, in_channels=len(enhance.PLANES)):
    import segmentation_models_pytorch as smp
    return smp.UnetPlusPlus(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                            in_channels=in_channels, classes=1, activation=None)


def loss_fn(logit, t):
    """0.5 BCE + 0.5 Tversky(alpha 0.3, beta 0.7): missing filament pixels costs more than spilling over."""
    import torch.nn.functional as F
    p = logit.float().sigmoid()
    tp, fp, fn = (p * t).sum(), (p * (1 - t)).sum(), ((1 - p) * t).sum()
    tversky = (tp + 1e-6) / (tp + 0.3 * fp + 0.7 * fn + 1e-6)
    return 0.5 * F.binary_cross_entropy_with_logits(logit.float(), t) + 0.5 * (1 - tversky)


def train(fold, name, epochs=15, batch=8, size=512, pad=0.2, jitter=0.12, lr=2e-4, device="cuda", workers=8,
          encoder="efficientnet-b4", init=None, trunc=0.0, pad_hi=None):
    """Train on the four other folds; keep the epoch with the best crop Dice on fold `fold` -> WORK/weights.
    init: name of an already trained segmenter of the same fold to start from (fine-tuning)."""
    import torch
    tr, va = data.split(fold)
    for s in tr + va:                                        # enhanced planes are computed once per photo
        enhance.planes(s)
    dl = lambda ds, shuffle: torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=shuffle, num_workers=workers, persistent_workers=workers > 0,
        worker_init_fn=_reseed if workers else None, drop_last=shuffle)
    tl = dl(CropDataset(tr, size, pad, jitter, True, seed=fold, trunc=trunc, pad_hi=pad_hi), True)
    vl = dl(CropDataset(va, size, pad, trunc=trunc, pad_hi=pad_hi), False)
    dev = torch.device(device)
    model = load(data.WEIGHTS / f"seg_{init}_f{fold}.pth", "cpu", encoder).train().to(dev) if init else build_model(encoder).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    out = data.WEIGHTS / f"seg_{name}_f{fold}.pth"
    out.parent.mkdir(parents=True, exist_ok=True)
    best, hist, t0 = -1.0, [], time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot = n = 0
        for x, t in tl:
            x, t = x.to(dev, non_blocking=True), t.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                logit = model(x)
            loss = loss_fn(logit, t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot, n = tot + loss.item() * len(x), n + len(x)
        sched.step()
        model.eval()
        inter = psum = tsum = 0.0
        with torch.no_grad():
            for x, t in vl:
                with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                    p = (model(x.to(dev)).float().sigmoid() > 0.5).float()
                t = t.to(dev)
                inter += float((p * t).sum()); psum += float(p.sum()); tsum += float(t.sum())
        dice = 2 * inter / max(psum + tsum, 1)
        hist.append(dict(epoch=ep, train_loss=round(tot / max(n, 1), 5), val_crop_dice=round(dice, 5),
                         minutes=round((time.time() - t0) / 60, 1)))
        print(f"seg {name} fold {fold}", hist[-1], flush=True)
        if dice > best:
            best = dice
            torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, out)
        json.dump(hist, open(out.with_suffix(".history.json"), "w"), indent=0)
    return out


def load(weights, device="cuda", encoder="efficientnet-b4"):
    import torch
    m = build_model(encoder, pretrained=False)
    m.load_state_dict({k: v.float() if v.is_floating_point() else v
                       for k, v in torch.load(weights, map_location="cpu").items()})
    return m.to(device).eval()


def predict(models, planes, windows, size=512, batch=32, tta=True, device="cuda"):
    """Probability of "the filament this box points at" for each crop window: mean over models and (tta) the four
    flips, resized back to the window. -> list of uint8 maps (0-255), one per window, at window size."""
    import torch
    dev = torch.device(device)
    out = []
    for i in range(0, len(windows), batch):
        ws = windows[i:i + batch]
        x = torch.from_numpy(np.stack([(crop_input(planes, b, size) / 255.0 - 0.5) / 0.5 for b in ws])).to(dev)
        acc, n = 0, 0
        with torch.no_grad():
            for m in models:
                for k in range(4 if tta else 1):
                    xx = x.flip(3) if k in (1, 3) else x
                    xx = xx.flip(2) if k in (2, 3) else xx
                    with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                        y = m(xx).float().sigmoid()
                    y = y.flip(2) if k in (2, 3) else y
                    acc = acc + (y.flip(3) if k in (1, 3) else y)
                    n += 1
        for p, b in zip((acc / n).cpu().numpy()[:, 0], ws):
            out.append((cv2.resize(p, (b[2] - b[0], b[3] - b[1])) * 255 + 0.5).astype(np.uint8))
    return out


def grow_windows(windows, probs, thr=0.5, grow=0.5, edge=2, pad=0.2):
    """A mask that touches the border of its crop belongs to a filament that continues outside (the box covered only
    part of it). Such a window is rebuilt around the mask found so far, enlarged by `grow` of the mask's extent on
    every touched side, then padded as usual. -> (new windows, indices of the windows that changed)."""
    out, changed = list(windows), []
    for i, ((x1, y1, x2, y2), p) in enumerate(zip(windows, probs)):
        m = p > int(thr * 255)
        if not m.any():
            continue
        left, right = m[:, :edge].any() and x1 > 0, m[:, -edge:].any() and x2 < FULL
        top, bottom = m[:edge].any() and y1 > 0, m[-edge:].any() and y2 < FULL
        if not (left or right or top or bottom):
            continue
        ys, xs = np.nonzero(m)
        bx1, bx2, by1, by2 = x1 + xs.min(), x1 + xs.max() + 1, y1 + ys.min(), y1 + ys.max() + 1
        w, h = bx2 - bx1, by2 - by1
        bx1, bx2 = bx1 - grow * w * left, bx2 + grow * w * right
        by1, by2 = by1 - grow * h * top, by2 + grow * h * bottom
        new = crop_box(bx1, by1, bx2 - bx1, by2 - by1, pad)
        if new != (x1, y1, x2, y2):
            out[i] = new
            changed.append(i)
    return out, changed
