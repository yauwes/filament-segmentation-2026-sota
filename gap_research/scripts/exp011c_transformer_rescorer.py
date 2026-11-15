"""EXP-011c: Multimodal Physics ViT + Relational Set-Transformer + GBDT Hurdle Ensemble.

Trained strictly out-of-fold across all 707 OOF stems (date-grouped 5-fold CV) on the clamped
14-stream symmetric candidate pool (`exp011b_pool_clamped.pkl` + 4x48x48 physical/mask patches):

1. Branch A (`VisionCropTransformer`, Apple M1 MPS GPU):
   - Hybrid early-conv + 4-layer Pre-LN Transformer Encoder reading 4x48x48 aspect-preserving
     solar physics & mask patches [flat_z, bth_z, pmap, mask] conditioned on tabular scale/context
     with D4 dihedral equivariant training augmentation and 4-view inference TTA.
2. Branch B (`RelationalSetTransformer`, Graphormer w/ Edge Attention Bias, MPS GPU):
   - 4-layer Set Transformer attending across all candidates within the same solar disk photo
     using pairwise geometric/IoU/containment edge biases + [DISK_CLS] context + pairwise ranking
     loss on overlapping candidate variants (`ta_base` vs `split` vs `cons62` vs `cons68`).
3. Branch C (Calibrated GBDT Hurdle: LGBM + XGB + CatBoostClassifier/Regressor, batched predict).

Uses precomputed fold statistics + exact inner-fold `filseg.metric` re-ranking inside Nested 5-Fold
Date-CV (ablating GBDT-only, ViT-only, Relational-only, and joint Transformer+GBDT blend, plus
optional solo-candidate gating and fine tau_keep/anchor_bonus).
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "gap_research" / "scripts"))

import eval_wave23_joint as ej  # noqa: E402
import exp011b_hurdle_antichain as e11b  # noqa: E402
import test_shift_audit as tsa  # noqa: E402
from filseg import data, metric, submission  # noqa: E402

FULL = 2048
KIND_LIST = ("ta_base", "ta_rescue", "cons68", "cons62", "cons74", "split", "solo", "merge")
KIND_TO_IDX = {k: i for i, k in enumerate(KIND_LIST)}


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _fit_transform_tab(X_tr: np.ndarray, X_list: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Leak-free robust train-fold standardization with clipping."""
    med = np.median(X_tr, axis=0)
    std = np.std(X_tr, axis=0)
    std = np.where(std < 1e-4, 1.0, std)
    z_tr = np.clip((X_tr - med) / std, -8.0, 8.0).astype(np.float32)
    z_out = [np.clip((x - med) / std, -8.0, 8.0).astype(np.float32) if len(x) else np.zeros((0, X_tr.shape[1]), dtype=np.float32) for x in X_list]
    return z_tr, z_out


def _node_features(entry: dict, x_norm: np.ndarray) -> np.ndarray:
    n = len(entry["X"])
    k_oh = np.zeros((n, len(KIND_LIST)), dtype=np.float32)
    for i, k in enumerate(entry["kinds"]):
        if k in KIND_TO_IDX:
            k_oh[i, KIND_TO_IDX[k]] = 1.0
    return np.hstack([x_norm, k_oh]).astype(np.float32)


def _stem_edge_tensor(entry: dict) -> np.ndarray:
    """Build (N, N, 9) pairwise geometric/IoU/containment matrix for Relational Attention."""
    n = len(entry["X"])
    E = np.zeros((n, n, 9), dtype=np.float32)
    if n == 0:
        return E
    boxes = entry["boxes"].astype(np.float32)
    cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
    cy = 0.5 * (boxes[:, 1] + boxes[:, 3])
    log_area = np.log1p(entry["areas"].astype(np.float32))
    max_side = np.log1p(np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]))
    conf = entry["X"][:, 0].astype(np.float32)

    dx = (cx[None, :] - cx[:, None]) / 2048.0
    dy = (cy[None, :] - cy[:, None]) / 2048.0
    E[:, :, 3] = dx
    E[:, :, 4] = dy
    E[:, :, 5] = np.hypot(dx, dy)
    E[:, :, 6] = log_area[None, :] - log_area[:, None]
    E[:, :, 7] = max_side[None, :] - max_side[:, None]
    E[:, :, 8] = conf[None, :] - conf[:, None]
    for e in entry["edges"]:
        i, j = int(e[0]), int(e[1])
        ifrac, iou_ij = (float(e[2]), float(e[3])) if len(e) >= 4 else (1.0, 0.5)
        E[i, j, 0] = iou_ij
        E[j, i, 0] = iou_ij
        E[i, j, 1] = ifrac
        E[j, i, 1] = ifrac
        same = 1.0 if (iou_ij >= 0.15 or ifrac >= 0.30) else 0.0
        E[i, j, 2] = same
        E[j, i, 2] = same
    return E


# ---------------------------------------------------------------------------
# Branch A: VisionCropTransformer (4x48x48 Physics + Probability ViT)
# ---------------------------------------------------------------------------

class VisionCropTransformer(nn.Module):
    """Hybrid Conv-Stem + Pre-LN Vision Transformer conditioned on tabular context."""

    def __init__(self, tab_dim: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 3, drop: float = 0.15):
        super().__init__()
        # 48x48 -> 24x24 -> 6x6 (36 overlapping patch tokens for 4x faster MPS throughput)
        self.stem = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, d_model, kernel_size=3, stride=4, padding=1),
            nn.GELU(),
        )
        self.cls_tok = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_patch = nn.Parameter(torch.randn(1, 36, d_model) * 0.02)
        self.tab_proj = nn.Sequential(
            nn.Linear(tab_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d_model, d_model),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(d_model * 2.5),
            dropout=drop,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.out_h = nn.Linear(d_model * 2, 1)
        self.out_m = nn.Linear(d_model * 2, 1)
        self.out_u = nn.Linear(d_model * 2, 1)

    def forward(self, img_u8: torch.Tensor, tab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # img_u8: (B, 4, 48, 48) float in [0, 1]
        x = img_u8 * 2.0 - 1.0
        p = self.stem(x).flatten(2).transpose(1, 2) + self.pos_patch  # (B, 36, D)
        mask_w = F.avg_pool2d(img_u8[:, 3:4], kernel_size=8, stride=8).flatten(1)  # (B, 36)
        mask_w = mask_w / mask_w.sum(dim=1, keepdim=True).clamp_min(1e-4)

        b = img_u8.size(0)
        cls_t = self.cls_tok.expand(b, -1, -1)
        tab_t = self.tab_proj(tab).unsqueeze(1)
        seq = torch.cat([cls_t, tab_t, p], dim=1)  # (B, 66, D)
        z = self.norm(self.encoder(seq))

        z_cls = z[:, 0]
        z_tab = z[:, 1]
        z_pat = z[:, 2:]
        z_mask = (z_pat * mask_w.unsqueeze(-1)).sum(dim=1)
        z_avg = z_pat.mean(dim=1)

        feat = self.head(torch.cat([z_cls, z_tab, z_mask, z_avg], dim=-1))
        lh = self.out_h(feat).squeeze(-1)
        pm = 0.50 + 0.50 * torch.sigmoid(self.out_m(feat).squeeze(-1))
        pu = torch.sigmoid(self.out_u(feat).squeeze(-1))
        return lh, pm, pu


def _apply_d4_batch(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    k = int(rng.integers(4))
    if k:
        x = torch.rot90(x, k, dims=(-2, -1))
    if rng.random() < 0.5:
        x = torch.flip(x, dims=(-1,))
    return x


def _fit_predict_vit_fold(
    patches_oof: np.ndarray,
    patches_test: np.ndarray,
    oof: dict,
    test: dict,
    tr_stems: list[str],
    va_stems: list[str],
    test_stems: list[str],
    device: torch.device,
    epochs: int = 24,
    seeds: tuple[int, ...] = (0, 42),
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    X_tr_raw = np.vstack([oof[s]["X"] for s in tr_stems])
    med = np.median(X_tr_raw, axis=0)
    std = np.where(np.std(X_tr_raw, axis=0) < 1e-4, 1.0, np.std(X_tr_raw, axis=0))

    def _norm_entry(ent: dict) -> np.ndarray:
        if len(ent["X"]) == 0:
            return np.zeros((0, ent["X"].shape[1] + len(KIND_LIST)), dtype=np.float32)
        xn = np.clip((ent["X"] - med) / std, -8.0, 8.0).astype(np.float32)
        return _node_features(ent, xn)

    tr_tabs = np.vstack([_norm_entry(oof[s]) for s in tr_stems])
    tr_imgs = np.concatenate([patches_oof[oof[s]["patch_slice"][0] : oof[s]["patch_slice"][1]] for s in tr_stems], axis=0)
    tr_h = np.concatenate([oof[s]["h"] for s in tr_stems]).astype(np.float32)
    tr_m = np.concatenate([oof[s]["m"] for s in tr_stems]).astype(np.float32)
    tr_u = np.concatenate([oof[s]["u"] for s in tr_stems]).astype(np.float32)
    tr_w = np.concatenate([np.full(len(oof[s]["X"]), float(oof[s]["r_count"]), dtype=np.float32) for s in tr_stems])

    tab_dim = tr_tabs.shape[1]
    n_tr = len(tr_tabs)
    batch_sz = 512

    tr_tab_t = torch.from_numpy(tr_tabs).to(device)
    tr_img_u8 = torch.from_numpy(tr_imgs).to(device)
    tr_h_t = torch.from_numpy(tr_h).to(device)
    tr_m_t = torch.from_numpy(tr_m).to(device)
    tr_u_t = torch.from_numpy(tr_u).to(device)
    tr_w_t = torch.from_numpy(tr_w).to(device)

    def _infer_split(models: list[VisionCropTransformer], data_dict: dict, patch_arr: np.ndarray, stems: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        all_tabs, all_imgs, slices = [], [], []
        off = 0
        for s in stems:
            n = len(data_dict[s]["X"])
            slices.append((s, off, off + n))
            off += n
            if n > 0:
                all_tabs.append(_norm_entry(data_dict[s]))
                a, b = data_dict[s]["patch_slice"]
                all_imgs.append(patch_arr[a:b])
        if off == 0:
            z = np.zeros(0, dtype=np.float32)
            return {s: (z, z, z) for s in stems}
        cat_tab = torch.from_numpy(np.vstack(all_tabs))
        cat_img = torch.from_numpy(np.concatenate(all_imgs, axis=0))

        ph_acc = np.zeros(off, dtype=np.float32)
        pm_acc = np.zeros(off, dtype=np.float32)
        pu_acc = np.zeros(off, dtype=np.float32)
        for model in models:
            model.eval()
            with torch.no_grad():
                for st in range(0, off, 512):
                    en = min(off, st + 512)
                    xb = cat_tab[st:en].to(device)
                    ib = cat_img[st:en].to(device=device, dtype=torch.float32) / 255.0
                    views = [
                        ib,
                        torch.flip(ib, dims=(-1,)),
                        torch.flip(ib, dims=(-2,)),
                        torch.rot90(ib, 1, dims=(-2, -1)),
                    ]
                    vh, vm, vu = 0.0, 0.0, 0.0
                    for v_ib in views:
                        lh, pm_v, pu_v = model(v_ib, xb)
                        vh = vh + torch.sigmoid(lh)
                        vm = vm + pm_v
                        vu = vu + pu_v
                    ph_acc[st:en] += (vh / 4.0).cpu().numpy()
                    pm_acc[st:en] += (vm / 4.0).cpu().numpy()
                    pu_acc[st:en] += (vu / 4.0).cpu().numpy()
        inv = 1.0 / len(models)
        ph_acc *= inv
        pm_acc *= inv
        pu_acc *= inv
        return {s: (ph_acc[a:b].copy(), pm_acc[a:b].copy(), pu_acc[a:b].copy()) for s, a, b in slices}

    fitted_models = []
    for sd in seeds:
        torch.manual_seed(sd)
        rng = np.random.default_rng(sd)
        model = VisionCropTransformer(tab_dim=tab_dim, d_model=128, n_heads=4, n_layers=3, drop=0.15).to(device)
        swa_state = None
        swa_n = 0
        opt = torch.optim.AdamW(model.parameters(), lr=1.1e-3, weight_decay=2.5e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=2e-5)

        for ep in range(epochs):
            model.train()
            perm = torch.from_numpy(rng.permutation(n_tr)).to(device)
            for st in range(0, n_tr, batch_sz):
                idx = perm[st : st + batch_sz]
                ib = _apply_d4_batch(tr_img_u8[idx].float() / 255.0, rng)
                xb = tr_tab_t[idx]
                yh = tr_h_t[idx]
                ym = tr_m_t[idx]
                yu = tr_u_t[idx]
                wr = tr_w_t[idx]

                opt.zero_grad(set_to_none=True)
                lh, pm, pu = model(ib, xb)
                loss_h = (wr * F.binary_cross_entropy_with_logits(lh, yh, reduction="none")).mean()
                pos_w = (yh > 0).float() * yh * wr
                denom = pos_w.sum().clamp_min(1.0)
                loss_m = (pos_w * F.smooth_l1_loss(pm, ym, beta=0.05, reduction="none")).sum() / denom
                loss_u = (wr * F.smooth_l1_loss(pu, yu, beta=0.05, reduction="none")).mean()
                loss = loss_h + 0.65 * loss_m + 0.85 * loss_u
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            sched.step()
            if ep >= epochs - 6:
                cur = {k: v.detach().clone() for k, v in model.state_dict().items()}
                if swa_state is None:
                    swa_state = cur
                else:
                    for k in swa_state:
                        swa_state[k] += cur[k]
                swa_n += 1
        if swa_state is not None and swa_n > 0:
            for k in swa_state:
                swa_state[k] /= float(swa_n)
            model.load_state_dict(swa_state)
        fitted_models.append(model)

    va_preds = _infer_split(fitted_models, oof, patches_oof, va_stems)
    te_preds = _infer_split(fitted_models, test, patches_test, test_stems)
    va_h = {s: va_preds[s][0] for s in va_stems}
    va_m = {s: va_preds[s][1] for s in va_stems}
    va_u = {s: va_preds[s][2] for s in va_stems}
    return va_h, va_m, va_u, te_preds


# ---------------------------------------------------------------------------
# Branch B: RelationalSetTransformer (Graphormer w/ Pairwise Edge Attention Bias)
# ---------------------------------------------------------------------------

class RelationalAttnBlock(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, edge_dim: int = 9, drop: float = 0.12):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.edge_bias = nn.Sequential(
            nn.Linear(edge_dim, n_heads * 2),
            nn.GELU(),
            nn.Linear(n_heads * 2, n_heads),
        )
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * 2.5)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(d_model * 2.5), d_model),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, edges: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) where index 0 is [DISK_CLS] and 1..L-1 are candidates
        # edges: (B, L-1, L-1, edge_dim), valid_mask: (B, L) bool
        b, l, _ = x.shape
        h_in = self.norm1(x)
        qkv = self.qkv(h_in).view(b, l, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, L, d_h)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, H, L, L)
        if l > 1:
            eb = self.edge_bias(edges).permute(0, 3, 1, 2)  # (B, H, L-1, L-1)
            attn[:, :, 1:, 1:] = attn[:, :, 1:, 1:] + eb
        attn = attn.masked_fill(~valid_mask[:, None, None, :], -1e9)
        w = torch.softmax(attn, dim=-1)
        w = self.drop(w)
        out = (w @ v).transpose(1, 2).contiguous().view(b, l, self.d_model)
        x = x + self.drop(self.proj(out))
        x = x + self.mlp(self.norm2(x))
        return x


class RelationalSetTransformer(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int = 9, d_model: int = 128, n_heads: int = 4, n_layers: int = 4, drop: float = 0.12):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.disk_cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList([RelationalAttnBlock(d_model, n_heads, edge_dim, drop) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.out_h = nn.Linear(d_model * 2, 1)
        self.out_m = nn.Linear(d_model * 2, 1)
        self.out_u = nn.Linear(d_model * 2, 1)

    def forward(self, x_nodes: torch.Tensor, edges: torch.Tensor, valid_nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_nodes: (B, N, in_dim), edges: (B, N, N, edge_dim), valid_nodes: (B, N) bool
        b, n, _ = x_nodes.shape
        h = self.in_proj(x_nodes)
        cls_t = self.disk_cls.expand(b, -1, -1)
        seq = torch.cat([cls_t, h], dim=1)  # (B, 1+N, D)
        v_all = torch.cat([torch.ones((b, 1), dtype=torch.bool, device=valid_nodes.device), valid_nodes], dim=1)
        for blk in self.blocks:
            seq = blk(seq, edges, v_all)
        seq = self.norm(seq)
        cls_ctx = seq[:, 0:1].expand(-1, n, -1)
        cand_z = seq[:, 1:]
        feat = self.head(torch.cat([cand_z, cls_ctx], dim=-1))
        lh = self.out_h(feat).squeeze(-1)
        pm = 0.50 + 0.50 * torch.sigmoid(self.out_m(feat).squeeze(-1))
        pu = torch.sigmoid(self.out_u(feat).squeeze(-1))
        return lh, pm, pu


def _fit_predict_rel_fold(
    oof: dict,
    test: dict,
    edge_oof: dict[str, np.ndarray],
    edge_test: dict[str, np.ndarray],
    tr_stems: list[str],
    va_stems: list[str],
    test_stems: list[str],
    device: torch.device,
    epochs: int = 38,
    seeds: tuple[int, ...] = (0, 42),
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    X_tr_raw = np.vstack([oof[s]["X"] for s in tr_stems])
    med = np.median(X_tr_raw, axis=0)
    std = np.where(np.std(X_tr_raw, axis=0) < 1e-4, 1.0, np.std(X_tr_raw, axis=0))

    def _pack_stems(data_dict: dict, edge_dict: dict, stems: list[str], train_flag: bool):
        items = []
        for s in stems:
            ent = data_dict[s]
            n = len(ent["X"])
            if n == 0:
                continue
            xn = np.clip((ent["X"] - med) / std, -8.0, 8.0).astype(np.float32)
            nf = _node_features(ent, xn)
            items.append((s, nf, edge_dict[s], ent["h"], ent["m"], ent["u"], float(ent["r_count"]) if train_flag else 1.0))
        return items

    tr_items = _pack_stems(oof, edge_oof, tr_stems, True)
    in_dim = tr_items[0][1].shape[1]

    def _collate(batch_items):
        b = len(batch_items)
        nmax = max(it[1].shape[0] for it in batch_items)
        xb = np.zeros((b, nmax, in_dim), dtype=np.float32)
        eb = np.zeros((b, nmax, nmax, 9), dtype=np.float32)
        vb = np.zeros((b, nmax), dtype=bool)
        yh = np.zeros((b, nmax), dtype=np.float32)
        ym = np.zeros((b, nmax), dtype=np.float32)
        yu = np.zeros((b, nmax), dtype=np.float32)
        wr = np.zeros((b, nmax), dtype=np.float32)
        for bi, (_, nf, ed, h, m, u, rc) in enumerate(batch_items):
            n = len(nf)
            xb[bi, :n] = nf
            eb[bi, :n, :n] = ed
            vb[bi, :n] = True
            if len(h):
                yh[bi, :n] = h
                ym[bi, :n] = m
                yu[bi, :n] = u
                wr[bi, :n] = rc
        return (
            torch.from_numpy(xb).to(device),
            torch.from_numpy(eb).to(device),
            torch.from_numpy(vb).to(device),
            torch.from_numpy(yh).to(device),
            torch.from_numpy(ym).to(device),
            torch.from_numpy(yu).to(device),
            torch.from_numpy(wr).to(device),
        )

    def _infer_rel(models: list[RelationalSetTransformer], data_dict: dict, edge_dict: dict, stems: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        out = {s: (np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)) for s in stems}
        items = _pack_stems(data_dict, edge_dict, stems, False)
        if not items:
            return out
        for st in range(0, len(items), 32):
            sub = items[st : st + 32]
            xb, eb, vb, *_ = _collate(sub)
            ph, pm, pu = 0.0, 0.0, 0.0
            for model in models:
                model.eval()
                with torch.no_grad():
                    lh, pm_v, pu_v = model(xb, eb, vb)
                    ph = ph + torch.sigmoid(lh)
                    pm = pm + pm_v
                    pu = pu + pu_v
            ph_np = (ph / len(models)).cpu().numpy()
            pm_np = (pm / len(models)).cpu().numpy()
            pu_np = (pu / len(models)).cpu().numpy()
            for bi, (s, nf, *_) in enumerate(sub):
                n = len(nf)
                out[s] = (ph_np[bi, :n].copy(), pm_np[bi, :n].copy(), pu_np[bi, :n].copy())
        return out

    fitted = []
    batch_sz = 32
    tr_xb, tr_eb, tr_vb, tr_yh, tr_ym, tr_yu, tr_wr = _collate(tr_items)
    for sd in seeds:
        torch.manual_seed(sd)
        rng = np.random.default_rng(sd)
        model = RelationalSetTransformer(in_dim=in_dim, edge_dim=9, d_model=128, n_heads=4, n_layers=4, drop=0.12).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=2.5e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=2e-5)
        swa_state = None
        swa_n = 0

        for ep in range(epochs):
            model.train()
            perm = torch.from_numpy(rng.permutation(len(tr_items))).to(device)
            for st in range(0, len(tr_items), batch_sz):
                idx = perm[st : st + batch_sz]
                xb, eb, vb = tr_xb[idx], tr_eb[idx], tr_vb[idx]
                yh, ym, yu, wr = tr_yh[idx], tr_ym[idx], tr_yu[idx], tr_wr[idx]
                opt.zero_grad(set_to_none=True)
                lh, pm, pu = model(xb, eb, vb)

                vw = vb.float() * wr
                vdenom = vw.sum().clamp_min(1.0)
                loss_h = (vw * F.binary_cross_entropy_with_logits(lh, yh, reduction="none")).sum() / vdenom
                pos_w = vw * (yh > 0).float() * yh
                loss_m = (pos_w * F.smooth_l1_loss(pm, ym, beta=0.05, reduction="none")).sum() / pos_w.sum().clamp_min(1.0)
                loss_u = (vw * F.smooth_l1_loss(pu, yu, beta=0.05, reduction="none")).sum() / vdenom

                # Overlapping candidate pairwise utility ranking loss (teaches split/cons vs ta_base)
                ov_pair = (eb[:, :, :, 2] > 0.5) & vb[:, :, None] & vb[:, None, :]
                du_true = yu[:, :, None] - yu[:, None, :]
                rank_mask = ov_pair & (du_true > 0.08)
                if rank_mask.any():
                    du_pred = pu[:, :, None] - pu[:, None, :]
                    pair_w = (du_true.clamp_max(0.5)) * rank_mask.float()
                    loss_rank = (pair_w * F.softplus(-du_pred * 6.0)).sum() / pair_w.sum().clamp_min(1e-4)
                else:
                    loss_rank = 0.0

                loss = loss_h + 0.65 * loss_m + 0.85 * loss_u + 0.30 * loss_rank
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            sched.step()
            if ep >= epochs - 8:
                cur = {k: v.detach().clone() for k, v in model.state_dict().items()}
                if swa_state is None:
                    swa_state = cur
                else:
                    for k in swa_state:
                        swa_state[k] += cur[k]
                swa_n += 1
        if swa_state is not None and swa_n > 0:
            for k in swa_state:
                swa_state[k] /= float(swa_n)
            model.load_state_dict(swa_state)
        fitted.append(model)

    va_preds = _infer_rel(fitted, oof, edge_oof, va_stems)
    te_preds = _infer_rel(fitted, test, edge_test, test_stems)
    return (
        {s: va_preds[s][0] for s in va_stems},
        {s: va_preds[s][1] for s in va_stems},
        {s: va_preds[s][2] for s in va_stems},
        te_preds,
    )


# ---------------------------------------------------------------------------
# Fast GBDT (zero OpenMP thrashing + batched predict)
# ---------------------------------------------------------------------------

def _fit_fold_gbdt_fast(X_tr: np.ndarray, y_h: np.ndarray, y_m: np.ndarray, y_u: np.ndarray, w_r: np.ndarray, seeds: tuple[int, ...] = (0, 42)) -> dict:
    pos = y_h > 0
    X_pos = X_tr[pos]
    y_m_pos = y_m[pos]
    w_m_pos = (y_h[pos] * w_r[pos]).astype(np.float32)

    mh_list, mm_list, mu_list = [], [], []
    for sd in seeds:
        mh_lgb = LGBMRegressor(
            objective="cross_entropy", n_estimators=420, learning_rate=0.03, num_leaves=31,
            min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.75,
            reg_alpha=1.0, reg_lambda=3.0, n_jobs=1, random_state=sd, verbose=-1,
        ).fit(X_tr, y_h, sample_weight=w_r)
        mh_xgb = XGBRegressor(
            objective="reg:logistic", n_estimators=420, learning_rate=0.03, max_depth=5,
            min_child_weight=15, subsample=0.8, colsample_bytree=0.75, reg_alpha=1.0, reg_lambda=3.0,
            tree_method="hist", n_jobs=1, random_state=sd,
        ).fit(X_tr, y_h, sample_weight=w_r, verbose=False)
        mh_cat = CatBoostClassifier(
            loss_function="CrossEntropy", iterations=550, learning_rate=0.035, depth=6,
            l2_leaf_reg=5.0, thread_count=2, random_seed=sd, verbose=0,
        ).fit(X_tr, y_h, sample_weight=w_r)
        mh_list.extend([("lgb", mh_lgb), ("xgb", mh_xgb), ("cat", mh_cat)])

        mm_lgb = LGBMRegressor(
            objective="regression", n_estimators=360, learning_rate=0.03, num_leaves=23,
            min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.75,
            reg_alpha=0.5, reg_lambda=3.0, n_jobs=1, random_state=sd + 100, verbose=-1,
        ).fit(X_pos, y_m_pos, sample_weight=w_m_pos)
        mm_xgb = XGBRegressor(
            objective="reg:squarederror", n_estimators=360, learning_rate=0.03, max_depth=4,
            min_child_weight=12, subsample=0.8, colsample_bytree=0.75, reg_alpha=0.5, reg_lambda=3.0,
            tree_method="hist", n_jobs=1, random_state=sd + 100,
        ).fit(X_pos, y_m_pos, sample_weight=w_m_pos, verbose=False)
        mm_cat = CatBoostRegressor(
            loss_function="RMSE", iterations=480, learning_rate=0.035, depth=5,
            l2_leaf_reg=5.0, thread_count=2, random_seed=sd + 100, verbose=0,
        ).fit(X_pos, y_m_pos, sample_weight=w_m_pos)
        mm_list.extend([("lgb", mm_lgb), ("xgb", mm_xgb), ("cat", mm_cat)])

        mu_lgb = LGBMRegressor(
            objective="regression", n_estimators=420, learning_rate=0.03, num_leaves=31,
            min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.75,
            reg_alpha=1.0, reg_lambda=3.0, n_jobs=1, random_state=sd + 200, verbose=-1,
        ).fit(X_tr, y_u, sample_weight=w_r)
        mu_xgb = XGBRegressor(
            objective="reg:squarederror", n_estimators=420, learning_rate=0.03, max_depth=5,
            min_child_weight=15, subsample=0.8, colsample_bytree=0.75, reg_alpha=1.0, reg_lambda=3.0,
            tree_method="hist", n_jobs=1, random_state=sd + 200,
        ).fit(X_tr, y_u, sample_weight=w_r, verbose=False)
        mu_list.extend([("lgb", mu_lgb), ("xgb", mu_xgb)])

    return {"mh": mh_list, "mm": mm_list, "mu": mu_list}


def _predict_gbdt_batch(models: dict, data_dict: dict, stems: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    Xs, slices = [], []
    off = 0
    for s in stems:
        x = data_dict[s]["X"]
        slices.append((s, off, off + len(x)))
        off += len(x)
        if len(x):
            Xs.append(x)
    if off == 0:
        z = np.zeros(0, dtype=np.float32)
        return {s: (z, z, z) for s in stems}
    X_all = np.vstack(Xs)
    ph = np.mean([np.clip(m.predict_proba(X_all)[:, 1] if k == "cat" else m.predict(X_all), 0.0, 1.0) for k, m in models["mh"]], axis=0).astype(np.float32)
    pm = np.mean([np.clip(m.predict(X_all), 0.50, 1.0) for _, m in models["mm"]], axis=0).astype(np.float32)
    pu = np.mean([np.clip(m.predict(X_all), 0.0, 1.0) for _, m in models["mu"]], axis=0).astype(np.float32)
    return {s: (ph[a:b].copy(), pm[a:b].copy(), pu[a:b].copy()) for s, a, b in slices}


# ---------------------------------------------------------------------------
# Fast Policy Weights & Precomputed Per-Fold MWIS Search
# ---------------------------------------------------------------------------

def _node_weights_policy(entry: dict, u_score: np.ndarray, params: tuple[float, float, float, float, bool]) -> np.ndarray:
    tau_keep, tau_add, anchor_bonus, tau_base, allow_solo = params
    kinds = entry["kinds"]
    n = len(kinds)
    w = np.full(n, -1.0, dtype=np.float32)
    if n == 0:
        return w
    iou_ta = entry["X"][:, 61]
    for i in range(n):
        ui = float(u_score[i])
        k = kinds[i]
        if k in ("ta_base", "ta_rescue"):
            if ui >= tau_keep:
                val = (ui - tau_base) + anchor_bonus
                if tau_keep <= 0.0:
                    val = max(1e-4 + 1e-3 * ui, val)
                w[i] = val
        elif iou_ta[i] >= 0.25:
            if anchor_bonus < 5.0 and ui >= tau_base:
                w[i] = ui - tau_base
        else:
            if k == "solo" and not allow_solo:
                continue
            if ui >= tau_add:
                w[i] = max(1e-4 + 1e-3 * ui, ui - tau_base)
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=Path("/private/tmp/exp011b_pool_clamped.pkl"))
    ap.add_argument("--out-csv", type=Path, default=REPO / "gap_research" / "submissions" / "exp011c_transformer_clean.csv")
    ap.add_argument("--vit-epochs", type=int, default=14)
    ap.add_argument("--rel-epochs", type=int, default=24)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    device = _get_device()
    print(f"EXP-011c Multimodal Vision+Relational Transformer + GBDT | device={device}", flush=True)

    pool = pickle.load(open(args.pool, "rb"))
    oof = pool["oof"]
    test = pool["test"]
    patches_oof = np.load(pool["patches_oof_path"], mmap_mode="r")
    patches_test = np.load(pool["patches_test_path"], mmap_mode="r")

    folds = data.folds()
    all_stems = sorted(oof)
    test_stems = sorted(test)
    f0_stems = [s for s in all_stems if folds[s] == 0]
    gt_recs = metric.load_records(all_stems)

    adj_oof = {s: e11b._build_adj(oof[s]) for s in all_stems}
    adj_test = {s: e11b._build_adj(test[s]) for s in test_stems}
    edge_oof = {s: _stem_edge_tensor(oof[s]) for s in all_stems}
    edge_test = {s: _stem_edge_tensor(test[s]) for s in test_stems}

    # Sanity check Track-A inside clamped pool
    ta_u_dummy = {s: np.array([1.0 if k in ("ta_base", "ta_rescue") else 0.0 for k in oof[s]["kinds"]], dtype=np.float32) for s in all_stems}
    ta_params = {s: (0.5, 9.9, 9.9, 0.24) for s in all_stems}
    sm_ta_pool, _, _ = e11b._eval_exact_stems(oof, adj_oof, all_stems, gt_recs, ta_u_dummy, ta_params)
    print(f"[Sanity Track-A inside clamped pool]: PQ={sm_ta_pool['pq']:.5f} SQ={sm_ta_pool['sq']:.5f} RQ={sm_ta_pool['rq']:.5f} TP={sm_ta_pool['tp']} FP={sm_ta_pool['fp']} FN={sm_ta_pool['fn']}", flush=True)

    ta_cache = pickle.load(open("/private/tmp/tracka_clean_oof_cache.pkl", "rb"))
    print(f"[Track-A Clean Baseline Cache]:      PQ={ta_cache['sm_all']['pq']:.5f} | Fold-0={ta_cache['sm_f0']['pq']:.5f}", flush=True)

    if args.smoke:
        vit_ep, rel_ep, seeds = 3, 4, (0,)
    else:
        vit_ep, rel_ep, seeds = args.vit_epochs, args.rel_epochs, (0, 42)

    preds = {k: {"oof_h": {}, "oof_m": {}, "oof_u": {}, "test_h": [], "test_m": [], "test_u": []} for k in ("gbdt", "vit", "rel")}

    for fk in range(5):
        tr_s = [s for s in all_stems if folds[s] != fk and len(oof[s]["X"]) > 0]
        va_s = [s for s in all_stems if folds[s] == fk]
        if not tr_s or not va_s:
            continue

        # Branch C: Fast GBDT
        tf0 = time.time()
        X_tr = np.vstack([oof[s]["X"] for s in tr_s])
        y_h = np.concatenate([oof[s]["h"] for s in tr_s])
        y_m = np.concatenate([oof[s]["m"] for s in tr_s])
        y_u = np.concatenate([oof[s]["u"] for s in tr_s])
        w_r = np.concatenate([np.full(len(oof[s]["X"]), float(oof[s]["r_count"]), dtype=np.float32) for s in tr_s])
        md_gb = _fit_fold_gbdt_fast(X_tr, y_h, y_m, y_u, w_r, seeds=seeds)
        va_gb = _predict_gbdt_batch(md_gb, oof, va_s)
        te_gb = _predict_gbdt_batch(md_gb, test, test_stems)
        for s in va_s:
            preds["gbdt"]["oof_h"][s], preds["gbdt"]["oof_m"][s], preds["gbdt"]["oof_u"][s] = va_gb[s]
        preds["gbdt"]["test_h"].append({s: te_gb[s][0] for s in test_stems})
        preds["gbdt"]["test_m"].append({s: te_gb[s][1] for s in test_stems})
        preds["gbdt"]["test_u"].append({s: te_gb[s][2] for s in test_stems})
        t_gb = time.time() - tf0
        print(f"  Fold {fk} gbdt={t_gb:.1f}s...", end=" ", flush=True)

        # Branch A: Multimodal Physics ViT (MPS)
        tf1 = time.time()
        vh, vm, vu, te_vit = _fit_predict_vit_fold(
            patches_oof, patches_test, oof, test, tr_s, va_s, test_stems, device=device, epochs=vit_ep, seeds=seeds
        )
        preds["vit"]["oof_h"].update(vh)
        preds["vit"]["oof_m"].update(vm)
        preds["vit"]["oof_u"].update(vu)
        preds["vit"]["test_h"].append({s: te_vit[s][0] for s in test_stems})
        preds["vit"]["test_m"].append({s: te_vit[s][1] for s in test_stems})
        preds["vit"]["test_u"].append({s: te_vit[s][2] for s in test_stems})
        t_vit = time.time() - tf1
        print(f"vit={t_vit:.1f}s...", end=" ", flush=True)

        # Branch B: Relational Set-Transformer (Graphormer w/ Edge Bias, MPS)
        tf2 = time.time()
        rh, rm, ru, te_rel = _fit_predict_rel_fold(
            oof, test, edge_oof, edge_test, tr_s, va_s, test_stems, device=device, epochs=rel_ep, seeds=seeds
        )
        preds["rel"]["oof_h"].update(rh)
        preds["rel"]["oof_m"].update(rm)
        preds["rel"]["oof_u"].update(ru)
        preds["rel"]["test_h"].append({s: te_rel[s][0] for s in test_stems})
        preds["rel"]["test_m"].append({s: te_rel[s][1] for s in test_stems})
        preds["rel"]["test_u"].append({s: te_rel[s][2] for s in test_stems})
        t_rel = time.time() - tf2
        if device.type == "mps":
            torch.mps.empty_cache()

        print(f"rel={t_rel:.1f}s | elapsed={time.time()-t0:.1f}s", flush=True)

    if args.smoke:
        print("SMOKE TEST COMPLETE OK", flush=True)
        return

    # Average 5-fold Test predictions per branch
    test_branch = {}
    for br in ("gbdt", "vit", "rel"):
        test_branch[br] = {
            s: (
                np.mean([d[s] for d in preds[br]["test_h"]], axis=0).astype(np.float32),
                np.mean([d[s] for d in preds[br]["test_m"]], axis=0).astype(np.float32),
                np.mean([d[s] for d in preds[br]["test_u"]], axis=0).astype(np.float32),
            )
            for s in test_stems
        }

    # Print per-branch standalone OOF correlations
    all_ut = np.concatenate([oof[s]["u"] for s in all_stems if len(oof[s]["X"])])
    all_ht = np.concatenate([oof[s]["h"] for s in all_stems if len(oof[s]["X"])])
    for br in ("gbdt", "vit", "rel"):
        ph = np.concatenate([preds[br]["oof_h"][s] for s in all_stems if len(oof[s]["X"])])
        pm = np.concatenate([preds[br]["oof_m"][s] for s in all_stems if len(oof[s]["X"])])
        pu = np.concatenate([preds[br]["oof_u"][s] for s in all_stems if len(oof[s]["X"])])
        uh = ph * pm
        print(f"  Branch {br:4s}: true_h={all_ht.mean():.4f} pred_h={ph.mean():.4f} | corr(h*m)={np.corrcoef(all_ut, uh)[0,1]:.4f} corr(dir)={np.corrcoef(all_ut, pu)[0,1]:.4f}", flush=True)

    # Save raw branch predictions for instant downstream inspection/ablations
    with open("/private/tmp/exp011c_branch_preds.pkl", "wb") as f:
        pickle.dump({"oof": preds, "test": test_branch}, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 2. Precompute fold-wise soft PQ stats + exact Top-K Nested 5-Fold Selection
    blend_weights = [
        (1.00, 0.00, 0.00),  # GBDT only
        (0.00, 1.00, 0.00),  # ViT only
        (0.00, 0.00, 1.00),  # RelSet only
        (0.00, 0.50, 0.50),  # Pure Transformer (ViT + RelSet, zero GBDT)
        (0.65, 0.20, 0.15),  # Hybrid 1
        (0.50, 0.25, 0.25),  # Hybrid 2
        (0.40, 0.30, 0.30),  # Hybrid 3
    ]
    alphas = (0.60, 0.80)
    grid = [
        (tk, ta_v, ab, tb, solo_ok)
        for tk in (0.00, 0.06, 0.10, 0.14)
        for ta_v in (0.25, 0.28, 0.31)
        for ab in (0.02, 0.05, 0.09, 9.9)
        for tb in (0.24,)
        for solo_ok in (False, True)
    ]
    stems_by_fold = {fk: [s for s in all_stems if folds[s] == fk] for fk in range(5)}
    gt_by_fold = {fk: sum(float(sum(len(gr) for gr in gt_recs[s])) for s in stems_by_fold[fk]) for fk in range(5)}

    def _make_u_oof(bw: tuple[float, float, float], alp: float) -> dict[str, np.ndarray]:
        wg, wv, wr = bw
        umap = {}
        for s in all_stems:
            uh = wg * (preds["gbdt"]["oof_h"][s] * preds["gbdt"]["oof_m"][s]) + wv * (preds["vit"]["oof_h"][s] * preds["vit"]["oof_m"][s]) + wr * (preds["rel"]["oof_h"][s] * preds["rel"]["oof_m"][s])
            ud = wg * preds["gbdt"]["oof_u"][s] + wv * preds["vit"]["oof_u"][s] + wr * preds["rel"]["oof_u"][s]
            umap[s] = alp * uh + (1.0 - alp) * ud
        return umap

    def _eval_exact_custom(stems: list[str], umap: dict[str, np.ndarray], params_map: dict[str, tuple]) -> tuple[dict, dict[str, dict], dict[str, list[str]]]:
        rows, rles_out = {}, {}
        for s in stems:
            w = _node_weights_policy(oof[s], umap[s], params_map[s])
            sel = e11b._solve_mwis_exact(w, adj_oof[s])
            rstrs = e11b._decode_rles_disjoint(oof[s], sel, w)
            rles_out[s] = rstrs
            pr = [{"size": [FULL, FULL], "counts": r.encode("ascii")} for r in rstrs]
            rows[s] = metric.counts(pr, gt_recs[s])
        return metric.summarize(list(rows.values())), rows, rles_out

    # Step 2a: Precompute per-fold soft stats ONCE across the whole grid (4x fewer calls!)
    u_cache = {(bw, alp): _make_u_oof(bw, alp) for bw in blend_weights for alp in alphas}
    fold_stats = {}  # (fk, bw, alp, p) -> (tp_a, fp_a, iou_a)
    for bw in blend_weights:
        for alp in alphas:
            umap = u_cache[(bw, alp)]
            for p in grid:
                for fk in range(5):
                    tp_a, fp_a, iou_a = 0.0, 0.0, 0.0
                    for s in stems_by_fold[fk]:
                        ent = oof[s]
                        if len(ent["u"]) == 0:
                            continue
                        w = _node_weights_policy(ent, umap[s], p)
                        sel = e11b._solve_mwis_exact(w, adj_oof[s])
                        rc = float(ent["r_count"])
                        for idx in sel:
                            tp_a += rc * float(ent["h"][idx])
                            fp_a += rc * (1.0 - float(ent["h"][idx]))
                            iou_a += rc * float(ent["u"][idx])
                    fold_stats[(fk, bw, alp, p)] = (tp_a, fp_a, iou_a)
    print(f"Precomputed fold soft stats in elapsed={time.time()-t0:.1f}s", flush=True)

    # Step 2b: Inner-fold Top-12 exact filseg.metric ranking (zero leakage, exact metric!)
    def _rank_nested(allowed_bws: list[tuple[float, float, float]], label: str) -> tuple[dict, dict, dict, list]:
        exact_stem_cache = {}
        u_n, p_n, chosen = {}, {}, []
        for fk in range(5):
            inner_gt = sum(gt_by_fold[k] for k in range(5) if k != fk)
            inner_stems = [s for s in all_stems if folds[s] != fk]
            cands_in = []
            for bw in allowed_bws:
                for alp in alphas:
                    for p in grid:
                        tp = sum(fold_stats[(k, bw, alp, p)][0] for k in range(5) if k != fk)
                        fp = sum(fold_stats[(k, bw, alp, p)][1] for k in range(5) if k != fk)
                        iou = sum(fold_stats[(k, bw, alp, p)][2] for k in range(5) if k != fk)
                        fn = inner_gt - tp
                        den = tp + 0.5 * (fp + fn)
                        cands_in.append((iou / den if den > 0 else 0.0, bw, alp, p))
            cands_in.sort(reverse=True, key=lambda x: x[0])

            best_exact, best_cfg = -1.0, (allowed_bws[0], alphas[0], grid[0])
            for _, bw, alp, p in cands_in[:10]:
                key = (bw, alp, p)
                if key not in exact_stem_cache:
                    _, rmap, _ = _eval_exact_custom(all_stems, u_cache[(bw, alp)], {s: p for s in all_stems})
                    exact_stem_cache[key] = rmap
                pq_in = metric.summarize([exact_stem_cache[key][s] for s in inner_stems])["pq"]
                if pq_in > best_exact:
                    best_exact, best_cfg = pq_in, key
            chosen.append(best_cfg)
            for s in stems_by_fold[fk]:
                u_n[s] = u_cache[(best_cfg[0], best_cfg[1])][s]
                p_n[s] = best_cfg[2]
        sm, rows, rles = _eval_exact_custom(all_stems, u_n, p_n)
        sm_f0 = metric.summarize([rows[s] for s in f0_stems])
        boot = e11b._paired_date_bootstrap(rows, ta_cache["counts"], all_stems, n_boot=1000, seed=42)
        print(f"[{label:22s}] OOF PQ={sm['pq']:.5f} SQ={sm['sq']:.5f} RQ={sm['rq']:.5f} TP={sm['tp']} FP={sm['fp']} FN={sm['fn']} | F0={sm_f0['pq']:.5f} | boot={boot['mean_gain']:+.5f} P={boot['p_win']:.4f}", flush=True)
        return sm, rows, rles, chosen

    print("\n--- Strictly Nested 5-Fold Architecture Ablations ---", flush=True)
    _rank_nested([(1.0, 0.0, 0.0)], "Nested GBDT-Only")
    _rank_nested([(0.0, 1.0, 0.0)], "Nested ViT-Only")
    _rank_nested([(0.0, 0.0, 1.0)], "Nested RelSet-Only")
    _rank_nested([(0.0, 0.5, 0.5)], "Nested Pure-Transf")
    sm_all, rows_all, rles_all, chosen_all = _rank_nested(blend_weights, "Nested Joint-All")

    print("Joint Nested chosen configs per fold:", chosen_all, flush=True)
    sim80_m, sim80_s = e11b._sim_r80(all_stems, rles_all, gt_recs, trials=10)
    bins_all = ej.bin_pq_summary(rles_all, all_stems, gt_recs)
    print(f"[Sim_r80_PQ (10x 1-ann)]: {sim80_m:.5f} +/- {sim80_s:.5f}", flush=True)
    print("Joint Area Bins:", bins_all, flush=True)

    # Consensus Test export from joint nested fold choices
    cons_bw = tuple(float(np.mean([c[0][i] for c in chosen_all])) for i in range(3))
    s_bw = sum(cons_bw)
    cons_bw = tuple(v / s_bw for v in cons_bw)
    cons_alp = float(np.median([c[1] for c in chosen_all]))
    cons_p = (
        float(np.median([c[2][0] for c in chosen_all])),
        float(np.median([c[2][1] for c in chosen_all])),
        float(np.median([c[2][2] for c in chosen_all])),
        float(np.median([c[2][3] for c in chosen_all])),
        bool(np.mean([1.0 if c[2][4] else 0.0 for c in chosen_all]) >= 0.5),
    )
    print(f"Consensus Test blend={tuple(round(x,3) for x in cons_bw)}, alpha={cons_alp}, params={cons_p}", flush=True)

    rles_test_csv = {}
    for s in test_stems:
        if len(test[s]["X"]) == 0:
            rles_test_csv[s] = []
            continue
        wg, wv, wr = cons_bw
        uh = wg * (test_branch["gbdt"][s][0] * test_branch["gbdt"][s][1]) + wv * (test_branch["vit"][s][0] * test_branch["vit"][s][1]) + wr * (test_branch["rel"][s][0] * test_branch["rel"][s][1])
        ud = wg * test_branch["gbdt"][s][2] + wv * test_branch["vit"][s][2] + wr * test_branch["rel"][s][2]
        u_t = cons_alp * uh + (1.0 - cons_alp) * ud
        wt = _node_weights_policy(test[s], u_t, cons_p)
        sel = e11b._solve_mwis_exact(wt, adj_test[s])
        rstrs = e11b._decode_rles_disjoint(test[s], sel, wt)
        rles_test_csv[s] = [{"size": [FULL, FULL], "counts": r.encode("ascii")} for r in rstrs]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.write_csv(args.out_csv, rles_test_csv)
    probs = submission.validate(args.out_csv, data.test_stems())
    assert probs == [], f"Validation failed: {probs}"
    audit = tsa.audit_csv(args.out_csv, REPO / "gap_research" / "submissions" / "exp006_007_tracka_clean.csv")
    print(f"Saved {args.out_csv} (validate=[] OK), Test shift audit:\n{json.dumps(audit, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
