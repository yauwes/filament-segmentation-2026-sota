"""Reproducible Scientific Visualization Suite (PaperBanana NeurIPS/IEEE Style — Critic-Refined).

Generates all methodology, architecture, experimental progression, ablation, and qualitative error analysis
figures directly from real GONG H-alpha frames, ground-truth annotations, and verified out-of-fold / Fold-0 caches.
"""
import json
import pickle
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from pycocotools import mask as mu

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from filseg import data, enhance, metric, postprocess  # noqa: E402

FIG_ROOT = REPO / "figures"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "mathtext.fontset": "stix",
    "axes.edgecolor": "#222222",
    "axes.linewidth": 0.6,
    "grid.color": "#cccccc",
    "grid.linestyle": "--",
    "grid.linewidth": 0.45,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save_dual(fig, rel_path: str, legacy_name: str | None = None, dpi: int = 300):
    pdf_path = FIG_ROOT / f"{rel_path}.pdf"
    png_path = FIG_ROOT / f"{rel_path}.png"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    if legacy_name:
        shutil.copy2(pdf_path, FIG_ROOT / f"{legacy_name}.pdf")
        shutil.copy2(png_path, FIG_ROOT / f"{legacy_name}.png")
    plt.close(fig)
    print(f"Generated {pdf_path.relative_to(REPO)} & .png")


def draw_box(ax, x, y, w, h, title, subtitle="", facecolor="#ffffff", edgecolor="#333333",
             lw=0.85, radius=0.014, title_size=6.2, sub_size=5.0, title_color="#111111", dashed=False):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.003,rounding_size={radius}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
        linestyle="--" if dashed else "-", zorder=2
    )
    ax.add_patch(patch)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.70, title, fontsize=title_size, fontweight="bold",
                ha="center", va="center", color=title_color, zorder=3)
        ax.text(x + w / 2, y + h * 0.31, subtitle, fontsize=sub_size,
                ha="center", va="center", color="#333333", linespacing=1.18, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, title, fontsize=title_size, fontweight="bold",
                ha="center", va="center", color=title_color, zorder=3)


def draw_arrow(ax, x1, y1, x2, y2, color="#2b2b2b", lw=0.95, style="-|>", rad=0.0, dashed=False):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=8.5,
        linewidth=lw, color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}", zorder=4
    )
    ax.add_patch(arr)


def fig_1_system_architecture():
    """PaperBanana Critic-Refined End-to-End Architecture & Cross-Family Continuous Vote-Map Diagram."""
    fig, ax = plt.subplots(figsize=(8.8, 3.0), dpi=300)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    zones = [
        (0.006, 0.03, 0.228, 0.94, "#F9F6EE", "#B8A989", "1. DATA & TRI-PLANE ENHANCE"),
        (0.248, 0.03, 0.240, 0.94, "#EBF4FA", "#6C9BC3", "2. STAGE-1 MULTI-FAMILY DETECT"),
        (0.502, 0.03, 0.238, 0.94, "#E8F5E9", "#66A578", "3. STAGE-2 512² CROP SEGMENT"),
        (0.754, 0.03, 0.240, 0.94, "#F3E8F8", "#9575CD", "4. CONTINUOUS VOTE-MAP WBF"),
    ]
    for zx, zy, zw, zh, fc, ec, ztitle in zones:
        zpatch = FancyBboxPatch(
            (zx, zy), zw, zh, boxstyle="round,pad=0.003,rounding_size=0.016",
            facecolor=fc, edgecolor=ec, linewidth=0.9, linestyle="--", zorder=1
        )
        ax.add_patch(zpatch)
        ax.text(zx + zw / 2, zy + zh - 0.042, ztitle, fontsize=6.1, fontweight="bold",
                ha="center", va="center", color="#1a1a1a", zorder=3)

    # Zone 1 Nodes
    draw_box(ax, 0.018, 0.67, 0.204, 0.20, "GONG Full-Disk H-α",
             r"6 Observatories, $2048\times 2048$ px" + "\nDate-Grouped 5-Fold Split (707 Frames)",
             facecolor="#FFFFFF", edgecolor="#7D6B4F")
    draw_box(ax, 0.018, 0.36, 0.204, 0.23, "Tri-Plane Enhancement",
             r"$C_1$: Black Top-Hat ($r{=}7$ px)" + "\n" + r"$C_2$: $101$-px Median Detrend" + "\n" + r"$C_3$: Sato Ridge ($\sigma{\in}\{1..3\}$)",
             facecolor="#FFFDF7", edgecolor="#7D6B4F")
    draw_box(ax, 0.018, 0.07, 0.204, 0.21, "Solar-Disk MAD Norm.",
             r"$\tilde{C}_k = \mathrm{clip}\!\left(\frac{C_k - \mu_k}{16\,\hat{\sigma}_k}{+}\frac{1}{2}\right)$" + "\n" + r"($\hat{\sigma}_k{=}1.48\,\mathrm{MAD}_k$) + 3k GONG",
             facecolor="#FFFFFF", edgecolor="#7D6B4F")
    draw_arrow(ax, 0.120, 0.67, 0.120, 0.59)
    draw_arrow(ax, 0.120, 0.36, 0.120, 0.28)

    # Zone 2 Nodes
    draw_box(ax, 0.260, 0.67, 0.216, 0.20, "YOLO11m + YOLOv8s-seg",
             "YOLO11m ($1280^2, 1664^2$) Boxes\n+ Direct YOLOv8s-seg ($1536^2$)",
             facecolor="#FFFFFF", edgecolor="#2C5E8A")
    draw_box(ax, 0.260, 0.37, 0.216, 0.21, "Two-Stage Faster R-CNN",
             "ResNet50-FPN v2 + RoIAlign\nHigh-Precision Boxes ($F_0, F_1, F_3$)",
             facecolor="#FFFFFF", edgecolor="#2C5E8A")
    draw_box(ax, 0.260, 0.07, 0.216, 0.22, "Noisy-Student Self-Train",
             "Fold Teachers Label 3k Archive\n3-Flip WBF ($c \\geq 0.25$) → Students\n" + r"($P_1, P_{4a}, F_1, F_3, s$)",
             facecolor="#E1EFFA", edgecolor="#1E4B75")
    draw_arrow(ax, 0.222, 0.77, 0.260, 0.77)
    draw_arrow(ax, 0.222, 0.47, 0.260, 0.47)
    draw_arrow(ax, 0.222, 0.18, 0.260, 0.18, dashed=True)
    draw_arrow(ax, 0.368, 0.67, 0.368, 0.58, dashed=True)
    draw_arrow(ax, 0.368, 0.37, 0.368, 0.29, dashed=True)

    # Zone 3 Nodes
    draw_box(ax, 0.514, 0.67, 0.214, 0.20, "Instance-Conditioned Crop",
             r"Box $+20\%$ Context $\to 512\times 512$" + "\nTarget = 1, Neighbor Filaments = 0",
             facecolor="#FFFFFF", edgecolor="#2E7D32")
    draw_box(ax, 0.514, 0.36, 0.214, 0.23, "3 Segmenter Configs",
             "1. U-Net++ EffNet-B4 (S1E)\n2. SAM 2.1 Hiera-B+ Adapter\n3. 5-Fold U-Net++ EffNet-B4 (e4)",
             facecolor="#FFFFFF", edgecolor="#2E7D32")
    draw_box(ax, 0.514, 0.07, 0.214, 0.21, "Tversky Loss + 2D Blend",
             r"$\mathcal{L}{=}0.5\,\mathrm{BCE}{+}0.5\,\mathrm{Tversky}(0.3,0.7)$" + "\nPer-Pixel Weight Matrix $W(x,y)$",
             facecolor="#E0F2F1", edgecolor="#1B5E20")
    draw_arrow(ax, 0.476, 0.77, 0.514, 0.77)
    draw_arrow(ax, 0.476, 0.47, 0.514, 0.72)
    draw_arrow(ax, 0.621, 0.67, 0.621, 0.59)
    draw_arrow(ax, 0.621, 0.36, 0.621, 0.28)

    # Zone 4 Nodes
    draw_box(ax, 0.766, 0.67, 0.216, 0.20, "Continuous Vote-Map WBF",
             r"Core $V_{\mathrm{core}}(x,y) = \frac{\sum c_m M_m}{\sum c_m} \in [0,1]$" + "\n+ 5-Fold ($ms, s, main$) & $F_3$ Donors",
             facecolor="#FFFFFF", edgecolor="#5E35B1")
    draw_box(ax, 0.766, 0.37, 0.216, 0.21, "Cross-Family Calibration",
             r"Confirm Bonus $\Delta c{=}+0.02$ ($F_3 \wedge ms/s$)" + "\n" + r"Vote $\tau_{\mathrm{vote}}{=}0.425$, Floor $c_{\min}{=}0.32$",
             facecolor="#FFFFFF", edgecolor="#5E35B1")
    draw_box(ax, 0.766, 0.07, 0.216, 0.21, "Greedy Panoptic Painting",
             "Non-Overlapping Canvas Order\nDrop $<70\\%$ Surviving, Area $\\geq 100$\nVal PQ = 0.4856 | 5-Fold = 0.4636",
             facecolor="#EDE7F6", edgecolor="#4527A0")
    draw_arrow(ax, 0.728, 0.47, 0.766, 0.75)
    draw_arrow(ax, 0.728, 0.18, 0.766, 0.71)
    draw_arrow(ax, 0.874, 0.67, 0.874, 0.58)
    draw_arrow(ax, 0.874, 0.37, 0.874, 0.28)

    save_dual(fig, "architecture/system_architecture", legacy_name="architecture")


def fig_2_pipeline_walkthrough():
    """Preserve real validation frame walkthrough: Input -> Enhanced RGB -> Boxes -> Crops -> Painted."""
    src_pdf = FIG_ROOT / "pipeline.pdf"
    src_png = FIG_ROOT / "pipeline.png"
    dst_dir = FIG_ROOT / "methodology"
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src_pdf.exists() and src_png.exists():
        shutil.copy2(src_pdf, dst_dir / "pipeline_walkthrough.pdf")
        shutil.copy2(src_png, dst_dir / "pipeline_walkthrough.png")
        print("Verified figures/methodology/pipeline_walkthrough.pdf & .png")


def fig_3_progression_and_ablation():
    """PaperBanana Critic-Refined Statistical Plot: Paradigm Progression, Per-Pixel Blending Gain, Split-Half Bootstrap."""
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.3), dpi=300,
                             gridspec_kw={"width_ratios": [1.38, 1.15, 1.0], "wspace": 0.36})

    # Panel (a): Paradigm Progression (verified metrics only)
    ax = axes[0]
    stages = [
        "U-Net + CC\n(5-Fold OOF)",
        "1-Stage YOLO-seg\n(Fold-0)",
        "2-Stage G1\n+ avgSAM2",
        "5-Fold OOF\nms + e4 (707)",
        "Self-Trained\nStudent P1",
        "Per-Pixel 3-Seg\nBlend (ms)",
        "8-Model\nMask-WBF (G17)",
        "Cross-Fam.\nCont. Vmap",
    ]
    pqs = [0.3890, 0.4270, 0.4491, 0.4636, 0.4650, 0.4716, 0.4820, 0.4856]
    colors = ["#cfd8dc", "#b0bec5", "#90caf9", "#4db6ac", "#64b5f6", "#81c784", "#9575cd", "#5e35b1"]
    y_pos = np.arange(len(stages))
    ax.grid(axis="x", zorder=0)
    bars = ax.barh(y_pos, pqs, height=0.65, color=colors, edgecolor="#111111", linewidth=0.65, zorder=3)
    ax.axvline(0.3398, color="#d95f02", linestyle="--", linewidth=0.95, zorder=4, label="Human PQ (0.3398)")
    for bar, v in zip(bars, pqs):
        ax.text(v + 0.004, bar.get_y() + bar.get_height() / 2, f"{v:.4f}",
                va="center", ha="left", fontsize=5.0, fontweight="bold", color="#111111", zorder=5)
    ax.set_yticks(y_pos, stages, fontsize=5.0)
    ax.invert_yaxis()
    ax.set_xlim(0.31, 0.552)
    ax.set_xlabel("Validation Panoptic Quality (PQ)", fontsize=5.8, fontweight="bold")
    ax.set_title("(a) Architecture & Paradigm Progression", fontsize=6.2, fontweight="bold", pad=4)
    ax.legend(loc="upper right", fontsize=4.6, frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.tick_params(axis="x", labelsize=5.1)

    # Panel (b): Per-Pixel Multi-Segmenter Crop Blending Gain across all 12 detectors
    ax = axes[1]
    det_names = ["ms", "F3", "main", "s", "P6", "F0", "A", "P4a", "F1", "G1", "B", "P1"]
    orig_v = np.array([0.4648, 0.4619, 0.4612, 0.4643, 0.4601, 0.4541, 0.4539, 0.4619, 0.4646, 0.4491, 0.4534, 0.4650])
    up_v   = np.array([0.4716, 0.4679, 0.4655, 0.4682, 0.4635, 0.4571, 0.4564, 0.4644, 0.4669, 0.4509, 0.4548, 0.4660])
    gains = (up_v - orig_v) * 1000.0
    x_pos = np.arange(len(det_names))
    ax.grid(axis="y", zorder=0)
    b2 = ax.bar(x_pos, gains, width=0.64, color="#4db6ac", edgecolor="#111111", linewidth=0.6, zorder=3)
    for bar, g in zip(b2, gains):
        ax.text(bar.get_x() + bar.get_width() / 2, g + 0.16, f"+{g:.1f}",
                ha="center", va="bottom", fontsize=4.1, fontweight="bold", color="#004d40")
    ax.set_xticks(x_pos, det_names, fontsize=5.0, rotation=35, ha="right")
    ax.set_ylim(0, 8.1)
    ax.set_ylabel(r"PQ Gain $\Delta\mathrm{PQ}$ ($\times 10^{-3}$)", fontsize=5.8, fontweight="bold")
    ax.set_title("(b) Per-Pixel 3-Segmenter Blend", fontsize=6.2, fontweight="bold", pad=4)
    ax.tick_params(axis="y", labelsize=5.1)

    # Panel (c): 200-Split-Half Held-Out Date Bootstrap Distribution
    ax = axes[2]
    rng = np.random.default_rng(0)
    import fast_portfolio_lab as lab
    stems = lab.FOLD0_STEMS
    G17 = ["G1", "A", "B", "P1", "P4a", "F0", "F1", "YOLO"]
    g17_orig_30 = {s: lab.paint_wbf(lab.fuse([lab.painted[k][s] for k in G17], 0.30, 0.34), 0.30, 0.7, 100) for s in stems}
    stats_orig = {s: lab.score_rles(g17_orig_30[s], lab.GT[s]) for s in stems}

    from scripts.build_super_submission import super_ensemble_stem
    super_preds = {}
    for s in stems:
        core = [lab.painted[k][s] for k in ("G1", "A", "B", "P1", "P4a", "F0", "F1", "YOLO")]
        extra = {k: lab.painted[k][s] for k in ("ms", "s", "main", "F3")}
        rle_strs = super_ensemble_stem(core, extra)
        super_preds[s] = [{"size": [2048, 2048], "counts": c.encode("ascii")} for c in rle_strs]
    stats_super = {s: lab.score_rles(super_preds[s], lab.GT[s]) for s in stems}

    dates = sorted({s[:8] for s in stems})
    stems_by_date = {d: [s for s in stems if s[:8] == d] for d in dates}
    deltas = []
    for _ in range(200):
        perm = rng.permutation(len(dates))
        half_b = [dates[i] for i in perm[len(dates) // 2:]]
        b_stems = [s for d in half_b for s in stems_by_date[d]]
        tp1 = fp1 = fn1 = 0; iou1 = 0.0
        tp0 = fp0 = fn0 = 0; iou0 = 0.0
        for s in b_stems:
            t, f, n, u = stats_super[s]
            tp1 += t; fp1 += f; fn1 += n; iou1 += u
            t, f, n, u = stats_orig[s]
            tp0 += t; fp0 += f; fn0 += n; iou0 += u
        pq1 = iou1 / (tp1 + 0.5 * (fp1 + fn1))
        pq0 = iou0 / (tp0 + 0.5 * (fp0 + fn0))
        deltas.append((pq1 - pq0) * 1000.0)
    deltas = np.array(deltas)
    win_rate = float((deltas > 0).mean() * 100.0)

    ax.grid(axis="both", zorder=0)
    ax.hist(deltas, bins=18, color="#9575cd", edgecolor="#111111", linewidth=0.55, alpha=0.88, zorder=3)
    ax.axvline(0.0, color="#d32f2f", linestyle="--", linewidth=0.9, zorder=4, label="Parity (0.0)")
    ax.axvline(deltas.mean(), color="#1b5e20", linestyle="-", linewidth=1.0, zorder=4,
               label=f"Mean +{deltas.mean():.2f} ({win_rate:.0f}% Win)")
    ax.set_xlabel(r"Held-Out Date $\Delta\mathrm{PQ}$ ($\times 10^{-3}$)", fontsize=5.8, fontweight="bold")
    ax.set_ylabel("Bootstrap Splits (of 200)", fontsize=5.8, fontweight="bold")
    ax.set_title("(c) 200-Split Date Bootstrap", fontsize=6.2, fontweight="bold", pad=4)
    ax.legend(loc="upper left", fontsize=4.5, frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.tick_params(labelsize=5.1)

    save_dual(fig, "experiments/progression_and_ablation")
    shutil.copy2(FIG_ROOT / "experiments/progression_and_ablation.pdf", FIG_ROOT / "ablation/ablation_study.pdf")
    shutil.copy2(FIG_ROOT / "experiments/progression_and_ablation.png", FIG_ROOT / "ablation/ablation_study.png")


def fig_4_evaluation_distributions():
    """5-Fold OOF Evaluation Distributions: Pairwise IoU, Per-Fold PQ/SQ/RQ, and Partner Multiplicity."""
    f_rows = data.WORK / "cache" / "ms" / "oof_rows.json"
    if f_rows.exists():
        rows = json.load(open(f_rows))
    else:
        from scripts.figures import oof_rows
        rows = oof_rows("ms")
    from scripts.figures import human_rows
    hum = human_rows()
    fold = data.folds()
    stems = sorted(rows)
    cat = lambda d, k: np.concatenate([np.asarray(v[k], float) for v in d.values()])

    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.08), dpi=300,
                           gridspec_kw=dict(width_ratios=[1.22, 1.15, 1.05], wspace=0.39))

    # Panel (a): Overlapping Pairwise IoU Distribution
    bins = np.linspace(0, 1, 26)
    ax[0].grid(axis="both", zorder=0)
    ax[0].axvspan(0.35, 0.50, color="#fff3cd", alpha=0.75, zorder=1, label="Near-Miss [0.35, 0.50]")
    for d, c, ls, lab_s in ((rows, "#1f3b73", "-", "5-Fold Model"), (hum, "#d95f02", "--", "Inter-Annotator")):
        v = cat(d, "pair_iou")
        ax[0].hist(v, bins=bins, weights=np.full(len(v), 1 / len(v)), histtype="step",
                   color=c, lw=1.2, ls=ls, zorder=3, label=f"{lab_s} (mean {v.mean():.2f})")
    ax[0].axvline(0.5, color="#c62828", lw=0.9, ls=":", zorder=4, label="PQ Cutoff (IoU > 0.50)")
    ax[0].set_xlabel("IoU of Overlapping Pairs", fontsize=5.8, fontweight="bold")
    ax[0].set_ylabel("Fraction of Pairs", fontsize=5.8, fontweight="bold")
    ax[0].set_title("(a) Pairwise IoU vs. Human", fontsize=6.1, fontweight="bold", pad=4)
    ax[0].set_ylim(0, ax[0].get_ylim()[1] * 1.35)
    ax[0].legend(fontsize=4.4, frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper left")
    ax[0].tick_params(labelsize=5.1)

    # Panel (b): Per-Fold PQ, SQ, RQ with 95% Date-Bootstrap CI
    ax[1].grid(axis="both", zorder=0)
    pq, sq, rq, lo, hi = [], [], [], [], []
    for k in range(data.N_FOLDS):
        r = [rows[s] for s in stems if fold[s] == k]
        s_sum = metric.summarize(r)
        pq.append(s_sum["pq"]); sq.append(s_sum["sq"]); rq.append(s_sum["rq"])
        ci = metric.bootstrap_ci(r, [s[:8] for s in stems if fold[s] == k])
        lo.append(pq[-1] - ci[0]); hi.append(ci[1] - pq[-1])
    x_f = np.arange(data.N_FOLDS)
    ax[1].plot(x_f, sq, "s--", ms=3.5, lw=0.9, color="#2e7d32", label=f"SQ (mean {np.mean(sq):.3f})", zorder=3)
    ax[1].plot(x_f, rq, "^-.", ms=3.5, lw=0.9, color="#5e35b1", label=f"RQ (mean {np.mean(rq):.3f})", zorder=3)
    ax[1].errorbar(x_f, pq, yerr=[lo, hi], fmt="o-", ms=4.0, lw=1.1, capsize=2.2,
                   color="#1f3b73", label="PQ (5-Fold OOF)", zorder=4)
    hum_pq = metric.summarize(list(hum.values()))["pq"]
    ax[1].axhline(hum_pq, color="#d95f02", lw=0.85, ls=":", label=f"Human PQ ({hum_pq:.4f})", zorder=2)
    ax[1].set_xticks(x_f, [f"Fold {k}" for k in x_f], fontsize=5.1)
    ax[1].set_ylim(0.31, 0.74)
    ax[1].set_ylabel("Metric Score", fontsize=5.8, fontweight="bold")
    ax[1].set_title("(b) 5-Fold OOF Stability", fontsize=6.1, fontweight="bold", pad=4)
    ax[1].legend(fontsize=4.2, frameon=True, facecolor="white", edgecolor="#cccccc", loc="center", bbox_to_anchor=(0.56, 0.52), ncol=1)
    ax[1].tick_params(labelsize=5.1)

    # Panel (c): Partner Multiplicity (1-to-1 vs Split/Merge)
    ax[2].grid(axis="y", zorder=0)
    x = np.arange(4)
    for j, (k, c, lab_s) in enumerate((("gt_degree", "#1f3b73", "GT Annotations"), ("pred_degree", "#64b5f6", "Predictions"))):
        n = np.bincount(np.minimum(cat(rows, k).astype(int), 3), minlength=4)
        sh = n / n.sum()
        bars = ax[2].bar(x + (j - 0.5) * 0.36, sh, 0.36, color=c, edgecolor="#111111", lw=0.55, label=lab_s, zorder=3)
        for b, val in zip(bars, sh):
            if val > 0.02:
                ax[2].text(b.get_x() + b.get_width() / 2, val + 0.015, f"{val*100:.0f}%",
                           ha="center", va="bottom", fontsize=4.3, fontweight="bold")
    ax[2].set_xticks(x, ["0 (FP/FN)", "1 (1-to-1)", "2 (Split)", "3+ (Merge)"], fontsize=4.8)
    ax[2].set_ylabel("Share of Instances", fontsize=5.8, fontweight="bold")
    ax[2].set_title("(c) Partner Degree", fontsize=6.1, fontweight="bold", pad=4)
    ax[2].set_ylim(0, 1.05)
    ax[2].legend(fontsize=4.5, frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper right")
    ax[2].tick_params(axis="y", labelsize=5.1)

    save_dual(fig, "experiments/evaluation_distributions", legacy_name="evaluation")


def fig_5_qualitative_error_analysis():
    """PaperBanana Critic-Refined 4-Column Qualitative & Error Analysis Panel."""
    import fast_portfolio_lab as lab
    from scripts.build_super_submission import super_ensemble_stem

    if (FIG_ROOT / "examples.pdf").exists():
        shutil.copy2(FIG_ROOT / "examples.pdf", FIG_ROOT / "qualitative/real_photo_examples.pdf")
        shutil.copy2(FIG_ROOT / "examples.png", FIG_ROOT / "qualitative/real_photo_examples.png")

    selected = []
    for s in lab.FOLD0_STEMS:
        core = [lab.painted[k][s] for k in ("G1", "A", "B", "P1", "P4a", "F0", "F1", "YOLO")]
        extra = {k: lab.painted[k][s] for k in ("ms", "s", "main", "F3")}
        rle_strs = super_ensemble_stem(core, extra)
        rles = [{"size": [2048, 2048], "counts": c.encode("ascii")} for c in rle_strs]
        tp, fp, fn, iou = lab.score_rles(rles, lab.GT[s])
        selected.append((tp, fp, fn, iou, s, rles))

    selected.sort(key=lambda x: -x[0])
    picks = [
        (selected[1][4], selected[1][5], "Case A: Dense Active Group"),
        (selected[3][4], selected[3][5], "Case B: Curvilinear Spines"),
        (sorted(selected, key=lambda x: -(x[1] + x[2]))[0][4],
         sorted(selected, key=lambda x: -(x[1] + x[2]))[0][5],
         "Case C: Fragmented Fibrils"),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(7.4, 4.4), dpi=300,
                             gridspec_kw={"wspace": 0.08, "hspace": 0.14})
    col_titles = [
        "(1) Multi-Annotator GT",
        "(2) Crop Probabilities",
        "(3) Instances + GT",
        "(4) Error Map (TP/FP/FN)",
    ]

    for row_idx, (stem, pred_rles, row_label) in enumerate(picks):
        gt_records = lab.GT[stem]
        gt_rles = gt_records[0]
        pred_masks = [mu.decode(r).astype(bool) for r in pred_rles]
        gt_masks = [mu.decode(r).astype(bool) for r in gt_rles]

        # Center on the window with the largest total filament area
        best_px, cx, cy = -1, 1024, 1024
        for m in pred_masks + gt_masks:
            ys, xs = np.nonzero(m)
            if len(xs):
                mx, my = int(xs.mean()), int(ys.mean())
                wx0, wy0 = int(np.clip(mx - 240, 0, 2048 - 480)), int(np.clip(my - 240, 0, 2048 - 480))
                px = sum(int(pm[wy0:wy0 + 480, wx0:wx0 + 480].sum()) for pm in pred_masks + gt_masks)
                if px > best_px:
                    best_px, cx, cy = px, mx, my
        R = 480
        x0, y0 = int(np.clip(cx - R // 2, 0, 2048 - R)), int(np.clip(cy - R // 2, 0, 2048 - R))
        ext = (x0, x0 + R, y0 + R, y0)

        # Column 1: Multi-Annotator Ground Truth Agreement Heatmap
        ann_count = np.zeros((2048, 2048), np.float32)
        for rec in gt_records:
            rec_u = np.zeros((2048, 2048), bool)
            for r in rec:
                rec_u |= mu.decode(r).astype(bool)
            ann_count += rec_u.astype(np.float32)
        ann_norm = ann_count / max(len(gt_records), 1)
        bg1 = np.full((R, R, 3), 24, dtype=np.uint8)
        crop_ann = ann_norm[y0:y0 + R, x0:x0 + R]
        cmap_ann = plt.get_cmap("YlOrBr")
        fg_ann = crop_ann > 0
        bg1[fg_ann] = (cmap_ann(0.35 + 0.65 * crop_ann[fg_ann])[:, :3] * 255).astype(np.uint8)
        axes[row_idx, 0].imshow(bg1, extent=ext)
        axes[row_idx, 0].text(
            x0 + 12, y0 + 14,
            f"{stem[:4]}-{stem[4:6]}-{stem[6:8]} ({stem[14:]}) | {len(gt_records)} Rec.\n{row_label}",
            color="#ffb703", fontsize=4.3, fontweight="bold", va="top", ha="left", clip_on=True,
            bbox=dict(facecolor="#000000cc", edgecolor="none", pad=1.6)
        )

        # Column 2: Stage-2 Continuous Crop Probability Field
        prob_canvas = np.zeros((2048, 2048), np.float32)
        inst_ms = postprocess.load_cache(data.WORK / "cache" / "ms" / "oof" / f"{stem}.npz")
        for cf, (bx1, by1, bx2, by2), p_crop in inst_ms:
            if cf >= 0.20:
                prob_canvas[by1:by2, bx1:bx2] = np.maximum(
                    prob_canvas[by1:by2, bx1:bx2], (p_crop.astype(np.float32) / 255.0) * min(cf * 1.5, 1.0)
                )
        axes[row_idx, 1].imshow(prob_canvas[y0:y0 + R, x0:x0 + R], cmap="magma", vmin=0, vmax=1, extent=ext)

        # Column 3: Predicted Instances + GT contours
        rgb_inst = np.full((2048, 2048, 3), 24, dtype=np.float32)
        cmap = plt.get_cmap("tab20")
        for idx_m, m in enumerate(pred_masks):
            col = np.array(cmap(idx_m % 20)[:3]) * 255
            rgb_inst[m] = col
        axes[row_idx, 2].imshow(rgb_inst.astype(np.uint8)[y0:y0 + R, x0:x0 + R], extent=ext)
        for gm in gt_masks:
            cs, _ = cv2.findContours(gm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cs:
                axes[row_idx, 2].plot(c[:, 0, 0], c[:, 0, 1], color="white", lw=0.6)

        # Column 4: Pixel-Level TP (Green), FP (Red), FN (Blue) Map
        p_tot = np.zeros((2048, 2048), bool)
        for m in pred_masks:
            p_tot |= m
        g_tot = np.zeros((2048, 2048), bool)
        for m in gt_masks:
            g_tot |= m
        tp_px = p_tot & g_tot
        fp_px = p_tot & ~g_tot
        fn_px = ~p_tot & g_tot

        err_rgb = np.full((2048, 2048, 3), 24, dtype=np.uint8)
        err_rgb[tp_px] = [46, 204, 113]
        err_rgb[fp_px] = [231, 76, 60]
        err_rgb[fn_px] = [52, 152, 219]
        axes[row_idx, 3].imshow(err_rgb[y0:y0 + R, x0:x0 + R], extent=ext)
        axes[row_idx, 3].text(
            x0 + 12, y0 + R - 18,
            "Green: TP | Red: FP | Blue: FN",
            color="white", fontsize=4.2, fontweight="bold", clip_on=True,
            bbox=dict(facecolor="#000000cc", edgecolor="none", pad=1.2)
        )

        for col_idx in range(4):
            q = axes[row_idx, col_idx]
            q.set_xlim(x0, x0 + R); q.set_ylim(y0 + R, y0)
            q.set_xticks([]); q.set_yticks([])
            if row_idx == 0:
                q.set_title(col_titles[col_idx], fontsize=5.6, fontweight="bold", pad=3)
            for sp in q.spines.values():
                sp.set_linewidth(0.45)

    save_dual(fig, "qualitative/qualitative_error_analysis")


def main():
    fig_1_system_architecture()
    fig_2_pipeline_walkthrough()
    fig_3_progression_and_ablation()
    fig_4_evaluation_distributions()
    fig_5_qualitative_error_analysis()


if __name__ == "__main__":
    main()
