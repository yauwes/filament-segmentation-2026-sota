"""Figures of the technical report, drawn from cached proposals (WORK/cache/<run>) and the annotations.

    python scripts/figures.py pipeline --run main --stem 20140607195854Bh
    python scripts/figures.py examples --run main
    python scripts/figures.py evaluation --run main
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg import data, enhance, metric, postprocess   # noqa: E402

OUT = data.REPO / "figures"
PAD = 0.2


def cell_of(run):
    return json.load(open(data.WORK / "cache" / run / "tune.json"))["cell"]


def painted(run, stem, split="oof"):
    return postprocess.paint(postprocess.load_cache(data.WORK / "cache" / run / split / f"{stem}.npz"), **cell_of(run))


def colour_overlay(img, inst, alpha=0.55):
    """Grey photo + one colour per painted instance."""
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.float32)
    cmap = plt.get_cmap("tab20")
    for i, (_, (x1, y1, x2, y2), m) in enumerate(inst):
        c = np.array(cmap(i % 20)[:3]) * 255
        rgb[y1:y2, x1:x2][m] = (1 - alpha) * rgb[y1:y2, x1:x2][m] + alpha * c
    return rgb.astype(np.uint8)


def outlines(ax, rles, colour, lw=0.6):
    for r in rles:
        cs, _ = cv2.findContours(mu.decode(r), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            ax.plot(c[:, 0, 0], c[:, 0, 1], color=colour, lw=lw)


def fig_pipeline(a):
    """One frame through the pipeline: input -> enhanced planes -> boxes -> crop probabilities -> painted instances."""
    img = data.read_gray(a.stem)
    inst = postprocess.load_cache(data.WORK / "cache" / a.run / a.split / f"{a.stem}.npz")
    cell = cell_of(a.run)
    kept = postprocess.paint(inst, **cell)
    R = a.region
    c = np.array([[(x1 + x2) / 2, (y1 + y2) / 2] for _, (x1, y1, x2, y2), _ in kept] or [[1024, 1024]])
    cx, cy = (int(v) for v in (a.centre.split(":") if a.centre else np.median(c, 0)))
    x0, y0 = int(np.clip(cx - R // 2, 0, 2048 - R)), int(np.clip(cy - R // 2, 0, 2048 - R))
    win = lambda x: x[y0:y0 + R, x0:x0 + R]
    ext = (x0, x0 + R, y0 + R, y0)
    fig = plt.figure(figsize=(7.1, 1.62), dpi=300)
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.52, 1], wspace=0.04, left=0.004, right=0.996, top=0.89, bottom=0.01)
    ax = [fig.add_subplot(gs[0, k]) for k in (0, 1, 2, 4)]
    ax[0].imshow(cv2.resize(img, (768, 768), interpolation=cv2.INTER_AREA), cmap="gray", vmin=0, vmax=255, extent=(0, 2048, 2048, 0))
    ax[0].add_patch(plt.Rectangle((x0, y0), R, R, fill=False, ec="#ffcc00", lw=0.7))
    ax[1].imshow(win(np.moveaxis(enhance.enhance(img), 0, -1)), extent=ext)
    ax[2].imshow(win(img), cmap="gray", vmin=0, vmax=255, extent=ext)
    for cf, (x1, y1, x2, y2), _ in inst:
        w, h = x2 - x1, y2 - y1
        on = cf >= cell["conf"]
        bx, by, bw, bh = x1 + w * PAD / (1 + 2 * PAD), y1 + h * PAD / (1 + 2 * PAD), w / (1 + 2 * PAD), h / (1 + 2 * PAD)
        ax[2].add_patch(plt.Rectangle((bx, by), bw, bh, fill=False, lw=0.6, ec="#ffcc00" if on else "#ff5555", ls="-" if on else ":"))
        if on and x0 < bx < x0 + R - 60 and y0 + 30 < by < y0 + R:
            ax[2].text(bx, by - 6, f"{cf:.2f}", color="#ffcc00", fontsize=3.6, va="bottom")
    ax[3].imshow(win(colour_overlay(img, kept, 0.65)), extent=ext)
    for q, t in zip(ax, ["(a) H-$\\alpha$ frame, 2048$^2$ px", "(b) enhanced planes (region)", "(c) boxes and confidences",
                         f"(e) painted instances"]):
        q.set_title(t, fontsize=6.3, pad=2)
        q.set_xticks([]); q.set_yticks([])
        if q is not ax[0]:
            q.set_xlim(x0, x0 + R); q.set_ylim(y0 + R, y0)
        for sp in q.spines.values():
            sp.set_linewidth(0.3)
    sub = gs[0, 3].subgridspec(4, 2, wspace=0.06, hspace=0.06)
    inside = [i for i in inst if i[0] >= cell["conf"] and x0 <= (i[1][0] + i[1][2]) / 2 <= x0 + R and y0 <= (i[1][1] + i[1][3]) / 2 <= y0 + R]
    for k, (cf, (x1, y1, x2, y2), p) in enumerate(inside[:4]):
        a1, a2 = fig.add_subplot(sub[k, 0]), fig.add_subplot(sub[k, 1])
        a1.imshow(cv2.resize(img[y1:y2, x1:x2], (96, 96)), cmap="gray", vmin=0, vmax=255)
        a2.imshow(cv2.resize(p, (96, 96)), cmap="magma", vmin=0, vmax=255)
        for q in (a1, a2):
            q.set_xticks([]); q.set_yticks([])
            for sp in q.spines.values():
                sp.set_linewidth(0.3)
    fig.text(0.5 * (ax[2].get_position().x1 + ax[3].get_position().x0), 0.905, "(d) crops, mask prob.", fontsize=6.3, ha="center", va="bottom")
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "pipeline.png", dpi=300)
    fig.savefig(OUT / "pipeline.pdf", dpi=300)
    print("wrote", OUT / "pipeline.png", "centre", cx, cy)


def _pairs(st):
    """overlap_stats keys that would collide with the counts ("iou" is the IoU sum there)."""
    return dict(pair_iou=st["iou"], pair_dice=st["dice"], gt_degree=st["gt_degree"], pred_degree=st["pred_degree"])


def oof_rows(run):
    """Per-photo official counts and overlap statistics of a run at its tuned cell (cached in WORK/cache/<run>)."""
    f = data.WORK / "cache" / run / "oof_rows.json"
    if f.exists():
        return json.load(open(f))
    root = data.WORK / "cache" / run / "oof"
    stems = sorted(p.stem for p in root.glob("*.npz"))
    recs = metric.load_records(stems)
    out = {}
    for s in stems:
        rl = postprocess.to_rles(painted(run, s))
        out[s] = dict(metric.counts(rl, recs[s]), n=len(rl), **_pairs(metric.overlap_stats(rl, recs[s])))
    json.dump(out, open(f, "w"))
    return out


def human_rows():
    """Reference: on every frame with several records, the first annotator as "prediction" against the others."""
    recs = metric.load_records()
    return {s: dict(metric.counts(r[0], r[1:]), **_pairs(metric.overlap_stats(r[0], r[1:]))) for s, r in recs.items() if len(r) > 1}


def fig_evaluation(a):
    rows, hum, fold = oof_rows(a.run), human_rows(), data.folds()
    stems = sorted(rows)
    cat = lambda d, k: np.concatenate([np.asarray(v[k], float) for v in d.values()])
    fig, ax = plt.subplots(1, 3, figsize=(3.4, 1.5), dpi=300, gridspec_kw=dict(width_ratios=[1.25, 1, 1.1], wspace=0.62))
    bins = np.linspace(0, 1, 26)
    for d, c, lab in ((rows, "#1f3b73", "model"), (hum, "#d9822b", "annotator")):
        v = cat(d, "pair_iou")
        ax[0].hist(v, bins=bins, weights=np.full(len(v), 1 / len(v)), histtype="step", color=c, lw=0.9, label=f"{lab}, mean {v.mean():.2f}")
    ax[0].axvline(0.5, color="gray", lw=0.5, ls=":")
    ax[0].set_xlabel("IoU of overlapping pairs"); ax[0].set_ylabel("share of pairs")
    ax[0].set_ylim(0, ax[0].get_ylim()[1] * 1.38); ax[0].legend(fontsize=4.5, frameon=False, loc="upper left", borderaxespad=0.2)
    pq, lo, hi = [], [], []
    for k in range(data.N_FOLDS):
        r = [rows[s] for s in stems if fold[s] == k]
        pq.append(metric.summarize(r)["pq"])
        ci = metric.bootstrap_ci(r, [s[:8] for s in stems if fold[s] == k])
        lo.append(pq[-1] - ci[0]); hi.append(ci[1] - pq[-1])
    ax[1].errorbar(range(data.N_FOLDS), pq, yerr=[lo, hi], fmt="o", ms=2.5, lw=0.8, capsize=1.5, color="#1f3b73")
    ax[1].axhline(metric.summarize(list(rows.values()))["pq"], color="#1f3b73", lw=0.5, ls="--")
    ax[1].axhline(metric.summarize(list(hum.values()))["pq"], color="#d9822b", lw=0.5, ls="--")
    ax[1].set_xticks(range(data.N_FOLDS)); ax[1].set_xlabel("fold"); ax[1].set_ylabel("PQ")
    x = np.arange(4)
    for j, (k, c, lab) in enumerate((("gt_degree", "#1f3b73", "annotations"), ("pred_degree", "#7fa3d7", "predictions"))):
        n = np.bincount(np.minimum(cat(rows, k).astype(int), 3), minlength=4)
        ax[2].bar(x + (j - 0.5) * 0.38, n / n.sum(), 0.38, color=c, label=lab)
    ax[2].set_xticks(x, ["0", "1", "2", "3+"]); ax[2].set_xlabel("partners overlapped"); ax[2].set_ylabel("share")
    ax[2].set_ylim(0, 1.18); ax[2].legend(fontsize=4.5, frameon=False, loc="upper right", borderaxespad=0.2)
    for q in ax:
        q.tick_params(labelsize=5, length=2, width=0.4); q.xaxis.label.set_size(5.5); q.yaxis.label.set_size(5.5)
        for sp in q.spines.values():
            sp.set_linewidth(0.4)
    fig.subplots_adjust(left=0.115, right=0.99, bottom=0.26, top=0.97)
    fig.savefig(OUT / "evaluation.pdf"); fig.savefig(OUT / "evaluation.png", dpi=300)
    m, h = metric.summarize(list(rows.values())), metric.summarize(list(hum.values()))
    deg = {k: np.bincount(np.minimum(cat(rows, k).astype(int), 3), minlength=4) for k in ("gt_degree", "pred_degree")}
    res = dict(run=a.run, photos=len(rows), model={k: m[k] for k in ("pq", "sq", "rq", "tp", "fp", "fn")},
               ci95=metric.bootstrap_ci([rows[s] for s in stems], [s[:8] for s in stems]), per_fold=pq,
               human=dict({k: h[k] for k in ("pq", "sq", "rq", "tp", "fp", "fn")}, photos=len(hum)),
               mean_iou=float(cat(rows, "pair_iou").mean()), mean_dice=float(cat(rows, "pair_dice").mean()),
               human_mean_iou=float(cat(hum, "pair_iou").mean()), human_mean_dice=float(cat(hum, "pair_dice").mean()),
               gt_degree=deg["gt_degree"].tolist(), pred_degree=deg["pred_degree"].tolist(),
               instances_per_photo=float(np.mean([v["n"] for v in rows.values()])))
    (data.REPO / "results").mkdir(exist_ok=True)
    json.dump(res, open(data.REPO / "results" / f"oof_{a.run}.json", "w"), indent=1)
    json.dump({k: np.round(cat(rows, "pair_" + k), 3).tolist() for k in ("iou", "dice")} | {k: cat(rows, k).astype(int).tolist() for k in ("gt_degree", "pred_degree")},
              open(data.REPO / "results" / "overlap_stats.json", "w"))
    print(json.dumps({k: v for k, v in res.items() if k != "per_fold"}, indent=1))


def fig_examples(a):
    """Six 420-px regions of out-of-fold predictions: the four best-matched, one with a miss, one with a false alarm."""
    rows = oof_rows(a.run)
    recs = metric.load_records(sorted(rows))
    pick = a.stems.split(",") if a.stems else sorted(rows, key=lambda s: -rows[s]["tp"])[:6]
    fig, ax = plt.subplots(2, 3, figsize=(3.4, 2.3), dpi=300)
    for q, s in zip(ax.ravel(), pick):
        s, *xy = s.split("@")
        inst = painted(a.run, s)
        img = colour_overlay(data.read_gray(s), inst, alpha=0.6)
        if xy:
            cx, cy = (int(v) for v in xy[0].split(":"))
        else:                                                   # centre on the densest group of predictions
            c = np.array([[(x1 + x2) / 2, (y1 + y2) / 2] for _, (x1, y1, x2, y2), _ in inst] or [[1024, 1024]])
            cx, cy = c[np.argmin(((c[:, None] - c[None]) ** 2).sum(-1).sum(1))].astype(int)
        x0, y0 = int(np.clip(cx - 210, 0, 2048 - 420)), int(np.clip(cy - 210, 0, 2048 - 420))
        q.imshow(img[y0:y0 + 420, x0:x0 + 420], extent=(x0, x0 + 420, y0 + 420, y0))
        outlines(q, recs[s][0], "white", 0.45)
        q.set_xlim(x0, x0 + 420); q.set_ylim(y0 + 420, y0)
        q.set_xticks([]); q.set_yticks([])
        q.set_title(f"{s[:4]}-{s[4:6]}-{s[6:8]} {s[14:]}", fontsize=4.5, pad=1.5)
        for sp in q.spines.values():
            sp.set_linewidth(0.3)
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.955, wspace=0.03, hspace=0.12)
    fig.savefig(OUT / "examples.pdf", dpi=300); fig.savefig(OUT / "examples.png", dpi=300)
    print("wrote", OUT / "examples.png", pick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["pipeline", "examples", "evaluation"])
    ap.add_argument("--run", default="main"); ap.add_argument("--stem"); ap.add_argument("--split", default="oof")
    ap.add_argument("--stems", help="examples: comma-separated stems, optionally stem@cx:cy for the region centre")
    ap.add_argument("--region", type=int, default=900, help="pipeline: side of the zoomed region in px")
    ap.add_argument("--centre", help="pipeline: cx:cy of the zoomed region (default: median of the kept instances)")
    a = ap.parse_args()
    globals()[f"fig_{a.what}"](a)


if __name__ == "__main__":
    main()
