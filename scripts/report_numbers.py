"""Write report/numbers.tex: every number quoted in the technical report, taken from results/*.json.

    python scripts/report_numbers.py

results/tune_<run>.json is the output of `run.py tune` for a run, results/oof_<run>.json the output of
`scripts/figures.py evaluation`. Edit AUTHOR, FINAL, PUBLIC and ROWS below; nothing else in the report is typed by hand.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AUTHOR = dict(name="AUTHOR NAME", city="CITY", country="COUNTRY", email="author@example.com")
FINAL = "main"                                               # the run behind the submitted file
PUBLIC = {"main": "0.40"}                                    # public leaderboard scores (two decimals are shown)
# (label, run or None, fixed (pq, sq, rq) for rows that are not runs of this repository)
ROWS = [
    ("Detector without flips", "nodettta", None),
    ("Segmenter without flips", "nosegtta", None),
    ("Two-stage pipeline (baseline detectors)", "main", None),
]
EARLIER = [                                                  # our earlier attempts on the same folds (previous code base)
    ("Full-frame U-Net + connected components", (0.389, 0.668, 0.582), "0.35"),
    ("YOLO instance masks, no crop segmenter", (0.427, 0.671, 0.636), "0.37"),
]


def tune(run):
    return json.load(open(REPO / "results" / f"tune_{run}.json"))


def main():
    fin, oof = tune(FINAL), json.load(open(REPO / "results" / f"oof_{FINAL}.json"))
    hum, cell = oof["human"], fin["cell"]
    rows = []
    for label, pqs, lb in EARLIER:
        rows.append(f"{label}$^\\dagger$ & {pqs[0]:.3f} & {pqs[1]:.3f} & {pqs[2]:.3f} & {lb} \\\\")
    rows.append("\\midrule")
    for label, run, _ in ROWS:
        f = REPO / "results" / f"tune_{run}.json"
        if not f.exists():
            continue
        t = tune(run)
        name = f"\\textbf{{{label}}}" if run == FINAL else label
        rows.append(f"{name} & {t['pq']:.3f} {{\\scriptsize[{t['ci95'][0]:.3f}, {t['ci95'][1]:.3f}]}} & {t['sq']:.3f} & {t['rq']:.3f} & {PUBLIC.get(run, '--')} \\\\")
    rows.append("\\midrule")
    rows.append(f"One annotator against another ({hum['photos']} frames) & {hum['pq']:.3f} & {hum['sq']:.3f} & {hum['rq']:.3f} & -- \\\\")
    gd, pd = oof["gt_degree"], oof["pred_degree"]
    macros = dict(
        authorname=AUTHOR["name"], authorcity=AUTHOR["city"], authorcountry=AUTHOR["country"], authoremail=AUTHOR["email"],
        oofpq=f"{fin['pq']:.3f}", oofci=f"{fin['ci95'][0]:.3f}--{fin['ci95'][1]:.3f}", oofsq=f"{fin['sq']:.3f}", oofrq=f"{fin['rq']:.3f}",
        ooftp=f"{fin['tp']:,}".replace(",", "{,}"), ooffp=f"{fin['fp']:,}".replace(",", "{,}"), ooffn=f"{fin['fn']:,}".replace(",", "{,}"),
        lbpq=PUBLIC[FINAL], humanpq=f"{hum['pq']:.2f}", humansq=f"{hum['sq']:.2f}", humanrq=f"{hum['rq']:.2f}",
        multiphotos=str(hum["photos"]), meaniou=f"{oof['mean_iou']:.2f}", meandice=f"{oof['mean_dice']:.2f}",
        humaniou=f"{oof['human_mean_iou']:.2f}", humandice=f"{oof['human_mean_dice']:.2f}",
        gtone=f"{100 * gd[1] / sum(gd):.0f}", gtzero=f"{100 * gd[0] / sum(gd):.0f}", gtmany=f"{100 * (gd[2] + gd[3]) / sum(gd):.1f}",
        predone=f"{100 * pd[1] / sum(pd):.0f}", predzero=f"{100 * pd[0] / sum(pd):.0f}", predmany=f"{100 * (pd[2] + pd[3]) / sum(pd):.1f}",
        perphoto=f"{oof['instances_per_photo']:.1f}",
        foldmin=f"{min(oof['per_fold']):.3f}", foldmax=f"{max(oof['per_fold']):.3f}",
        cellthr=f"{cell['thr']:g}", cellconf=f"{cell['conf']:g}", celldrop=f"{cell['drop']:g}",
        resultrows="\n    ".join(rows),
    )
    out = ["% Written by scripts/report_numbers.py from results/*.json -- do not edit by hand."]
    out += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    (REPO / "report" / "numbers.tex").write_text("\n".join(out) + "\n")
    print("\n".join(out[:1] + [o for o in out if "resultrows" not in o][1:8]), "\n...\n", macros["resultrows"])


if __name__ == "__main__":
    main()
