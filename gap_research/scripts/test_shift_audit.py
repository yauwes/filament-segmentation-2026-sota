"""Test distribution & non-transductive invariance audit for Wave-2 candidate submissions.

Checks:
  1. Strict schema & validation via filseg.submission.validate
  2. Size distribution parity (<580px bbox max-side share on Test pred vs ref)
  3. Prediction density (filaments/photo) and empty-photo count
  4. Observatory parity (mean predictions & small-fraction across Ch/Bh/Mh/Th/Uh/Lh stations)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "gap_research" / "scripts"))
import eval_harness as eh  # noqa: E402
from filseg import submission  # noqa: E402


def _decode_areas_and_sides(rle_strs: list) -> tuple[list[int], list[float]]:
    areas, sides = [], []
    for s in rle_strs:
        if not s:
            continue
        m = mu.decode(s if isinstance(s, dict) else {"size": [2048, 2048], "counts": s.encode()})
        area = int(m.sum())
        if area > 0:
            ys, xs = np.nonzero(m)
            areas.append(area)
            sides.append(float(max(int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1))))
    return areas, sides


def audit_csv(csv_path: Path, ref_csv: Path | None = None) -> dict:
    stems = list(eh.TEST_STEMS)
    submission.validate(csv_path, stems)
    by_stem = submission.read_csv(csv_path)

    counts = []
    areas_all, sides_all = [], []
    obs_counts: dict[str, list[int]] = defaultdict(list)
    obs_small: dict[str, list[int]] = defaultdict(list)
    for s in stems:
        rs = by_stem.get(s, [])
        counts.append(len(rs))
        obs = s[-2:]
        obs_counts[obs].append(len(rs))
        ar, ss = _decode_areas_and_sides(rs)
        areas_all.extend(ar)
        sides_all.extend(ss)
        obs_small[obs].extend([1 if a < 580 else 0 for a in ar])

    small_area_share = float(np.mean([a < 580 for a in areas_all])) if areas_all else 0.0
    out = {
        "csv": str(csv_path),
        "test_images": len(stems),
        "total_preds": int(sum(counts)),
        "mean_per_photo": round(float(np.mean(counts)), 3),
        "empty_photos": int(sum(1 for c in counts if c == 0)),
        "small_area_lt580_share": round(small_area_share, 4),
        "median_area_px": round(float(np.median(areas_all)), 1) if areas_all else 0.0,
        "median_max_side": round(float(np.median(sides_all)), 1) if sides_all else 0.0,
        "observatory_means": {k: round(float(np.mean(v)), 2) for k, v in sorted(obs_counts.items())},
        "observatory_small_area_share": {k: round(float(np.mean(v)) if v else 0.0, 3) for k, v in sorted(obs_small.items())},
    }
    if ref_csv and Path(ref_csv).is_file():
        ref_by = submission.read_csv(ref_csv)
        ref_counts = [len(ref_by.get(s, [])) for s in stems]
        ref_areas = [a for s in stems for a in _decode_areas_and_sides(ref_by.get(s, []))[0]]
        ref_small = float(np.mean([a < 580 for a in ref_areas])) if ref_areas else 0.0
        out["ref_csv"] = str(ref_csv)
        out["ref_total_preds"] = int(sum(ref_counts))
        out["ref_mean_per_photo"] = round(float(np.mean(ref_counts)), 3)
        out["ref_small_area_lt580_share"] = round(ref_small, 4)
        out["delta_preds"] = int(sum(counts) - sum(ref_counts))
        out["delta_small_area_share"] = round(small_area_share - ref_small, 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--ref", type=Path, default=REPO / "results" / "submission.csv")
    args = ap.parse_args()
    rep = audit_csv(args.csv, args.ref)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
