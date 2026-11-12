"""Submission CSV: one row per predicted filament, `filament_id,segmentation_rle` (COCO compressed counts, 2048^2)."""
import csv
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

FULL = 2048


def write_csv(path, rles_by_stem):
    """rles_by_stem: stem -> [RLE dict]. Photos without a prediction simply have no row."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filament_id", "segmentation_rle"])
        for stem in sorted(rles_by_stem):
            for n, r in enumerate(rles_by_stem[stem], 1):
                c = r["counts"]
                w.writerow([f"{stem}_{n}", c.decode("ascii") if isinstance(c, bytes) else c])


def read_csv(path):
    """-> stem -> [RLE dict]."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.setdefault(row["filament_id"].rsplit("_", 1)[0], []).append(
                {"size": [FULL, FULL], "counts": row["segmentation_rle"].encode("ascii")})
    return out


def validate(path, stems):
    """The checks the leaderboard enforces: known photo ids, unique rows, decodable non-empty masks, and no pixel
    claimed by two filaments of the same photo. -> list of problems (empty = valid)."""
    problems, seen = [], set()
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or list(rows[0]) != ["filament_id", "segmentation_rle"]:
        return ["header must be filament_id,segmentation_rle"]
    stems = set(stems)
    for row in rows:
        fid = row["filament_id"]
        if fid in seen:
            problems.append(f"duplicate id {fid}")
        seen.add(fid)
        if fid.rsplit("_", 1)[0] not in stems:
            problems.append(f"unknown photo in id {fid}")
    for stem, rles in read_csv(path).items():
        try:
            area = mu.area(rles)
            union = mu.area(mu.merge(rles))
        except Exception as e:                               # noqa: BLE001 - any decode failure is a format problem
            problems.append(f"{stem}: undecodable RLE ({e})")
            continue
        if (area == 0).any():
            problems.append(f"{stem}: empty mask")
        if int(area.sum()) != int(union):
            problems.append(f"{stem}: overlapping masks ({int(area.sum()) - int(union)} px)")
    return problems


def demo():
    """Self-check: an RLE written to CSV comes back identical, and an overlap is caught."""
    import tempfile
    a = np.zeros((FULL, FULL), np.uint8); a[10:20, 10:40] = 1
    b = np.zeros((FULL, FULL), np.uint8); b[15:30, 30:60] = 1
    enc = lambda m: mu.encode(np.asfortranarray(m))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.csv"
        write_csv(p, {"x": [enc(a)]})
        assert (mu.decode(read_csv(p)["x"][0]) == a).all() and validate(p, ["x"]) == []
        write_csv(p, {"x": [enc(a), enc(b)]})
        assert any("overlapping" in s for s in validate(p, ["x"]))
    print("submission self-check ok")


if __name__ == "__main__":
    demo()
