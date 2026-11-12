"""Download the 3,000 unlabeled GONG H-alpha frames used for self-training (public NSO/NISP archive).

    python scripts/fetch_gong.py OUT_DIR [--delay 3.5]

The frames are listed in scripts/gong_unlabeled.txt: one or two per day, days of 2011-2025 stratified by year, never
a competition frame and never within two days of a test date. They are JPEGs without annotations; the pipeline only
uses the images (filseg.detector.pseudo_label). One request every `delay` seconds, resumable.
"""
import argparse
import subprocess
import time
from pathlib import Path

BASE = "https://gong2.nso.edu/HA/hag"
UA = "filament-research-download/1.0 (unlabeled GONG H-alpha for a scientific ML study; <=1 request/3 s)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--delay", type=float, default=3.5)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stems = Path(__file__).with_name("gong_unlabeled.txt").read_text().split()
    for i, s in enumerate(stems, 1):
        f = out / f"{s}.jpg"
        if f.exists() and f.stat().st_size > 100_000:
            continue
        time.sleep(a.delay)
        subprocess.run(["curl", "-s", "-m", "180", "-A", UA, "-o", str(f), f"{BASE}/{s[:6]}/{s[:8]}/{s}.jpg"])
        if not f.exists() or f.stat().st_size < 100_000:        # missing on the server: skip, the list is a superset
            f.unlink(missing_ok=True)
        if i % 100 == 0:
            print(f"{i}/{len(stems)}", flush=True)
    print("DONE", len(list(out.glob("*.jpg"))), "frames in", out)


if __name__ == "__main__":
    main()
