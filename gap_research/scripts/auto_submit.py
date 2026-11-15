"""Validated Kaggle Auto-Submission & Public LB Poller for filament-segmentation-2026."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "gap_research" / "scripts"))
import eval_harness as eh  # noqa: E402
import test_shift_audit  # noqa: E402
from filseg import submission  # noqa: E402


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(65536), b""):
            h.update(ch)
    return h.hexdigest()


def list_submissions() -> list[dict]:
    out = subprocess.check_output(
        ["kaggle", "competitions", "submissions", "-c", "filament-segmentation-2026", "-v"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    lines = [ln for ln in out.splitlines() if ln.strip() and not ln.startswith("Warning:")]
    if not lines:
        return []
    return list(csv.DictReader(lines))


def submit_and_poll(csv_path: Path, message: str, metrics_json: Path | None = None, dry_run: bool = False) -> dict:
    assert csv_path.exists(), f"Missing submission CSV: {csv_path}"
    submission.validate(csv_path, list(eh.TEST_STEMS))
    shift_rep = test_shift_audit.audit_csv(csv_path, REPO / "results" / "submission.csv")
    print("Shift Audit:", json.dumps(shift_rep, indent=2), flush=True)

    if metrics_json and metrics_json.exists():
        m = json.loads(metrics_json.read_text())
        print("Local Gate Metrics:", json.dumps(m, indent=2), flush=True)
        if m.get("pq", 0.0) <= 0.48563 or m.get("vs_e19_p_win", 0.0) < 0.95:
            raise RuntimeError(f"Blocked by Nested Date-CV gate: pq={m.get('pq')}, p_win={m.get('vs_e19_p_win')}")

    sha = _sha256(csv_path)
    subs = list_submissions()
    print(f"Existing recent submissions ({len(subs)}): {subs[:3]}", flush=True)
    if dry_run:
        return {"dry_run": True, "csv": str(csv_path), "sha256": sha, "shift": shift_rep}

    cmd = ["kaggle", "competitions", "submit", "-c", "filament-segmentation-2026", "-f", str(csv_path), "-m", message]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(res.stdout, flush=True)

    # Poll until latest submission matching fileName / description finishes
    target_name = csv_path.name
    latest = None
    for _ in range(40):
        time.sleep(15)
        rows = list_submissions()
        cand = next((r for r in rows if r.get("fileName") == target_name), rows[0] if rows else None)
        if cand:
            st = str(cand.get("status", "")).upper()
            print(f"[{dt.datetime.now(dt.timezone.utc).strftime('%H:%M:%S')}] {cand}", flush=True)
            if "COMPLETE" in st or "ERROR" in st:
                latest = cand
                break

    log_entry = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "csv": str(csv_path),
        "sha256": sha,
        "message": message,
        "shift": shift_rep,
        "kaggle_row": latest,
    }
    log_p = REPO / "gap_research" / "submissions" / "submission_log.jsonl"
    log_p.parent.mkdir(parents=True, exist_ok=True)
    with open(log_p, "a") as fh:
        fh.write(json.dumps(log_entry) + "\n")
    return log_entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("-m", "--message", required=True)
    ap.add_argument("--metrics", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out = submit_and_poll(args.csv, args.message, args.metrics, args.dry_run)
    print("RESULT:", json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
