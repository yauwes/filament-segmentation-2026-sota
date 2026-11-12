"""Kaggle kernel entry point: run one jobs/kaggle/<JOB>.sh of the attached code dataset on the kernel's GPUs."""
import os
import subprocess
import sys
from pathlib import Path

JOB = "@@JOB@@"
os.environ.update(dict(kv.split("=", 1) for kv in "@@ENV@@".split() if "=" in kv))     # job parameters
inp = Path("/kaggle/input")
repo = next(p for p in inp.rglob("run.py") if (p.parent / "filseg").is_dir()).parent
root = next(inp.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json")).parents[1]
env = dict(os.environ, MAGFILO_ROOT=str(root), FILSEG_WORK="/kaggle/working/work", PYTHONUNBUFFERED="1",
           PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(repo))
print("repo", repo, "| data", root, flush=True)
subprocess.run("pip install -q ultralytics==8.4.160 segmentation-models-pytorch==0.5.0 pycocotools scikit-image 2>&1 | tail -2; nvidia-smi -L; nproc",
               shell=True)
sys.exit(subprocess.run(["bash", str(repo / "jobs" / "kaggle" / f"{JOB}.sh")], cwd="/kaggle/working", env=dict(env, REPO=str(repo))).returncode)
