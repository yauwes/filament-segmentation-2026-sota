"""Colab side of the remote training loop (no browser needed after the first cell).

The laptop publishes the repository as a private Kaggle dataset; this runner polls it, starts every new jobs/colab/*.sh
once (in the background, log to work/logs), and publishes logs, weights and prediction caches back as private
Kaggle datasets. When nothing has been running for IDLE minutes it returns, and the notebook cell that launched it
releases the GPU runtime, so compute units are only spent while a job is running.

Control files (push them with the code): jobs/colab/KILL_<job> stops that job, jobs/colab/STOP ends the runner.
"""
import json
import os
import signal
import subprocess
import time
from pathlib import Path

USER = "ajikajadeh"
CODE, OUT, WEIGHTS, CACHE = (f"{USER}/filseg-a100-{n}" for n in ("code", "out", "weights", "cache"))
C = Path("/content")
REPO, WORK, DL = C / "repo", C / "work", C / "code_dl"
LOGS, STARTED, SYNC = WORK / "logs", WORK / "started", WORK / "sync"
ENV = dict(os.environ, MAGFILO_ROOT=str(C / "data" / "MAGFiLO_1.0_Kaggle_2026"), FILSEG_WORK=str(WORK), PYTHONUNBUFFERED="1")
IDLE = 8                                                     # minutes without a running job before releasing the GPU


def sh(cmd, timeout=3600):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=ENV)
        return r.returncode, (r.stdout + r.stderr)[-3000:]
    except Exception as e:                                   # noqa: BLE001 - the loop must survive any failure
        return 1, repr(e)


def pull():
    rc, out = sh(f"rm -rf {DL} && kaggle datasets download {CODE} -p {DL} --unzip -q")
    if rc == 0 and (DL / "run.py").exists():
        sh(f"mkdir -p {REPO} && cp -r {DL}/. {REPO}/")
    return rc, out


def push(folder, slug, title):
    (Path(folder) / "dataset-metadata.json").write_text(json.dumps({"id": slug, "title": title, "licenses": [{"name": "CC0-1.0"}]}))
    exists = sh(f"kaggle datasets status {slug}")[0] == 0
    return sh(f"kaggle datasets version -p {folder} -m sync --dir-mode zip -q -d" if exists
              else f"kaggle datasets create -p {folder} --dir-mode zip -q")


def signature(folder):
    return sorted((str(p.relative_to(folder)), p.stat().st_size, int(p.stat().st_mtime)) for p in Path(folder).rglob("*")
                  if p.is_file() and p.name != "dataset-metadata.json")


def setup():
    cmds = ["pip install -q ultralytics==8.4.160 segmentation-models-pytorch==0.5.0 pycocotools scikit-image 2>&1 | tail -2",
            f"mkdir -p {C}/data && kaggle competitions download -c filament-segmentation-2026 -p {C}/data -q "
            f"&& unzip -q -o {C}/data/filament-segmentation-2026.zip -d {C}/data && rm {C}/data/filament-segmentation-2026.zip",
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc; free -g | head -2 | tail -1; "
            f"ls {C}/data/MAGFiLO_1.0_Kaggle_2026/train/train_images | wc -l; ls {C}/data/MAGFiLO_1.0_Kaggle_2026/test/test_images | wc -l"]
    return [dict(cmd=c[:80], rc=rc, out=out[-600:]) for c in cmds for rc, out in [sh(c)]]


def report(procs, rcs, extra=None):
    """Logs (tails), job states, GPU load and small result files -> the `out` dataset."""
    SYNC.mkdir(parents=True, exist_ok=True)
    for f in LOGS.glob("*.log"):
        lines = f.read_text(errors="replace").replace("\r", "\n").splitlines()
        (SYNC / f"log_{f.stem}.txt").write_text("\n".join(l for l in lines[-400:] if l.strip())[-60000:])
    for f in list((WORK / "cache").glob("*/tune.json")) + list((WORK / "weights").glob("*.history.json")) \
            + list((WORK / "det").glob("*/fold*/results.csv")) + list(WORK.glob("*.csv")):
        (SYNC / "__".join(f.relative_to(WORK).parts)).write_bytes(f.read_bytes())
    gpu = sh("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader")[1].strip()
    status = dict(time=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), gpu=gpu,
                  running=sorted(k for k, p in procs.items() if p.poll() is None), finished=rcs,
                  weights=sorted(p.name for p in (WORK / "weights").glob("*") if p.suffix in (".pt", ".pth")),
                  disk=sh("df -h /content | tail -1")[1].split()[-3:-1], extra=extra)
    (SYNC / "status.json").write_text(json.dumps(status, indent=1))
    print("STATUS", json.dumps(status)[:600], flush=True)
    return push(SYNC, OUT, "filseg a100 out")


def main():
    for d in (LOGS, STARTED, WORK / "weights", WORK / "cache"):
        d.mkdir(parents=True, exist_ok=True)
    print("setup ...", flush=True)
    extra = dict(pull=pull()[0], setup=setup(),
                 restored=sh(f"kaggle datasets download {WEIGHTS} -p {WORK}/weights --unzip -q")[0])   # resume
    procs, rcs, sigs = {}, {}, {}
    print("report", report(procs, rcs, extra)[1][-300:], flush=True)
    last_report = last_busy = time.time()
    while True:
        pull()
        jobs = REPO / "jobs" / "colab"
        for f in sorted(jobs.glob("KILL_*")):
            p = procs.get(f.name[5:])
            if p is not None and p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for f in sorted(jobs.glob("*.sh")):
            if not (STARTED / f.stem).exists():
                (STARTED / f.stem).write_text(time.ctime())
                procs[f.stem] = subprocess.Popen(["bash", str(f)], cwd=REPO, env=ENV, start_new_session=True,
                                                 stdout=open(LOGS / f"{f.stem}.log", "w"), stderr=subprocess.STDOUT)
                print("START", f.stem, flush=True)
        done = [k for k, p in procs.items() if p.poll() is not None and k not in rcs]
        for k in done:
            rcs[k] = procs[k].returncode
            print("END", k, rcs[k], flush=True)
        busy = any(p.poll() is None for p in procs.values())
        if busy:
            last_busy = time.time()
        if done or time.time() - last_report > 240:
            for folder, slug, title in ((WORK / "weights", WEIGHTS, "filseg a100 weights"), (WORK / "cache", CACHE, "filseg a100 cache")):
                sig = signature(folder)
                if sig and sig != sigs.get(slug) and (slug == WEIGHTS or done or not busy):   # caches: job boundaries only
                    rc, out = push(folder, slug, title)
                    print("PUSH", slug, rc, out[-200:], flush=True)
                    if rc == 0:
                        sigs[slug] = sig
            report(procs, rcs, extra)
            last_report = time.time()
        if (jobs / "STOP").exists() or (not busy and time.time() - last_busy > IDLE * 60):
            report(procs, rcs, dict(extra, stopped=time.ctime()))
            print("runner stops: no job running", flush=True)
            return
        time.sleep(45)


if __name__ == "__main__":
    main()
