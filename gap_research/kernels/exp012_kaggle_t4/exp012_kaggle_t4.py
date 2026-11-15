"""Kaggle 2 x T4 GPU Runner for EXP-012: Full-Frame Dense Physics (patience=5).

Runs parallelized across GPU 0 (folds 0,1,2) and GPU 1 (folds 3,4):
  Stage A: 5-Fold `mit_b3` UNet (@1024 bs=1 accum=2) -> packs `exp012_mitb3_*.tar.gz` immediately.
  Stage B: 5-Fold `cnx_upp` (`tu-convnext_small.fb_in22k`) UNet (@1280 bs=2, time-guarded) ->
           packs combined 2-backbone `exp012_dense_probs.tar.gz` + `exp012_cc_proposals.tar.gz`.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

MAX_WALL_MIN = 465.0


def _sh(cmd: list[str] | str, check: bool = True, env: dict | None = None) -> int:
    shell = isinstance(cmd, str)
    print("+", cmd if shell else " ".join(cmd), flush=True)
    res = subprocess.run(cmd, shell=shell, env=env)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed rc={res.returncode}: {cmd}")
    return res.returncode


def _find_comp_root() -> Path:
    for p in Path("/kaggle/input").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"):
        if "magfilo-extra" not in str(p).lower():
            return p.parents[1]
    raise FileNotFoundError("MAGFiLO_1.0_Kaggle_2026 not found under /kaggle/input")


def _find_code_root() -> Path:
    for p in Path("/kaggle/input").rglob("exp012_fullframe_dense_physics.py"):
        for cand in p.parents:
            if (cand / "filseg").is_dir() and (cand / "folds" / "folds.json").is_file():
                return cand
    raise FileNotFoundError("filseg-gap-wave4-code not found under /kaggle/input")


def _pack(src_dir: Path, dst_tgz: Path) -> None:
    tmp = dst_tgz.with_suffix(".tmp.gz")
    with tarfile.open(tmp, "w:gz", compresslevel=1) as tf:
        tf.add(src_dir, arcname=src_dir.name)
    tmp.replace(dst_tgz)
    print(f"Packed {dst_tgz.name}: {dst_tgz.stat().st_size / 1e6:.1f} MB", flush=True)


def _patch_exp012_for_kaggle(src: Path, dst: Path) -> None:
    txt = src.read_text()
    # Prevent /dev/shm overflow across 2 parallel T4 processes on Kaggle Docker
    txt = txt.replace("num_workers=4, drop_last=True, pin_memory=True", "num_workers=2, drop_last=True, pin_memory=False")
    txt = txt.replace("num_workers=4, drop_last=(b_try > 1), pin_memory=True", "num_workers=2, drop_last=(b_try > 1), pin_memory=False")
    txt = txt.replace("num_workers=2, pin_memory=True", "num_workers=2, pin_memory=False")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(txt)


def main() -> None:
    t0 = time.time()
    _sh("nvidia-smi -L; nproc; df -h /kaggle/working /tmp /dev/shm")

    res_pip = subprocess.run(
        "pip install -q segmentation-models-pytorch==0.5.0 pycocotools scikit-image 2>&1 | tail -5",
        shell=True,
        capture_output=True,
        text=True,
    )
    print(res_pip.stdout, flush=True)

    import torch

    n_gpu = torch.cuda.device_count()
    print(f"Detected GPUs: {n_gpu}", flush=True)

    data_root = _find_comp_root()
    code_src = _find_code_root()
    repo_dir = Path("/tmp/w5_repo")
    shutil.rmtree(repo_dir, ignore_errors=True)
    shutil.copytree(code_src, repo_dir, dirs_exist_ok=True)

    exp012_src = repo_dir / "gap_research" / "kernels" / "exp012_fullframe_dense_physics" / "exp012_fullframe_dense_physics.py"
    exp012_py = Path("/tmp/exp012_t4.py")
    _patch_exp012_for_kaggle(exp012_src, exp012_py)

    work_dir = Path("/tmp/exp012_work")
    cache_dir = Path("/tmp/exp012_cache")
    out_dir = Path("/tmp/exp012_out")
    for d in (work_dir, cache_dir, out_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Pre-warm pretrained encoder weights ONCE on main thread (avoid concurrent download race)
    import segmentation_models_pytorch as smp

    arches_avail = []
    for a, enc in (("mit_b3", "mit_b3"), ("cnx_upp", "tu-convnext_small.fb_in22k")):
        try:
            m = smp.Unet(encoder_name=enc, encoder_weights="imagenet", in_channels=4, classes=2)
            del m
            arches_avail.append(a)
            print(f"[prewarm OK] {a} ({enc})", flush=True)
        except Exception as exc:
            print(f"[prewarm WARN] {a}: {exc}", flush=True)
    assert arches_avail, "No EXP-012 architectures available"

    py = sys.executable
    env_base = dict(
        os.environ,
        PYTHONPATH=str(repo_dir),
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    common = [
        "--repo", str(repo_dir),
        "--data-root", str(data_root),
        "--work-dir", str(work_dir),
        "--cache-dir", str(cache_dir),
        "--out-dir", str(out_dir),
    ]

    _sh([py, str(exp012_py), "--worker", "prepare", "--procs", "4"] + common, env=env_base)

    def _run_arch_5f(arch: str, epochs: int = 32, patience: int = 5) -> bool:
        fold_groups = [(0, [0, 1, 2]), (1, [3, 4])] if n_gpu >= 2 else [(0, [0, 1, 2, 3, 4])]
        errs: list[BaseException] = []

        def _grp(gpu: int, folds: list[int]) -> None:
            try:
                for fk in folds:
                    if (time.time() - t0) / 60.0 > MAX_WALL_MIN:
                        print(f"[{arch} f{fk}] wall-clock guard ({MAX_WALL_MIN}m) reached -> stopping", flush=True)
                        return
                    _sh(
                        [py, str(exp012_py), "--worker", "train", "--arch", arch, "--fold", str(fk),
                         "--epochs", str(epochs), "--patience", str(patience)] + common,
                        env=dict(env_base, CUDA_VISIBLE_DEVICES=str(gpu)),
                    )
            except BaseException as exc:
                errs.append(exc)

        ts = [threading.Thread(target=_grp, args=(g, fks)) for g, fks in fold_groups]
        for th in ts:
            th.start()
        for th in ts:
            th.join()
        if errs:
            raise errs[0]
        return all((work_dir / "weights" / f"dense_{arch}_f{k}.done.json").exists() for k in range(5))

    done_arches: list[str] = []
    kw = Path("/kaggle/working")
    kw.mkdir(parents=True, exist_ok=True)

    # Phase 1: fast high-diversity MiT-B3 (@1024 bs=1 accum=2) -> pack checkpoint immediately
    if "mit_b3" in arches_avail:
        ok_m = _run_arch_5f("mit_b3", epochs=20, patience=5)
        if ok_m:
            done_arches.append("mit_b3")
            _sh([py, str(exp012_py), "--worker", "infer", "--arches", "mit_b3"] + common, env=dict(env_base, CUDA_VISIBLE_DEVICES="0"))
            _pack(out_dir / "dense_probs", kw / "exp012_mitb3_dense_probs.tar.gz")
            _pack(out_dir / "cc_proposals", kw / "exp012_mitb3_cc_proposals.tar.gz")
            if (out_dir / "exp012_solo_oof.json").exists():
                shutil.copy2(out_dir / "exp012_solo_oof.json", kw / "exp012_mitb3_summary.json")

    # Phase 2: ConvNeXt-Small UNet (@1280 bs=2) -> combined ensemble inference
    if "cnx_upp" in arches_avail and (time.time() - t0) / 60.0 < MAX_WALL_MIN - 35:
        ok_c = _run_arch_5f("cnx_upp", epochs=18, patience=5)
        if ok_c:
            done_arches.insert(0, "cnx_upp")
            shutil.rmtree(out_dir / "dense_probs", ignore_errors=True)
            shutil.rmtree(out_dir / "cc_proposals", ignore_errors=True)
            _sh([py, str(exp012_py), "--worker", "infer", "--arches", *done_arches] + common, env=dict(env_base, CUDA_VISIBLE_DEVICES="0"))
            _pack(out_dir / "dense_probs", kw / "exp012_dense_probs.tar.gz")
            _pack(out_dir / "cc_proposals", kw / "exp012_cc_proposals.tar.gz")
            if (out_dir / "exp012_solo_oof.json").exists():
                shutil.copy2(out_dir / "exp012_solo_oof.json", kw / "exp012_summary.json")

    # Export trained weights compactly too
    w_tgz = kw / "exp012_weights.tar.gz"
    if (work_dir / "weights").exists():
        _pack(work_dir / "weights", w_tgz)

    shutil.rmtree(cache_dir, ignore_errors=True)
    print(f"EXP-012 Kaggle T4x2 COMPLETE arches={done_arches} in {(time.time() - t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
