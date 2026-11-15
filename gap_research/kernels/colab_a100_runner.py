"""Colab A100 Accelerator Runner for Wave-2/Wave-3 Experiments (EXP-012 -> EXP-009a -> EXP-008a -> EXP-009b -> EXP-008b).

Stages read-only Kaggle inputs under /content/kaggle_input/ in two phases:
  1. stage_core (<20s): competition zip + filseg-a100-code -> immediately runs EXP-012 Full-Frame Dense Physics!
  2. stage_wave2_extra: remaining weights/caches -> runs EXP-009a Enh-YOLO11l, EXP-008a FRCNN-5F, etc.
Publishes compact output archives after EACH completed experiment to `ajikajadeh/filseg-gap-a100-out`.
Zero credentials are stored or logged in this script.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time


K_IN = pathlib.Path("/content/kaggle_input")
K_WORK = pathlib.Path("/content/kaggle_working")
PKG_DIR = pathlib.Path("/tmp/a100_out_pkg")


def _run(cmd: list[str], check: bool = True, quiet_log: str | None = None) -> int:
    print("+", " ".join(cmd), flush=True)
    if quiet_log:
        with open(quiet_log, "a") as lf:
            lf.write(f"\n+ {' '.join(cmd)}\n")
            res = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
        if check and res.returncode != 0:
            tail = pathlib.Path(quiet_log).read_text(errors="ignore").splitlines()[-40:]
            print("\n".join(tail), flush=True)
            raise RuntimeError(f"Command failed (rc={res.returncode}): {' '.join(cmd)}")
        return res.returncode
    res = subprocess.run(cmd)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed (rc={res.returncode}): {' '.join(cmd)}")
    return res.returncode


def stage_core() -> None:
    t0 = time.time()
    comp_dir = K_IN / "competitions" / "filament-segmentation-2026"
    if not (comp_dir / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").exists():
        comp_dir.mkdir(parents=True, exist_ok=True)
        _run(["kaggle", "competitions", "download", "-c", "filament-segmentation-2026", "-p", str(comp_dir)], quiet_log="/tmp/stage.log")
        for z in comp_dir.glob("*.zip"):
            _run(["unzip", "-qo", str(z), "-d", str(comp_dir)], quiet_log="/tmp/stage.log")
            z.unlink()

    dst = K_IN / "datasets" / "ajikajadeh/filseg-a100-code"
    if not dst.exists() or not any(dst.iterdir()):
        dst.mkdir(parents=True, exist_ok=True)
        _run(["kaggle", "datasets", "download", "-d", "ajikajadeh/filseg-a100-code", "-p", str(dst), "--unzip"], quiet_log="/tmp/stage.log")
    print(f"Core inputs staged cleanly in {(time.time()-t0):.1f}s", flush=True)


def stage_inputs() -> None:
    import tarfile

    t0 = time.time()
    stage_core()

    datasets = [
        "ajikajadeh/filseg-a100-weights",
        "ajikajadeh/filseg-tpu-code",
        "ajikajadeh/filseg-ts-weights",
    ]
    for ds in datasets:
        dst = K_IN / "datasets" / ds
        if not dst.exists() or not any(dst.iterdir()):
            dst.mkdir(parents=True, exist_ok=True)
            _run(["kaggle", "datasets", "download", "-d", ds, "-p", str(dst), "--unzip"], quiet_log="/tmp/stage.log")

    # Fast single-archive staging for g19 (tsF3, tsP6) and tsP4a (<10s vs 40m sequential API calls)
    g19_dst = K_IN / "notebooks" / "ajikajadeh" / "filseg-twostage-g19"
    fp_dst = K_IN / "notebooks" / "ajikajadeh" / "filseg-final-pipeline"
    if not (g19_dst / "tsF3").exists() or not (fp_dst / "final" / "tsP4a").exists():
        bundle_dir = pathlib.Path("/tmp/g19_p4a_dl")
        bundle_dir.mkdir(parents=True, exist_ok=True)
        _run(["kaggle", "datasets", "download", "-d", "ajikajadeh/filseg-gap-g19-p4a", "-p", str(bundle_dir), "--unzip"], quiet_log="/tmp/stage.log")
        tar_p = bundle_dir / "g19_p4a.tar"
        ext_tmp = pathlib.Path("/tmp/g19_p4a_ext")
        ext_tmp.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_p, "r") as tf:
            tf.extractall(ext_tmp)
        g19_dst.mkdir(parents=True, exist_ok=True)
        fp_dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ext_tmp / "g19" / "tsF3", g19_dst / "tsF3", dirs_exist_ok=True)
        shutil.copytree(ext_tmp / "g19" / "tsP6", g19_dst / "tsP6", dirs_exist_ok=True)
        shutil.copytree(ext_tmp / "fp" / "final", fp_dst / "final", dirs_exist_ok=True)
        shutil.rmtree(bundle_dir, ignore_errors=True)
        shutil.rmtree(ext_tmp, ignore_errors=True)

    for ks in ("ajikajadeh/filseg-infer-ms", "ajikajadeh/filseg-infer-s", "ajikajadeh/filseg-infer-main"):
        dst = K_IN / "notebooks" / ks
        if not dst.exists() or not any(dst.iterdir()):
            dst.mkdir(parents=True, exist_ok=True)
            _run(["kaggle", "kernels", "output", ks, "-p", str(dst)], quiet_log="/tmp/stage.log")

    print(f"Extra Wave-2 inputs staged cleanly under {K_IN} in {(time.time()-t0)/60:.1f}m", flush=True)


def _patch_for_colab(src_script: pathlib.Path, dst_script: pathlib.Path, single_gpu: bool) -> None:
    txt = src_script.read_text()
    # Route ALL /kaggle/input* & /kaggle/working* (including f-strings) into writable Colab /content directories
    txt = txt.replace("/kaggle/input", str(K_IN))
    txt = txt.replace("/kaggle/working", str(K_WORK))
    # Pre-installed in Colab bootstrap cell; avoid failing on nonexistent Kaggle wheel pins
    txt = txt.replace(
        '"pip install -q ultralytics==8.4.160 segmentation-models-pytorch==0.5.0 pycocotools scikit-image \'transformers==5.14.1\' 2>&1 | tail -2"',
        '"echo deps_already_installed"',
    )
    txt = txt.replace(
        '"pip install -q ultralytics==8.4.160 segmentation-models-pytorch==0.5.0 pycocotools scikit-image 2>&1 | tail -2"',
        '"echo deps_already_installed"',
    )
    txt = txt.replace(
        '"pip install -q segmentation-models-pytorch==0.5.0 pycocotools scikit-image 2>&1 | tail -2"',
        '"echo deps_already_installed"',
    )
    if single_gpu:
        txt = txt.replace("CUDA_VISIBLE_DEVICES=str(g)", 'CUDA_VISIBLE_DEVICES="0"')
        txt = txt.replace("CUDA_VISIBLE_DEVICES=str(gpu)", 'CUDA_VISIBLE_DEVICES="0"')
        txt = txt.replace('CUDA_VISIBLE_DEVICES="1"', 'CUDA_VISIBLE_DEVICES="0"')
    dst_script.write_text(txt)


def publish_outputs(note: str) -> None:
    PKG_DIR.mkdir(parents=True, exist_ok=True)
    # Keep ONLY compact deliverables & summaries inside upload dataset
    copied = []
    for pat in ("exp008*", "exp009*", "exp012*"):
        for p in sorted(K_WORK.glob(pat)):
            if p.is_file():
                shutil.copy2(p, PKG_DIR / p.name)
                copied.append(f"{p.name} ({p.stat().st_size/1e6:.1f}MB)")
    meta = {
        "title": "filseg gap a100 out",
        "id": "ajikajadeh/filseg-gap-a100-out",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (PKG_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Uploading {len(copied)} artifacts to ajikajadeh/filseg-gap-a100-out: {copied}", flush=True)
    rc = subprocess.call(["kaggle", "datasets", "version", "-p", str(PKG_DIR), "-m", note])
    if rc != 0:
        _run(["kaggle", "datasets", "create", "-p", str(PKG_DIR)])


def main() -> None:
    import torch

    t0 = time.time()
    n_gpu = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0) if n_gpu else "cpu"
    print(f"GPU: {gpu_name} x{n_gpu} | CPUs={os.cpu_count()}", flush=True)

    K_IN.mkdir(parents=True, exist_ok=True)
    K_WORK.mkdir(parents=True, exist_ok=True)
    code_root = pathlib.Path(__file__).resolve().parent

    # 1. Stage lightweight core (<20s) & run orthogonal EXP-012 Full-Frame Dense Physics FIRST
    stage_core()
    extra_staged = False
    for name, fname, need_extra in (
        ("EXP-012 FullFrame-DensePhysics", "exp012_fullframe_dense_physics.py", False),
        ("EXP-009a Enh-YOLO11l", "exp009a_enh_yolo.py", True),
        ("EXP-008a FRCNN-5F", "exp008a_frcnn5f.py", True),
        ("EXP-009b SegFormer-B3", "exp009b_segformer.py", True),
        ("EXP-008b ConvNeXt-Topo", "exp008b_convnext_topo.py", True),
    ):
        if need_extra and not extra_staged:
            stage_inputs()
            extra_staged = True
        src = code_root / fname
        patched = pathlib.Path("/tmp") / f"colab_{fname}"
        _patch_for_colab(src, patched, single_gpu=(n_gpu <= 1))
        print(f"\n=== Starting {name} on {gpu_name} at {(time.time()-t0)/60:.1f}m ===", flush=True)
        tk = time.time()
        _run([sys.executable, str(patched)])
        print(f"=== Finished {name} in {(time.time()-tk)/60:.1f}m ===", flush=True)
        publish_outputs(f"Colab {gpu_name}: {name} complete")

    print(f"\nALL A100 TASKS COMPLETE in {(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
