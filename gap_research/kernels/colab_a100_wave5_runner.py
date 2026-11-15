"""Wave-5 Colab A100-80GB High-RAM Orchestrator.

Maximizes A100-80GB compute cleanly within wall-clock limits, reusing High-RAM caches
and shared Restormer enhancements, publishing lightweight prediction streams incrementally
after EVERY completed step to `ajikajadeh/filseg-w5-a100-out` (and weights to
`ajikajadeh/filseg-w5-a100-weights`), then cleanly unassigning the Colab runtime at completion.

Priority order:
1. EXP-013a Stage-1 `yolo26x-seg @1536` (`dl_enh_copypaste`, Restormer + Copy-Paste + GONG SSL, patience=5)
2. EXP-013a Stage-1 Transformer `rtdetr-x @1280` (`dl_enh_copypaste`, patience=5)
3. EXP-013c Stage-2 High-RAM `unet-mitb4` & `convnext-edge` (`dl_enh`, patience=5) on [yolo26x_seg, rtdetr_x, ms]
4. EXP-013b Box-Free Transformer `Mask2Former Swin-Base` (`dl_enh_copypaste`, patience=5)
5. Gap-close checks: `EXP-012 dense_mit_b3` Fold 2 + 5-fold infer & `EXP-013a yolo26x-seg baseline` if Kaggle still incomplete.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import time
import traceback


K_IN = pathlib.Path("/content/kaggle_input")
K_WORK = pathlib.Path("/content/kaggle_working")
PKG_OUT = pathlib.Path("/tmp/w5_a100_out_pkg")
PKG_WT = pathlib.Path("/tmp/w5_a100_wt_pkg")
PREWARM_DIR = pathlib.Path("/tmp/w5_prewarm")


def _run(cmd: list[str], check: bool = True, quiet_log: str | None = None, env: dict | None = None, cwd: str | None = None) -> int:
    print("+", " ".join(cmd), flush=True)
    if quiet_log:
        with open(quiet_log, "a") as lf:
            lf.write(f"\n+ {' '.join(cmd)}\n")
            res = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=cwd)
        if check and res.returncode != 0:
            tail = pathlib.Path(quiet_log).read_text(errors="ignore").splitlines()[-50:]
            print("\n".join(tail), flush=True)
            raise RuntimeError(f"Command failed (rc={res.returncode}): {' '.join(cmd)}")
        return res.returncode
    res = subprocess.run(cmd, env=env, cwd=cwd)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed (rc={res.returncode}): {' '.join(cmd)}")
    return res.returncode


def _extract_tar_if_needed(tgz: pathlib.Path, dst_dir: pathlib.Path) -> None:
    if tgz.exists() and not any(dst_dir.rglob("*.npz")):
        dst_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tgz, "r:gz") as tf:
            tf.extractall(dst_dir, filter="data")


def _pack_dir(src_dir: pathlib.Path, out_tar: pathlib.Path) -> float:
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tar, "w:gz") as tf:
        tf.add(src_dir, arcname=src_dir.name)
    mb = out_tar.stat().st_size / (1024 * 1024)
    print(f"[packed] {out_tar} ({mb:.1f} MB)", flush=True)
    return round(mb, 2)


def _patch_script(src: pathlib.Path, dst: pathlib.Path, single_gpu: bool = False) -> pathlib.Path:
    txt = src.read_text()
    txt = txt.replace("/kaggle/input", str(K_IN))
    txt = txt.replace("/kaggle/working", str(K_WORK))
    for pip_s in (
        '"pip install -q ultralytics==8.4.160 segmentation-models-pytorch==0.5.0 pycocotools scikit-image \'transformers==5.14.1\' 2>&1 | tail -2"',
        '"pip install -q ultralytics==8.4.160 segmentation-models-pytorch==0.5.0 pycocotools scikit-image 2>&1 | tail -2"',
        '"pip install -q segmentation-models-pytorch==0.5.0 pycocotools scikit-image 2>&1 | tail -2"',
    ):
        txt = txt.replace(pip_s, '"echo deps_already_installed"')
    if single_gpu:
        txt = txt.replace("fold_groups = [(0, [0, 1, 2]), (1, [3, 4])]", "fold_groups = [(0, [0, 1, 2, 3, 4])]")
        txt = txt.replace("fold_groups = [(0, [0, 1, 2]), (1, [3, 4])] if n_gpu >= 2 else [(0, [0, 1, 2, 3, 4])]", "fold_groups = [(0, [0, 1, 2, 3, 4])]")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(txt)
    return dst


def _publish_dataset(slug: str, title: str, pkg_dir: pathlib.Path, files_to_copy: list[pathlib.Path], note: str) -> None:
    shutil.rmtree(pkg_dir, ignore_errors=True)
    pkg_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for p in files_to_copy:
        if p.exists() and p.is_file():
            shutil.copy2(p, pkg_dir / p.name)
            copied.append(f"{p.name} ({p.stat().st_size / 1e6:.1f}MB)")
    if not copied:
        return
    meta = {"title": title, "id": slug, "licenses": [{"name": "CC0-1.0"}]}
    (pkg_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[publish] Uploading {len(copied)} file(s) -> {slug}: {copied}", flush=True)
    rc = subprocess.call(["kaggle", "datasets", "version", "-p", str(pkg_dir), "-m", note])
    if rc != 0:
        rc_c = subprocess.call(["kaggle", "datasets", "create", "-p", str(pkg_dir)])
        if rc_c != 0:
            print(f"[warn] dataset push failed for {slug} (continuing so local/runtime work is not lost)", flush=True)


def _publish_out(note: str, include_weights: bool = False) -> None:
    out_files: list[pathlib.Path] = []
    wt_files: list[pathlib.Path] = []
    for p in sorted(K_WORK.iterdir()):
        if not p.is_file():
            continue
        if p.name.endswith("_weights.tar.gz"):
            wt_files.append(p)
        elif p.suffix in (".gz", ".json", ".txt") and p.name != "dataset-metadata.json":
            out_files.append(p)
    _publish_dataset("ajikajadeh/filseg-w5-a100-out", "filseg w5 a100 out", PKG_OUT, out_files, note)
    if include_weights and wt_files:
        _publish_dataset("ajikajadeh/filseg-w5-a100-weights", "filseg w5 a100 weights", PKG_WT, wt_files, note)


def _kernel_status(slug: str) -> str:
    try:
        out = subprocess.check_output(["kaggle", "kernels", "status", slug], text=True, stderr=subprocess.STDOUT)
        return out.strip()
    except Exception as exc:
        return f"UNKNOWN: {exc}"


def _render_phys(arg: tuple[str, str, str]) -> None:
    """Module-level worker for Python 3.13 ProcessPoolExecutor pickling."""
    src_p, dst_p, w4s = arg
    if w4s not in sys.path:
        sys.path.insert(0, w4s)
    import cv2
    import solar_dl_enhance as sde
    im = cv2.imread(src_p, cv2.IMREAD_GRAYSCALE)
    if im is not None:
        cv2.imwrite(dst_p, sde.physics_3ch_from_gray(im))


def stage_fast_wave5(repo_root: pathlib.Path, w4_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
    t0 = time.time()
    comp_dir = K_IN / "competitions" / "filament-segmentation-2026"
    ann_hits = [p for p in comp_dir.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json") if "magfilo-extra" not in str(p).lower()] if comp_dir.exists() else []
    if not ann_hits:
        comp_dir.mkdir(parents=True, exist_ok=True)
        _run(["kaggle", "competitions", "download", "-c", "filament-segmentation-2026", "-p", str(comp_dir)], quiet_log="/tmp/stage_w5.log")
        for z in comp_dir.glob("*.zip"):
            _run(["unzip", "-qo", str(z), "-d", str(comp_dir)], quiet_log="/tmp/stage_w5.log")
            z.unlink()
        ann_hits = [p for p in comp_dir.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json") if "magfilo-extra" not in str(p).lower()]
    assert ann_hits, f"MAGFiLO_1.0_Annotations_kaggle2026_train.json not found under {comp_dir}"
    comp_root = ann_hits[0].parents[1]

    code_dst = K_IN / "datasets" / "ajikajadeh" / "filseg-a100-code"
    if not (code_dst / "filseg").exists():
        code_dst.mkdir(parents=True, exist_ok=True)
        _run(["kaggle", "datasets", "download", "-d", "ajikajadeh/filseg-a100-code", "-p", str(code_dst), "--unzip"], quiet_log="/tmp/stage_w5.log")

    ms_dir = K_IN / "notebooks" / "ajikajadeh" / "filseg-infer-ms"
    ms_extract = ms_dir / "ms_cache"
    if not any(ms_extract.rglob("*.npz")):
        ms_dir.mkdir(parents=True, exist_ok=True)
        if not any(ms_dir.glob("*.tar.gz")) and not any(ms_dir.glob("*.tgz")):
            _run(["kaggle", "kernels", "output", "ajikajadeh/filseg-infer-ms", "-p", str(ms_dir)], check=False, quiet_log="/tmp/stage_w5.log")
        for tgz in list(ms_dir.glob("*.tar.gz")) + list(ms_dir.glob("*.tgz")):
            ms_extract.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tgz, "r:gz") as tf:
                tf.extractall(ms_extract, filter="data")
    ms_oof_dir = next((p for p in ms_extract.rglob("oof") if p.is_dir() and any(p.glob("*.npz"))), None)
    ms_root = ms_oof_dir.parent if ms_oof_dir is not None else ms_extract

    sahi_dir = K_IN / "notebooks" / "ajikajadeh" / "filseg-gap-exp007-sahi1024"
    sahi_extract = sahi_dir / "sahi_cache"
    if not any(sahi_extract.rglob("*.npz")):
        sahi_dir.mkdir(parents=True, exist_ok=True)
        if not any(sahi_dir.glob("*.tar.gz")):
            _run(["kaggle", "kernels", "output", "ajikajadeh/filseg-gap-exp007-sahi1024", "-p", str(sahi_dir)], check=False, quiet_log="/tmp/stage_w5.log")
        for tgz in sahi_dir.glob("exp007_sahi_boxes.tar.gz"):
            sahi_extract.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tgz, "r:gz") as tf:
                tf.extractall(sahi_extract, filter="data")
    sahi_oof_dir = next((p for p in sahi_extract.rglob("oof") if p.is_dir() and any(p.glob("*.npz"))), None)
    sahi_root = sahi_oof_dir.parent if sahi_oof_dir is not None else sahi_extract

    K_WORK.mkdir(parents=True, exist_ok=True)
    prev_w5 = pathlib.Path("/tmp/prev_w5_a100_out")
    shutil.rmtree(prev_w5, ignore_errors=True)
    if subprocess.call(["kaggle", "datasets", "download", "-d", "ajikajadeh/filseg-w5-a100-out", "-p", str(prev_w5), "--unzip"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
        for p in prev_w5.iterdir():
            if p.is_file() and p.name not in ("dataset-metadata.json", "w5_a100_error.txt", "w5_a100_status.json") and not (K_WORK / p.name).exists():
                shutil.copy2(p, K_WORK / p.name)
        print(f"Restored existing filseg-w5-a100-out: {[p.name for p in sorted(K_WORK.glob('*.tar.gz'))]}", flush=True)

    # Prebuild shared baseline physics_3ch PNGs once (~25s across 12 CPUs)
    shared_dir = pathlib.Path("/tmp/shared_phys")
    shared_dir.mkdir(parents=True, exist_ok=True)
    for p in (str(repo_root), str(w4_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ["MAGFILO_ROOT"] = str(comp_root)
    from filseg import data as _fdata
    _fdata.REPO = repo_root
    _fdata.ROOT = comp_root
    _fdata.TRAIN_JSON = comp_root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    _fdata.TRAIN_IMAGES = comp_root / "train" / "train_images"
    _fdata.TEST_IMAGES = comp_root / "test" / "test_images"
    _fdata.FOLDS = repo_root / "folds" / "folds.json"
    folds_map = _fdata.folds()
    te_stems = sorted(_fdata.test_stems())
    assert folds_map and te_stems, f"Empty dataset under {comp_root}: folds={len(folds_map)}, test={len(te_stems)}"
    assert _fdata.image_path(sorted(folds_map)[0]).exists(), f"Missing train image under {_fdata.TRAIN_IMAGES}"
    all_stems = sorted(folds_map) + te_stems
    tasks = [(str(_fdata.image_path(s)), str(shared_dir / f"{s}.png"), str(w4_dir)) for s in all_stems if not (shared_dir / f"{s}.png").exists()]
    print(f"[stage] comp_root={comp_root} (train={len(folds_map)}, test={len(te_stems)}, phys_render={len(tasks)})", flush=True)
    if tasks:
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
            list(ex.map(_render_phys, tasks, chunksize=16))

    # Prewarm pretrained backbones sequentially so subprocesses start clean & offline-ready
    PREWARM_DIR.mkdir(parents=True, exist_ok=True)
    old_cwd = os.getcwd()
    os.chdir(PREWARM_DIR)
    import segmentation_models_pytorch as smp
    from transformers import Mask2FormerForUniversalSegmentation
    _ = smp.Unet(encoder_name="mit_b4", encoder_weights="imagenet", in_channels=5, classes=1)
    _ = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-instance", ignore_mismatched_sizes=True)
    os.chdir(old_cwd)
    print(f"[stage+prewarm] Complete in {time.time() - t0:.1f}s (ms_root={ms_root}, sahi_root={sahi_root})", flush=True)
    return ms_root, sahi_root, shared_dir, comp_root


def _sync_det_from_kaggle() -> tuple[pathlib.Path | None, pathlib.Path | None]:
    """Pull completed or partial Stage-1 boxes (yolo26x_seg, rtdetr_x) from Kaggle if available."""
    det_dl = pathlib.Path("/tmp/k_exp013_det")
    if not any(K_WORK.glob("exp013_yolo26x_seg_*_boxes.tar.gz")):
        st_det = _kernel_status("ajikajadeh/filseg-w5-exp013-det")
        print(f"[det-sync] filseg-w5-exp013-det status: {st_det}", flush=True)
        if "COMPLETE" in st_det:
            det_dl.mkdir(parents=True, exist_ok=True)
            _run(["kaggle", "kernels", "output", "ajikajadeh/filseg-w5-exp013-det", "-p", str(det_dl)], check=False, quiet_log="/tmp/stage_w5.log")
            for tgz in det_dl.glob("*.tar.gz"):
                if not (K_WORK / tgz.name).exists():
                    shutil.copy2(tgz, K_WORK / tgz.name)

    y26_tar = next(iter(sorted(K_WORK.glob("exp013_yolo26x_seg_*_boxes.tar.gz"))), None)
    rtd_tar = next(iter(sorted(K_WORK.glob("exp013_rtdetr_x_*_boxes.tar.gz"))), None)
    y26_dir, rtd_dir = None, None
    if y26_tar is not None:
        ydst = pathlib.Path("/tmp/s1_boxes_y26")
        if not any(ydst.rglob("*.npz")):
            _extract_tar_if_needed(y26_tar, ydst)
        yoof = next((p for p in ydst.rglob("oof") if p.is_dir() and any(p.glob("*.npz"))), None)
        if yoof is not None:
            y26_dir = yoof.parent
    if rtd_tar is not None:
        rdst = pathlib.Path("/tmp/s1_boxes_rtd")
        if not any(rdst.rglob("*.npz")):
            _extract_tar_if_needed(rtd_tar, rdst)
        roof = next((p for p in rdst.rglob("oof") if p.is_dir() and any(p.glob("*.npz"))), None)
        if roof is not None:
            rtd_dir = roof.parent
    return y26_dir, rtd_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-wall-min", type=float, default=120.0)
    ap.add_argument("--no-unassign", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    import torch
    n_gpu = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Wave-5 Fast A100 Orchestrator | GPU={gpu_name} x{n_gpu} | CPUs={os.cpu_count()}", flush=True)

    here = pathlib.Path(__file__).resolve().parent
    repo_root = here.parents[1] if (here.parents[1] / "filseg" / "data.py").exists() else pathlib.Path("/content/gap_wave4_code")
    w4_dir = here / "exp013_transformer_a100"
    assert (repo_root / "filseg" / "data.py").exists() and (w4_dir / "exp013c_stage2_seg_sota.py").exists()

    ms_cache, sahi_cache, shared_dir, comp_root = stage_fast_wave5(repo_root, w4_dir)
    env_w5 = {
        **os.environ,
        "PYTHONPATH": f"{repo_root}:{w4_dir}",
        "EXP013_SHARED_PHYS_DIR": str(shared_dir),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }

    status_rec: dict = {"gpu": gpu_name, "n_gpu": n_gpu, "steps": {}, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (K_WORK / "w5_a100_error.txt").unlink(missing_ok=True)

    def _within_budget(min_needed: float) -> bool:
        elapsed = (time.time() - t0) / 60.0
        ok = (elapsed + min_needed) <= args.max_wall_min
        if not ok:
            print(f"[budget] Stopping before step needing {min_needed:.0f}m (elapsed={elapsed:.1f}m / cap={args.max_wall_min:.0f}m)", flush=True)
        return ok

    # Sync updated repo_root into /tmp/filseg_repo_w4 so all scripts use latest filseg & patches cleanly
    shutil.rmtree("/tmp/filseg_repo_w4", ignore_errors=True)
    shutil.copytree(repo_root, "/tmp/filseg_repo_w4")

    try:
        s1_script = _patch_script(w4_dir / "exp013a_stage1_det_sota.py", w4_dir / "_colab_w5_exp013a.py", single_gpu=(n_gpu <= 1))
        s2_script = _patch_script(w4_dir / "exp013c_stage2_seg_sota.py", w4_dir / "_colab_w5_exp013c.py", single_gpu=(n_gpu <= 1))
        m2f_script = _patch_script(w4_dir / "exp013b_mask2former_swin.py", w4_dir / "_colab_w5_exp013b.py", single_gpu=(n_gpu <= 1))

        win_sc = "baseline"

        # [1/4 & 2/4] EXP-013c Stage-2 unet-mitb4 & convnext-edge (baseline, fast 5F train + infer, ~12m each)
        for step_idx, s2_arch in ((1, "unet-mitb4"), (2, "convnext-edge")):
            safe_s2 = s2_arch.replace("-", "_")
            seg_tar = K_WORK / f"exp013c_{safe_s2}_{win_sc}_segs.tar.gz"
            if not seg_tar.exists() and _within_budget(14.0):
                ts = time.time()
                try:
                    box_args: list[str] = []
                    if ms_cache.exists():
                        box_args.extend(["--boxes-root", f"ms={ms_cache}"])
                    if sahi_cache.exists():
                        box_args.extend(["--boxes-root", f"sahi={sahi_cache}"])
                    print(f"\n=== [{step_idx}/4] EXP-013c Stage-2 {s2_arch} ({win_sc}, boxes={[b.split('=')[0] for b in box_args[1::2]]}) ===", flush=True)
                    _run([sys.executable, str(s2_script), "--arch", s2_arch, "--scenario", win_sc, "--epochs", "8", "--patience", "3", *box_args], env=env_w5, cwd=str(PREWARM_DIR))
                    s2_w = pathlib.Path(f"/tmp/exp013c_{s2_arch}_{win_sc}/weights")
                    if s2_w.exists():
                        _pack_dir(s2_w, K_WORK / f"exp013c_{safe_s2}_{win_sc}_weights.tar.gz")
                    status_rec["steps"][f"s2_{safe_s2}"] = {"status": "complete", "min": round((time.time() - ts) / 60.0, 2)}
                    (K_WORK / "w5_a100_status.json").write_text(json.dumps(status_rec, indent=2))
                    _publish_out(f"Wave-5 A100 [{step_idx}/4]: EXP-013c {s2_arch} 5F complete")
                except Exception as e_s2:
                    status_rec["steps"][f"s2_{safe_s2}"] = {"status": "error", "err": str(e_s2)}
                    print(f"[warn] Step {step_idx}/4 {s2_arch} error: {e_s2}", flush=True)

        # [3/4] Complete EXP-013a yolo26x-seg (Folds 0,1,2,3 from Kaggle + Fold 4 + 5F infer on A100, ~14m)
        y26_tar = K_WORK / f"exp013_yolo26x_seg_{win_sc}_boxes.tar.gz"
        if not y26_tar.exists() and _within_budget(15.0):
            ts = time.time()
            try:
                print("\n=== [3/4] EXP-013a yolo26x-seg (Resume Folds 0,1,2,3 + Fold 4 + 5F Infer) ===", flush=True)
                from ultralytics import YOLO
                old_cwd = os.getcwd()
                os.chdir(PREWARM_DIR)
                YOLO("yolo26x-seg.pt")
                os.chdir(old_cwd)
                y26_work = pathlib.Path(f"/tmp/exp013_yolo26x-seg_{win_sc}")
                y26_w = y26_work / "weights"
                y26_w.mkdir(parents=True, exist_ok=True)
                dl_det = pathlib.Path("/tmp/k_exp013_det")
                if not (y26_w / f"det_yolo26x-seg_{win_sc}_f3.pt").exists():
                    dl_det.mkdir(parents=True, exist_ok=True)
                    if not any(dl_det.rglob("exp013_yolo26x*weights.tar.gz")):
                        _run(["kaggle", "kernels", "output", "ajikajadeh/filseg-w5-exp013-det", "-p", str(dl_det)], check=False, quiet_log="/tmp/stage_w5.log")
                    wt_y26 = next(dl_det.rglob("exp013_yolo26x*weights.tar.gz"), None)
                    if wt_y26 is not None:
                        with tarfile.open(wt_y26, "r:gz") as tf:
                            tf.extractall(y26_w, filter="data")
                        if (y26_w / "weights").exists():
                            for sub in (y26_w / "weights").iterdir():
                                shutil.move(str(sub), str(y26_w / sub.name))
                        for fk in range(5):
                            pt = y26_w / f"det_yolo26x-seg_{win_sc}_f{fk}.pt"
                            done_p = y26_w / f"det_yolo26x-seg_{win_sc}_f{fk}.done.json"
                            if pt.exists() and not done_p.exists():
                                done_p.write_text(json.dumps({"arch": "yolo26x-seg", "scenario": win_sc, "fold": fk, "restored": True}))
                        print(f"[y26] Restored folds: {[p.name for p in sorted(y26_w.glob('det_*.pt'))]}", flush=True)
                    shutil.rmtree(dl_det, ignore_errors=True)

                _run([sys.executable, str(s1_script), "--arch", "yolo26x-seg", "--scenario", win_sc, "--epochs", "10", "--patience", "3", "--pseudo-conf", "0.0"], env=env_w5, cwd=str(PREWARM_DIR))
                status_rec["steps"]["yolo26x_seg_base"] = {"status": "complete", "min": round((time.time() - ts) / 60.0, 2)}
                (K_WORK / "w5_a100_status.json").write_text(json.dumps(status_rec, indent=2))
                _publish_out("Wave-5 A100 [3/4]: YOLO26x-seg 5F complete")

                # Run Stage-2 inference on newly generated yolo26x_seg boxes (~2m per arch)
                y26_dir, _ = _sync_det_from_kaggle()
                if y26_dir is not None and y26_dir.exists():
                    for s2_arch in ("unet-mitb4", "convnext-edge"):
                        w_s2 = pathlib.Path(f"/tmp/exp013c_{s2_arch}_{win_sc}/weights")
                        if w_s2.exists() and len(list(w_s2.glob("*.pth"))) == 5:
                            _run([sys.executable, str(s2_script), "--arch", s2_arch, "--scenario", win_sc, "--epochs", "8", "--patience", "3", "--boxes-root", f"yolo26x_seg={y26_dir}"], env=env_w5, cwd=str(PREWARM_DIR))
                    _publish_out("Wave-5 A100 [3b/4]: Stage-2 segs updated with yolo26x_seg boxes")
            except Exception as e_s1:
                status_rec["steps"]["yolo26x_seg_base"] = {"status": "error", "err": str(e_s1)}
                print(f"[warn] Step 3/4 yolo26x-seg error: {e_s1}", flush=True)

        # [4/4] Complete EXP-013b Mask2Former Swin-Base (Folds 0,1,3,4 from Kaggle + Fold 2 3-epoch + 5F infer on A100, ~24m)
        m2f_tar = K_WORK / f"exp013b_{win_sc}_m2f_probs.tar.gz"
        m2f_inst_tar = K_WORK / f"exp013b_{win_sc}_m2f_inst.tar.gz"
        if (not m2f_tar.exists() or not m2f_inst_tar.exists()) and _within_budget(22.0):
            ts = time.time()
            try:
                print("\n=== [4/4] EXP-013b Mask2Former Swin-Base (Resume Folds 0,1,3,4 + Fold 2 3-ep + 5F Infer) ===", flush=True)
                m2f_work = pathlib.Path(f"/tmp/exp013b_m2f_{win_sc}")
                m2f_w = m2f_work / "weights"
                m2f_w.mkdir(parents=True, exist_ok=True)
                dl_m2f = pathlib.Path("/tmp/k_exp013b_m2f")
                if not any(m2f_w.glob("m2f_baseline_f*")):
                    dl_m2f.mkdir(parents=True, exist_ok=True)
                    _run(["kaggle", "kernels", "output", "ajikajadeh/filseg-w5-exp013b-m2f", "-p", str(dl_m2f)], check=False, quiet_log="/tmp/stage_w5.log")
                    wt_m2f = next(dl_m2f.rglob("exp013b_m2f_weights.tar.gz"), None)
                    if wt_m2f is not None:
                        with tarfile.open(wt_m2f, "r:gz") as tf:
                            tf.extractall(m2f_w, filter="data")
                        if (m2f_w / "weights").exists():
                            for sub in (m2f_w / "weights").iterdir():
                                if not (m2f_w / sub.name).exists():
                                    shutil.move(str(sub), str(m2f_w / sub.name))
                        for fk in range(5):
                            ck_d = m2f_w / f"m2f_{win_sc}_f{fk}"
                            done_p = ck_d.with_suffix(".done.json")
                            if ck_d.exists() and any(ck_d.iterdir()) and not done_p.exists():
                                done_p.write_text(json.dumps({"scenario": win_sc, "fold": fk, "restored": True}))
                        print(f"[m2f] Restored folds: {[p.name for p in sorted(m2f_w.glob('m2f_baseline_f*')) if p.is_dir()]}", flush=True)
                    shutil.rmtree(dl_m2f, ignore_errors=True)

                _run([sys.executable, str(m2f_script), "--scenario", win_sc, "--epochs", "3", "--patience", "2"], env=env_w5, cwd=str(PREWARM_DIR))
                status_rec["steps"]["m2f_swin"] = {"status": "complete", "min": round((time.time() - ts) / 60.0, 2)}
                (K_WORK / "w5_a100_status.json").write_text(json.dumps(status_rec, indent=2))
                _publish_out("Wave-5 A100 [4/4]: Mask2Former Swin-Base 5F complete")
            except Exception as e_m2f:
                status_rec["steps"]["m2f_swin"] = {"status": "error", "err": str(e_m2f)}
                print(f"[warn] Step 4/4 m2f error: {e_m2f}", flush=True)

        status_rec["completed_cleanly"] = True
    except BaseException as exc:
        status_rec["completed_cleanly"] = False
        status_rec["error"] = traceback.format_exc()
        (K_WORK / "w5_a100_error.txt").write_text(status_rec["error"])
        print(status_rec["error"], flush=True)
        raise
    finally:
        status_rec["total_min"] = round((time.time() - t0) / 60.0, 2)
        (K_WORK / "w5_a100_status.json").write_text(json.dumps(status_rec, indent=2))
        _publish_out(f"Wave-5 A100 Final ({status_rec['total_min']}m)", include_weights=True)
        print(f"Wave-5 A100 Orchestrator finished in {status_rec['total_min']}m -> status={status_rec.get('completed_cleanly')}", flush=True)
        if not args.no_unassign and os.environ.get("COLAB_NO_UNASSIGN", "0") != "1":
            print("Flush + auto-unassigning Colab runtime to stop CU burn...", flush=True)
            sys.stdout.flush()
            time.sleep(10)
            try:
                from google.colab import runtime  # type: ignore
                runtime.unassign()
            except Exception as u_exc:
                print(f"[warn] runtime.unassign skipped: {u_exc}", flush=True)


if __name__ == "__main__":
    main()
