"""Fast Colab A100 Orchestrator for Wave-4 Transformer + Hybrid SOTA + Leak-Free DL Scenarios (+ EXP-012).

Designed to avoid 162-minute Kaggle file-by-file staging bottlenecks:
- Core stage (~25s): competition zip + filseg-a100-code + filseg-infer-ms (4 files only) + wave4 scripts.
- Runs in priority order, publishing compact archives after EVERY completed step to `ajikajadeh/filseg-gap-a100-out`:
  1. `EXP-012`: Full-Frame Dense Physics UNet++ (`exp012_fullframe_dense_physics.py`) — unlocks dense pixel prior for ALL 14+ streams.
  2. `EXP-013a A/B Scenario Check`: Fold-0 untouched real `va` comparison of `baseline` vs `dl_enh_copypaste` (SolarRestormer + tr-only Copy-Paste) on `RT-DETR-X` & `YOLO12-X`.
  3. `EXP-013a Stage-1 5-Fold`: Pure Transformer `RT-DETR-X` + Hybrid Area-Attention `YOLO12-X` w/ SAHI native-1x tiling.
  4. `EXP-013b Box-Free Transformer`: 5-Fold `Mask2Former Swin-Base` (`1152x1152` 5-tile overlap -> `m2f_probs` + `m2f_inst`).
  5. `EXP-013c Stage-2 Hybrid Transformer Segmentation`: 5-Fold `SegFormer MiT-B4` & `ConvNeXt-UNet` (w/ Box-Guided 4th ch + ECCV 2024 Skeleton Recall Loss) refining `ms`, `rtdetr-x`, and `yolo12x` boxes!
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import time


K_IN = pathlib.Path("/content/kaggle_input")
K_WORK = pathlib.Path("/content/kaggle_working")
PKG_DIR = pathlib.Path("/tmp/a100_out_pkg")


def _run(cmd: list[str], check: bool = True, quiet_log: str | None = None, env: dict | None = None) -> int:
    print("+", " ".join(cmd), flush=True)
    if quiet_log:
        with open(quiet_log, "a") as lf:
            lf.write(f"\n+ {' '.join(cmd)}\n")
            res = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        if check and res.returncode != 0:
            tail = pathlib.Path(quiet_log).read_text(errors="ignore").splitlines()[-40:]
            print("\n".join(tail), flush=True)
            raise RuntimeError(f"Command failed (rc={res.returncode}): {' '.join(cmd)}")
        return res.returncode
    res = subprocess.run(cmd, env=env)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed (rc={res.returncode}): {' '.join(cmd)}")
    return res.returncode


def stage_fast_wave4() -> tuple[pathlib.Path, pathlib.Path]:
    t0 = time.time()
    comp_dir = K_IN / "competitions" / "filament-segmentation-2026"
    if not (comp_dir / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").exists():
        comp_dir.mkdir(parents=True, exist_ok=True)
        _run(["kaggle", "competitions", "download", "-c", "filament-segmentation-2026", "-p", str(comp_dir)], quiet_log="/tmp/stage_w4.log")
        for z in comp_dir.glob("*.zip"):
            _run(["unzip", "-qo", str(z), "-d", str(comp_dir)], quiet_log="/tmp/stage_w4.log")
            z.unlink()

    code_dst = K_IN / "datasets" / "ajikajadeh" / "filseg-a100-code"
    if not (code_dst / "filseg").exists():
        code_dst.mkdir(parents=True, exist_ok=True)
        _run(["kaggle", "datasets", "download", "-d", "ajikajadeh/filseg-a100-code", "-p", str(code_dst), "--unzip"], quiet_log="/tmp/stage_w4.log")

    # Stage legal unlabeled GONG archive (~3000 frames; filtered strictly per fold to exclude test +-1h/dates & va dates)
    gong_dst = K_IN / "datasets" / "ajikajadeh" / "filseg-gong-unlabeled"
    if not (gong_dst / "images").exists():
        gong_dst.mkdir(parents=True, exist_ok=True)
        _run(["kaggle", "datasets", "download", "-d", "ajikajadeh/filseg-gong-unlabeled", "-p", str(gong_dst), "--unzip"], check=False, quiet_log="/tmp/stage_w4.log")

    # Resume any previously published Wave-4/5 archives so interrupted sessions never lose progress
    prev_out = pathlib.Path("/tmp/prev_a100_out")
    if not any(K_WORK.glob("exp013*")):
        prev_out.mkdir(parents=True, exist_ok=True)
        rc_prev = _run(["kaggle", "datasets", "download", "-d", "ajikajadeh/filseg-gap-a100-out", "-p", str(prev_out), "--unzip"], check=False, quiet_log="/tmp/stage_w4.log")
        if rc_prev == 0:
            for p in prev_out.iterdir():
                if p.is_file() and p.name != "dataset-metadata.json" and not (K_WORK / p.name).exists():
                    shutil.copy2(p, K_WORK / p.name)
            print(f"Restored prior outputs from filseg-gap-a100-out: {[p.name for p in K_WORK.glob('*.tar.gz')]}", flush=True)

    # Fast ms Stage-1/2 boxes cache (only 4 files in filseg-infer-ms!) for direct Stage-2 comparison
    ms_dir = K_IN / "notebooks" / "ajikajadeh" / "filseg-infer-ms"
    ms_extract = pathlib.Path("/tmp/ms_cache_extract")
    if not (ms_extract / "cache" / "ms" / "oof").exists():
        ms_dir.mkdir(parents=True, exist_ok=True)
        _run(["kaggle", "kernels", "output", "ajikajadeh/filseg-infer-ms", "-p", str(ms_dir)], check=False, quiet_log="/tmp/stage_w4.log")
        tgz = ms_dir / "cache_ms.tgz"
        if tgz.exists():
            ms_extract.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tgz, "r:gz") as tf:
                tf.extractall(ms_extract)

    print(f"Fast Wave-5 staging complete in {(time.time()-t0):.1f}s", flush=True)
    return ms_extract / "cache" / "ms", gong_dst


def publish_outputs(note: str) -> None:
    PKG_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for pat in ("exp008*", "exp009*", "exp012*", "exp013*", "*_val.json", "*.json"):
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
        txt = txt.replace("CUDA_VISIBLE_DEVICES=str(g)", 'CUDA_VISIBLE_DEVICES="0"')
        txt = txt.replace("CUDA_VISIBLE_DEVICES=str(gpu)", 'CUDA_VISIBLE_DEVICES="0"')
        txt = txt.replace('CUDA_VISIBLE_DEVICES="1"', 'CUDA_VISIBLE_DEVICES="0"')
    dst.write_text(txt)
    return dst


def _extract_tar_if_needed(tgz: pathlib.Path, extract_to: pathlib.Path) -> pathlib.Path | None:
    if tgz.exists():
        extract_to.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tgz, "r:gz") as tf:
            tf.extractall(extract_to)
        return extract_to
    return None


def main() -> None:
    import torch

    t0 = time.time()
    n_gpu = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0) if n_gpu else "cpu"
    print(f"Wave-5 A100 Orchestrator | GPU={gpu_name} x{n_gpu} | CPUs={os.cpu_count()}", flush=True)

    K_IN.mkdir(parents=True, exist_ok=True)
    K_WORK.mkdir(parents=True, exist_ok=True)
    ms_cache, gong_dst = stage_fast_wave4()

    here = pathlib.Path(__file__).resolve().parent
    w4_dir = here / "exp013_transformer_a100" if (here / "exp013_transformer_a100").exists() else here
    repo_root = here.parents[1] if (here.parents[1] / "filseg").exists() else (K_IN / "datasets" / "ajikajadeh" / "filseg-a100-code")
    exp012_src = here / "exp012_fullframe_dense_physics" / "exp012_fullframe_dense_physics.py"
    if not exp012_src.exists():
        exp012_src = here / "exp012_fullframe_dense_physics.py"
    assert exp012_src.exists(), f"Missing {exp012_src}"
    # Ensure Kaggle-style worker fallback always uses freshest filseg repo cleanly
    shutil.rmtree("/tmp/filseg_repo_w4", ignore_errors=True)
    if (repo_root / "filseg").exists():
        shutil.copytree(repo_root, K_WORK / "filseg_repo", dirs_exist_ok=True)

    gong_args = ["--gong-dir", str(gong_dst)] if (gong_dst / "images").exists() else []
    s1_script = _patch_script(w4_dir / "exp013a_stage1_det_sota.py", pathlib.Path("/tmp/colab_exp013a.py"), single_gpu=(n_gpu <= 1))
    env_w4 = dict(os.environ, PYTHONPATH=f"{repo_root}:{w4_dir}")

    # 1. EXP-013a Fold-0 Leak-Free Scenario A/B (Baseline CLV-CLAHE vs DL-Restormer + Legal GONG SSL + Rotated CopyPaste)
    ab_marker = K_WORK / "wave5_ab_f0_summary.json"
    if ab_marker.exists():
        win_sc = json.loads(ab_marker.read_text()).get("win_sc", "dl_enh_copypaste")
        print(f"Resuming with saved Fold-0 A/B winning scenario: {win_sc}", flush=True)
    else:
        print("\n=== [1/7] EXP-013a Fold-0 Leak-Free Scenario A/B (YOLO26x-seg, patience=5) ===", flush=True)
        _run([sys.executable, str(s1_script), "--arch", "yolo26x-seg", "--smoke-fold0-ab", "--epochs", "24", "--patience", "5"] + gong_args, env=env_w4)
        ab_scores = {}
        for sc in ("baseline", "dl_enh_copypaste"):
            vp = pathlib.Path(f"/tmp/exp013_yolo26x-seg_{sc}/weights/det_yolo26x-seg_{sc}_f0_val.json")
            if vp.exists():
                d = json.loads(vp.read_text())
                ab_scores[sc] = d.get("va_orig_phys", {}).get("metrics/mAP50(B)", -1.0)
        shutil.copytree("/tmp/exp013_yolo26x-seg_baseline/weights", K_WORK / "exp013_ab_f0_baseline", dirs_exist_ok=True)
        shutil.copytree("/tmp/exp013_yolo26x-seg_dl_enh_copypaste/weights", K_WORK / "exp013_ab_f0_dl", dirs_exist_ok=True)
        win_sc = "dl_enh_copypaste" if ab_scores.get("dl_enh_copypaste", -1.0) >= ab_scores.get("baseline", 0.0) else "baseline"
        ab_marker.write_text(json.dumps({"ab_scores": ab_scores, "win_sc": win_sc}, indent=2))
        print(f"Fold-0 untouched original va mAP50 scores={ab_scores} -> selected 5-fold scenario={win_sc}", flush=True)
        publish_outputs(f"Wave-5 [1/7]: Fold-0 A/B complete (winner={win_sc})")

    # 2. EXP-013a Stage-1 5-Fold YOLO26x-seg @1536 (polygon rotated degrees=180, patience=5 -> boxes + s1masks)
    y26_tar = K_WORK / f"exp013_yolo26x_seg_{win_sc}_boxes.tar.gz"
    if not y26_tar.exists():
        print(f"\n=== [2/7] EXP-013a 5-Fold YOLO26x-seg ({win_sc}, patience=5) ===", flush=True)
        _run([sys.executable, str(s1_script), "--arch", "yolo26x-seg", "--scenario", win_sc, "--epochs", "50", "--patience", "5"] + gong_args, env=env_w4)
        publish_outputs(f"Wave-5 [2/7]: EXP-013a YOLO26x-seg ({win_sc}) 5-fold complete")

    # 3. EXP-013a Stage-1 5-Fold Pure Transformer RT-DETR-X @1280 (patience=5)
    rtd_tar = K_WORK / f"exp013_rtdetr_x_{win_sc}_boxes.tar.gz"
    if not rtd_tar.exists():
        print(f"\n=== [3/7] EXP-013a 5-Fold RT-DETR-X ({win_sc}, patience=5) ===", flush=True)
        _run([sys.executable, str(s1_script), "--arch", "rtdetr-x", "--scenario", win_sc, "--epochs", "45", "--patience", "5"] + gong_args, env=env_w4)
        publish_outputs(f"Wave-5 [3/7]: EXP-013a RT-DETR-X ({win_sc}) 5-fold complete")

    # Ensure Stage-1 boxes directories are present even if loaded from prior published archive
    y26_dir = pathlib.Path(f"/tmp/exp013_yolo26x-seg_{win_sc}/boxes_yolo26x-seg_{win_sc}")
    if not y26_dir.exists() and y26_tar.exists():
        _extract_tar_if_needed(y26_tar, pathlib.Path(f"/tmp/exp013_yolo26x-seg_{win_sc}"))
    rtd_dir = pathlib.Path(f"/tmp/exp013_rtdetr-x_{win_sc}/boxes_rtdetr-x_{win_sc}")
    if not rtd_dir.exists() and rtd_tar.exists():
        _extract_tar_if_needed(rtd_tar, pathlib.Path(f"/tmp/exp013_rtdetr-x_{win_sc}"))

    # 4. EXP-013c Stage-2 Full-Res 1/1 Skip 5-ch (unet-mitb4 Transformer + convnext-edge EdgeGuidedMHSA w/ Empty-Target FP Suppression)
    s2_script = _patch_script(w4_dir / "exp013c_stage2_seg_sota.py", pathlib.Path("/tmp/colab_exp013c.py"), single_gpu=(n_gpu <= 1))
    box_args: list[str] = []
    if y26_dir.exists():
        box_args.extend(["--boxes-root", f"yolo26x_seg={y26_dir}"])
    if rtd_dir.exists():
        box_args.extend(["--boxes-root", f"rtdetr_x={rtd_dir}"])
    if ms_cache.exists():
        box_args.extend(["--boxes-root", f"ms={ms_cache}"])

    s2_sc = "dl_enh" if win_sc == "dl_enh_copypaste" else "baseline"
    for step_tag, s2_arch in (("4a/7", "unet-mitb4"), ("4b/7", "convnext-edge")):
        safe_s2 = s2_arch.replace("-", "_")
        if not (K_WORK / f"exp013c_{safe_s2}_{s2_sc}_segs.tar.gz").exists():
            print(f"\n=== [{step_tag}] EXP-013c 5-Fold Stage-2 {s2_arch} ({s2_sc}, patience=5) ===", flush=True)
            _run([sys.executable, str(s2_script), "--arch", s2_arch, "--scenario", s2_sc, "--epochs", "24", "--patience", "5", *box_args], env=env_w4)
            publish_outputs(f"Wave-5 [{step_tag}]: EXP-013c Stage-2 {s2_arch} ({s2_sc}) complete")

    # 5. EXP-013b 5-Fold Mask2Former Swin-Base Box-Free Transformer (5-fold test consensus, patience=5)
    m2f_sc = "dl_enh_copypaste" if win_sc == "dl_enh_copypaste" else "baseline"
    if not (K_WORK / f"exp013b_{m2f_sc}_m2f_probs.tar.gz").exists():
        print(f"\n=== [5/7] EXP-013b 5-Fold Mask2Former Swin-Base ({m2f_sc}, patience=5) ===", flush=True)
        m2f_script = _patch_script(w4_dir / "exp013b_mask2former_swin.py", pathlib.Path("/tmp/colab_exp013b.py"), single_gpu=(n_gpu <= 1))
        _run([sys.executable, str(m2f_script), "--scenario", m2f_sc, "--epochs", "24", "--patience", "5"], env=env_w4)
        publish_outputs(f"Wave-5 [5/7]: EXP-013b Mask2Former Swin-Base ({m2f_sc}) complete")

    # 6. EXP-012 Full-Frame Dense Physics (mit_b3 first for rapid checkpoint, then cnx_upp+mit_b3 blend, patience=5)
    p12 = _patch_script(exp012_src, pathlib.Path("/tmp/colab_exp012.py"), single_gpu=(n_gpu <= 1))
    if not (K_WORK / "exp012_dense_probs.tar.gz").exists():
        print("\n=== [6a/7] EXP-012 Full-Frame Dense Physics mit_b3 (patience=5) ===", flush=True)
        _run([sys.executable, str(p12), "--arches", "mit_b3", "--epochs", "30", "--patience", "5"])
        publish_outputs("Wave-5 [6a/7]: EXP-012 mit_b3 dense probs complete")

    done_marker_cnx = K_WORK / "exp012_cnx_upp_complete.json"
    if not done_marker_cnx.exists():
        print("\n=== [6b/7] EXP-012 Full-Frame Dense Physics cnx_upp+mit_b3 blend (patience=5) ===", flush=True)
        _run([sys.executable, str(p12), "--arches", "cnx_upp", "mit_b3", "--epochs", "30", "--patience", "5"])
        done_marker_cnx.write_text(json.dumps({"complete": True}))
        publish_outputs("Wave-5 [6b/7]: EXP-012 full cnx_upp+mit_b3 blend complete")

    # 7. Legal External GONG Semi-Supervised Pseudo-Label Refinement for YOLO26x-seg (regression-gated per fold)
    gong_tar = K_WORK / f"exp013_yolo26x_seg_{win_sc}_gong_ssl_boxes.tar.gz"
    base_w = pathlib.Path(f"/tmp/exp013_yolo26x-seg_{win_sc}/weights")
    if gong_args and base_w.exists() and not gong_tar.exists():
        print("\n=== [7/7] EXP-013a Legal GONG SSL Pseudo-Labeling YOLO26x-seg (regression-gated, patience=5) ===", flush=True)
        rc_ssl = _run(
            [sys.executable, str(s1_script), "--arch", "yolo26x-seg", "--scenario", win_sc,
             "--epochs", "30", "--patience", "5", "--pseudo-conf", "0.38",
             "--out-tag", "gong_ssl", "--init-weights-dir", str(base_w), *gong_args],
            check=False,
            env=env_w4,
        )
        if rc_ssl == 0:
            publish_outputs("Wave-5 [7/7]: EXP-013a GONG SSL complete")

    (K_WORK / "wave5_complete.json").write_text(json.dumps({"elapsed_min": round((time.time() - t0) / 60, 2), "win_sc": win_sc}, indent=2))
    publish_outputs("Wave-5 COMPLETE: All transformer/hybrid stages finished")
    print(f"\nALL WAVE-5 A100 TASKS COMPLETE in {(time.time()-t0)/60:.1f}m!", flush=True)

    if os.environ.get("FILSEG_KEEP_ALIVE", "0") != "1":
        try:
            from google.colab import runtime  # type: ignore

            print("Disconnecting Colab A100 runtime in 5s to save compute units...", flush=True)
            time.sleep(5)
            runtime.unassign()
        except Exception as exc:
            print(f"Note: auto-unassign unavailable ({exc}); please disconnect runtime manually.", flush=True)


if __name__ == "__main__":
    main()
