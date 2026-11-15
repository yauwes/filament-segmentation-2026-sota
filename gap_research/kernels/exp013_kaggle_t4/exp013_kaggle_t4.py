"""Kaggle 2 x T4 GPU Runner for EXP-013 Stage-1/2 SOTA Transformers + Hybrids + Legal GONG SSL (patience=5).

Order of execution (checkpointed to /kaggle/working after each completed phase):
  Phase 1 (~35-45m): EXP-013c Full-Res 1/1 Stage-2 (`unet-mitb4` & `EdgeAttNet convnext-edge`, 5 folds, patience=5)
                     refining existing `ms` and `sahi` Stage-1 boxes -> `.tar.gz` outputs.
  Phase 2 (~25m):    EXP-013a Fold-0 Leak-Free A/B (`yolo26x-seg @1536`, `baseline` vs `dl_enh_copypaste` on GPU0/GPU1)
                     with legal `filseg-gong-unlabeled` Restormer + rotated Copy-Paste -> `wave5_ab_f0_summary.json`.
  Phase 3 (~1.5-2h): EXP-013a 5-Fold `yolo26x-seg @1536` (`degrees=180`, winning scenario) -> boxes + s1masks `.tar.gz`,
                     followed by Stage-2 `unet-mitb4` + `convnext-edge` inference on `yolo26x_seg` boxes!
  Phase 4 (if time): EXP-013a 5-Fold Pure Transformer `RT-DETR-X @1280` (time-guarded) + Stage-2 inference.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

MAX_WALL_MIN = 470.0


def _render_shared_one(task: tuple[str, str, str, str]) -> str:
    src_p, dst_p, repo_s, w4_s = task
    if os.path.exists(dst_p):
        return dst_p
    for p in (repo_s, w4_s):
        if p not in sys.path:
            sys.path.insert(0, p)
    import cv2
    import solar_dl_enhance as sde

    im = cv2.imread(src_p, cv2.IMREAD_GRAYSCALE)
    bgr = sde.physics_3ch_from_gray(im)
    cv2.imwrite(dst_p, bgr)
    return dst_p


def _prebuild_shared_phys(repo_dir: Path, w4_dir: Path, data_root: Path, shared_dir: Path) -> Path:
    t0 = time.time()
    shared_dir.mkdir(parents=True, exist_ok=True)
    for p in (str(repo_dir), str(w4_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from filseg import data

    os.environ["MAGFILO_ROOT"] = str(data_root)
    data.REPO = repo_dir
    data.ROOT = data_root
    data.TRAIN_IMAGES = data_root / "train" / "train_images"
    data.TEST_IMAGES = data_root / "test" / "test_images"
    data.TRAIN_JSON = data_root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    folds = data.folds()
    all_stems = sorted(folds) + sorted(data.test_stems())
    tasks = []
    for s in all_stems:
        src = data.image_path(s)
        assert src.exists(), f"Missing image {src}"
        dst = shared_dir / f"{s}.png"
        if not dst.exists():
            tasks.append((str(src), str(dst), str(repo_dir), str(w4_dir)))
    if tasks:
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
            list(ex.map(_render_shared_one, tasks, chunksize=8))
    print(f"[shared_phys] Prepared {len(all_stems)} baseline physics_3ch PNGs in {time.time() - t0:.1f}s", flush=True)
    return shared_dir


def _sh(cmd: list[str] | str, check: bool = True, env: dict | None = None, cwd: str | None = None) -> int:
    shell = isinstance(cmd, str)
    print("+", cmd if shell else " ".join(cmd), flush=True)
    res = subprocess.run(cmd, shell=shell, env=env, cwd=cwd)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed rc={res.returncode}: {cmd}")
    return res.returncode


def _find_comp_root() -> Path:
    for p in Path("/kaggle/input").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"):
        if "magfilo-extra" not in str(p).lower():
            return p.parents[1]
    raise FileNotFoundError("MAGFiLO_1.0_Kaggle_2026 not found under /kaggle/input")


def _find_code_root() -> Path:
    for p in Path("/kaggle/input").rglob("exp013a_stage1_det_sota.py"):
        for cand in p.parents:
            if (cand / "filseg").is_dir() and (cand / "folds" / "folds.json").is_file():
                return cand
    raise FileNotFoundError("filseg-gap-wave4-code not found under /kaggle/input")


def _find_gong_dir() -> Path | None:
    for p in Path("/kaggle/input").rglob("filseg-gong-unlabeled"):
        if p.is_dir():
            return p
    for p in Path("/kaggle/input").glob("*gong*"):
        if p.is_dir():
            return p
    return None


def _pack(src_dir: Path, dst_tgz: Path) -> None:
    tmp = dst_tgz.with_suffix(".tmp.gz")
    with tarfile.open(tmp, "w:gz", compresslevel=1) as tf:
        tf.add(src_dir, arcname=src_dir.name)
    tmp.replace(dst_tgz)
    print(f"Packed {dst_tgz.name}: {dst_tgz.stat().st_size / 1e6:.1f} MB", flush=True)


def _patch_scripts_for_kaggle(w4_dir: Path) -> None:
    for fname in ("exp013a_stage1_det_sota.py", "exp013c_stage2_seg_sota.py", "solar_dl_enhance.py"):
        p = w4_dir / fname
        txt = p.read_text()
        txt = txt.replace("num_workers=4, drop_last=True, pin_memory=True", "num_workers=2, drop_last=True, pin_memory=False")
        txt = txt.replace("num_workers=2, pin_memory=True", "num_workers=2, pin_memory=False")
        txt = txt.replace("workers=4,", "workers=2,")
        txt = txt.replace("for b_try in (batch, max(2, batch // 2), 2):", "for b_try in (batch, max(2, batch // 2), 2, 1):")
        txt = txt.replace('if "out of memory" in str(exc).lower() and b_try > 2:', 'if "out of memory" in str(exc).lower() and b_try > 1:')
        p.write_text(txt)


def _stage_box_caches(boxes_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in Path("/kaggle/input").rglob("cache_ms.tgz"):
        dst = boxes_root / "ms_extract"
        dst.mkdir(parents=True, exist_ok=True)
        with tarfile.open(p, "r:gz") as tf:
            tf.extractall(dst)
        if (dst / "cache" / "ms" / "oof").exists():
            out["ms"] = dst / "cache" / "ms"
            break
    for p in Path("/kaggle/input").rglob("exp007_sahi_boxes.tar.gz"):
        dst = boxes_root / "sahi_extract"
        dst.mkdir(parents=True, exist_ok=True)
        with tarfile.open(p, "r:gz") as tf:
            tf.extractall(dst)
        for sub in dst.rglob("oof"):
            if any(sub.glob("*.npz")):
                out["sahi"] = sub.parent
                break
        if "sahi" in out:
            break
    print(f"Staged Stage-1 box caches: { {k: str(v) for k, v in out.items()} }", flush=True)
    return out


def _index_boxes(named_roots: dict[str, Path], idx_path: Path) -> Path:
    idx: dict[str, dict[str, dict[str, str]]] = {}
    for sname, rp in named_roots.items():
        if not rp.exists():
            continue
        idx[sname] = {}
        for sp in ("oof", "val", "test"):
            sub = rp / sp
            if sub.exists():
                idx[sname][sp] = {f.stem: str(f) for f in sorted(sub.glob("*.npz"))}
    idx_path.write_text(json.dumps(idx))
    return idx_path


def main() -> None:
    t0 = time.time()
    scratch = Path("/tmp/w5_scratch")
    scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)

    _sh("nvidia-smi -L; nproc; df -h /kaggle/working /tmp /dev/shm")
    res_pip = subprocess.run(
        "pip install -q ultralytics==8.4.160 segmentation-models-pytorch==0.5.0 pycocotools scikit-image 'transformers==5.14.1' 2>&1 | tail -5",
        shell=True,
        capture_output=True,
        text=True,
    )
    print(res_pip.stdout, flush=True)

    import torch
    import segmentation_models_pytorch as smp
    from ultralytics import RTDETR, YOLO

    n_gpu = torch.cuda.device_count()
    print(f"Detected GPUs: {n_gpu}", flush=True)

    data_root = _find_comp_root()
    code_src = _find_code_root()
    gong_dir = _find_gong_dir()
    print(f"GONG dir: {gong_dir}", flush=True)

    repo_dir = Path("/tmp/w5_repo")
    shutil.rmtree(repo_dir, ignore_errors=True)
    shutil.copytree(code_src, repo_dir, dirs_exist_ok=True)
    w4_dir = repo_dir / "gap_research" / "kernels" / "exp013_transformer_a100"
    _patch_scripts_for_kaggle(w4_dir)

    # Pre-download pretrained backbones ONCE on main thread to avoid dual-GPU concurrent download race
    for enc in ("mit_b4", "tu-convnext_small.fb_in22k"):
        m = smp.Unet(encoder_name=enc, encoder_weights="imagenet", in_channels=5, classes=1)
        del m
    for wname, cls in (("yolo26x-seg.pt", YOLO), ("rtdetr-x.pt", RTDETR)):
        try:
            mm = cls(wname)
            del mm
            print(f"[prewarm OK] {wname}", flush=True)
        except Exception as exc:
            print(f"[prewarm WARN] {wname}: {exc}", flush=True)

    py = sys.executable
    kw = Path("/kaggle/working")
    kw.mkdir(parents=True, exist_ok=True)
    shared_phys = _prebuild_shared_phys(repo_dir, w4_dir, data_root, Path("/tmp/shared_phys"))
    env_base = dict(
        os.environ,
        MAGFILO_ROOT=str(data_root),
        PYTHONPATH=f"{repo_dir}:{w4_dir}",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        EXP013_SHARED_PHYS_DIR=str(shared_phys),
    )

    s2_py = w4_dir / "exp013c_stage2_seg_sota.py"
    s1_py = w4_dir / "exp013a_stage1_det_sota.py"

    staged_boxes = _stage_box_caches(Path("/tmp/w5_boxes"))

    # =========================================================================
    # PHASE 1: EXP-013c Stage-2 Full-Res 1/1 Models (unet-mitb4 & convnext-edge)
    # =========================================================================
    fold_groups = [(0, [0, 1, 2]), (1, [3, 4])] if n_gpu >= 2 else [(0, [0, 1, 2, 3, 4])]
    s2_trained: list[tuple[str, Path, Path]] = []

    def _run_s2_infer(arch: str, work_dir: Path, out_dir: Path, boxes_dict: dict[str, Path], out_tar: Path) -> None:
        if not boxes_dict:
            return
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        idx_p = _index_boxes(boxes_dict, work_dir / f"idx_{out_tar.stem}.json")
        ps = [
            subprocess.Popen(
                [py, str(s2_py), "--worker", "infer", "--repo", str(repo_dir), "--data-root", str(data_root),
                 "--work-dir", str(work_dir), "--arch", arch, "--scenario", "baseline",
                 "--boxes-index", str(idx_p), "--out-dir", str(out_dir), "--part", str(p)],
                env=dict(env_base, CUDA_VISIBLE_DEVICES=str(p if n_gpu >= 2 else 0)),
                cwd=str(scratch),
            )
            for p in (0, 1)
        ]
        assert all(p.wait() == 0 for p in ps), f"{arch} infer failed"
        _pack(out_dir, out_tar)

    for s2_arch in ("unet-mitb4", "convnext-edge"):
        if (time.time() - t0) / 60.0 > MAX_WALL_MIN - 30:
            break
        wdir = Path(f"/tmp/exp013c_{s2_arch}_baseline")
        odir = Path(f"/tmp/exp013c_out_{s2_arch}_baseline")
        wdir.mkdir(parents=True, exist_ok=True)
        errs: list[BaseException] = []

        def _grp_s2(gpu: int, fks: list[int], a=s2_arch, wd=wdir) -> None:
            try:
                for fk in fks:
                    _sh(
                        [py, str(s2_py), "--worker", "train_fold", "--repo", str(repo_dir), "--data-root", str(data_root),
                         "--work-dir", str(wd), "--arch", a, "--scenario", "baseline", "--fold", str(fk),
                         "--epochs", "24", "--patience", "5"],
                        env=dict(env_base, CUDA_VISIBLE_DEVICES=str(gpu)),
                        cwd=str(scratch),
                    )
            except BaseException as exc:
                errs.append(exc)

        ts = [threading.Thread(target=_grp_s2, args=(g, fks)) for g, fks in fold_groups]
        for th in ts:
            th.start()
        for th in ts:
            th.join()
        assert not errs, errs

        safe_s2 = s2_arch.replace("-", "_")
        _run_s2_infer(s2_arch, wdir, odir, staged_boxes, kw / f"exp013c_{safe_s2}_baseline_segs.tar.gz")
        s2_trained.append((s2_arch, wdir, odir))

    # =========================================================================
    # PHASE 2: EXP-013a Fold-0 Leak-Free Scenario A/B (YOLO26x-seg @1536)
    # =========================================================================
    gong_args = ["--gong-dir", str(gong_dir)] if gong_dir is not None else []
    win_sc = "dl_enh_copypaste"
    if (time.time() - t0) / 60.0 < MAX_WALL_MIN - 60:
        ab_errs: list[BaseException] = []

        def _ab_run(gpu: int, sc: str) -> None:
            try:
                g_args = gong_args if sc == "dl_enh_copypaste" else []
                _sh(
                    [py, str(s1_py), "--worker", "train_fold", "--repo", str(repo_dir), "--data-root", str(data_root),
                     "--work-dir", f"/tmp/exp013_yolo26x-seg_{sc}", "--arch", "yolo26x-seg",
                     "--scenario", sc, "--fold", "0", "--epochs", "32", "--patience", "5"] + g_args,
                    env=dict(env_base, CUDA_VISIBLE_DEVICES=str(gpu)),
                    cwd=str(scratch),
                )
            except BaseException as exc:
                ab_errs.append(exc)

        t_ab0 = threading.Thread(target=_ab_run, args=(0, "baseline"))
        t_ab1 = threading.Thread(target=_ab_run, args=(1 if n_gpu >= 2 else 0, "dl_enh_copypaste"))
        if n_gpu >= 2:
            t_ab0.start(); t_ab1.start(); t_ab0.join(); t_ab1.join()
        else:
            t_ab0.start(); t_ab0.join(); t_ab1.start(); t_ab1.join()
        assert not ab_errs, ab_errs

        ab_scores = {}
        for sc in ("baseline", "dl_enh_copypaste"):
            vp = Path(f"/tmp/exp013_yolo26x-seg_{sc}/weights/det_yolo26x-seg_{sc}_f0_val.json")
            if vp.exists():
                ab_scores[sc] = json.loads(vp.read_text()).get("va_orig_phys", {}).get("metrics/mAP50(B)", -1.0)
        win_sc = "dl_enh_copypaste" if ab_scores.get("dl_enh_copypaste", -1.0) >= ab_scores.get("baseline", 0.0) else "baseline"
        lose_sc = "baseline" if win_sc == "dl_enh_copypaste" else "dl_enh_copypaste"
        shutil.rmtree(f"/tmp/exp013_yolo26x-seg_{lose_sc}/det_data", ignore_errors=True)
        (kw / "wave5_ab_f0_summary.json").write_text(json.dumps({"ab_scores": ab_scores, "win_sc": win_sc}, indent=2))
        print(f"[Phase 2 A/B] scores={ab_scores} -> win_sc={win_sc}", flush=True)

    # =========================================================================
    # PHASE 3: EXP-013a Full 5-Fold YOLO26x-seg @1536 + Stage-2 Refinement
    # =========================================================================
    if (time.time() - t0) / 60.0 < MAX_WALL_MIN - 60:
        y26_wdir = Path(f"/tmp/exp013_yolo26x-seg_{win_sc}")
        g_args = gong_args if win_sc == "dl_enh_copypaste" else []
        y26_errs: list[BaseException] = []

        def _grp_y26(gpu: int, fks: list[int]) -> None:
            try:
                for fk in fks:
                    if (time.time() - t0) / 60.0 > MAX_WALL_MIN - 25:
                        print(f"[yolo26x-seg f{fk}] wall-clock guard hit", flush=True)
                        return
                    _sh(
                        [py, str(s1_py), "--worker", "train_fold", "--repo", str(repo_dir), "--data-root", str(data_root),
                         "--work-dir", str(y26_wdir), "--arch", "yolo26x-seg",
                         "--scenario", win_sc, "--fold", str(fk), "--epochs", "32", "--patience", "5"] + g_args,
                        env=dict(env_base, CUDA_VISIBLE_DEVICES=str(gpu)),
                        cwd=str(scratch),
                    )
                    shutil.rmtree(y26_wdir / "det_data" / f"yolo26x-seg_{win_sc}_f{fk}", ignore_errors=True)
            except BaseException as exc:
                y26_errs.append(exc)

        ts = [threading.Thread(target=_grp_y26, args=(g, fks)) for g, fks in fold_groups]
        for th in ts:
            th.start()
        for th in ts:
            th.join()
        assert not y26_errs, y26_errs

        if all((y26_wdir / "weights" / f"det_yolo26x-seg_{win_sc}_f{k}.done.json").exists() for k in range(5)):
            boxes_dir = y26_wdir / f"boxes_yolo26x_seg_{win_sc}"
            s1_inst_dir = y26_wdir / f"s1_inst_yolo26x_seg_{win_sc}"
            ps = [
                subprocess.Popen(
                    [py, str(s1_py), "--worker", "infer", "--repo", str(repo_dir), "--data-root", str(data_root),
                     "--work-dir", str(y26_wdir), "--arch", "yolo26x-seg", "--scenario", win_sc,
                     "--boxes-dir", str(boxes_dir), "--s1-inst-dir", str(s1_inst_dir), "--part", str(p)],
                    env=dict(env_base, CUDA_VISIBLE_DEVICES=str(p if n_gpu >= 2 else 0)),
                    cwd=str(scratch),
                )
                for p in (0, 1)
            ]
            assert all(p.wait() == 0 for p in ps), "yolo26x-seg infer failed"
            _pack(boxes_dir, kw / f"exp013_yolo26x_seg_{win_sc}_boxes.tar.gz")
            _pack(s1_inst_dir, kw / f"exp013_yolo26x_seg_{win_sc}_s1masks.tar.gz")
            _pack(y26_wdir / "weights", kw / f"exp013_yolo26x_seg_{win_sc}_weights.tar.gz")

            # Refine new YOLO26x-seg boxes using our already-trained 5-fold Stage-2 SOTA segmenters
            for s2_arch, s2_wdir, s2_odir in s2_trained:
                safe_s2 = s2_arch.replace("-", "_")
                _run_s2_infer(
                    s2_arch, s2_wdir, s2_odir,
                    {f"y26_{safe_s2}": boxes_dir},
                    kw / f"exp013c_y26_{safe_s2}_segs.tar.gz",
                )

    # =========================================================================
    # PHASE 4 (time-guarded): EXP-013a 5-Fold Pure Transformer RT-DETR-X @1280
    # =========================================================================
    if (time.time() - t0) / 60.0 < MAX_WALL_MIN - 95:
        rtd_wdir = Path(f"/tmp/exp013_rtdetr-x_{win_sc}")
        g_args = gong_args if win_sc == "dl_enh_copypaste" else []
        rtd_errs: list[BaseException] = []

        def _grp_rtd(gpu: int, fks: list[int]) -> None:
            try:
                for fk in fks:
                    if (time.time() - t0) / 60.0 > MAX_WALL_MIN - 20:
                        return
                    _sh(
                        [py, str(s1_py), "--worker", "train_fold", "--repo", str(repo_dir), "--data-root", str(data_root),
                         "--work-dir", str(rtd_wdir), "--arch", "rtdetr-x",
                         "--scenario", win_sc, "--fold", str(fk), "--epochs", "28", "--patience", "5"] + g_args,
                        env=dict(env_base, CUDA_VISIBLE_DEVICES=str(gpu)),
                        cwd=str(scratch),
                    )
                    shutil.rmtree(rtd_wdir / "det_data" / f"rtdetr-x_{win_sc}_f{fk}", ignore_errors=True)
            except BaseException as exc:
                rtd_errs.append(exc)

        ts = [threading.Thread(target=_grp_rtd, args=(g, fks)) for g, fks in fold_groups]
        for th in ts:
            th.start()
        for th in ts:
            th.join()
        assert not rtd_errs, rtd_errs

        if all((rtd_wdir / "weights" / f"det_rtdetr-x_{win_sc}_f{k}.done.json").exists() for k in range(5)):
            boxes_rtd = rtd_wdir / f"boxes_rtdetr_x_{win_sc}"
            ps = [
                subprocess.Popen(
                    [py, str(s1_py), "--worker", "infer", "--repo", str(repo_dir), "--data-root", str(data_root),
                     "--work-dir", str(rtd_wdir), "--arch", "rtdetr-x", "--scenario", win_sc,
                     "--boxes-dir", str(boxes_rtd), "--part", str(p)],
                    env=dict(env_base, CUDA_VISIBLE_DEVICES=str(p if n_gpu >= 2 else 0)),
                    cwd=str(scratch),
                )
                for p in (0, 1)
            ]
            assert all(p.wait() == 0 for p in ps), "rtdetr-x infer failed"
            _pack(boxes_rtd, kw / f"exp013_rtdetr_x_{win_sc}_boxes.tar.gz")
            for s2_arch, s2_wdir, s2_odir in s2_trained:
                safe_s2 = s2_arch.replace("-", "_")
                _run_s2_infer(
                    s2_arch, s2_wdir, s2_odir,
                    {f"rtd_{safe_s2}": boxes_rtd},
                    kw / f"exp013c_rtd_{safe_s2}_segs.tar.gz",
                )

    print(f"EXP-013 Kaggle T4x2 COMPLETE in {(time.time() - t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
