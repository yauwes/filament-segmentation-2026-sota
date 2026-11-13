# GAP RESEARCH MASTER — Kaggle Filament Segmentation 2026 (`0.41++` Mission)

- **Branch**: `gap-research/0.41-plus` (branched cleanly from `main` @ `b4acdb4157279babc59c2909df5459aabef16778`, zero changes to `main`, `report/`, `results/submission.csv`, or `weights/`)
- **Objective**: Systematically audit and execute untested/under-explored gaps across the full pipeline to push **Kaggle Public LB > `0.4100` (`0.41++`, aiming for `0.41–0.42+`)** with strict local nested date-grouped validation and format verification before any submission.
- **Competition Constraints & Quota (Checked `2026-11-13`)**:
  - Deadline: `2026-11-15 06:00:00 UTC` (~44 hours remaining)
  - Daily submission quota: `5/5` available for `2026-11-13` UTC
  - Official Metric: Pooled **Panoptic Quality (PQ)** at `IoU > 0.50` across all annotator records (`PQ = Sum(IoU_TP) / (TP + 0.5*(FP + FN))`), maximize `PQ`.
  - Compute Available:
    1. **Local Mac (M1, 8 cores, 8 GB RAM, MPS)**: Fast vectorized CPU/MPS post-processing, meta-learning, continuous probability matrix fusion across ALL cached Fold-0 (`145` frames) and Test (`180` frames) `.npz` predictions + `707` 5-fold OOF frames. (Local free disk `~4.9 GB`: keep big checkpoints/datasets off local root).
    2. **Remote Kaggle `2 × Tesla T4` GPU Kernels (`kaggle kernels push`, private slugs under `ajikajadeh`)**: Full access to `/kaggle/input/filament-segmentation-2026` (`707` train + `180` test native `2048²` frames), `ajikajadeh/filseg-gong-unlabeled` (`3,000` unlabeled GONG frames), `ajikajadeh/filseg-ts-weights`, `ajikajadeh/filseg-twostage-g19`, etc.
    3. *Note on Colab (`filseg-a100-out`)*: Earlier L4 session (`NVIDIA L4 23 GiB`) stopped at `2026-11-12 13:43 UTC` after finishing epoch 1 of `seg_e4x_f0`; not actively connected right now. All heavy remote GPU work in this branch runs natively on isolated Kaggle `2 × T4` (or user-reconnected Colab if triggered later).

---

## 1. Audited Baseline State (`EXP-000` / `E19`)

| Dimension | Audited Current State |
|---|---|
| **Best Kaggle Public LB** | **`0.40` displayed (2-decimal API rounding; `0.402` logged in `report/numbers.tex`)** via `results/submission.csv` (Sub `#58280243`, E19, `1,148` test instances, `6.38/frame`, `2` empty test frames) |
| **Best Local Val PQ** | **Fold 0 (`145` frames, `221` records, `1,488` GT)**: `PQ = 0.4856`, `SQ = 0.6924`, `RQ = 0.7013`, `TP = 978`, `FP = 323`, `FN = 510` |
| **Best 5-Fold Core OOF** | **707 frames (`1,154` records, `8,199` GT)**: `ms` (`YOLO11m main+s + e4`) `PQ = 0.4636` (`95% CI [0.4538, 0.4716]`, `SQ = 0.6920`, `RQ = 0.6699`, `TP=5288, FP=2301, FN=2911`) |
| **Single Streams on Fold-0** | `P1` `0.4650`, `ms` `0.4648`, `F1` `0.4646`, `s` `0.4643`, `P4a` `0.4619`, `F3` `0.4619`, `main` `0.4612`, `P6` `0.4601`, `F0` `0.4541`, `A(1664)` `0.4539`, `B` `0.4534`, `G1` `0.4491`, `YOLOv8s-seg(1536)` `0.4270` |
| **Preprocessing (`filseg/enhance.py`)** | Disk-MAD `[-8,+8]sigma` standardized 3-plane enhancement (`C1`: black top-hat `r=7`, `C2`: `101²` median subtraction, `C3`: half-res Sato dark ridge `sigmas=(1,2,3)`) |
| **Stage 1 Detection** | `YOLO11m` (`1280/1664`), `Faster R-CNN R50-FPN v2` (`1280` raw grayscale? Check!), `YOLOv8s-seg` (`1536`), trained per annotator record (`1,154` records) + `3,000` GONG Noisy-Student (`ms`, `s`, `P1`, `P4a`, `P6`, `F1`, `F3`) |
| **Stage 2 Segmentation** | `512×512` padded crop (`pad=0.20`, `jitter=±0.12`), owner-only supervision, `U-Net++ EfficientNet-B4` (`e4`, `S1E`) + `SAM 2.1 Hiera-B+` (`avgSAM2`), `0.5 BCE + 0.5 Tversky(0.3, 0.7)`, 4-flip TTA |
| **Stage 3 Fusion (`build_super_submission.py`)** | Early binary threshold (`thr=0.60`) + per-stream greedy painting -> Mask-WBF clustering (`G1..F1+YOLO`, `match=0.40, vote=0.40`) -> confirm bonus (`+0.02` if `F3` and `ms/s` `IoU>=0.40`) -> 2D binary donor injection (`ms/s/main/F3`) -> `vr=0.425`, `c_floor=0.32`, greedy non-overlap (`drop=0.70`, `min_area=100`) |
| **Already Tested & Excluded** *(DO NOT repeat)* | All-data 5-fold refit (`0.39` LB); wide naive pad `p35` (`0.4436`); non-trunc `--grow 2` (`0.4582`); dropping det/seg TTA (`-0.0050/-0.0016`); RT-DETR; crop DINOv3 / SegFormer / DeepLab / HRNet / naive clDice; custom fresh-head FRCNN anchors (`0` AP50 at 80 iters) |

---

## 2. Empirical Diagnostic Discoveries (`audit_data_and_streams.py` & `diagnose_baseline.py`)

1. **Test vs. Train Observatory & Date Overlap Shift**:
   - **Train (`707` fr)**: `B:118, C:125, L:127, M:125, T:110, U:102`; years `2011–2022`.
   - **Test (`180` fr, `176` dates)**: `B:33 (18.3%), C:33 (18.3%), L:34 (18.9%), M:26, T:27, U:27`.
   - **`31` Test dates overlap with Train dates (`31/176 = 17.6%`)** — distributed across Folds `0..4`, meaning **Fold-0-only streams (`9` of `12` in E19: `G1,A,B,P1,P4a,F0,F1,F3,P6`) miss cross-fold generalization and same-day disk coverage from Folds `1..4`!**
   - **Severe Station & Solar-Cycle Variance on Fold 0 (`E19`)**:
     - `U` (`0.5216`), `L` (`0.5082`), `T` (`0.5015`), `C` (`0.4868`), vs **`B` (`0.4511`, `RQ=0.6390`)** and **`M` (`0.4274`, `SQ=0.6711, RQ=0.6368`, `98` FNs!)**, and **Solar Min `2019` (`0.0000`, `0` TP, `7` FN) / `2021` (`0.4287`)**!
2. **Test vs. Val Area Shift & Small-Filament Crisis**:
   - Fold-0 GT area `< 580 px`: **`20.2%`** (`q5=319, q10=408, q25=659, q50=1170 px`).
   - E19 Test Preds `< 580 px`: **`13.2%`** (`q5=432, q10=528, q25=771, q50=1388 px`).
   - Prior `nearmiss.py` (`ms` OOF): small GT (`9–580 px`, `n=1642`) has only **`49.8%` match rate**, **`39.0%` completely untouched**, and **`2.96x` median area over-prediction** (`purity=0.30`)!
3. **5-Fold Test Confidence Shift BUG in `ms/s/main` Donors**:
   - On Val (`1`-fold 3-TTA WBF, `V=3`), `ms` median `conf=0.122`, `s` median `conf=0.419`.
   - On Test (`5`-fold 3-TTA WBF, `V=15`), `ms` median `conf=0.039`, `s` median `conf=0.203`, yet `build_super_submission.py` uses fixed `conf=0.30` on both! Thus donor coverage & confirmation behave asymmetrically between Val and Test.
4. **Hard Binary Bottleneck + Missing `P6` + Raw Uncalibrated Confidence Mixing**:
   - `build_super_submission.py` binarizes every stream at `thr=0.60` (`paint_ts_cropped`) **before** ensemble fusion, discarding `8`-bit continuous mask geometry despite `E15/E16` proving soft 2D blending lifts single detectors `+0.0060..+0.0068 PQ`!
   - `P6` (`PQ=0.4601`, `145` Val + `180` Test cached!) was omitted entirely.
   - `F0/F1/F3` softmax scores (`0.80–0.99`) vs `YOLO` observer-plurality scores (`0.20–0.70`) are averaged raw in `mean_c * sup/8` and pixel vote weights, while `F0 < 0.80`, `F1 < 0.85`, `F3 < 0.85` are clipped away pre-clustering.
5. **Large Truncated / Split Filaments (`21%` double-predicted) & Radial Limb Falloff (`r > 0.75`)**:
   - Precision drops `73.4% (r<0.50)` -> `57.2% (0.75..0.90)` -> `0 (r>0.90)`.
   - `70%` of large (`>=1582 px`) near-misses extend outside crop windows; Colab truncated-aware `e4x` stopped at Fold-0 Epoch 1 without full 5-fold completion.

---

## 3. Prioritized 10-Stage Research Gap Matrix

| ID | Stage | Gap & Concrete Hypothesis | Pre-Checks / Safeguards | Cost / Compute | Priority |
|---|---|---|---|---|---|
| **EXP-001** | **Validation** | Build leak-free **5-Fold Date-Grouped Nested CV + 200 Split-Half Date Bootstrap** harness on Fold 0; test E19's nested CV score vs `0.4856` | Date-grouped splits only | <1 min CPU | **P0 (Gate)** |
| **EXP-002** | **H/J Fusion** | **True Continuous Soft-Probability Fusion + Include `P6` + Quantile-Calibrated Confidences + Relaxed Lower Floors Pre-Fusion + Optimal Test-Matched Donor Calibration**: Keep `[0,1]` uint8 probability maps (`p1/p4/p`) all the way into cluster vote-map accumulation and donor injection, rank-align `F0/F1/F3` vs `YOLO` confidences, and include `P6` | Evaluate strictly on Nested Date CV + Date Bootstrap (`P>0.95`) vs E19; verify Test distribution invariance (`ms/s/main` quantile matching) | ~2 min CPU | **P0 (Wave 1)** |
| **EXP-003** | **H Postproc** | **Scale- & Radius-Adaptive Probability Thresholding + Soft-IoU Optimal Thin-Spine Calibration**: Use higher probability threshold / subtle boundary tightening on small (`<600 px`) high-purity-error filaments vs lower threshold on faint/large spines (`vr(area, r/R_sun, edge_grad)`), fixing `2.96x` fat small near-misses and test small-filament shift | Nested Date CV + SQ/RQ per area quintile | ~2 min CPU | **P0 (Wave 1)** |
| **EXP-004** | **H Postproc** | **Collinear Curvilinear Fragment Re-Stitching (Tangent + Gap Linking)**: Merge near-adjacent collinear cluster fragments belonging to the same broken spine (principal axis angle `<20°`, endpoint gap `<=14 px`, non-duplicate parallel guard) before/during greedy painting to lift large split near-misses (`21%`) | Check `IoU>0.5` merge rate carefully so separate parallel filaments aren't merged | ~2 min CPU | **P0 (Wave 1)** |
| **EXP-005** | **H/J Meta-Rescoring** | **Cross-Family Date-CV Meta-Rescorer (`Calibrated HGBR / Logistic`)**: Predict expected `TP` probability `p_hat` from family agreement count (`YOLO11_teacher`, `YOLO11_student`, `FRCNN`, `YOLOv8seg`, `5F_ms`), `P6`, normalized rank confidence, `r/R_sun`, station-invariant image stats proxy / area / aspect / solidity / soft-IoU uncertainty; threshold directly around theoretical PQ Bayes cutoff `PQ / (2 * E[IoU_TP]) ≈ 0.35`, fixing `r > 0.75` limb FP and `2019` empty-frame suppression | Strictly OOF nested date-grouped folds on Fold 0! Check calibration curve & feature stability across stations | ~2 min CPU | **P0 (Wave 1)** |
| **EXP-006** | **G/D 5-Fold & Trunc** | **5-Fold Cross-Fitting of Top Detectors & Truncation-Aware Segmenters on Kaggle `2×T4`**: (a) Train **5-fold `Faster R-CNN R50-FPN v2`** + GONG student (`F_5fold` instead of Fold-0 only!) — check whether enhanced tri-plane input helps FRCNN vs raw grayscale; (b) Complete **5-fold `seg_e4x` (`--trunc 0.4 --pad-hi 0.6 --grow`)**; (c) Train **5-fold `YOLO11l` / `YOLO11x` or `YOLO11m@1664`** so 100% of training data (`31` matching test dates!) informs test predictions | Isolated Kaggle kernel `ajikajadeh/filseg-gap-*`, check Fold-0 OOF PQ gain paired vs single-fold `F3`/`A`/`e4` | GPU (`2×T4`) | **P1 (Wave 2)** |
| **EXP-007** | **C/A/D High-Res & Limb** | **SAHI Overlapped `1024×1024` Tiled Detector + Radial Limb Contrast Calibration**: Train & infer a Stage-1 detector on native `1024×1024` crops (preserving `1.0x` pixel resolution on `9–580 px` small filaments!) + test Mauna Loa/Big Bear (`M/B`) contrast/CLAHE-band normalization to recover the **`39%` untouched small filaments** and **`M/B` + `r>0.75` limb recall** | Compare Fold-0 `<580 px` and `r>0.75` recall directly on nested CV + inject into Super-Ensemble | GPU (`2×T4`) | **P1 (Wave 2)** |
| **EXP-008** | **D/E/I New Segmenter & Soft Pseudo** | **Modern Crop Segmenter Head (`ConvNeXt-Base U-Net++ / UPerNet` or `SwinV2/EVA02` / `SAM2.1` 5-fold) with `Focal-Tversky + Lovász + Boundary Loss` at `640×640`**, PLUS Round-2 ensemble-distilled **consensus pseudo-labeling** on 3,000 GONG frames | Must strictly beat Fold-0 `S1E+e4+SAM2` (`0.4716` on `ms` boxes) on paired bootstrap! | GPU (`2×T4`) | **P2 (Wave 2/3)** |
