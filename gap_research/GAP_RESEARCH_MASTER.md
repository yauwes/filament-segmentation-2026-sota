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

---

## Wave-1 Empirical Audit & Local Suite Results (2026-11-13)

### Prior Kaggle Kernel Audits Verified Before Wave-2 Launch
- **`G4` (`filseg-twostage-g4`)**: Arm `E` (`bth7+flat+ridge` at `imgsz=1280`, 29 ep) achieved box `mAP50=0.608` vs standard raw `YOLO11m` (`det_b`/`det_a2`: `mAP50=0.659..0.685`). Full replacement with heavy morphological planes hurts Stage-1 localization compared to raw `[Y]`, whereas `C` (`imgsz=1664`) only reached `mAP50=0.642` at batch 1 in `2.6h`.
- **`G30` / `G31` (`rfdetr` sliced queries / mixed-tile control)**: RF-DETR alone capped at `PQ=0.3737` (global) vs `0.3712` (mixed, `TP` jumped `790 -> 875`, `+85` true positives with higher `FP`), confirming native-resolution tiling boosts recall on small/faint filaments when paired with strong WBF/precision filtering (`SAHI@1024`).
- **`seg_e4` / `det_m1280` / `student_b` / `infer-ms` / `infer-s`**: All 5 folds of `det_m1280`, `det_m1280s`, `seg_e4` completed cleanly (`infer-ms` 5-fold OOF `PQ=0.4636`, `infer-s` `0.4599`, `oof_main` `0.4582`), while `ajikajadeh/filseg-a100-weights` already stores `seg_e4x_f0.pth` alongside `seg_e4_f0..4.pth`, `det_m1280_f0..4.pt`, and `det_m1280s_f0,2,4.pt` (`f1,3` in `filseg-student-b`). Only Fold-0 had `seg_e4x`; Folds `1..4` and multi-stream `SAM2+S1E+seg_e4x` application across `ms/s/main/F3/P4a/P6` remain untapped.

### Wave-1 Local Suite (`EXP-001b..005b`) — Conclusive Ceiling on Fixed E19 Caches
- **E19 Baseline (`EXP-000`, Fold-0 `145` fr / `221` recs / `1,488` GT)**: `PQ=0.4856, SQ=0.6924, RQ=0.7013 (TP=978, FP=323, FN=510)`, Public LB `0.402` (CLI displays `0.40`).
- **`EXP-001b` (Nested 5-Fold Date-CV Cluster Floor)**: `PQ=0.4845` (`[0.32]*5`) vs `0.4856`.
- **`EXP-002` / `EXP-002b` (Continuous `[0,1]` Soft Vote-Map / Logit + `P6`)**: `PQ=0.4760..0.4820` (`+P6: 0.4815`) — binary per-member consensus in E19 already outperforms continuous averaging on fixed caches.
- **`EXP-003` (Scale-Adaptive Mask Probability Threshold under Nested Date-CV)**: Best `PQ=0.4851` vs `0.4856`.
- **`EXP-004` (Collinear Fragment Stitching `8/12/18 px`)**: `PQ=0.4844` — geometric post-stitching merges/degrades PQ on fixed masks.
- **`EXP-005b` (Nested Date-CV Meta-Rescorer: LogReg/Ridge/HGBR TP & IoU)**: Every sweep (`a=0.10..1.00`) strictly lost to E19 (`logreg_tp` `0.4786..0.4818`; `ridge_iou` `0.4790..0.4820`; `hgbr_tp a=0.10` `0.4833, vsE19=-0.0023 CI=[-0.0042,-0.0004] P=0.010`; `hgbr_iou a=0.10` `0.4828`).
- **Wave-1 Verdict**: Post-processing and rescoring on fixed E19 caches is strictly bounded (`<= 0.4856`). Moving Kaggle LB from `0.402` to **`0.4100+` (`0.41++`)** requires Wave-2 upstream GPU execution (`EXP-006..EXP-009`).

---

## Wave-2 Clean-Only Baseline & Remote `2 × T4` GPU Execution (Started 2026-11-13 ~10:41 UTC+7)

### Clean-Only & Full-Train Baseline Audit (Verified Before Wave-2 Launch)
- **Clean-Only Reference (`E19_no_F1`, excluding `MAGFiLO_extra` / `F1` cache)**:
  - `E19_no_F1` (`Fold-0, 145 fr`): `PQ=0.4771 (TP=943, FP=312, FN=545)` vs `E19 (0.4856, TP=978, FP=323, FN=510)` — `-0.0085 PQ` (`-35 TP`).
  - `E19_no_F1 + P6_core`: `PQ=0.4741 (TP=943, FP=328, FN=545)`.
  - `E19_no_F1 + ms/P6 rescue` (`build_gap_submission.py` `super_ensemble_upgraded_stem`, verified bit-for-bit parity `0.48563` on full `E19`!): `PQ=0.47706 (rescue>=0.75..0.80: TP=943, FP=312, FN=545)` — passive `ms/P6` rescue adds `0 TP` on top of existing global detectors, proving genuine new recall/precision sources (`SAHI@1024`, `FRCNN-5F`, `seg_e4x`/`SAM2`/`S1E` + `ConvNeXt-Topo`) are mandatory.
- **Full `707`-Photo 5-Fold OOF Baseline (`ms/s/main` on `/private/tmp/gt_records.pkl`)**:
  - `ms alone`: `PQ=0.46155, SQ=0.69295, RQ=0.66607 (TP=5185, FP=2185, fn=3014)`
  - `ms+s+main`: `PQ=0.46171, SQ=0.69306, RQ=0.66619 (TP=5167, FP=2146, fn=3032)`
- **Test Distribution Audit (`results/submission.csv`, 180 Test images)**:
  - `total_preds=1148`, `mean_per_photo=6.378`, `empty_photos=2`, `small_area_lt580_share=0.1315` (vs `~20%` GT `<580 px`), `median_area_px=1388.0`, `median_max_side=98.0`.
  - Observatory means: `Bh: 7.12 (0.123 small), Ch: 7.27 (0.158 small), Lh: 6.09 (0.150 small), Mh: 5.69 (0.101 small), Th: 6.11 (0.139 small), Uh: 5.67 (0.098 small)`.

### Active Wave-2 GPU Kernels (`2 × T4` Quota: 2/2 Running, 2 Queued)
1. **`EXP-006` (`ajikajadeh/filseg-gap-exp006-seg-upgrade`) — `RUNNING`** (started ~10:41, est. ~4.3h):
   - Phase 1: Parallel CPU `planes --procs 3` + `SAM2` (`g15/SAM2/best.pth`) over `ms, F3, P6, P4a, s, main` + `S1E` (`S1E_best.pth`) over `ms, s, main` (`707 OOF + 180 Test`).
   - Phase 2: 5-fold `seg_e4x` (`--init e4 --trunc 0.4 --pad-hi 0.6 --lr 1e-4 --epochs 6`) across `folds 0..4`.
   - Phase 3-4: Sharded `2 × T4` `e4x` & `e4x_g2` (`grow_windows` up to 2 rounds) + `SAM2_g2`/`S1E_g2` delta crops.
2. **`EXP-007` (`ajikajadeh/filseg-gap-exp007-sahi1024`) — `RUNNING`** (started ~10:43, est. ~2.3-2.8h):
   - 5-fold `YOLO11m` (`det_m1280_f0..4.pt` init) on overlapping `3×3` `1024×1024` native tiles (`starts=[0,512,1024]`, `min_vis=0.40` + 2 neg tiles/rec, 15 ep, `AdamW lr0=3e-4`), visibility-normalized cross-tile WBF (`cov_tiles`) + Stage-2 `seg_e4 (+g1), S1E, SAM2`.
3. **`EXP-008a` (`ajikajadeh/filseg-gap-exp008a-frcnn5f`) — `QUEUED` (Compiled, ready for first free slot)**:
   - Clean 5-fold `Faster R-CNN R50-FPN v2` @ `imgsz=1408` on all annotator records, dihedral aug, 3-flip WBF + Stage-2 `e4 (+g1), S1E, SAM2`.
4. **`EXP-008b` (`ajikajadeh/filseg-gap-exp008b-convnext-topo`) — `QUEUED` (Compiled + `topo_loss` unit-tested, ready for second free slot)**:
   - 5-fold `smp.UnetPlusPlus("tu-convnext_small.fb_in22k")` w/ `0.40*BCE + 0.35*Tversky(0.4,0.6) + 0.25*soft_clDice(6)` (`--trunc 0.4 --pad-hi 0.6`, 9 ep) across `ms, F3, P6, P4a, s, main`.

---

## Final Clean Competition Submissions & Wave-5 Summary
- **Baseline Clean Improvement Chain (`submission_log.jsonl`, strictly leak-free `5F 707-OOF Nested Date-CV`)**:
  - `EXP-006+007 Track-A Clean` (`exp006_007_tracka_clean.csv`, Kaggle `ref 58313893`): `PQ=0.46613`, Public LB `0.40`.
  - `EXP-014b` (`exp014b_hyst_gbdt_clean.csv`, Kaggle `ref 58331699`): `PQ=0.46819`, `Sim80=0.47135`, Public LB `0.41`.
  - `EXP-015a` (`exp015a_dense_clean.csv`, Kaggle `ref 58341800`): `PQ=0.47093`, `Sim80=0.47373`, Public LB `0.41`.
  - `EXP-015a LogitH` (`exp015a_logit_clean.csv`, Kaggle `ref 58348277`): `PQ=0.47191` (`TP=5311, FP=2116, FN=2888, SQ=0.69422, RQ=0.67976`), `Sim80=0.47462`, Public LB `0.41`.
  - **`EXP-015b LogitH + PurePhysViT 12%` (`exp015b_gv_pp12_clean.csv` / `results/submission.csv`, Official Final Submission, Kaggle `ref 58371665`, `SubmissionStatus.COMPLETE`)**:
    - Formula: `0.55*GBDT + 0.33*ViT + 0.12*PurePhysViT` (`alpha=0.72`, `cons_p=[0.195, 0.265, 0.06, 0.232, 0.015, true]`).
    - Nested 5-Fold Date-CV (`707` stems): `PQ=0.47196` (`SQ=0.69398, RQ=0.68008, TP=5322, FP=2130, FN=2877`), `Sim80=0.47460 ± 0.00169`, `mean_gain=+0.00005 vs exp015a` (`p=0.550`).
    - Validation & Test Shift Audit: `0` validation errors, `1192` predictions across `180` test images (`mean=6.622`, `empty=3`, `lt580=0.1753`, `median_area=1188.0`), **Public LB `0.41`**.
  - **`EXP-016 v4_y26_cnx` (`exp016_v4_tab_raw_clean.csv`, Kaggle `ref 58414256`, `SubmissionStatus.COMPLETE`)**:
    - Pool expansion (`43,356` OOF cands, `84.99%` OracleRecall@0.5 on `5,209` record-0 GT: `+ YOLO26x-seg @ 1536px + 5-ch EdgeGuidedMHSAUNet (ConvNeXt-Edge) + 11 MiT-B3 dense/CC features`).
    - 3-Family Tabular (`Tab3Fam_HeavyLGB`: `0.55*LightGBM + 0.28*CatBoost + 0.17*XGBoost`): Nested 5-Fold Date-CV `PQ=0.47256` (`SQ=0.6939, RQ=0.6810, TP=5301, FP=2068, FN=2898`), `1rec PQ=0.47848`, `Sim80=0.47470`, `vsTA=+0.00637` (`p=1.0000`).
    - Public LB probe (`ab=0.025, allow=False`, trees-only without `VisionCropTransformer`): scored `0.40` (`1196` test preds), confirming that pairing `VisionCropTransformer` patch attention with conservative Track-A anchoring (`ab=0.06, allow=True` in `EXP-015b`) is essential against cross-station test shift.
- **Sweep Variants Tested & Discarded (`5F Nested Date-CV` vs `0.47196`)**: `gv_pp08 (0.47130)`, `gv_pp10 (0.47186)`, `gv_pp15 (0.47036)`, `gv_pp12_gl2_05 (0.47061)`, `v5_y26_3cnx (0.46960)`.
