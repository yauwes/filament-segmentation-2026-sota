# Literature Review — Gap Research Extension (0.41++ Target)

This document builds directly on `research/literature_review.md` to map state-of-the-art techniques addressing the empirical failure modes identified on the E19 baseline (`Val PQ = 0.4856`, `Public LB = 0.402`):
1. **39.0% untouched small filaments** (`9–580 px`, plus `2.96x` over-dilation on matched small near-misses, purity `0.30`).
2. **Radial limb degradation**: precision falls monotonically from `73.4%` (`r < 0.50 R_sun`) to `57.2%` (`0.75 <= r < 0.90 R_sun`) and zero high-confidence detections near the limb (`r > 0.90`).
3. **Large-filament truncation & splitting** (`21%` of large `>=1582 px` near-misses split into `>=2` overlapping or disjoint pieces; `70%` clipped at the crop margin).
4. **Post-fusion discretization & uncalibrated score mixing** across YOLO11, Faster R-CNN, YOLOv8-seg, and 5-fold WBF streams (`ms/s/main` 5-fold vs 1-fold WBF score shift on test).

---

## 1. High-Resolution Small-Object Detection via Tiled Inference (SAHI / Slicing-Aided Fine-Tuning)
- **Reference**: Akyon, Altinuc & Temizel, *Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection*, IEEE ICIP 2022 ([SAHI](https://arxiv.org/abs/2202.06934)).
- **Mechanism**: Native $2048 \times 2048$ GONG frames resized to $1280 \times 1280$ compress a $15\text{–}30\,\text{px}$ filament box ($<580\,\text{px}^2$ area) down to $\sim 9\text{–}18\,\text{px}$, losing critical P3 feature-pyramid contrast. Partitioning the disk into overlapping $1024 \times 1024$ tiles (e.g. $25\%$ overlap, plus $2048$-px global context) preserves $1.0\times$ native resolution and triples effective pixel density on faint, compact filaments.
- **Adaptation to Filament PQ**: Unlike naive SAHI NMS which suppresses adjacent collinear tiles or creates boundary slivers, sliced detections must be merged via Confidence-Weighted IoU/IoS Merging (NMM) *or* passed into Stage-2 full-frame coordinate crop segmentation (`2048²` context) and resolved with soft probability vote maps.

## 2. Continuous Pixel Probability Accumulation vs. Early Binarization in Ensemble Fusion
- **Reference**: Solovyev et al., *Weighted Boxes Fusion*, Image and Vision Computing 2021; Kirillov et al., *Panoptic Segmentation*, CVPR 2019.
- **Mechanism**: E19 (`scripts/build_super_submission.py`) applies `p > 0.60` hard binarization inside `paint_ts_cropped` to all 12 two-stage streams before cluster accumulation (`fuse_with_continuous_acc`), reducing rich `uint8` (`[0, 255]`) spatial uncertainty from `S1E+SAM2` and `5-fold e4` into binary `{0, 1}` stamps.
- **Expected Benefit for Thin Curvilinear Masks**: Because median small near-misses are `2.96x` too wide (purity `0.30`), soft probabilities along `2–5 px` filament flanks carry strong signal (`0.35..0.59` vs `0.65..0.95` core). Preserving exact continuous `uint8 / 255.0` probabilities per stream—and allowing calibrated threshold $\tau(A, r)$ as a function of filament area and disk radius $r/R_{\odot}$—optimizes the sharp $\mathrm{IoU} > 0.50$ cliff directly via higher SQ on thin objects without destroying thin cores.

## 3. Cross-Family Score Calibration & Learned Panoptic Rescoring
- **Reference**: Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017; Platt Scaling / Rank-Quantile Calibration; Kirillov et al. (2019) PQ optimality derivation.
- **Problem**: `Faster R-CNN` outputs sharp RPN/RoI softmax scores (median `0.64–0.70`, `q90 = 0.994`), whereas `YOLO11` outputs observer-plurality scores (`q50 = 0.27–0.49`), AND 5-fold WBF at test time (`V = 15`) depresses `ms/s/main` confidences relative to 1-fold OOF (`V = 3`: `s` median drops `0.419 -> 0.203`).
- **Remedy**: Monotonic percentile / logit alignment across detectors, **plus** nested date-grouped cross-validation of a lightweight GBDT / calibrated logistic emission classifier over cluster features: calibrated member scores, cross-family support indicators (`YOLO11`, `FRCNN`, `YOLOv8-seg`, `SAM2+S1E`, `5-fold`), normalized radial coordinate $r/R_{\odot}$, aspect ratio / eccentricity, mask area $\log(A)$, and boundary-to-core probability contrast. Directly converts expected TP probability $p(x) > \mathrm{PQ}/(2\overline{\mathrm{IoU}}_{\mathrm{TP}}) \approx 0.35$ into an optimal ranking/gating score across radii and families.

## 4. Curvilinear Fragment Re-Stitching & Boundary-Aware / Truncation-Aware Objectives
- **Reference**: Hu et al., *MORDEN / Automated Solar Filament Instance Segmentation*, 2026 (DBSCAN fragment linking); Shit et al., *clDice*, CVPR 2021; Berman et al., *Lovász-Softmax*, CVPR 2018; Ravi et al., *SAM 2*, 2024.
- **Mechanism**: On `21%` of large near-misses (`>= 1582 px`), one continuous filament spine is split across adjacent boxes into separate fragments that overlap cleanly or leave a narrow `2–12 px` gap along a shared principal-axis tangent, causing both pieces to miss `IoU > 0.50`. Geometric collinear compatibility (principal orientation cosine + minimal endpoint distance + bridge response along `C3` Sato ridge plane) can safely link split tail fragments *without* merging parallel neighboring filaments, while **full 5-fold truncation-aware fine-tuning (`--trunc --grow`, incomplete on Colab after epoch 1 Fold 0)** trains the Stage-2 segmenter itself to follow broken/clipped ends outside initial box bounds.

## 5. Polar / Limb-Aware Chromospheric Normalization & Strong Cross-Family Scaling (5-Fold Expansion)
- **Observation**: Currently 9 of 12 streams on test inference come from **Fold-0-only splits** (`79.5%` of train labels), whereas Test contains `31` same-day frames distributed across Folds `1..4` and diverse seeing/limb contrast conditions.
- **Gaps**:
  1. Expanding top single-fold families (`Faster R-CNN R50-FPN v2` / `YOLO11-1664` / `YOLO11x` or `YOLO12` / `Mask2Former` or `SAHI` tiled detector / `SAM 2.1`) to full **5-fold ensembles** gives both $+25\%$ labeled supervision and variance reduction across Test dates.
  2. Evaluating **polar / radial contrast equalization** or **Center-to-Limb Variance Normalization** inside concentric solar rings ($r/R_{\odot} \in [0, 1]$) alongside dual-view **enhanced tri-plane input** for Faster R-CNN / YOLOv8-seg (remember Stage-1 `FRCNN` and `YOLO` in `filseg/detector.py`: check whether `FRCNN` used raw grayscale vs enhanced planes!).

---

## 6. Wave-4 Transformer, Hybrid SOTA & Topology-Preserving Literature (2024–2026)

1. **Universal Masked-Attention & End-to-End Detection Transformers (Mask2Former, RT-DETR, RF-DETR, D-FINE, YOLO12)**:
   - **Mask2Former** (Cheng et al., CVPR 2022; widely adopted in 2024–2026 thin-structure segmentation): Masked-attention Transformer decoder + multi-scale deformable pixel decoder retaining `1/4` resolution spatial details (`P2`). When paired with `1152x1152` overlapping tile inference (`256 px` seam overlap), eliminates Stage-1 box truncation and separates filaments from sunspots via global Swin-Base context.
   - **RT-DETR** (Lv et al., CVPR 2024) & **D-FINE / RF-DETR** (2025–2026): IoU-aware query selection + hybrid Transformer encoders remove heuristic NMS bottlenecks; **YOLO12** (Tian et al., arXiv Feb 2025) introduces Area-Attention (`R-ELAN`) unifying CNN local inductive bias with Transformer long-range context for thin filaments (`rtdetr-x` + `yolo12x` Stage-1 + SAHI native-1x tiling).
2. **Resource-Efficient Topology Preservation (`Skeleton Recall Loss` & `cbDice`, 2024)**:
   - **Skeleton Recall Loss** (Kirchhoff et al., ECCV 2024): Precomputes CPU tubed ground-truth skeleton $Y_{ts} = D(S(Y)) \odot Y$ and minimizes $1 - \frac{\sum Y_{ts} \cdot P + \epsilon}{\sum Y_{ts} + \epsilon}$ alongside BCE, Tversky ($\alpha=0.40, \beta=0.60$), and `Soft-clDice`. Slashes compute overhead vs full differentiable GPU skeletonization while directly defending the $\mathrm{IoU} > 0.50$ fragmentation cliff on thin filaments.
   - **cbDice** (Shi et al., MICCAI 2024): Complements `clDice` by weighting centerline and boundary radii uniformly across filament widths.
3. **Self-Supervised Transformer Image Restoration & Train-Only Augmentation (`Restormer` & Object Copy-Paste)**:
   - **Restormer** (Zamir et al., CVPR 2022; solar cross-band/seeing restoration 2024–2026): Linear-complexity Multi-Dconv Head Transposed Attention (`MDTA`) + Gated-Dconv Feed-Forward Network (`GDFN`) trained strictly on sharp in-fold `tr` frames with synthetic PSF seeing blur + haze restores fine filament absorption edges without fitting on validation/test frames.
   - **Object-Aware Small-Filament Copy-Paste**: Strictly restricted to `folds != fold` training photos (`<1100 px` filaments pasted onto clean quiet-Sun regions) while holding out 100% original untouched competition images & labels for zero-leakage validation.

---

## 7. Verified Deep Browser Research: Latest Papers & Kaggle Writeups (Nov 2026)

1. **MAGFiLO v1.0 Label Ceiling & Temporal CV Hygiene (`ameyypawar/filament-seg`, `Agnuxo1/solar-filament-segmentation-2026`, `arXiv:2610.34502`)**:
   - Pairwise inter-annotator human agreement on MAGFiLO multi-drawn frames is only **`PQ = 0.343`** (`kappa = 0.47`), making calibration of threshold/score directly against single-annotator (`Sim_r80`) & consensus targets essential.
   - Solar rotation (`~13 deg/day`) causes adjacent-day frames to share filaments, inflating non-grouped random-split Dice by `+0.10 to +0.25` (`arXiv:2610.34502`). All Gap-Research evaluations strictly use 5-fold date-grouped splits.
   - Aggressive external contrast stretching / histogram equalization collapses U-Net Val Dice (`0.75-0.80 -> 0.50-0.60`) by corrupting soft skip-connection intensity gradients (`arXiv:2610.34502`); keep raw CLV-flattened grayscale in Channel 0 and feed Frangi/Sato/TopHat/CLAHE strictly as auxiliary channels or tabular features.
2. **Bright-Halo Limb Rejection & Radial CLV Normalization (`filament_seg/disk.py`, `arXiv:2607.01095`)**:
   - GONG frames exhibit a scattered-light halo annulus outside the photosphere limb (`35-150 px` overshoot under naive Otsu). Ray-casting inward from multi-seed disk estimates + sub-pixel gradient snapping + MAD-Kasa circle fit + `shrink=0.98` masking eliminates off-disk FPs for free.
3. **Curvilinear & Topology-Preserving Literature (2024–2026)**:
   - **WPRF (Widest-Path Reachability Fields, `arXiv:2607.07123`)**: Multi-scale Max-Min DP reachability + bottleneck hard-negative BCE improves Mask2Former Swin-T & SegFormer MiT-B1/B3 `clDice` (`+3.0 to +7.2 pp`) with zero extra inference overhead.
   - **MORDEN (`arXiv:2607.24525v1`) & DenseCRF**: Replacing BatchNorm with **Instance/Group Normalization**, `5x5` stems, and full-res `2048px` DenseCRF edge crispness preserves local H-alpha filament contrast (`IoU 0.678 -> 0.696`).
   - **Pavement-DETR (`MDPI Sensors 2025, 25(8):2426`)**: Channel-Spatial Shuffle + Conv3XC neck + PIoU v2 + Normalized Wasserstein Distance (NWD) boosts RT-DETR thin curvilinear detection (`mAP@0.5: 0.794 -> 0.871`) where standard box IoU drops under 1-px diagonal jitter.
   - **HuBMAP Top-3 & SenNet 1st (`0.589` / `0.835` Private LB)**: Extra full-res stem (`GroupNorm/GELU`), multi-scale mask loss `2x-4x`, 2-stage Noisy->Clean/Consensus curriculum (`+4-6% LB`), quantile NMS/mask pruning, and warning that blind morphological mask dilation hurts hidden Private LB (`-0.062`) even when inflating Public LB.
   - **PQ-Optimal Post-Processing (`ameyypawar`, `Agnuxo1`)**: Pre-sigmoid logit averaging, cross-detector cluster mean confidence (counting misses as `0`), **`open_radius=0`** (never erase thin barbs), **non-inflating gap bridging (`bridge_and_label`: dilate ONLY to group components, keep un-dilated mask pixels)**, exclusive greedy U-Net/dense-map pixel claiming (`grow=4..8 px`), and MWIS conflict graph optimization.
