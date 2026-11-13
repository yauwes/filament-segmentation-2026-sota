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
