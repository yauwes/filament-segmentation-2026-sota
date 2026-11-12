# Comprehensive Methodology: Cross-Family Detect-Crop-Segment-Paint Architecture for Solar Filament Instance Segmentation

This document details every major technical component of our solution to the **Solar Filament Segmentation Challenge 2026** using the 9-part scientific explanation framework: **Problem, Intuition, Mechanism, Mathematics, Implementation, Justification, Alternatives, Trade-offs, and Empirical Evidence**.

---

## 1. Domain-Specific Tri-Plane Chromospheric Enhancement

1. **Problem**: Ground-based $\text{H-}\alpha$ observations from six GONG stations (`Bb`, `Ct`, `Le`, `Ml`, `Td`, `Ud`) exhibit severe center-to-limb intensity darkening, site-dependent exposure variations, and atmospheric seeing blur, while solar filaments span widths from $2\,\text{px}$ spines to $80\,\text{px}$ quiescent structures.
2. **Intuition**: Instead of forcing a convolutional encoder to learn radial background subtraction and multi-scale curvilinear ridge detection from raw 8-bit grayscale intensities, we explicitly decompose the solar disk into three physically interpretable planes: narrow dark structures, detrended medium/broad structures, and multi-scale dark curvilinear ridges.
3. **Mechanism**:
   - **Plane 1 (Black Top-Hat)**: Applies morphological closing with an elliptical structuring element $B_7$ of radius $r=7\,\text{px}$ ($15 \times 15\,\text{px}$) and subtracts the original image, isolating dark absorption features narrower than $15\,\text{px}$.
   - **Plane 2 (Large-Scale Median Detrending)**: Subtracts a $101 \times 101\,\text{px}$ median-filtered background (computed via $1/4$-scale fast bilinear downsampling and upsampling) to remove radial limb darkening while preserving broad quiescent filaments.
   - **Plane 3 (Multi-Scale Sato Curvilinear Ridge Filter)**: Evaluates the Hessian matrix eigenvalues at scales $\sigma \in \{1, 2, 3\}\,\text{px}$ on the half-resolution frame to amplify continuous dark tubular spines and faint barbs [Sato et al., 1998].
   - **Solar-Disk Robust Normalization**: Each plane is centered by its median over the active solar disk $\Omega_{\odot} = \{p : I(p) > 20\}$ and scaled by its Median Absolute Deviation ($\text{MAD}$), mapped into $[0, 255]$ via $\pm 4\,\text{MAD}$ clipping.
4. **Mathematics**:
   Let $I: \Omega \to [0, 255]$ denote the full-disk $2048 \times 2048$ grayscale image and $\Omega_{\odot} = \{x \in \Omega : I(x) > 20\}$ denote the solar disk mask. The three raw planes $C_1, C_2, C_3$ are defined as:
   $$C_1(x) = (I \bullet B_7)(x) - I(x), \qquad C_2(x) = \mathcal{U}_4\!\left(\text{Med}_{25}\!\left(\mathcal{D}_4(I)\right)\right)(x) - I(x), \qquad C_3(x) = \mathcal{U}_2\!\left(\max_{\sigma \in \{1,2,3\}} \mathcal{S}_{\sigma}(\mathcal{D}_2(I))\right)(x)$$
   where $\bullet$ denotes morphological closing, $\mathcal{D}_s, \mathcal{U}_s$ denote downsampling and upsampling by factor $s$, and $\mathcal{S}_{\sigma}$ is the Sato dark-ridge response. Each channel $k \in \{1,2,3\}$ is normalized via:
   $$\tilde{C}_k(x) = \text{clip}\!\left(\frac{C_k(x) - \mu_k}{8\,\text{MAD}_k + \epsilon} + \frac{1}{2},\; 0,\; 1\right) \times 255, \quad \mu_k = \text{median}_{x \in \Omega_{\odot}} C_k(x), \quad \text{MAD}_k = \text{median}_{x \in \Omega_{\odot}} |C_k(x) - \mu_k|$$
5. **Implementation**: Implemented in `filseg/enhance.py` (`enhance(gray)`), executing in $<45\,\text{ms}$ per $2048 \times 2048$ frame using OpenCV C++ SIMD routines and scikit-image's `sato` filter.
6. **Justification**: Preserves sharp filament boundaries across all six GONG observatories and eliminates inter-station intensity shift without requiring station-ID metadata at inference.
7. **Alternatives**: Raw 3-channel grayscale replication (`[I, I, I]`) or CLAHE histogram equalization. CLAHE amplifies chromospheric granulation noise in quiet-Sun regions, increasing false positives.
8. **Trade-offs**: Adds $\sim 45\,\text{ms}$ CPU overhead per frame, which is negligible compared to GPU forward passes and completely cached during training.
9. **Evidence**: Combining tri-plane enhancement with native-resolution crop segmentation lifted single-model validation $\text{PQ}$ from $0.4050$ (raw single-stage YOLO11x-seg) to $0.4491\text{--}0.4650$.

---

## 2. Multi-Annotator Plurality Preservation and Date-Grouped Cross-Validation

1. **Problem**: Among the $707$ training frames, $355$ frames were independently annotated by two or three observers ($1{,}154$ total records, $8{,}199$ filament polygons). Furthermore, multiple frames captured on the same calendar date from different GONG stations observe the same physical solar rotation disk.
2. **Intuition**: Collapsing multiple annotators into a single consensus or union mask destroys valuable uncertainty information: a filament marked by 3 of 3 annotators has near-certain existence probability, whereas one marked by 1 of 3 lies near the human decision threshold. Meanwhile, splitting same-day frames across train and validation folds leaks solar disk identity.
3. **Mechanism**:
   - **Observation-Date Grouping**: All frames sharing an 8-digit observation date prefix (`YYYYMMDD`) are assigned to the same cross-validation fold (`folds/folds.json`, 5 folds), guaranteeing zero temporal or structural leakage.
   - **Record-Level Training Multiplicity**: During detector training, every annotator record $(I_i, \mathcal{B}_{i,a})$ is presented as a distinct training sample. A structure annotated by $k$ of $M_i$ observers appears as a positive target in $k/M_i$ of the exposures of image $I_i$.
4. **Mathematics**:
   Under binary cross-entropy objectness supervision over $M_i$ annotator records for frame $I_i$ where $k$ records label bounding box $b$, the expected per-frame objectness loss for predicted probability $\hat{p}$ is:
   $$\mathcal{L}_{\text{obj}}(\hat{p}) = -\frac{k}{M_i}\log \hat{p} - \left(1 - \frac{k}{M_i}\right)\log(1 - \hat{p})$$
   which is uniquely minimized at the Bayes-optimal human agreement probability $\hat{p}^\star = \frac{k}{M_i}$.
5. **Implementation**: Implemented in `filseg/data.py` (`make_folds`, `records`) and `filseg/metric.py` (`score_rles`, `bootstrap_ci`).
6. **Justification**: The official evaluation metric pools $\text{TP}, \text{FP}, \text{FN}$ across all annotator records of every frame. Learning $\hat{p}^\star = k / M_i$ calibrates detector confidence directly to expected multi-annotator precision.
7. **Alternatives**: Union merging (over-predicts faint fibrils, hurting precision on stricter annotators) or intersection merging (misses valid filaments, hurting recall).
8. **Trade-offs**: Increases epoch length by $1154 / 707 = 1.63\times$, which is offset by faster convergence and superior confidence calibration.
9. **Evidence**: Scoring one human annotator against another on the $355$ multi-record frames yields $\text{PQ} = 0.4494$ ($\text{SQ} = 0.6766, \text{RQ} = 0.6642$), whereas our 5-fold out-of-fold teacher-student model reaches $\text{PQ} = 0.4728$ ($+0.0234$ above inter-human agreement).

---

## 3. Stage 1: Heterogeneous Full-Disk Detection and Noisy-Student Self-Training

1. **Problem**: Localizing filaments across a $2048 \times 2048$ disk requires global context to distinguish true filaments from sunspots and chromospheric plages, while labeled training data is limited to $707$ frames.
2. **Intuition**: Combining anchor-free single-stage detectors (YOLO11m/l/x at $1280\text{--}1664\,\text{px}$) with two-stage region-proposal detectors (Faster R-CNN R50-FPN v2) provides complementary recall (YOLO excels on elongated diagonal filaments) and precision (Faster R-CNN RoIAlign excels on compact high-contrast filaments). Expanding supervision via Noisy Student self-training on $3{,}000$ unlabeled GONG archive frames (2011–2025) regularizes feature representations across solar cycle phases.
3. **Mechanism**:
   - **Fold-Isolated Pseudo-Labeling**: For each fold $k \in \{0,\dots,4\}$, the teacher detector trained exclusively on folds $\neq k$ predicts on $3{,}000$ unlabeled GONG archive frames (filtered to exclude any competition date $\pm 2$ days) using $3\times$ flip Test-Time Augmentation (identity, horizontal, vertical) fused via Weighted Boxes Fusion ($\text{IoU}=0.50$, confidence floor $0.25$).
   - **Student Training**: Student detectors (YOLO11m/l/x and Faster R-CNN R50-FPN v2) are trained jointly on the ground-truth fold split plus pseudo-labeled archive frames with mosaic, flip, and scale augmentations.
4. **Mathematics**:
   Given $V$ augmented views/models producing candidate boxes $\{b_j, s_j\}_{j=1}^K$ in a spatial cluster $\mathcal{C}$ with pairwise $\text{IoU} \ge \tau_{\text{box}}$, Weighted Boxes Fusion [Solovyev et al., 2021] computes the fused box coordinates $\bar{b}$ and view-penalized confidence $\bar{s}$:
   $$\bar{b} = \frac{\sum_{j \in \mathcal{C}} s_j b_j}{\sum_{j \in \mathcal{C}} s_j}, \qquad \bar{s} = \left(\frac{1}{|\mathcal{C}|}\sum_{j \in \mathcal{C}} s_j\right) \cdot \frac{\min(|\mathcal{C}|, V)}{V}$$
5. **Implementation**: Implemented in `filseg/detect.py` (`train`, `pseudo`, `propose`).
6. **Justification**:Strictly out-of-fold fold-specific teachers prevent any indirect information leakage from validation frames through the unlabeled archive.
7. **Alternatives**: End-to-end Mask R-CNN or Mask2Former on full $2048 \times 2048$ frames, which exhaust GPU memory or downsample RoI features to $28 \times 28\,\text{px}$.
8. **Trade-offs**: Self-training adds one offline pseudo-labeling pass and 20 student epochs per fold on A100 GPU, zero additional inference latency per single model.
9. **Evidence**: Self-training consistently improves 5-fold out-of-fold $\text{PQ}$ from $0.4694$ (`main` teacher) to $0.4718$ (`s` student) and $0.4728$ (`ms` teacher+student), and lifts Fold-0 single detectors to $0.4650$ (`P1`) and $0.4646$ (`F1`).

---

## 4. Stage 2: Instance-Conditioned Native-Resolution Crop Segmentation and Per-Pixel Multi-Segmenter Blending

1. **Problem**: Standard RoI mask heads ($28 \times 28$) or full-frame prototype masks ($320 \times 320$) destroy the fine $2\,\text{px}$ boundary geometry required to push instance $\text{IoU}$ across the strict $0.50$ Panoptic Quality threshold, and frequently merge adjacent filaments that share overlapping bounding boxes.
2. **Intuition**: By cropping each detected bounding box with $20\%$ context padding from the native $2048 \times 2048$ tri-plane image and magnifying it to $512 \times 512$, the median filament ($130\,\text{px}$) is segmented at nearly $4\times$ super-resolution. Crucially, training the segmenter to predict **only the target filament whose bounding box defines the crop**—while treating any neighboring filaments invading the padded margin as background—teaches the network instance-conditioned separation.
3. **Mechanism & Mathematics**:
   - **Instance-Conditioned Crop Extraction**: Given box $b = (x_1, y_1, x_2, y_2)$ with width $w = x_2 - x_1$ and height $h = y_2 - y_1$, the padded crop window is:
     $$\tilde{b} = \left(\max(0, \lfloor x_1 - \rho w \rfloor),\; \max(0, \lfloor y_1 - \rho h \rfloor),\; \min(W, \lceil x_2 + \rho w \rceil),\; \min(H, \lceil y_2 + \rho h \rceil)\right), \quad \rho = 0.20$$
     During training, $b$ is perturbed by uniform random box jitter $\delta \sim \mathcal{U}(-0.12, 0.12)$ so the segmenter becomes invariant to Stage-1 localization noise.
   - **Asymmetric Tversky + BCE Objective**: To prevent thin filament spines from being eroded by background class imbalance, we minimize a composite Binary Cross-Entropy and asymmetric Tversky loss [Salehi et al., 2017] with false-negative weight $\beta = 0.7$ and false-positive weight $\alpha = 0.3$:
     $$\mathcal{L}_{\text{seg}}(p, y) = \frac{1}{2}\mathcal{L}_{\text{BCE}}(p, y) + \frac{1}{2}\left(1 - \frac{\sum_i p_i y_i + \epsilon}{\sum_i p_i y_i + \alpha \sum_i p_i(1 - y_i) + \beta \sum_i (1 - p_i)y_i + \epsilon}\right)$$
   - **Per-Pixel Multi-Segmenter Probability Blending**: Across our three diverse segmentation backbones—(i) U-Net++ with EfficientNet-B4 (`S1E`), (ii) SAM 2.1 Hiera-B+ fine-tuned adapter (`SAM2`), and (iii) 5-fold U-Net++ with ResNet-34 (`e4`)—donor crops $d$ matching target crop $t$ ($\text{IoU} > 0.50$) blend their continuous probability maps $P_d(x,y)$ only within their spatial intersection $\tilde{b}_t \cap \tilde{b}_d$ using a 2D per-pixel weight accumulator $W(x,y)$:
     $$\bar{P}_t(x,y) = \frac{P_t(x,y) + \sum_{d \in \mathcal{D}(t)} w_d \,\mathbb{I}[(x,y) \in \tilde{b}_d]\, P_d(x,y)}{1 + \sum_{d \in \mathcal{D}(t)} w_d \,\mathbb{I}[(x,y) \in \tilde{b}_d]}$$
4. **Implementation**: Implemented in `filseg/segment.py` and `filseg/postprocess.py`.
5. **Justification**: Scalar normalization across non-identical crop boxes attenuates boundary probabilities outside the donor intersection; our 2D per-pixel weight matrix $W(x,y)$ preserves $100\%$ boundary fidelity.
6. **Evidence**: Per-pixel multi-segmenter blending improved **every single detector** on Fold-0 validation (`ms`: $0.4648 \to 0.4716$, $+0.0068\,\text{PQ}$; `F3`: $0.4619 \to 0.4679$, $+0.0060\,\text{PQ}$).

---

## 5. Stage 3: Panoptic-Optimal Cross-Family Continuous Vote-Map Fusion and Non-Overlapping Painting

1. **Problem**: Panoptic Quality ($\text{PQ}$) enforces a hard step-function reward at $\text{IoU} > 0.50$ and penalizes both False Positives and False Negatives with weight $0.5$ in the denominator. Standard multi-model ensembling faces two failure modes: (i) binarizing each member mask before WBF loses sub-threshold boundary consensus on near-miss filaments ($0.35 \le \text{IoU} \le 0.50$), and (ii) uncalibrated softmax confidences from Faster R-CNN ($\text{median} \approx 0.96$) vs. YOLO ($\text{median} \approx 0.58$) admit low-precision marginal clusters near the confidence floor.
2. **Intuition & Mathematical Derivation of PQ-Optimal Emission**:
   Consider an ensemble with current totals $(\text{TP}, \text{FP}, \text{FN})$ and $\text{PQ} = \frac{\sum \text{IoU}}{\text{TP} + 0.5(\text{FP} + \text{FN})}$. Suppose a candidate marginal cluster has probability $p$ of being a True Positive ($\text{IoU} > 0.50$, converting $1\,\text{FN} \to 1\,\text{TP}$ with expected $\overline{\text{IoU}}_{\text{TP}} \approx 0.692$) and probability $1-p$ of being a False Positive (adding $1\,\text{FP}$). Emitting this candidate increases expected $\text{PQ}$ if and only if:
   $$\Delta \text{PQ} > 0 \iff p \cdot \overline{\text{IoU}}_{\text{TP}} > \frac{1}{2}(1 - 2p)\,\text{PQ} \iff p > \frac{0.5\,\text{PQ}}{\overline{\text{IoU}}_{\text{TP}} + \text{PQ}} \approx \frac{0.242}{0.692 + 0.485} = 20.6\% \quad (\text{or } p > 35.0\% \text{ against multi-annotator pooled records})$$
   Empirically, in the marginal confidence zone $c \in [0.30, 0.32)$, clusters lacking confirmation from both our highest-precision Faster R-CNN student (`F3`, precision $0.714$) and our 5-fold teacher-student ensemble (`ms`, precision $0.703$) exhibit precision $< 29.5\%$—below the PQ-neutral threshold—whereas clusters confirmed by both `F3` and `ms`/`s` exceed $68\%$ precision!
3. **Mechanism**:
   - **Continuous WBF Vote-Map Accumulation**: For the 8-member core detector portfolio $\mathcal{M}_{\text{core}} = \{\text{G1}, \text{A}, \text{B}, \text{P1}, \text{P4a}, \text{F0}, \text{F1}, \text{YOLO}\}$, greedy mask-IoU clustering ($\tau_{\text{match}} = 0.40$) retains the **continuous confidence-weighted vote fraction** $V_{\text{core}}(x,y) = \frac{\sum_{m \in \mathcal{C}} c_m M_m(x,y)}{\sum_{m \in \mathcal{C}} c_m} \in [0, 1]$.
   - **Cross-Family Continuous Geometry Refinement**: Matched instances from the 5-fold ResNet34-UNet++ teacher-student models (`ms`, `s`, `main`) and Faster R-CNN student (`F3`) inject their masks into $V_{\text{core}}(x,y)$ via per-pixel weights $(w_{\text{ms}}, w_{\text{s}}, w_{\text{main}}, w_{\text{F3}}) = (0.18, 0.18, 0.12, 0.06)$ before thresholding at $\tau_{\text{vote}} = 0.425$.
   - **Cross-Family Confirmation Calibration & Greedy Non-Overlapping Painting**: Clusters confirmed by both `F3` and `ms`/`s` ($\text{IoU} \ge 0.40$) receive a calibration bonus $\Delta c = +0.020$ against a strict emission floor $c_{\min} = 0.320$, and surviving masks are painted greedily onto the $2048 \times 2048$ canvas in descending confidence order (`drop=0.70`, `min_area=100`).
4. **Evidence**: Across 200 random date-grouped split-half bootstrap evaluations on Fold 0, this cross-family continuous vote-map refinement + confirmation calibration improves validation $\text{PQ}$ from $0.4820$ to **$0.4856$** ($\text{TP}=978, \text{FP}=323, \text{FN}=510$), eliminating $20$ false positives at zero loss of true positives and winning on **$98.0\%$ of held-out date splits**!
