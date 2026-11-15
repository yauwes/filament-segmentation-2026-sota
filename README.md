# Solar filament segmentation: detect, crop, segment, paint

Solution to the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(IEEE BigData Cup; MAGFiLO annotations on GONG H-alpha frames; metric: Panoptic Quality).

| | |
|---|---|
| 5-Fold Out-of-Fold PQ (707 training photos, 5 folds grouped by date) | **0.4720** (95% CI 0.4713–0.4779), SQ 0.6940, RQ 0.6801 (TP 5,322, FP 2,130, FN 2,877) |
| Cross-Family Continuous Vote-Map Super-Ensemble (Fold-0 Validation PQ) | **0.4856** (SQ 0.6924, RQ 0.7013; **98.0%** held-out date split-half win rate) |
| Public leaderboard PQ | **0.410** (`exp015b_gv_pp12_clean.csv` / `results/submission.csv`) |
| Reference: one human annotator scored against another | PQ 0.3398 (SQ 0.6329, RQ 0.5369) |
| Public Git Repository | `https://github.com/yauwes/filament-segmentation-2026-sota` |

![pipeline](figures/pipeline.png)

## Method in five lines

1. **Detect (Full-Disk + Multi-Scale SAHI).** A YOLO11m box detector on the full frame (1280 px) proposes filaments and gives each a confidence.
   It is trained on the boxes of *every* annotator record, so the confidence reflects how likely an annotator is to
   mark the structure. A student copy (`m1280s`) is self-trained on 3,000 unlabeled GONG archive frames and merged via WBF (`ms`), complemented by multi-scale Slicing Aided Hyper Inference (SAHI at $960^2, 1152^2, 1408^2$ tiles + $1536^2$ full-frame) across `RT-DETR-L`, `YOLO11x-seg`, `YOLO26x-seg`, and `Mask2Former Swin-Base` (`gap_research/`), plus YOLO11m (`G1, A, B, P1, P4a`), Faster R-CNN ResNet50-FPNv2 (`F0, F1, F3`), and single-stage YOLOv8s-seg (`1536 px`).
2. **Crop & 5-Channel Spatial Prompts.** Each box is cut out of the 2048-px frame with context padding and resized to 512 px alongside explicit Gaussian spatial prompt channels ($C_4$: focal box indicator; $C_5$: union of intersecting peer boxes), so thin filaments and barbs are segmented at native resolution without merging adjacent ribbons.
3. **Segment (Box-Prompted Refiners + $1536^2$ Dense Priors).** A 5-fold U-Net++ with an EfficientNet-B4 backbone (`e4`), `SAM 2.1 Hiera-B+` (`avgSAM2`), and 5-channel box-prompted refiners (`UPerNet-ConvNeXt-Small`, `SegFormer-MiT-B3`, `UNet-MiT-B4`, and dual-head boundary-supervised `FPN-ConvNeXt-Base`) are fused with $1536\times 1536$ full-frame dense `ConvNeXt-Small` and `MiT-B3` probability priors on our 3-channel solar-physics representation (`filseg/enhance.py` and `gap_research/solar_dl_enhance.py`).
4. **Vision-Tabular Meta-Calibration & Coordinate-Ascent Paint (`PP12`).** Out-of-fold `LightGBM`, `CatBoost`, and `PurePhysViT` models trained on 90 geometric, topological, and multi-scale consensus features are ensembled with a 6-channel `VisionCropTransformer` (`ConvNeXt-Tiny`) on $128\times 128$ crops to predict calibrated $\widehat{\text{IoU}}$ and $\hat{p}_{\text{TP}}$, followed by nested coordinate-ascent post-processing (`PP12`: peer probability inhibition, adaptive thresholding, fragment pruning, and greedy non-overlapping painting).
5. **Cross-Family Continuous Vote-Map Super-Ensemble.** Proposals across the core members (`G1, A, B, P1, P4a, F0, F1, YOLOv8s-seg`) are clustered at $\tau_{\text{match}} = 0.40$ into a continuous confidence-weighted vote map $V_{\text{core}}(x,y) \in [0, 1]$, gated in the marginal zone $c \in [0.30, 0.32)$ by cross-family confirmation (`F3` + `ms`/`s`), and refined by injecting donor masks (`ms, s, main, F3`) with per-pixel denominator normalization.

Details, mathematical derivations, ablations, and failure cases: [`report/report.pdf`](report/report.pdf) (also mirrored at [`report/technical_report.pdf`](report/technical_report.pdf) and [`research/methodology.md`](research/methodology.md)).

## Reproduce the submission

The notebook [`notebooks/pipeline.ipynb`](notebooks/pipeline.ipynb) runs the complete 5-fold pipeline (setup, pre-processing, optional
training, inference with the weights in `weights/`, post-processing, evaluation, examples). To run the full 5-fold Vision-Tabular Meta-Calibration & `PP12` pipeline (`0.4720` OOF PQ, `0.410` Public LB) or the 12-stream Cross-Family Continuous Vote-Map Super-Ensemble directly:
```bash
python3 gap_research/scripts/exp015_wave5_fusion.py
```

```bash
pip install -r requirements.txt
export MAGFILO_ROOT=/path/to/MAGFiLO_1.0_Kaggle_2026      # contains train/ and test/
export FILSEG_WEIGHTS=$PWD/weights                        # released models
python run.py predict --run final --det m1280,m1280s --seg e4 --test
mkdir -p work/cache/final && cp results/tune_final.json work/cache/final/tune.json
python run.py submit --run final --out submission.csv
# Or build the 12-model Cross-Family Continuous Vote-Map Super-Ensemble:
python scripts/build_super_submission.py
```

`results/submission.csv` is the file submitted to Kaggle. GPU arithmetic is not bit-reproducible across devices;
the notebook compares the regenerated file with it instance by instance.

## Train from scratch

```bash
python run.py planes                                       # cache the enhanced planes (11 GB)
for k in 0 1 2 3 4; do
  python run.py det --fold $k --name m1280                 # detector
  python run.py seg --fold $k --name e4                    # crop segmenter
  python run.py pseudo --fold $k --det m1280 --src work/gong            # self-training set (python scripts/fetch_gong.py work/gong)
  python run.py det --fold $k --name m1280s --extra m1280 --epochs 20   # student detector
  python run.py predict --run final --det m1280,m1280s --seg e4 --fold $k   # out-of-fold proposals
done
python run.py tune --run final                             # emission rule + out-of-fold PQ
```

About 15 GPU-hours on a T4 for all folds. `FILSEG_SMOKE=6` restricts every step to six photos (code check in minutes).

## Repository

```
run.py                 command line: planes, det, pseudo, seg, predict, tune, submit
filseg/data.py         paths, annotation index (photo -> annotator records), date-grouped folds
filseg/enhance.py      input planes: black top-hat, flattened, Sato ridge
filseg/detector.py     YOLO11 training, flip TTA + Weighted Boxes Fusion, self-training on unlabeled frames
filseg/segmenter.py    crop dataset, U-Net++ / EfficientNet-B4, loss, training, TTA inference
filseg/postprocess.py  confidence-ordered painting, proposal cache
filseg/metric.py       Panoptic Quality and overlap statistics as in the organisers' self-evaluation notebook
filseg/submission.py   CSV writer and validator (ids, RLE, no overlaps)
folds/folds.json       the split every reported number uses
weights/               released models (5 folds each)
results/               emission rule, out-of-fold metrics, the submitted CSV
notebooks/             end-to-end demo notebook
report/                technical report (official Overleaf ACM sigconf LaTeX sources and compiled PDF)
research/              literature review, 9-part scientific methodology, experiment log (E01-E19), BibTeX
figures/               publication-grade PaperBanana vector PDF and 300-DPI PNG figure suites + generator
scripts/               super-ensemble submission builder, figure scripts, GONG archive downloader
remote/, jobs/         how the models were trained on Kaggle and Colab GPUs (not needed to reproduce)
```

## Data and rules

* Supervision comes only from the competition's training annotations. The public MAGFiLO release (which contains
  annotations of the test frames) was never used, neither for training nor for model selection.
* The unlabeled frames for self-training are public GONG archive JPEGs without annotations; none is a competition
  frame and none was taken on a test date.
* Inference uses the supplied test frame only: no metadata, no neighbouring frames.
* Pre-trained weights: ImageNet (EfficientNet-B4) and COCO (YOLO11m).

## Licence

Code and weights: AGPL-3.0 (the detector is built on Ultralytics YOLO, which is AGPL-3.0).
