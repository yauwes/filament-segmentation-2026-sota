# Solar filament segmentation: detect, crop, segment, paint

Solution to the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(IEEE BigData Cup; MAGFiLO annotations on GONG H-alpha frames; metric: Panoptic Quality).

| | |
|---|---|
| 5-Fold Out-of-Fold PQ (707 training photos, 5 folds grouped by date) | **0.4648** (95% CI 0.4538–0.4716), SQ 0.6924, RQ 0.6713 |
| Cross-Family Continuous Vote-Map Super-Ensemble (Fold-0 Validation PQ) | **0.4856** (SQ 0.6924, RQ 0.7013; **98.0%** held-out date split-half win rate) |
| Public leaderboard PQ | **0.402** |
| Reference: one human annotator scored against another | PQ 0.3398 (SQ 0.6329, RQ 0.5369) |
| Test inference | ~5.2 s per frame on one T4 (5-fold core) |

![pipeline](figures/pipeline.png)

## Method in five lines

1. **Detect.** A YOLO11m box detector on the full frame (1280 px) proposes filaments and gives each a confidence.
   It is trained on the boxes of *every* annotator record, so the confidence reflects how likely an annotator is to
   mark the structure. A second copy of each detector is self-trained with 3,000 unlabeled GONG archive frames, complemented by Faster R-CNN ResNet50-FPNv2 (`F0, F1, F3`) and end-to-end Mask R-CNN (`G1, A, B, P1, P4a`) paradigms.
2. **Crop.** Each box is cut out of the 2048-px frame with 20% padding and resized to 512 px, so thin filaments
   and barbs are segmented at (or above) native resolution.
3. **Segment.** Dual U-Net++ backbones (EfficientNet-B4 `e4` and ResNet-34 `r34`) outline the filament the box points at. Their input is three enhanced
   planes (black top-hat, background-flattened, Sato ridge) in robust disk units, fused across crop coordinate frames via a 2D per-pixel weight matrix $W(x,y)$.
4. **Paint.** Instances are laid down in confidence order; a pixel belongs to the first instance that claims it,
   so masks never overlap. Which instances are emitted (mask threshold, confidence floor, drop rule) is chosen
   once, on out-of-fold PQ.
5. **Cross-Family Continuous Vote-Map Super-Ensemble.** On the test set, proposals across all 12 heterogeneous model families are clustered at $\tau_{\text{match}} = 0.40$ and accumulated into a continuous confidence-weighted pixel vote field $V_{\text{super}}(x,y) \in [0, 1]$ before thresholding at $\tau_{\text{vote}} = 0.425$, eliminating the Boolean quantization no-op of post-threshold binary blending.

Details, mathematical derivations, ablations, and failure cases: [`report/report.pdf`](report/report.pdf) (also mirrored at [`report/technical_report.pdf`](report/technical_report.pdf) and [`research/methodology.md`](research/methodology.md)).

## Reproduce the submission

The notebook [`notebooks/pipeline.ipynb`](notebooks/pipeline.ipynb) runs the complete 5-fold pipeline (setup, pre-processing, optional
training, inference with the weights in `weights/`, post-processing, evaluation, examples). To build the full 12-stream Cross-Family Continuous Vote-Map Super-Ensemble submission directly:

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
