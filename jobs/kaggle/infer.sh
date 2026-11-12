# One pipeline run on 2 x T4: out-of-fold proposals for all folds, emission rule tuned on out-of-fold PQ, test
# proposals from the 5-fold ensemble, submission CSV. RUN / DET / SEG / EXTRA come from the environment (push_kernel.sh).
set -ex
RUN=${RUN:-main}; DET=${DET:-m1280}; SEG=${SEG:-e4}
W=$FILSEG_WORK/weights; mkdir -p $W /tmp/planes; ln -sfn /tmp/planes $FILSEG_WORK/planes
for d in ${DET//,/ }; do for k in 0 1 2 3 4; do cp "$(find /kaggle/input -name "det_${d}_f$k.pt" | head -1)" $W/; done; done
for s in ${SEG//,/ }; do for k in 0 1 2 3 4; do cp "$(find /kaggle/input -name "seg_${s}_f$k.pth" | head -1)" $W/; done; done
ls -la $W
python --version; pip list 2>/dev/null | grep -i -E "^(torch|torchvision|ultralytics|segmentation.models.pytorch|timm|numpy|opencv[a-z-]*|pycocotools|scikit-image|scipy|matplotlib|scikit-learn|polars|pillow) "
python $REPO/run.py planes --procs 4
P="python $REPO/run.py predict --run $RUN --det $DET --seg $SEG ${EXTRA//+/ }"      # EXTRA: more predict flags, + for spaces
(for k in 0 2 4; do CUDA_VISIBLE_DEVICES=0 $P --fold $k; done) &
(for k in 1 3; do CUDA_VISIBLE_DEVICES=1 $P --fold $k; done) &
wait
python $REPO/run.py tune --run $RUN --procs 4
CUDA_VISIBLE_DEVICES=0 $P --test &
CUDA_VISIBLE_DEVICES=1 $P --test --reverse &
wait
python $REPO/run.py submit --run $RUN --out /kaggle/working/submission_$RUN.csv
rm -f $FILSEG_WORK/planes
(cd $FILSEG_WORK && tar czf /kaggle/working/cache_$RUN.tgz cache/$RUN && rm -rf cache/$RUN/oof cache/$RUN/test weights)
