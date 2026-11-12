# Ablation on 2 x T4 (out-of-fold only, no test): the baseline run without detector flips / without segmenter flips.
set -ex
DET=${DET:-m1280}; SEG=${SEG:-e4}
W=$FILSEG_WORK/weights; mkdir -p $W /tmp/planes; ln -sfn /tmp/planes $FILSEG_WORK/planes
for k in 0 1 2 3 4; do cp "$(find /kaggle/input -name "det_${DET}_f$k.pt" | head -1)" "$(find /kaggle/input -name "seg_${SEG}_f$k.pth" | head -1)" $W/; done
python $REPO/run.py planes --procs 4
P="python $REPO/run.py predict --det $DET --seg $SEG"
(for k in 0 1 2 3 4; do CUDA_VISIBLE_DEVICES=0 $P --run nodettta --no-det-tta --fold $k; done) &
(for k in 0 1 2 3 4; do CUDA_VISIBLE_DEVICES=1 $P --run nosegtta --no-seg-tta --fold $k; done) &
wait
python $REPO/run.py tune --run nodettta --procs 4
python $REPO/run.py tune --run nosegtta --procs 4
rm -f $FILSEG_WORK/planes; rm -rf $FILSEG_WORK/cache/*/oof $W
