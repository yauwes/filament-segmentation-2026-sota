# Code check: the new inference + tuning code with two weights of the earlier project (detector G1, segmenter S1E,
# both trained on folds 1-4). The earlier code scored this pair at PQ 0.4589 on fold 0 with detector flip TTA.
set -ex
mkdir -p $FILSEG_WORK/weights
W=$(dirname "$(find /kaggle/input -name S1E_best.pth | head -1)")
cp $W/det_first_best.pt $FILSEG_WORK/weights/det_old_f0.pt
cp $W/S1E_best.pth $FILSEG_WORK/weights/seg_old_f0.pth
python $REPO/run.py predict --run old --det old --seg old --fold 0
python $REPO/run.py tune --run old --procs 4
