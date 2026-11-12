# Stage 2 on 2 x T4: U-Net++ / EfficientNet-B4 crop segmenter, five folds split over the two GPUs.
set -ex
mkdir -p /tmp/planes $FILSEG_WORK && ln -sfn /tmp/planes $FILSEG_WORK/planes
python $REPO/run.py planes --procs 4
(for k in 0 2 4; do CUDA_VISIBLE_DEVICES=0 python $REPO/run.py seg --fold $k --name e4 --workers 2; done) &
(for k in 1 3; do CUDA_VISIBLE_DEVICES=1 python $REPO/run.py seg --fold $k --name e4 --workers 2; done) &
wait
rm -f $FILSEG_WORK/planes
ls -la $FILSEG_WORK/weights/seg_e4_f0.pth $FILSEG_WORK/weights/seg_e4_f1.pth $FILSEG_WORK/weights/seg_e4_f2.pth $FILSEG_WORK/weights/seg_e4_f3.pth $FILSEG_WORK/weights/seg_e4_f4.pth
