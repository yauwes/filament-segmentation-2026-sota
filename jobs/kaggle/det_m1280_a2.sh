# Stage 1 on 2 x T4: YOLO11m box detector at 1280 px, folds 2 and 4 (one per GPU).
set -ex
CUDA_VISIBLE_DEVICES=0 python $REPO/run.py det --fold 2 --name m1280 --workers 2 &
sleep 120
CUDA_VISIBLE_DEVICES=1 python $REPO/run.py det --fold 4 --name m1280 --workers 2 &
wait
rm -rf $FILSEG_WORK/det_data
ls -la $FILSEG_WORK/weights/det_m1280_f2.pt $FILSEG_WORK/weights/det_m1280_f4.pt
