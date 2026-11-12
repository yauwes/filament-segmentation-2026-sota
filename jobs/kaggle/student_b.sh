# Self-training round on 2 x T4, folds 1 and 3: teacher = the fold's baseline detector (output of kernel
# filseg-det-m1280-b), 3,000 unlabeled GONG archive frames, student = same recipe on real + pseudo labels.
set -ex
mkdir -p $FILSEG_WORK/weights
cp $(find /kaggle/input -name "det_m1280_f1.pt" | head -1) $(find /kaggle/input -name "det_m1280_f3.pt" | head -1) $FILSEG_WORK/weights/
SRC=$(dirname "$(find /kaggle/input -name "20110110002214Mh.jpg" | head -1)")
CUDA_VISIBLE_DEVICES=0 python $REPO/run.py pseudo --fold 1 --det m1280 --src $SRC --procs 4
CUDA_VISIBLE_DEVICES=1 python $REPO/run.py pseudo --fold 3 --det m1280 --src $SRC --procs 4
CUDA_VISIBLE_DEVICES=0 python $REPO/run.py det --fold 1 --name m1280s --extra m1280 --epochs 20 --workers 2 --cache none &
sleep 120
CUDA_VISIBLE_DEVICES=1 python $REPO/run.py det --fold 3 --name m1280s --extra m1280 --epochs 20 --workers 2 --cache none &
wait
rm -rf $FILSEG_WORK/det_data $FILSEG_WORK/pseudo/images
ls -la $FILSEG_WORK/weights/det_m1280s_f1.pt $FILSEG_WORK/weights/det_m1280s_f3.pt
