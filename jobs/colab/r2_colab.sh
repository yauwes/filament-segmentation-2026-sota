# Colab lane: baseline detector for fold 0, then the self-training round for folds 0, 2, 4:
# the fold's own detector labels 3,000 unlabeled GONG archive frames, a student is trained on real + pseudo labels.
set -x
W=$FILSEG_WORK/weights
python run.py det --fold 0 --name m1280 --workers 8
[ -d /content/gong/images ] || kaggle datasets download ajikajadeh/filseg-gong-unlabeled -p /content/gong --unzip -q
python run.py pseudo --fold 0 --det m1280 --src /content/gong/images --procs 10
python run.py det --fold 0 --name m1280s --extra m1280 --epochs 20 --workers 8
# teachers of folds 2 and 4 are trained on Kaggle (kernel filseg-det-m1280-a2)
t0=$(date +%s)
until kaggle kernels status ajikajadeh/filseg-det-m1280-a2 | grep -q -i complete; do
  kaggle kernels status ajikajadeh/filseg-det-m1280-a2 | grep -q -i -E "error|cancel" && exit 1
  [ $(( $(date +%s) - t0 )) -gt 7200 ] && exit 1; sleep 120
done
kaggle kernels output ajikajadeh/filseg-det-m1280-a2 -p /content/kout/a2 > /dev/null
cp /content/kout/a2/work/weights/det_m1280_f?.pt $W/
for k in 2 4; do
  python run.py pseudo --fold $k --det m1280 --src /content/gong/images --procs 10
  python run.py det --fold $k --name m1280s --extra m1280 --epochs 20 --workers 8
done
ls -la $W
