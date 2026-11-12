# Truncation-aware crop segmenter "e4x": each fold's e4 model is fine-tuned for 6 epochs with partial boxes and a wider,
# variable padding, so that it follows a filament beyond a box that covers only its core. Then out-of-fold runs with
# growing crops at three paddings (compare with run main, PQ 0.4582). Starts when the GPU is free (student of fold 4 done).
set -x
W=$FILSEG_WORK/weights
until [ -f $W/det_m1280s_f4.pt ]; do sleep 60; done
for k in 0 1 2 3 4; do python run.py seg --fold $k --name e4x --init e4 --trunc 0.4 --pad-hi 0.6 --lr 1e-4 --epochs 6 --workers 10; done
for v in 40 25 20; do
  for k in 0 1 2 3 4; do python run.py predict --run x${v}g --det m1280 --seg e4x --fold $k --pad 0.$v --grow 2; done
  python run.py tune --run x${v}g --procs 10
done
