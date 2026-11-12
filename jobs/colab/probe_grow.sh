# Probe on out-of-fold predictions (baseline detectors and segmenters): crop windows that grow when a mask touches
# their border (run g2), and a larger crop padding (run p35). Compared with run main (PQ 0.4582).
set -x
W=$FILSEG_WORK/weights
[ -f $W/seg_e4_f0.pth ] || { kaggle kernels output ajikajadeh/filseg-seg-e4 -p /content/kout/seg > /dev/null; cp /content/kout/seg/work/weights/seg_e4_f?.pth $W/; }
[ -f $W/det_m1280_f1.pt ] || { kaggle kernels output ajikajadeh/filseg-det-m1280-b -p /content/kout/detb > /dev/null; cp /content/kout/detb/work/weights/det_m1280_f?.pt $W/; }
ls -la $W
python run.py planes --procs 6
for k in 0 1 2 3 4; do python run.py predict --run g2 --det m1280 --seg e4 --fold $k --grow 2 --seg-batch 8; done
python run.py tune --run g2 --procs 8
for k in 0 1 2 3 4; do python run.py predict --run p35 --det m1280 --seg e4 --fold $k --pad 0.35 --seg-batch 8; done
python run.py tune --run p35 --procs 8
