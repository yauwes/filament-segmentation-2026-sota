# Finish fold 4 self-trained student detector after probe_grow completes (sequential, --cache none to avoid OOM).
set -ex
while pgrep -f "probe_grow" >/dev/null; do sleep 30; done
python run.py det --fold 4 --name m1280s --extra m1280 --epochs 20 --workers 4 --cache none
ls -la $FILSEG_WORK/weights/det_m1280s_f4.pt
