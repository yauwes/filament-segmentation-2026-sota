#!/bin/bash
# Run jobs/kaggle/<JOB>.sh on a Kaggle GPU kernel:  [KSRC="owner/kernel ..."] remote/push_kernel.sh JOB [dataset slug ...]
# KSRC lists other kernels whose outputs (trained weights) are mounted under /kaggle/input; KENV="A=1 B=2" passes
# parameters to the job; KNAME overrides the kernel name (default: the job name).
set -e
cd "$(dirname "$0")/.."
JOB=$1; shift
K=$(mktemp -d)
sed -e "s/@@JOB@@/$JOB/" -e "s|@@ENV@@|$KENV|" remote/kernel_main.py > "$K/main.py"
DS='"ajikajadeh/filseg-a100-code"'; for d in "$@"; do DS="$DS, \"$d\""; done
KS=""; for k in $KSRC; do KS="$KS${KS:+, }\"$k\""; done
SLUG=$(echo "filseg-${KNAME:-$JOB}" | tr '_' '-')
cat > "$K/kernel-metadata.json" <<META
{"id": "ajikajadeh/$SLUG", "title": "$SLUG", "code_file": "main.py", "language": "python", "kernel_type": "script",
 "is_private": true, "enable_gpu": true, "enable_tpu": false, "enable_internet": true,
 "dataset_sources": [$DS], "competition_sources": ["filament-segmentation-2026"], "kernel_sources": [$KS], "model_sources": []}
META
kaggle -W kernels push -p "$K"
rm -rf "$K"
