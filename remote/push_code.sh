#!/bin/bash
# Publish the repository (code only) as the private Kaggle dataset the remote runners pull from.
set -e
cd "$(dirname "$0")/.."
STAGE=$(mktemp -d)
cp -r filseg folds jobs remote run.py "$STAGE"/
find "$STAGE" -name __pycache__ -prune -exec rm -rf {} +
echo '{"id": "ajikajadeh/filseg-a100-code", "title": "filseg a100 code", "licenses": [{"name": "CC0-1.0"}]}' > "$STAGE/dataset-metadata.json"
if kaggle -W datasets status ajikajadeh/filseg-a100-code >/dev/null 2>&1; then
  kaggle -W datasets version -p "$STAGE" -m "${1:-update}" --dir-mode zip -q -d
else
  kaggle -W datasets create -p "$STAGE" --dir-mode zip -q
fi
rm -rf "$STAGE"
