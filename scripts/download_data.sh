#!/usr/bin/env bash
# Download a small slice of the Open X-Embodiment `kuka` (QT-Opt) dataset and the
# UCI Robot Execution Failures dataset. Public HTTPS, no credentials needed.
set -euo pipefail
cd "$(dirname "$0")/.."
BASE="https://storage.googleapis.com/gresearch/robotics/kuka/0.1.0"
SHARDS="${SHARDS:-00000 00341 00682}"
mkdir -p data/raw/kuka data/raw/uci
curl -sSf -o data/raw/kuka/features.json "$BASE/features.json"
curl -sSf -o data/raw/kuka/dataset_info.json "$BASE/dataset_info.json"
for s in $SHARDS; do
  f="kuka-train.tfrecord-${s}-of-01024"
  [ -s "data/raw/kuka/$f" ] || curl -sSf -o "data/raw/kuka/$f" "$BASE/$f"
done
if [ ! -s data/raw/uci/robot_execution_failures.zip ]; then
  curl -sSfL -o data/raw/uci/robot_execution_failures.zip \
    "https://archive.ics.uci.edu/static/public/138/robot+execution+failures.zip"
fi
(cd data/raw/uci && unzip -oq robot_execution_failures.zip)
echo "done"
