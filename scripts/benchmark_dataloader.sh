#!/usr/bin/env bash

set -euo pipefail

# Pure DataLoader benchmark: measure only data loading time for different
# num_workers values, without running any model forward/backward.
#
# Usage (from project root, after activating `model` env):
#   bash scripts/benchmark_dataloader.sh
#
# You can override defaults via environment variables, e.g.:
#   MAX_SEQ_LEN=200 BATCH_SIZE=32768 NUM_WORKERS_LIST="0,4,8,12,16,24,32" \
#     bash scripts/benchmark_dataloader.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

DATA_PATH="data/processed/taobao_sequences.pt"

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Processed data not found at ${DATA_PATH}."
  echo "Please run the preprocessing script first."
  exit 1
fi

MAX_SEQ_LEN="${MAX_SEQ_LEN:-50}"
BATCH_SIZE="${BATCH_SIZE:-32768}"
NUM_WORKERS_LIST="${NUM_WORKERS_LIST:-1,2,4,8,12,16,24,32}"

python -m src.training.benchmark_dataloader \
  --data-path "${DATA_PATH}" \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers-list "${NUM_WORKERS_LIST}"


