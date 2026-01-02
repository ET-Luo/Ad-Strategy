#!/usr/bin/env bash

set -euo pipefail

# Quick benchmark script to find a good num_workers value for DataLoader.
# It runs a short training job (1 epoch) with different num_workers values
# and records the wall-clock time for each run.
#
# Usage (from project root, after activating `model` env):
#   bash scripts/benchmark_num_workers.sh
#
# You can optionally override defaults via environment variables:
#   MODEL=lstm MAX_SEQ_LEN=50 BATCH_SIZE=32768 EPOCHS=1 bash scripts/benchmark_num_workers.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

DATA_PATH="data/processed/taobao_sequences.pt"

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Processed data not found at ${DATA_PATH}."
  echo "Please run the preprocessing script first."
  exit 1
fi

# Default hyper-parameters for the benchmark (can be overridden from env).
MODEL="${MODEL:-lstm}"          # or transformer
MAX_SEQ_LEN="${MAX_SEQ_LEN:-50}"
BATCH_SIZE="${BATCH_SIZE:-32768}"
EPOCHS="${EPOCHS:-1}"

# Candidate num_workers values to test.
NUM_WORKERS_CANDIDATES=(0 4 8 12 16 24 32)

echo "Benchmarking num_workers for model=${MODEL}, max_seq_len=${MAX_SEQ_LEN}, batch_size=${BATCH_SIZE}, epochs=${EPOCHS}"
echo "Data path: ${DATA_PATH}"
echo

for NW in "${NUM_WORKERS_CANDIDATES[@]}"; do
  echo "---- num_workers=${NW} ----"
  start_ts=$(date +%s)

  python -m src.training.train \
    --data-path "${DATA_PATH}" \
    --model-type "${MODEL}" \
    --max-seq-len "${MAX_SEQ_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --num-workers "${NW}" \
    --log-dir "runs/benchmark_${MODEL}_len${MAX_SEQ_LEN}_nw${NW}" >/dev/null 2>&1

  end_ts=$(date +%s)
  elapsed=$((end_ts - start_ts))
  echo "num_workers=${NW} -> elapsed=${elapsed}s"
  echo
done

echo "Benchmark finished. Choose the smallest num_workers that gives near-minimum elapsed time."






