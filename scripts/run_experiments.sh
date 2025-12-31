#!/usr/bin/env bash

set -euo pipefail

# Simple driver script to compare LSTM and Transformer
# on Taobao User Behavior sequences with different max_seq_len.
#
# Usage (from project root, after activating `model` env):
#   bash scripts/run_experiments.sh
#
# Make sure you have already run:
#   python -m src.data.preprocess_taobao \
#     --raw-path data/raw/UserBehavior.csv \
#     --out-path data/processed/taobao_sequences.pt \
#     --max-users 100000

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

DATA_PATH="data/processed/taobao_sequences.pt"

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Processed data not found at ${DATA_PATH}."
  echo "Please run the preprocessing script first."
  exit 1
fi

MAX_SEQ_LENS=("50" "200")
MODELS=("lstm" "transformer")

for L in "${MAX_SEQ_LENS[@]}"; do
  for M in "${MODELS[@]}"; do
    LOG_DIR="runs/${M}_len${L}"
    echo "=== Running model=${M}, max_seq_len=${L}, log_dir=${LOG_DIR} ==="
    python -m src.training.train \
      --data-path "${DATA_PATH}" \
      --model-type "${M}" \
      --max-seq-len "${L}" \
      --batch-size 8192 \
      --epochs 2 \
      --log-dir "${LOG_DIR}"
  done
done

echo "All experiments completed."



