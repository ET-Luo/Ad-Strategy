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

MAX_SEQ_LENS=("250")
# MODELS=("lstm" "transformer")
MODELS=("lstm")
EPOCHS=10
TIME_LIMIT=25  # Minutes

for L in "${MAX_SEQ_LENS[@]}"; do
  for M in "${MODELS[@]}"; do
    LOG_DIR="runs/${M}_len${L}"
    echo "=== Starting/Resuming experiment: model=${M}, max_seq_len=${L}, log_dir=${LOG_DIR} ==="
    
    while :; do
      python -m src.training.train \
        --data-path "${DATA_PATH}" \
        --model-type "${M}" \
        --max-seq-len "${L}" \
        --batch-size 8192 \
        --epochs "${EPOCHS}" \
        --num-workers 4 \
        --gpu-id 0 \
        --log-dir "${LOG_DIR}" \
        --time-limit "${TIME_LIMIT}"
      
      # Check if training is completed
      CHECKPOINT="${LOG_DIR}/checkpoint.pt"
      if [[ -f "${CHECKPOINT}" ]]; then
        # Use a small python snippet to check if the saved epoch matches target epochs
        CURRENT_EPOCH=$(python -c "import torch; print(torch.load('${CHECKPOINT}', map_location='cpu')['epoch'])")
        if [[ "${CURRENT_EPOCH}" -ge "${EPOCHS}" ]]; then
          echo "=== Experiment model=${M}, max_seq_len=${L} COMPLETED at epoch ${CURRENT_EPOCH} ==="
          break
        fi
        echo "=== Experiment model=${M}, max_seq_len=${L} interrupted at epoch ${CURRENT_EPOCH}. Restarting... ==="
      else
        echo "=== No checkpoint found. Something went wrong. Restarting... ==="
      fi
      
      sleep 2
    done
  done
done

echo "All experiments completed."



