#!/usr/bin/env bash
# Batch launcher: calls scripts/run_hybrid_eval.py for each checkpoint directory below.
# Edit CKPT_DIRS to add/remove entries.

# bash scripts/run_hybrid_eval_batch.sh

set -euo pipefail

PROJECT_DIR="/data_fast/home/sihun/diffprotein/dplm"
PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
EVAL_SCRIPT="${PROJECT_DIR}/scripts/run_hybrid_eval.py"

CKPT_DIRS=(
    "${PROJECT_DIR}/train_logs/FT3L/emb-high_noise-16-ema_lora_lr-fs0/checkpoints"
    "${PROJECT_DIR}/train_logs/FT3L/emb-high_noise-16-ema_lora_lr-fs0.1/checkpoints"
    "${PROJECT_DIR}/train_logs/FT3L/emb-high_noise-16-ema_lora_lr-fs0.3/checkpoints"
    "${PROJECT_DIR}/train_logs/FT3L/emb-high_noise-16ln-ema_lora_lr-fs0/checkpoints"
    "${PROJECT_DIR}/train_logs/FT3L/emb-high_noise-16ln-ema_lora_lr-fs0.1/checkpoints"
    "${PROJECT_DIR}/train_logs/FT3L/emb-high_noise-16ln-ema_lora_lr-fs0.3/checkpoints"
    "${PROJECT_DIR}/train_logs/FT3L/emb-wide_noise-16-ema_lora_lr-fs0/checkpoints"
    "${PROJECT_DIR}/train_logs/FT3L/emb-wide_noise-16-ema_lora_lr-fs0.1/checkpoints"
    "${PROJECT_DIR}/train_logs/FT3L/emb-wide_noise-16-ema_lora_lr-fs0.3/checkpoints"
    "${PROJECT_DIR}/train_logs/FT3L/emb-wide_noise-16ln-ema_lora_lr-fs0/checkpoints"
    "${PROJECT_DIR}/train_logs/FT3L/emb-wide_noise-16ln-ema_lora_lr-fs0.1/checkpoints"
    "${PROJECT_DIR}/train_logs/FT3L/emb-wide_noise-16ln-ema_lora_lr-fs0.3/checkpoints"
    "${PROJECT_DIR}/train_logs/FT3L/emb-xhigh_noise-16-ema_lora_lr-fs0/checkpoints"
    "${PROJECT_DIR}/train_logs/FT3L/emb-xhigh_noise-16-ema_lora_lr-fs0.1/checkpoints"
    "${PROJECT_DIR}/train_logs/FT3L/emb-xhigh_noise-16-ema_lora_lr-fs0.3/checkpoints"
    "${PROJECT_DIR}/train_logs/FT3L/emb-xhigh_noise-16ln-ema_lora_lr-fs0/checkpoints"
    "${PROJECT_DIR}/train_logs/FT3L/emb-xhigh_noise-16ln-ema_lora_lr-fs0.1/checkpoints"
    # "${PROJECT_DIR}/train_logs/FT3L/emb-xhigh_noise-16ln-ema_lora_lr-fs0.3/checkpoints"
)

cd "${PROJECT_DIR}"

for ckpt_dir in "${CKPT_DIRS[@]}"; do
    echo "================================================================"
    echo "Launching: ${ckpt_dir}"
    echo "================================================================"
    if [[ ! -d "${ckpt_dir}" ]]; then
        echo "  [skip] not a directory"
        continue
    fi
    "${PYTHON_BIN}" "${EVAL_SCRIPT}" "${ckpt_dir}"
done
