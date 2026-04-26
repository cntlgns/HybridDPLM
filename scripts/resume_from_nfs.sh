#!/bin/bash
# Pull an NFS-archived run back to local disk and resume training with wandb.
# Usage:
#   bash scripts/resume_from_nfs.sh <NFS_SRC_DIR> [extra hydra overrides...]
# Example:
#   bash scripts/resume_from_nfs.sh \
#     /storage/sihun/diffprotein/dplm/train_logs/emb-high_noise-full-ema_lr-fs0 \
#     experiment=dplm2/dplm2_hybrid_650m
#
# The script:
#   1) rsyncs NFS_SRC to LOCAL_LOG_BASE/<name>
#   2) parses the wandb run id from wandb/latest-run (or the newest run-*)
#   3) execs train_and_sync.sh with name=, paths.log_dir=, logger=wandb,
#      logger.wandb.id=, plus any extra overrides passed through.

set -e

LOCAL_LOG_BASE="/data_large/unsynced_store/sihun/diffprotein/dplm/train_logs"
PROJECT_DIR="/data_fast/home/sihun/diffprotein/dplm"
TRAIN_AND_SYNC="${PROJECT_DIR}/scripts/train_and_sync.sh"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <NFS_SRC_DIR> [extra hydra overrides...]"
    exit 1
fi

NFS_SRC="${1%/}"
shift

if [ ! -d "$NFS_SRC" ]; then
    echo "[resume] ERROR: source directory not found: $NFS_SRC"
    exit 1
fi

EXP_NAME=$(basename "$NFS_SRC")
LOCAL_DST="${LOCAL_LOG_BASE}/${EXP_NAME}"

# --- 1) Copy NFS -> local -------------------------------------------------
echo "[resume] Syncing $NFS_SRC -> $LOCAL_DST"
mkdir -p "$LOCAL_DST"
rsync -a --info=progress2 "$NFS_SRC/" "$LOCAL_DST/"

LAST_CKPT="$LOCAL_DST/checkpoints/last.ckpt"
if [ ! -f "$LAST_CKPT" ]; then
    echo "[resume] ERROR: last.ckpt not found at $LAST_CKPT"
    exit 1
fi
echo "[resume] Found checkpoint: $LAST_CKPT"

# --- 2) Parse wandb run id ------------------------------------------------
WANDB_ID=""
WANDB_DIR="$LOCAL_DST/wandb"

if [ -L "$WANDB_DIR/latest-run" ] || [ -d "$WANDB_DIR/latest-run" ]; then
    LATEST_TARGET=$(readlink -f "$WANDB_DIR/latest-run" 2>/dev/null || echo "")
    if [ -n "$LATEST_TARGET" ]; then
        WANDB_ID="${LATEST_TARGET##*-}"
    fi
fi

if [ -z "$WANDB_ID" ] && [ -d "$WANDB_DIR" ]; then
    # fallback: newest run-* folder
    LATEST_RUN=$(ls -dt "$WANDB_DIR"/run-* 2>/dev/null | head -n 1 || true)
    if [ -n "$LATEST_RUN" ]; then
        WANDB_ID="${LATEST_RUN##*-}"
    fi
fi

if [ -z "$WANDB_ID" ]; then
    echo "[resume] WARNING: could not infer wandb run id; a new run will be created"
    WANDB_ARGS=(logger=wandb)
else
    echo "[resume] Resuming wandb run id: $WANDB_ID"
    WANDB_ARGS=(logger=wandb "logger.wandb.id=$WANDB_ID")
fi

# --- 3) Launch train_and_sync --------------------------------------------
echo "[resume] Launching train_and_sync for name=$EXP_NAME"
exec bash "$TRAIN_AND_SYNC" \
    "name=$EXP_NAME" \
    "paths.log_dir=$LOCAL_DST" \
    "${WANDB_ARGS[@]}" \
    "$@"
