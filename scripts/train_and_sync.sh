#!/bin/bash
# Train on local disk, then sync results to NFS storage.
# Usage: bash scripts/train_and_sync.sh <train.py args...>
#
# Expects paths.log_dir to point to LOCAL_LOG_BASE/<name> (set by sweep_hybrid.py).
# After training completes, rsyncs the entire log directory to NFS.

set -e

LOCAL_LOG_BASE="/data_large/unsynced_store/sihun/diffprotein/dplm/train_logs"
NFS_LOG_BASE="/storage/sihun/diffprotein/dplm/train_logs"

PROJECT_DIR="/data_fast/home/sihun/diffprotein/dplm"
PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
TRAIN_SCRIPT="${PROJECT_DIR}/train.py"

# --- Step 1: Run training ---
echo "[train_and_sync] Starting training..."
set +e
"$PYTHON_BIN" "$TRAIN_SCRIPT" "$@"
TRAIN_EXIT=$?
set -e

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "[train_and_sync] Training failed with exit code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

echo "[train_and_sync] Training finished successfully."

# --- Step 2: Find the log directory and sync to NFS ---
# Extract the experiment name from the hydra overrides.
# We look for paths.log_dir=<LOCAL_LOG_BASE>/<name> in the arguments.
LOG_DIR=""
for arg in "$@"; do
    if [[ "$arg" == paths.log_dir=* ]]; then
        LOG_DIR="${arg#paths.log_dir=}"
        break
    fi
done

if [ -z "$LOG_DIR" ]; then
    echo "[train_and_sync] Warning: paths.log_dir not found in args, skipping sync."
    exit 0
fi

if [ ! -d "$LOG_DIR" ]; then
    echo "[train_and_sync] Warning: log directory not found: $LOG_DIR, skipping sync."
    exit 0
fi

# Derive NFS destination: replace LOCAL_LOG_BASE prefix with NFS_LOG_BASE
EXP_NAME=$(basename "$LOG_DIR")
NFS_DEST="${NFS_LOG_BASE}/${EXP_NAME}"

echo "[train_and_sync] Syncing local -> NFS"
echo "  Source: $LOG_DIR"
echo "  Dest:   $NFS_DEST"

mkdir -p "$NFS_DEST"
rsync -a "$LOG_DIR/" "$NFS_DEST/"

echo "[train_and_sync] Sync complete. Cleaning up local copy..."
rm -rf "$LOG_DIR"
echo "[train_and_sync] Done."
