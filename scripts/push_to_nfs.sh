#!/bin/bash
# Push local train_logs back to NFS storage.
#   - .hydra/, wandb/, *.log: overwrite NFS with local
#   - checkpoints/: only transfer files missing on NFS (--ignore-existing)
#
# Usage:
#   bash scripts/push_to_nfs.sh                       # all 4 FT4 settings
#   bash scripts/push_to_nfs.sh <name1> [<name2> ...] # specific settings

set -e

LOCAL_BASE="/data_large/unsynced_store/sihun/diffprotein/dplm/train_logs"
NFS_BASE="/storage/sihun/diffprotein/dplm/train_logs/FT7"

# Throttle write throughput to NFS (KB/s for rsync; suffix M = MB/s).
# Override: BWLIMIT=100M bash scripts/push_to_nfs.sh
BWLIMIT="${BWLIMIT:-50M}"

# Optional: only transfer checkpoints whose filenames start with one of these
# prefixes. Edit the list below, or override via env var (space-separated):
#   CKPT_PREFIXES="step_6499.0 step_7000.0" bash scripts/push_to_nfs.sh
# Empty list = transfer all checkpoints.
DEFAULT_CKPT_PREFIXES=(
    step_14008.0
)

if [ -n "${CKPT_PREFIXES:-}" ]; then
    read -r -a CKPT_PREFIXES_ARR <<< "$CKPT_PREFIXES"
else
    CKPT_PREFIXES_ARR=("${DEFAULT_CKPT_PREFIXES[@]}")
fi

CKPT_FILTERS=()
if [ ${#CKPT_PREFIXES_ARR[@]} -gt 0 ]; then
    for prefix in "${CKPT_PREFIXES_ARR[@]}"; do
        CKPT_FILTERS+=(--include="${prefix}*")
    done
    CKPT_FILTERS+=(--exclude='*')
fi

DEFAULT_NAMES=(
    emb-2xhigh_noise-full-low_lr-fs0
    emb-xhigh_noise-full-low_lr-fs0
    emb-xhigh_noise-full-low_lr-fs0-normemb
    # emb-high_noise-full-low_lr-fs0-normemb
)

if [ $# -ge 1 ]; then
    NAMES=("$@")
else
    NAMES=("${DEFAULT_NAMES[@]}")
fi

for name in "${NAMES[@]}"; do
    SRC="${LOCAL_BASE}/${name}"
    DST="${NFS_BASE}/${name}"

    if [ ! -d "$SRC" ]; then
        echo "[push] SKIP: local missing $SRC"
        continue
    fi
    mkdir -p "$DST"

    echo "=============================================================="
    echo "[push] $name"
    echo "  src: $SRC"
    echo "  dst: $DST"
    echo "  bwlimit: $BWLIMIT"
    if [ ${#CKPT_PREFIXES_ARR[@]} -gt 0 ]; then
        echo "  ckpt prefixes: ${CKPT_PREFIXES_ARR[*]}"
    fi
    echo "=============================================================="

    # 1) Overwrite metadata: .hydra, wandb, top-level log files
    #    Exclude checkpoints/ here; handled separately below.
    echo "[push] (1/2) syncing .hydra, wandb, logs (overwrite)"
    rsync -a --info=progress2 --bwlimit="$BWLIMIT" \
        --exclude='checkpoints/' \
        "$SRC/" "$DST/"

    # 2) Checkpoints: only copy what's missing on NFS
    if [ -d "$SRC/checkpoints" ]; then
        echo "[push] (2/2) syncing checkpoints (ignore-existing)"
        mkdir -p "$DST/checkpoints"
        rsync -a --info=progress2 --bwlimit="$BWLIMIT" --ignore-existing \
            "${CKPT_FILTERS[@]}" \
            "$SRC/checkpoints/" "$DST/checkpoints/"
    else
        echo "[push] (2/2) no checkpoints dir at $SRC/checkpoints"
    fi
done

echo "[push] Done."
