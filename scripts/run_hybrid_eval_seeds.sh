#!/bin/bash
# Run hybrid-diffusion inverse-folding eval over multiple seeds for a single ckpt+dataset.
#
# Generation loads the model ONCE and iterates seeds; evaluation runs per-seed.
# Metrics are aggregated into seed_summary.csv and the whole tree is rsynced to NFS.
#
# Usage:
#   bash scripts/run_hybrid_eval_seeds.sh <ckpt_path> <dataset> <seeds_csv> \
#                                         [sampling] [max_iter] [batch_size] [result_subdir]
# Example:
#   bash scripts/run_hybrid_eval_seeds.sh \
#       train_logs/candi_const_weight_ft/checkpoints/step_63999.0-loss_0.00.ckpt \
#       cameo2022 42,43,44,45,46 argmax 500 5 maxiter500/hybrid_invfold_FT7_seedsweep

set -e

source /data_fast/home/sihun/diffprotein/dplm/.venv/bin/activate

CKPT_PATH="$1"
DATASET="$2"
SEEDS="$3"                       # comma-separated list, e.g. "42,43,44"
SAMPLING="${4:-argmax}"
MAX_ITER="${5:-100}"
BATCH_SIZE="${6:-10}"
RESULT_SUBDIR="${7:-hybrid_invfold_seedsweep}"   # path under generation-results/

if [ -z "$CKPT_PATH" ] || [ -z "$DATASET" ] || [ -z "$SEEDS" ]; then
    echo "Usage: bash run_hybrid_eval_seeds.sh <ckpt_path> <dataset> <seeds_csv> [sampling] [max_iter] [batch_size]"
    exit 1
fi
if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: checkpoint not found: $CKPT_PATH"
    exit 1
fi
if [ "$DATASET" = "all" ]; then
    echo "Error: 'all' not supported here; submit one job per dataset."
    exit 1
fi

# Parse experiment name and ckpt basename from the checkpoint path
CKPT_BASENAME=$(basename "$CKPT_PATH" .ckpt)
CKPT_DIR=$(dirname "$CKPT_PATH")
EXP_DIR=$(dirname "$CKPT_DIR")
EXP_NAME=$(basename "$EXP_DIR")

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

ORIG_HYDRA_CFG="${EXP_DIR}/.hydra/config.yaml"
if [ ! -f "$ORIG_HYDRA_CFG" ]; then
    echo "Error: .hydra/config.yaml not found at $ORIG_HYDRA_CFG"
    exit 1
fi

# ─── Paths ────────────────────────────────────────────────────────────────────
LOCAL_BASE="/data_large/unsynced_store/sihun/diffprotein/dplm"
REMOTE_RESULTS="generation-results"

LOCAL_SAVE_DIR="${LOCAL_BASE}/generation-results/${RESULT_SUBDIR}/${EXP_NAME}/${DATASET}/${CKPT_BASENAME}"
REMOTE_SAVE_DIR="${REMOTE_RESULTS}/${RESULT_SUBDIR}/${EXP_NAME}/${DATASET}/${CKPT_BASENAME}"

INPUT_FASTA="data-bin/${DATASET}/struct.fasta"
if [ "$DATASET" = "PDB_date" ]; then
    METADATA_CSV="${PROJECT_ROOT}/data-bin/metadata/pdb_date.csv"
    METADATA_DATA_DIR="${PROJECT_ROOT}/data-bin/PDB_date"
else
    METADATA_CSV="${PROJECT_ROOT}/data-bin/metadata/pdb_afdb_cameo.csv"
    METADATA_DATA_DIR="${PROJECT_ROOT}/data-bin"
fi
if [ ! -f "$INPUT_FASTA" ]; then
    echo "Error: input fasta not found: $INPUT_FASTA"
    exit 1
fi

echo "=============================================="
echo "  Dataset:    $DATASET"
echo "  Ckpt:       $CKPT_PATH"
echo "  Seeds:      $SEEDS"
echo "  Save base:  $LOCAL_SAVE_DIR (local)"
echo "  Sync to:    $REMOTE_SAVE_DIR (network)"
echo "  Sampling:   $SAMPLING   Max iter: $MAX_ITER   Batch: $BATCH_SIZE"
echo "=============================================="

mkdir -p "$LOCAL_SAVE_DIR"
mkdir -p "$REMOTE_SAVE_DIR"

touch /home/sihun/diffprotein/dplm/src/byprot/models/dplm2/dplm2_hybrid.py

# ─── Step 1: Generate (single model load, multi-seed) ─────────────────────────    --use_ema \
python generate_dplm2_hybrid.py \
    --ckpt_path "$CKPT_PATH" \
    --task inverse_folding \
    --input_fasta_path "$INPUT_FASTA" \
    --saveto "$LOCAL_SAVE_DIR" \
    --sampling_strategy "$SAMPLING" \
    --max_iter "$MAX_ITER" \
    --batch_size "$BATCH_SIZE" \
    --seeds "$SEEDS" \
    --skip_if_generated \
    --save_pdb True

# ─── Step 2: Evaluate each seed ───────────────────────────────────────────────
IFS=',' read -ra SEED_ARR <<< "$SEEDS"
for SEED in "${SEED_ARR[@]}"; do
    SEED_DIR="${LOCAL_SAVE_DIR}/seed_${SEED}"
    EVAL_FASTA_DIR="${SEED_DIR}/inverse_folding"
    DONE_MARKER="${EVAL_FASTA_DIR}/aatype/eval/inverse_fold_metrics.csv"

    if [ -f "$DONE_MARKER" ]; then
        echo "[skip-eval] seed=${SEED}: $DONE_MARKER exists"
        continue
    fi
    if [ ! -d "$EVAL_FASTA_DIR" ]; then
        echo "[warn] seed=${SEED}: no generation dir at $EVAL_FASTA_DIR — skipping"
        continue
    fi

    echo "---- Evaluating seed=${SEED} ----"
    python src/byprot/utils/protein/evaluator_dplm2.py \
        -cn inverse_folding \
        inference.input_fasta_dir="$EVAL_FASTA_DIR" \
        inference.metadata.csv_path="$METADATA_CSV" \
        inference.metadata.data_dir="$METADATA_DATA_DIR"
done

# ─── Step 3: Push new seed dirs to NFS first ──────────────────────────────────
# Aggregation runs on NFS so it sees the union of every seed ever generated
# (this run's new seeds + any prior runs' seed_*/ dirs already on NFS).
rsync -a "$LOCAL_SAVE_DIR/" "$REMOTE_SAVE_DIR/"

# ─── Step 4: Aggregate across all seeds present on NFS ────────────────────────
python scripts/aggregate_seed_metrics.py --root "$REMOTE_SAVE_DIR"
echo "[Synced] $LOCAL_SAVE_DIR -> $REMOTE_SAVE_DIR"

echo "All done for $EXP_NAME / $DATASET / $CKPT_BASENAME"
