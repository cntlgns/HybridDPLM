#!/bin/bash
# Usage: bash run_hybrid_eval.sh <ckpt_path> [dataset] [sampling_strategy] [max_iter]
#
# Example:
#   bash run_hybrid_eval.sh train_logs/candi_const_weight_ft/checkpoints/step_63999.0-loss_0.00.ckpt
#   bash run_hybrid_eval.sh train_logs/candi_const_weight_ft/checkpoints/step_63999.0-loss_0.00.ckpt cameo2022
#   bash run_hybrid_eval.sh train_logs/candi_const_weight_ft/checkpoints/step_63999.0-loss_0.00.ckpt all "annealing@2.0:0.1" 50

set -e

# Activate project virtualenv
source /data_fast/home/sihun/diffprotein/dplm/.venv/bin/activate

CKPT_PATH="$1"
DATASET="${2:-all}"           # cameo2022, PDB_date, or all (default: all)
SAMPLING="${3:-argmax}"
MAX_ITER="${4:-100}"
BATCH_SIZE="${5:-50}"

if [ -z "$CKPT_PATH" ]; then
    echo "Usage: bash run_hybrid_eval.sh <ckpt_path> [dataset] [sampling_strategy] [max_iter]"
    exit 1
fi

if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: checkpoint not found: $CKPT_PATH"
    exit 1
fi

# Parse experiment name and step name from the checkpoint path
# e.g. train_logs/candi_const_weight_ft/checkpoints/step_63999.0-loss_0.00.ckpt
#   -> exp_name=candi_const_weight_ft, step_name=step_63999.0-loss_0.00
CKPT_BASENAME=$(basename "$CKPT_PATH" .ckpt)
CKPT_DIR=$(dirname "$CKPT_PATH")
EXP_DIR=$(dirname "$CKPT_DIR")
EXP_NAME=$(basename "$EXP_DIR")

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# --- Local storage for results only ---
LOCAL_BASE="/data_large/unsynced_store/sihun/diffprotein/dplm"
REMOTE_RESULTS="generation-results"  # relative to PROJECT_ROOT (on network storage)

# Verify .hydra/config.yaml exists (needed by model loading)
ORIG_HYDRA_CFG="${EXP_DIR}/.hydra/config.yaml"
if [ ! -f "$ORIG_HYDRA_CFG" ]; then
    echo "Error: .hydra/config.yaml not found at $ORIG_HYDRA_CFG"
    exit 1
fi

DATASETS=()
if [ "$DATASET" = "all" ]; then
    DATASETS=("cameo2022" "PDB_date")
else
    DATASETS=("$DATASET")
fi

for DS in "${DATASETS[@]}"; do
    INPUT_FASTA="data-bin/${DS}/struct.fasta"
    # Save results to local storage
    LOCAL_SAVE_DIR="${LOCAL_BASE}/generation-results/hybrid_invfold_test/${EXP_NAME}/${DS}/${CKPT_BASENAME}"
    # Corresponding remote path for rsync target
    REMOTE_SAVE_DIR="${REMOTE_RESULTS}/hybrid_invfold_test/${EXP_NAME}/${DS}/${CKPT_BASENAME}"

    # Select metadata csv and data_dir matching the dataset
    if [ "$DS" = "PDB_date" ]; then
        METADATA_CSV="${PROJECT_ROOT}/data-bin/metadata/pdb_date.csv"
        METADATA_DATA_DIR="${PROJECT_ROOT}/data-bin/PDB_date"
    else
        METADATA_CSV="${PROJECT_ROOT}/data-bin/metadata/pdb_afdb_cameo.csv"
        METADATA_DATA_DIR="${PROJECT_ROOT}/data-bin"
    fi

    if [ ! -f "$INPUT_FASTA" ]; then
        echo "Warning: input fasta not found: $INPUT_FASTA, skipping $DS"
        continue
    fi

    echo "=============================================="
    echo "  Dataset:   $DS"
    echo "  Ckpt:      $CKPT_PATH (shared storage)"
    echo "  Save to:   $LOCAL_SAVE_DIR/inverse_folding (local)"
    echo "  Sync to:   $REMOTE_SAVE_DIR/inverse_folding (network)"
    echo "  Sampling:  $SAMPLING"
    echo "  Max iter:  $MAX_ITER"
    echo "=============================================="

    # Touch the hybrid module to force reload
    # touch /home/sihun/diffprotein/dplm/src/byprot/models/dplm2/dplm2_hybrid.py

    # Step 1: Generate (using local checkpoint, saving to local storage)
    python generate_dplm2_hybrid.py \
        --ckpt_path "$CKPT_PATH" \
        --task inverse_folding \
        --input_fasta_path "$INPUT_FASTA" \
        --saveto "$LOCAL_SAVE_DIR" \
        --sampling_strategy "$SAMPLING" \
        --max_iter "$MAX_ITER" \
        --batch_size "$BATCH_SIZE" \
        --save_pdb True

    echo "[Done] Generation saved to $LOCAL_SAVE_DIR/inverse_folding"

    # Step 2: Evaluate metrics (reading from local storage)
    EVAL_FASTA_DIR="$LOCAL_SAVE_DIR/inverse_folding"

    python src/byprot/utils/protein/evaluator_dplm2.py \
        -cn inverse_folding \
        inference.input_fasta_dir="$EVAL_FASTA_DIR" \
        inference.metadata.csv_path="$METADATA_CSV" \
        inference.metadata.data_dir="$METADATA_DATA_DIR"

    echo "[Done] Evaluation complete for $DS"

    # Step 3: Sync results back to network storage
    mkdir -p "$REMOTE_SAVE_DIR"
    rsync -a "$LOCAL_SAVE_DIR/" "$REMOTE_SAVE_DIR/"
    echo "[Synced] $LOCAL_SAVE_DIR -> $REMOTE_SAVE_DIR"

    echo ""
done

echo "All done!"
