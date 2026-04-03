#!/bin/bash
# Usage: bash run_candi_eval.sh <ckpt_path> [dataset] [sampling_strategy] [max_iter]
#
# Example:
#   bash run_candi_eval.sh train_logs/candi_const_weight_ft/checkpoints/step_63999.0-loss_0.00.ckpt
#   bash run_candi_eval.sh train_logs/candi_const_weight_ft/checkpoints/step_63999.0-loss_0.00.ckpt cameo2022
#   bash run_candi_eval.sh train_logs/candi_const_weight_ft/checkpoints/step_63999.0-loss_0.00.ckpt all "annealing@2.0:0.1" 50

set -e

# Activate project virtualenv
source /data_fast/home/sihun/diffprotein/dplm/.venv/bin/activate

CKPT_PATH="$1"
DATASET="${2:-all}"           # cameo2022, PDB_date, or all (default: all)
SAMPLING="${3:-argmax}"
MAX_ITER="${4:-100}"
BATCH_SIZE="${5:-50}"

if [ -z "$CKPT_PATH" ]; then
    echo "Usage: bash run_candi_eval.sh <ckpt_path> [dataset] [sampling_strategy] [max_iter]"
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

DATASETS=()
if [ "$DATASET" = "all" ]; then
    DATASETS=("cameo2022" "PDB_date")
else
    DATASETS=("$DATASET")
fi

for DS in "${DATASETS[@]}"; do
    INPUT_FASTA="data-bin/${DS}/struct.fasta"
    SAVE_DIR="generation-results/candi_invfold_test/${EXP_NAME}/${DS}/${CKPT_BASENAME}"

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
    echo "  Ckpt:      $CKPT_PATH"
    echo "  Save to:   $SAVE_DIR/inverse_folding"
    echo "  Sampling:  $SAMPLING"
    echo "  Max iter:  $MAX_ITER"
    echo "=============================================="

    # Touch the candi module to force reload
    touch /home/sihun/diffprotein/dplm/src/byprot/models/dplm2/dplm2_candi.py

    # Step 1: Generate
    python generate_dplm2_candi.py \
        --ckpt_path "$CKPT_PATH" \
        --task inverse_folding \
        --input_fasta_path "$INPUT_FASTA" \
        --saveto "$SAVE_DIR" \
        --sampling_strategy "$SAMPLING" \
        --max_iter "$MAX_ITER" \
        --batch_size "$BATCH_SIZE" \
        --save_pdb True

    echo "[Done] Generation saved to $SAVE_DIR/inverse_folding"

    # Step 2: Evaluate metrics
    EVAL_FASTA_DIR="$SAVE_DIR/inverse_folding"

    python src/byprot/utils/protein/evaluator_dplm2.py \
        -cn inverse_folding \
        inference.input_fasta_dir="$EVAL_FASTA_DIR" \
        inference.metadata.csv_path="$METADATA_CSV" \
        inference.metadata.data_dir="$METADATA_DATA_DIR"

    echo "[Done] Evaluation complete for $DS"
    echo ""
done

echo "All done!"
