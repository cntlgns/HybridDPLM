#!/bin/bash
# Run DPLM2 inverse-folding eval over multiple seeds for a single
# (model, dataset, sampling, remasking, max_iter) combo.
#
# Generation loads the model ONCE and iterates seeds; evaluation runs per-seed.
# Metrics are aggregated into seed_summary.csv via aggregate_seed_metrics.py.
#
# Usage:
#   bash scripts/run_dplm2_if_seeds.sh <dataset> <model_name> <seeds_csv> \
#                                       [sampling] [remasking] [max_iter] \
#                                       [batch_size] [result_subdir]
#
# Output layout:
#   generation-results/<result_subdir>/<dataset>/<sampling>_<remasking>_iter<N>/
#       seed_<S>/inverse_folding/aatype/{eval,...}
#       seed_summary.csv (after aggregation)

set -e

source /data_fast/home/sihun/diffprotein/dplm/.venv/bin/activate

DATASET="$1"
MODEL_NAME="$2"
SEEDS="$3"                       # comma-separated, e.g. "42,43,44"
SAMPLING="${4:-annealing@2.0:0.1}"
REMASKING="${5:-uncond}"
MAX_ITER="${6:-100}"
BATCH_SIZE="${7:-50}"
RESULT_SUBDIR="${8:-dplm2_650m_invfold_seedsweep}"

if [ -z "$DATASET" ] || [ -z "$MODEL_NAME" ] || [ -z "$SEEDS" ]; then
    echo "Usage: bash run_dplm2_if_seeds.sh <dataset> <model_name> <seeds_csv> \\"
    echo "       [sampling] [remasking] [max_iter] [batch_size] [result_subdir]"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

COMBO_DIR="${SAMPLING}_${REMASKING}_iter${MAX_ITER}"
SAVE_BASE="${PROJECT_ROOT}/generation-results/${RESULT_SUBDIR}/${DATASET}/${COMBO_DIR}"

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

BIT_MODEL_FLAG=""
if [[ "${MODEL_NAME}" == "dplm2_bit_650m" ]]; then
    BIT_MODEL_FLAG="--bit_model"
fi

echo "=============================================="
echo "  Dataset:    $DATASET"
echo "  Model:      airkingbd/${MODEL_NAME}"
echo "  Seeds:      $SEEDS"
echo "  Sampling:   $SAMPLING   Remask: $REMASKING   Iter: $MAX_ITER   BS: $BATCH_SIZE"
echo "  Save base:  $SAVE_BASE"
echo "=============================================="

mkdir -p "$SAVE_BASE"

# ─── Step 1: Generate (single model load, multi-seed) ─────────────────────────
python generate_dplm2.py \
    --model_name "airkingbd/${MODEL_NAME}" \
    --task inverse_folding \
    ${BIT_MODEL_FLAG} \
    --input_fasta_path "$INPUT_FASTA" \
    --batch_size "$BATCH_SIZE" \
    --max_iter "$MAX_ITER" \
    --unmasking_strategy deterministic \
    --sampling_strategy "$SAMPLING" \
    --remasking_strategy "$REMASKING" \
    --seeds "$SEEDS" \
    --skip_if_generated \
    --saveto "$SAVE_BASE"

# ─── Step 2: Evaluate each seed ───────────────────────────────────────────────
IFS=',' read -ra SEED_ARR <<< "$SEEDS"
for SEED in "${SEED_ARR[@]}"; do
    SEED_DIR="${SAVE_BASE}/seed_${SEED}"
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

# ─── Step 3: Aggregate across all seeds ───────────────────────────────────────
python scripts/aggregate_seed_metrics.py --root "$SAVE_BASE"

echo "All done for $MODEL_NAME / $DATASET / $COMBO_DIR"
