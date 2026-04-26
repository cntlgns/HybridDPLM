#!/bin/bash
# Evaluate hybrid-diffusion inference on the *base* (non-finetuned) DPLM2-650M,
# sweeping the hybrid noise schedule.
#
# The first two positional args are the noise-schedule range. Their meaning
# depends on NOISE_SPACE:
#   - NOISE_SPACE=embedding -> NOISE_MIN/MAX are sigma_min/sigma_max (VE-SDE).
#   - NOISE_SPACE=onehot    -> NOISE_MIN/MAX are r_min/r_max (Eq. 10).
#
# Usage: bash run_hybrid_eval_sigma.sh <noise_min> <noise_max> [dataset] [sampling] [max_iter] [batch_size] [noise_space] [tag] [cutoff_layer0_attn_residual] [use_normal_emb]
#   cutoff_layer0_attn_residual: 0 or 1 (default: 0)
#   use_normal_emb:              0 or 1 (default: 0)
#
# Example:
#   bash scripts/run_hybrid_eval_sigma.sh 0.5 5.0 cameo2022 argmax 100 50 embedding low 0 0
#   bash scripts/run_hybrid_eval_sigma.sh 0.01 0.25 cameo2022 argmax 100 50 onehot default 1 1

set -e

# Activate project virtualenv
source /data_fast/home/sihun/diffprotein/dplm/.venv/bin/activate

NOISE_MIN="$1"
NOISE_MAX="$2"
DATASET="${3:-all}"            # cameo2022, PDB_date, or all
SAMPLING="${4:-argmax}"
MAX_ITER="${5:-100}"
BATCH_SIZE="${6:-50}"
NOISE_SPACE="${7:-embedding}"
TAG="${8:-}"                   # optional human-readable tag (e.g. "mid_noise")
CUTOFF_L0_RES="${9:-0}"        # 1 = enable cutoff_layer0_attn_residual
USE_NORMAL_EMB="${10:-0}"      # 1 = enable row-wise L2 normalization of embedding rows
MODEL_NAME="${MODEL_NAME:-airkingbd/dplm2_650m}"

if [ -z "$NOISE_MIN" ] || [ -z "$NOISE_MAX" ]; then
    echo "Usage: bash run_hybrid_eval_sigma.sh <noise_min> <noise_max> [dataset] [sampling] [max_iter] [batch_size] [noise_space] [tag] [cutoff_layer0_attn_residual] [use_normal_emb]"
    exit 1
fi

# Dispatch noise args based on noise_space.
# onehot uses r_min/r_max; embedding uses sigma_min/sigma_max.
if [ "$NOISE_SPACE" = "onehot" ]; then
    NOISE_FLAGS="--r_min $NOISE_MIN --r_max $NOISE_MAX"
    NOISE_PREFIX="r"
else
    NOISE_FLAGS="--sigma_min $NOISE_MIN --sigma_max $NOISE_MAX"
    NOISE_PREFIX="s"
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Suffix when layer0 attn residual cutoff is on, so results don't collide
if [ "$CUTOFF_L0_RES" = "1" ]; then
    L0_SUFFIX="_cutL0"
    CUTOFF_FLAG="--cutoff_layer0_attn_residual"
else
    L0_SUFFIX=""
    CUTOFF_FLAG=""
fi

# Suffix when embedding-row normalization is on, so results don't collide
if [ "$USE_NORMAL_EMB" = "1" ]; then
    NORMEMB_SUFFIX="_normE"
    NORMEMB_FLAG="--use_normal_emb true"
else
    NORMEMB_SUFFIX=""
    NORMEMB_FLAG="--use_normal_emb false"
fi

# Directory name for this noise config (prefix is "s" for sigma, "r" for onehot r)
MODEL_TAG=$(basename "$MODEL_NAME")            # e.g. dplm2_650m
if [ -n "$TAG" ]; then
    RUN_NAME="${MODEL_TAG}_${NOISE_SPACE}_${TAG}_${NOISE_PREFIX}${NOISE_MIN}-${NOISE_MAX}${L0_SUFFIX}${NORMEMB_SUFFIX}"
else
    RUN_NAME="${MODEL_TAG}_${NOISE_SPACE}_${NOISE_PREFIX}${NOISE_MIN}-${NOISE_MAX}${L0_SUFFIX}${NORMEMB_SUFFIX}"
fi

# --- Local storage for results only ---
LOCAL_BASE="/data_large/unsynced_store/sihun/diffprotein/dplm"
REMOTE_RESULTS="generation-results"  # relative to PROJECT_ROOT (on network storage)
RESULT_NS="hybrid_noft_sigma_sweep2"

DATASETS=()
if [ "$DATASET" = "all" ]; then
    DATASETS=("cameo2022" "PDB_date")
else
    DATASETS=("$DATASET")
fi

for DS in "${DATASETS[@]}"; do
    INPUT_FASTA="data-bin/${DS}/struct.fasta"
    LOCAL_SAVE_DIR="${LOCAL_BASE}/generation-results/${RESULT_NS}/${RUN_NAME}/${DS}"
    REMOTE_SAVE_DIR="${REMOTE_RESULTS}/${RESULT_NS}/${RUN_NAME}/${DS}"

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
    echo "  Dataset:     $DS"
    echo "  Base model:  $MODEL_NAME (no finetuning)"
    echo "  noise_space: $NOISE_SPACE"
    echo "  noise_min:   $NOISE_MIN  (${NOISE_PREFIX}_min)"
    echo "  noise_max:   $NOISE_MAX  (${NOISE_PREFIX}_max)"
    echo "  cutoff_L0:   $CUTOFF_L0_RES"
    echo "  normal_emb:  $USE_NORMAL_EMB"
    echo "  Save to:     $LOCAL_SAVE_DIR/inverse_folding (local)"
    echo "  Sync to:     $REMOTE_SAVE_DIR/inverse_folding (network)"
    echo "  Sampling:    $SAMPLING"
    echo "  Max iter:    $MAX_ITER"
    echo "=============================================="

    touch /data_fast/home/sihun/diffprotein/dplm/src/byprot/models/dplm2/dplm2_hybrid.py

    # Step 1: Generate (HF weights, no ckpt)
    python generate_dplm2_hybrid.py \
        --model_name "$MODEL_NAME" \
        --task inverse_folding \
        --input_fasta_path "$INPUT_FASTA" \
        --saveto "$LOCAL_SAVE_DIR" \
        --sampling_strategy "$SAMPLING" \
        --max_iter "$MAX_ITER" \
        --batch_size "$BATCH_SIZE" \
        --noise_space "$NOISE_SPACE" \
        $NOISE_FLAGS \
        $CUTOFF_FLAG \
        $NORMEMB_FLAG \
        --save_pdb True

    echo "[Done] Generation saved to $LOCAL_SAVE_DIR/inverse_folding"

    # Step 2: Evaluate metrics
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
