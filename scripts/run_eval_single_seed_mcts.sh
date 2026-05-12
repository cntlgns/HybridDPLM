#!/bin/bash
# Per-seed MCTS generation + evaluation (mirrors run_eval_single_seed.sh).
#
# One SLURM job = one GPU = one (variant, ckpt, dataset, sampling, MCTS hp,
# seed) combo. Generates with that one seed via generate_dplm2_mcts.py,
# evaluates the produced fasta, rsyncs to NFS, then re-aggregates.
#
# Layout under generation-results/<result_subdir>/:
#     <EXP>/<DATASET>/<CKPT_BASENAME>/<strategy_tag>/seed_<s>/...
#                                    /<strategy_tag>/seed_summary.csv
#
# Usage:
#   bash scripts/run_eval_single_seed_mcts.sh \
#       <variant> <ckpt_path> <dataset> <seed> <sampling> \
#       [max_iter] [batch_size] [result_subdir] [strategy_tag] \
#       [remasking_strategy] [sample_noise_every_step] \
#       [mcts_iterations] [mcts_expansions] [mcts_c_uct] \
#       [mcts_reward_threshold] [mcts_num_outputs]
#
# variant:                 baseline | hybrid | noise
# remasking_strategy:      uncond | cond | no_remask (baseline only)
# sample_noise_every_step: true | false | "" (noise only)

set -e

source /data_fast/home/sihun/diffprotein/dplm/.venv/bin/activate

VARIANT="$1"
CKPT_PATH="$2"
DATASET="$3"
SEED="$4"
SAMPLING="$5"
MAX_ITER="${6:-100}"
BATCH_SIZE="${7:-1}"
RESULT_SUBDIR="${8:-mcts_eval_per_seed_sweep}"
STRATEGY_TAG="${9:-default}"
REMASKING_STRATEGY="${10:-no_remask}"
SAMPLE_NOISE_EVERY_STEP="${11:-}"
[ "$SAMPLE_NOISE_EVERY_STEP" = "_" ] && SAMPLE_NOISE_EVERY_STEP=""
MCTS_ITERATIONS="${12:-50}"
MCTS_EXPANSIONS="${13:-4}"
MCTS_C_UCT="${14:-0.01}"
MCTS_REWARD_THRESHOLD="${15:-0.99}"
MCTS_NUM_OUTPUTS="${16:-10}"

if [ -z "$VARIANT" ] || [ -z "$CKPT_PATH" ] || [ -z "$DATASET" ] \
   || [ -z "$SEED" ] || [ -z "$SAMPLING" ]; then
    echo "Usage: bash run_eval_single_seed_mcts.sh <variant> <ckpt_path> <dataset> <seed> <sampling> [max_iter] [batch_size] [result_subdir] [strategy_tag] [remasking_strategy] [sample_noise_every_step] [mcts_iterations] [mcts_expansions] [mcts_c_uct] [mcts_reward_threshold] [mcts_num_outputs]"
    exit 1
fi
if [ "$DATASET" = "all" ]; then
    echo "Error: 'all' not supported here; submit one job per dataset."
    exit 1
fi

EXTRA_GEN_ARGS=()
case "$VARIANT" in
    baseline)
        EXTRA_GEN_ARGS=(--unmasking_strategy deterministic
                        --remasking_strategy "$REMASKING_STRATEGY")
        ;;
    noise)
        if [ -n "$SAMPLE_NOISE_EVERY_STEP" ]; then
            EXTRA_GEN_ARGS=(--sample_noise_every_step "$SAMPLE_NOISE_EVERY_STEP")
        fi
        ;;
    hybrid)
        ;;
    *)
        echo "Error: unknown variant '$VARIANT' (expected: baseline | hybrid | noise)"
        exit 1
        ;;
esac

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# CKPT_PATH may be either absolute .ckpt or "hf:<huggingface_model_name>".
if [[ "$CKPT_PATH" == hf:* ]]; then
    HF_MODEL="${CKPT_PATH#hf:}"
    EXP_NAME="pretrained_hf"
    CKPT_BASENAME="${HF_MODEL##*/}"
    GEN_LOAD_ARGS=(--model_name "$HF_MODEL")
    echo "  Source:     huggingface ($HF_MODEL)"
else
    if [ ! -f "$CKPT_PATH" ]; then
        echo "Error: checkpoint not found: $CKPT_PATH"
        exit 1
    fi
    CKPT_BASENAME=$(basename "$CKPT_PATH" .ckpt)
    CKPT_DIR=$(dirname "$CKPT_PATH")
    EXP_DIR=$(dirname "$CKPT_DIR")
    EXP_NAME=$(basename "$EXP_DIR")
    ORIG_HYDRA_CFG="${EXP_DIR}/.hydra/config.yaml"
    if [ ! -f "$ORIG_HYDRA_CFG" ]; then
        echo "Error: .hydra/config.yaml not found at $ORIG_HYDRA_CFG"
        exit 1
    fi
    GEN_LOAD_ARGS=(--ckpt_path "$CKPT_PATH")
fi

# Suffix EXP_NAME for variant-specific knobs (matches run_eval_single_seed.sh).
if [ "$VARIANT" = "baseline" ]; then
    EXP_NAME="${EXP_NAME}_${REMASKING_STRATEGY}"
fi
if [ "$VARIANT" = "noise" ] && [ -n "$SAMPLE_NOISE_EVERY_STEP" ]; then
    case "$SAMPLE_NOISE_EVERY_STEP" in
        true)  EXP_NAME="${EXP_NAME}_redraw" ;;
        false) EXP_NAME="${EXP_NAME}_freeze" ;;
        *)
            echo "Error: sample_noise_every_step must be 'true' or 'false' (got: $SAMPLE_NOISE_EVERY_STEP)"
            exit 1
            ;;
    esac
fi

LOCAL_BASE="/data_large/unsynced_store/sihun/diffprotein/dplm"
REMOTE_RESULTS="generation-results"

LOCAL_STRATEGY_DIR="${LOCAL_BASE}/generation-results/${RESULT_SUBDIR}/${EXP_NAME}/${DATASET}/${CKPT_BASENAME}/${STRATEGY_TAG}"
REMOTE_STRATEGY_DIR="${REMOTE_RESULTS}/${RESULT_SUBDIR}/${EXP_NAME}/${DATASET}/${CKPT_BASENAME}/${STRATEGY_TAG}"

INPUT_FASTA="data-bin/${DATASET}/struct.fasta"
if [ "$DATASET" = "PDB_date" ]; then
    METADATA_CSV="${PROJECT_ROOT}/data-bin/metadata/pdb_date.csv"
    METADATA_DATA_DIR="${PROJECT_ROOT}/data-bin/PDB_date"
elif [[ "$DATASET" == cath_* ]]; then
    METADATA_CSV="${PROJECT_ROOT}/data-bin/metadata/${DATASET}.csv"
    METADATA_DATA_DIR="${PROJECT_ROOT}/data-bin"
elif [ "$DATASET" = "cameo2022_small" ]; then
    METADATA_CSV="${PROJECT_ROOT}/data-bin/metadata/cameo2022_small.csv"
    METADATA_DATA_DIR="${PROJECT_ROOT}/data-bin"
else
    METADATA_CSV="${PROJECT_ROOT}/data-bin/metadata/pdb_afdb_cameo.csv"
    METADATA_DATA_DIR="${PROJECT_ROOT}/data-bin"
fi
if [ ! -f "$INPUT_FASTA" ]; then
    echo "Error: input fasta not found: $INPUT_FASTA"
    exit 1
fi
if [ ! -f "$METADATA_CSV" ]; then
    echo "Error: metadata CSV not found: $METADATA_CSV"
    exit 1
fi

echo "=============================================="
echo "  Variant:      $VARIANT"
echo "  Ckpt:         $CKPT_PATH"
echo "  Dataset:      $DATASET"
echo "  Seed:         $SEED"
echo "  Sampling:     $SAMPLING   (tag: $STRATEGY_TAG)"
echo "  MCTS:         M=$MCTS_ITERATIONS K=$MCTS_EXPANSIONS c_uct=$MCTS_C_UCT τ=$MCTS_REWARD_THRESHOLD N=$MCTS_NUM_OUTPUTS"
if [ "$VARIANT" = "baseline" ]; then
    echo "  Remasking:    $REMASKING_STRATEGY"
fi
if [ "$VARIANT" = "noise" ] && [ -n "$SAMPLE_NOISE_EVERY_STEP" ]; then
    echo "  NoiseEvery:   $SAMPLE_NOISE_EVERY_STEP"
fi
echo "  Max iter:     $MAX_ITER   Batch: $BATCH_SIZE"
echo "  Save base:    $LOCAL_STRATEGY_DIR (local)"
echo "  Sync to:      $REMOTE_STRATEGY_DIR (network)"
echo "=============================================="

mkdir -p "$LOCAL_STRATEGY_DIR"
mkdir -p "$REMOTE_STRATEGY_DIR"

# ─── Step 1: MCTS generate (single seed) ──────────────────────────────────────
python generate_dplm2_mcts.py \
    --variant "$VARIANT" \
    "${GEN_LOAD_ARGS[@]}" \
    "${EXTRA_GEN_ARGS[@]}" \
    --task inverse_folding \
    --input_fasta_path "$INPUT_FASTA" \
    --metadata_csv "$METADATA_CSV" \
    --metadata_data_dir "$METADATA_DATA_DIR" \
    --saveto "$LOCAL_STRATEGY_DIR" \
    --sampling_strategy "$SAMPLING" \
    --max_iter "$MAX_ITER" \
    --batch_size "$BATCH_SIZE" \
    --mcts_iterations "$MCTS_ITERATIONS" \
    --mcts_expansions "$MCTS_EXPANSIONS" \
    --mcts_c_uct "$MCTS_C_UCT" \
    --mcts_reward_threshold "$MCTS_REWARD_THRESHOLD" \
    --mcts_num_outputs "$MCTS_NUM_OUTPUTS" \
    --seeds "$SEED" \
    --skip_if_generated

# ─── Step 2: Evaluate that seed ───────────────────────────────────────────────
SEED_DIR="${LOCAL_STRATEGY_DIR}/seed_${SEED}"
EVAL_FASTA_DIR="${SEED_DIR}/inverse_folding"
DONE_MARKER="${EVAL_FASTA_DIR}/aatype/eval/inverse_fold_metrics.csv"

if [ -f "$DONE_MARKER" ]; then
    echo "[skip-eval] seed=${SEED}: $DONE_MARKER exists"
elif [ ! -d "$EVAL_FASTA_DIR" ]; then
    echo "[warn] seed=${SEED}: no generation dir at $EVAL_FASTA_DIR — skipping eval"
else
    echo "---- Evaluating seed=${SEED} ----"
    python src/byprot/utils/protein/evaluator_dplm2.py \
        -cn inverse_folding \
        inference.input_fasta_dir="$EVAL_FASTA_DIR" \
        inference.metadata.csv_path="$METADATA_CSV" \
        inference.metadata.data_dir="$METADATA_DATA_DIR"
fi

# ─── Step 3: rsync local strategy dir to NFS ──────────────────────────────────
rsync -a "$LOCAL_STRATEGY_DIR/" "$REMOTE_STRATEGY_DIR/"

# ─── Step 4: Re-aggregate across all seeds present on NFS ─────────────────────
python scripts/aggregate_seed_metrics.py --root "$REMOTE_STRATEGY_DIR"
echo "[Synced] $LOCAL_STRATEGY_DIR -> $REMOTE_STRATEGY_DIR"

echo "All done for $VARIANT / $EXP_NAME / $DATASET / $CKPT_BASENAME / $STRATEGY_TAG / seed_$SEED"
