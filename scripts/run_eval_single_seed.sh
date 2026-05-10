#!/bin/bash
# Run inverse-folding evaluation for a single (model_kind, ckpt, dataset, seed,
# sampling) tuple — one SLURM job = one GPU = one combo.
#
# Generates with that one seed, evaluates it, rsyncs to NFS, then re-runs
# aggregate_seed_metrics.py against the strategy-level directory on NFS so the
# seed_summary.csv stays consistent with whatever seeds are currently present
# (idempotent — last finisher writes the complete summary).
#
# Layout under generation-results/<result_subdir>/:
#     <EXP>/<DATASET>/<CKPT_BASENAME>/<strategy_tag>/seed_<s>/...
#                                    /<strategy_tag>/seed_summary.csv
# For baseline kind, <EXP> is suffixed with _<remasking_strategy> so different
# remasking modes (no_remask vs uncond) live in sibling dirs.
#
# Usage:
#   bash scripts/run_eval_single_seed.sh \
#       <model_kind> <ckpt_path> <dataset> <seed> <sampling> \
#       [max_iter] [batch_size] [result_subdir] [strategy_tag] [remasking_strategy] \
#       [sample_noise_every_step]
#
# model_kind:              hybrid | noise | baseline
# strategy_tag:            filesystem-safe label (e.g. argmax, annealing2.0_0.1)
# remasking_strategy:      uncond | cond | no_remask (baseline only; default: no_remask)
# sample_noise_every_step: true | false | "" (noise only; "" = inherit from ckpt)

set -e

source /data_fast/home/sihun/diffprotein/dplm/.venv/bin/activate

MODEL_KIND="$1"
CKPT_PATH="$2"
DATASET="$3"
SEED="$4"
SAMPLING="$5"
MAX_ITER="${6:-100}"
BATCH_SIZE="${7:-10}"
RESULT_SUBDIR="${8:-eval_per_seed_sweep}"
STRATEGY_TAG="${9:-default}"
REMASKING_STRATEGY="${10:-no_remask}"  # baseline only; ignored otherwise
SAMPLE_NOISE_EVERY_STEP="${11:-}"       # noise only; "" = inherit from ckpt cfg

if [ -z "$MODEL_KIND" ] || [ -z "$CKPT_PATH" ] || [ -z "$DATASET" ] \
   || [ -z "$SEED" ] || [ -z "$SAMPLING" ]; then
    echo "Usage: bash run_eval_single_seed.sh <model_kind> <ckpt_path> <dataset> <seed> <sampling> [max_iter] [batch_size] [result_subdir] [strategy_tag] [remasking_strategy] [sample_noise_every_step]"
    echo "  ckpt_path: absolute .ckpt path, or 'hf:<huggingface_model_name>' (e.g. hf:airkingbd/dplm2_650m)"
    echo "  remasking_strategy (baseline only): uncond | cond | no_remask (default: no_remask)"
    echo "  sample_noise_every_step (noise only): true | false | '' (default: '' = inherit from ckpt)"
    exit 1
fi
if [ "$DATASET" = "all" ]; then
    echo "Error: 'all' not supported here; submit one job per dataset."
    exit 1
fi

EXTRA_GEN_ARGS=()
case "$MODEL_KIND" in
    hybrid)   GEN_SCRIPT="generate_dplm2_hybrid.py" ;;
    noise)
        GEN_SCRIPT="generate_dplm2_noise.py"
        # Forward sample_noise_every_step to the generator only when the
        # caller specified it. Empty -> inherit from the ckpt's training cfg.
        if [ -n "$SAMPLE_NOISE_EVERY_STEP" ]; then
            EXTRA_GEN_ARGS=(--sample_noise_every_step "$SAMPLE_NOISE_EVERY_STEP")
        fi
        ;;
    baseline)
        GEN_SCRIPT="generate_dplm2.py"
        # --unmasking_strategy deterministic matches run_dplm2_if_seeds.sh
        # protocol; without it generate_dplm2.py defaults to stochastic1.0,
        # which changes annealing behavior. Remasking is swept (no_remask vs
        # uncond, etc.) and reflected in the EXP_NAME suffix below.
        EXTRA_GEN_ARGS=(--unmasking_strategy deterministic --remasking_strategy "$REMASKING_STRATEGY")
        ;;
    *)
        echo "Error: unknown model_kind '$MODEL_KIND' (expected: hybrid | noise | baseline)"
        exit 1
        ;;
esac

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# CKPT_PATH may be either:
#   (a) absolute path to a .ckpt file from finetuning, or
#   (b) "hf:<huggingface_model_name>" sentinel (e.g. hf:airkingbd/dplm2_650m)
# (b) is used to evaluate untuned pretrained models for sanity / reproduction.
if [[ "$CKPT_PATH" == hf:* ]]; then
    HF_MODEL="${CKPT_PATH#hf:}"
    EXP_NAME="pretrained_hf"
    CKPT_BASENAME="${HF_MODEL##*/}"            # strip org, e.g. "dplm2_650m"
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

# Baseline sweeps remasking_strategy; tag it onto the EXP dir so the two modes
# (no_remask, uncond, ...) sit in sibling directories rather than colliding.
if [ "$MODEL_KIND" = "baseline" ]; then
    EXP_NAME="${EXP_NAME}_${REMASKING_STRATEGY}"
fi

# Noise sweeps sample_noise_every_step (true/false). Empty = inherit and no tag.
if [ "$MODEL_KIND" = "noise" ] && [ -n "$SAMPLE_NOISE_EVERY_STEP" ]; then
    case "$SAMPLE_NOISE_EVERY_STEP" in
        true)  EXP_NAME="${EXP_NAME}_redraw" ;;
        false) EXP_NAME="${EXP_NAME}_freeze" ;;
        *)
            echo "Error: sample_noise_every_step must be 'true', 'false', or empty (got: $SAMPLE_NOISE_EVERY_STEP)"
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
    # cath_4.2_all / cath_4.2_short / cath_4.2_single_chain / cath_4.3_* etc.
    # Built by scripts/prepare_cath_eval.py; metadata pdb_path is relative to data-bin.
    METADATA_CSV="${PROJECT_ROOT}/data-bin/metadata/${DATASET}.csv"
    METADATA_DATA_DIR="${PROJECT_ROOT}/data-bin"
else
    METADATA_CSV="${PROJECT_ROOT}/data-bin/metadata/pdb_afdb_cameo.csv"
    METADATA_DATA_DIR="${PROJECT_ROOT}/data-bin"
fi
if [ ! -f "$INPUT_FASTA" ]; then
    echo "Error: input fasta not found: $INPUT_FASTA"
    if [[ "$DATASET" == cath_* ]]; then
        # cath_4.2_all -> --cath_version 4.2 --subset all
        CATH_VER=$(echo "$DATASET" | sed -E 's/^cath_([0-9.]+)_.*/\1/')
        CATH_SUBSET=$(echo "$DATASET" | sed -E 's/^cath_[0-9.]+_(.*)/\1/')
        echo "Hint: build it with:"
        echo "  python scripts/prepare_cath_eval.py --cath_version $CATH_VER --subset $CATH_SUBSET"
    fi
    exit 1
fi
if [ ! -f "$METADATA_CSV" ]; then
    echo "Error: metadata CSV not found: $METADATA_CSV"
    if [[ "$DATASET" == cath_* ]]; then
        CATH_VER=$(echo "$DATASET" | sed -E 's/^cath_([0-9.]+)_.*/\1/')
        CATH_SUBSET=$(echo "$DATASET" | sed -E 's/^cath_[0-9.]+_(.*)/\1/')
        echo "Hint: build it with:"
        echo "  python scripts/prepare_cath_eval.py --cath_version $CATH_VER --subset $CATH_SUBSET"
    fi
    exit 1
fi

echo "=============================================="
echo "  Kind:       $MODEL_KIND   (gen: $GEN_SCRIPT)"
echo "  Ckpt:       $CKPT_PATH"
echo "  Dataset:    $DATASET"
echo "  Seed:       $SEED"
echo "  Sampling:   $SAMPLING   (tag: $STRATEGY_TAG)"
if [ "$MODEL_KIND" = "baseline" ]; then
    echo "  Remasking:  $REMASKING_STRATEGY   (EXP suffix applied)"
fi
if [ "$MODEL_KIND" = "noise" ] && [ -n "$SAMPLE_NOISE_EVERY_STEP" ]; then
    echo "  NoiseEvery: $SAMPLE_NOISE_EVERY_STEP   (EXP suffix applied)"
fi
echo "  Max iter:   $MAX_ITER   Batch: $BATCH_SIZE"
echo "  Save base:  $LOCAL_STRATEGY_DIR (local)"
echo "  Sync to:    $REMOTE_STRATEGY_DIR (network)"
echo "=============================================="

mkdir -p "$LOCAL_STRATEGY_DIR"
mkdir -p "$REMOTE_STRATEGY_DIR"

# ─── Step 1: Generate (single seed) ───────────────────────────────────────────
python "$GEN_SCRIPT" \
    "${GEN_LOAD_ARGS[@]}" \
    "${EXTRA_GEN_ARGS[@]}" \
    --task inverse_folding \
    --input_fasta_path "$INPUT_FASTA" \
    --saveto "$LOCAL_STRATEGY_DIR" \
    --sampling_strategy "$SAMPLING" \
    --max_iter "$MAX_ITER" \
    --batch_size "$BATCH_SIZE" \
    --seeds "$SEED" \
    --skip_if_generated \
    --save_pdb False

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
# Idempotent: each finisher writes a snapshot of seed_summary.csv covering
# whichever seed_*/ dirs exist on NFS at that moment. The last finisher in the
# (ckpt, ds, strategy) group writes the complete summary.
python scripts/aggregate_seed_metrics.py --root "$REMOTE_STRATEGY_DIR"
echo "[Synced] $LOCAL_STRATEGY_DIR -> $REMOTE_STRATEGY_DIR"

echo "All done for $MODEL_KIND / $EXP_NAME / $DATASET / $CKPT_BASENAME / $STRATEGY_TAG / seed_$SEED"
