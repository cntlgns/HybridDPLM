#!/bin/bash
# Usage: bash run_dplm2_if_ffdiff.sh <dataset> <model_name> <strategy_name> <hyperparam_tag> <decoding_strategy_string> <feedforward_mode> <mask_emb_mode>
# Example: bash run_dplm2_if_ffdiff.sh cameo2022 dplm2_650m dinfer_threshold_linear t0.8_embadd "dinfer_threshold@0.8" linear add
#
# Fixed settings: deterministic unmasking, argmax sampling, no_remask remasking, max_iter=500
# Output path: generation-results/ffdiff/{dataset}/{model_name}/{strategy_name}/{hyperparam_tag}/inverse_folding

DATASET=$1
MODEL_NAME=$2
STRATEGY_NAME=$3
HYPERPARAM_TAG=$4
DECODING_STRATEGY=$5
FEEDFORWARD_MODE=$6
MASK_EMB_MODE=$7

PROJECT_DIR=/data_fast/home/sihun/diffprotein/dplm
EXP_NAME=ffdiff
OUTPUT_DIR=${PROJECT_DIR}/generation-results/${EXP_NAME}/${DATASET}/${MODEL_NAME}/${STRATEGY_NAME}/${HYPERPARAM_TAG}
INPUT_FASTA=${PROJECT_DIR}/data-bin/${DATASET}/struct.fasta
EVAL_DIR=${OUTPUT_DIR}/inverse_folding

# Select metadata CSV and data_dir based on dataset
if [[ "${DATASET}" == "PDB_date" ]]; then
    METADATA_CSV=${PROJECT_DIR}/data-bin/metadata/pdb_date.csv
    METADATA_DATA_DIR=${PROJECT_DIR}/data-bin/PDB_date
else
    METADATA_CSV=${PROJECT_DIR}/data-bin/metadata/pdb_afdb_cameo.csv
    METADATA_DATA_DIR=${PROJECT_DIR}/data-bin
fi

PYTHON_BIN=${PROJECT_DIR}/.venv/bin/python

cd ${PROJECT_DIR}
mkdir -p ${OUTPUT_DIR}

BIT_MODEL_FLAG=""
if [[ "${MODEL_NAME}" == "dplm2_bit_650m" ]]; then
    BIT_MODEL_FLAG="--bit_model"
fi

${PYTHON_BIN} generate_dplm2.py \
    --model_name airkingbd/${MODEL_NAME} \
    --task inverse_folding \
    ${BIT_MODEL_FLAG} \
    --input_fasta_path ${INPUT_FASTA} \
    --max_iter 500 \
    --unmasking_strategy deterministic \
    --sampling_strategy argmax \
    --remasking_strategy no_remask \
    --decoding_strategy "${DECODING_STRATEGY}" \
    --feedforward_mode ${FEEDFORWARD_MODE} \
    --mask_emb_mode ${MASK_EMB_MODE} \
    --saveto ${OUTPUT_DIR} && \
${PYTHON_BIN} src/byprot/utils/protein/evaluator_dplm2.py \
    -cn inverse_folding \
    inference.input_fasta_dir=${EVAL_DIR} \
    inference.metadata.csv_path=${METADATA_CSV} \
    inference.metadata.data_dir=${METADATA_DATA_DIR} && \
${PYTHON_BIN} ${PROJECT_DIR}/summarize_results.py \
    --exp_name ${EXP_NAME} \
    --dataset ${DATASET} \
    --model_name ${MODEL_NAME} \
    --sampling_strategy ${STRATEGY_NAME} \
    --remasking_strategy ${HYPERPARAM_TAG}
