#!/bin/bash
# Usage: bash run_dplm2_if.sh <exp_name> <dataset> <model_name> <sampling_strategy> [remasking_strategy]
# Example: bash run_dplm2_if.sh reproduction cameo2022 dplm2_650m argmax uncond
# remasking_strategy options: uncond (default), cond, no_remask

EXP_NAME=$1
DATASET=$2
MODEL_NAME=$3
SAMPLING_STRATEGY=$4
REMASKING_STRATEGY=${5:-uncond}

PROJECT_DIR=/data_fast/home/sihun/diffprotein/dplm
OUTPUT_DIR=${PROJECT_DIR}/generation-results/${EXP_NAME}/${DATASET}/${MODEL_NAME}/${SAMPLING_STRATEGY}/${REMASKING_STRATEGY}
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
    --max_iter 100 \
    --unmasking_strategy deterministic \
    --sampling_strategy ${SAMPLING_STRATEGY} \
    --remasking_strategy ${REMASKING_STRATEGY} \
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
    --sampling_strategy ${SAMPLING_STRATEGY} \
    --remasking_strategy ${REMASKING_STRATEGY}
