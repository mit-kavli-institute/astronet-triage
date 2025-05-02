#!/bin/bash
set -e

#–– Configuration ––#
DATE=$(date +%Y%m%d)
CONFIG_NAME=cshallue
TFRECORD_PREFIX=tfrecords-vetting-v01-tois-triageJs-nocentroid-april2025
ENSEMBLE_NAME=$CONFIG_NAME
# N_TRIALS=30
N_TRIALS=300
N_RUNS=2 # Runs per trial (ensemble size)

PRETRAIN_MODEL_DIR=/pdo/users/cshallue/astronet/models/triage/20250420/cshallue-h5/AstroCNNModel_cshallue_20250420_174804_run2

#–– Paths ––#
CODE_DIR=/pdo/users/pablomer/Astronet-Triage/
DATA_DIR=/pdo/users/pablomer/mnt/tess/astronet
OUTPUT_DIR=/pdo/users/pablomer/mnt/tess/models/vetting/${DATE}/${ENSEMBLE_NAME}_optuna_${DATE}_round2

#–– Python & Env ––#
# PYTHON_BIN=/pdo/users/dmuth/miniconda3/envs/tf/bin/python
PYTHON_BIN=/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python


CONFIG_OVERRIDES="train_steps=100"

#–– Launch Optuna tuning ––#
echo "Starting Optuna tuning for ${ENSEMBLE_NAME} (n_trials=${N_TRIALS})"
/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python $CODE_DIR/astronet/tune_with_optuna.py \
    --model=AstroCNNModelVetting \
    --config_name=$CONFIG_NAME \
    --config_file=$CONFIG_FILE \
    --config_overrides="$CONFIG_OVERRIDES" \
    --pretrain_model_dir=$PRETRAIN_MODEL_DIR \
    --train_files="${DATA_DIR}/${TFRECORD_PREFIX}-train/*" \
    --eval_files="${DATA_DIR}/${TFRECORD_PREFIX}-val/*" \
    --model_dir="${OUTPUT_DIR}" \
    --n_trials=${N_TRIALS} \
    --n_runs=${N_RUNS}

echo "Optuna tuning finished. Results at ${OUTPUT_DIR}"
