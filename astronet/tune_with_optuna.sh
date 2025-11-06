#!/bin/bash
set -e

#–– Configuration ––#
DATE=$(date +%Y%m%d)
CONFIG_NAME=pablomer
TFRECORD_PREFIX=tfrecords-vetting-v01-tois-triageJs-nocentroid-april2025
ENSEMBLE_NAME=$CONFIG_NAME
# N_TRIALS=30
N_TRIALS=2
N_RUNS=1 # Runs per trial (ensemble size)
SAMPLER=${SAMPLER:-qmc} # QMCSampler (Sobol quasi-random)

# PRETRAIN_MODEL_DIR=/pdo/users/cshallue/astronet/models/triage/20250420/cshallue-h5/AstroCNNModel_cshallue_20250420_174804
PRETRAIN_MODEL_DIR=/pdo/users/pablomer/mnt/tess/models/triage/20250520/pablomer-h5/AstroCNNModel_pablomer_20250520_181651

#–– Paths ––#
CODE_DIR=/pdo/users/pablomer/Astronet-Triage/
DATA_DIR=/pdo/users/pablomer/mnt/tess/astronet
# OUTPUT_DIR=/pdo/users/pablomer/mnt/tess/models/vetting/${DATE}/${ENSEMBLE_NAME}_optuna_${DATE}
OUTPUT_DIR=/pdo/astronet-data/models/vetting/experimental/pablomer/oct2025_original/${DATE}-optuna/${ENSEMBLE_NAME}-2k-pretrained-cosine/


#–– Python & Env ––#
# PYTHON_BIN=/pdo/users/dmuth/miniconda3/envs/tf/bin/python
PYTHON_BIN=/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python


# CONFIG_OVERRIDES="train_steps=10"

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
    --n_runs=${N_RUNS} \
    --sampler=${SAMPLER}   # QMCSampler (Sobol quasi-random)

echo "Optuna tuning finished. Results at ${OUTPUT_DIR}"
