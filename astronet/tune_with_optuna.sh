#!/bin/bash
set -e

#–– Configuration ––#
DATE=$(date +%Y%m%d)
CONFIG_NAME=pablomer_base
TFRECORD_PREFIX=tfrecords-vetting-v01-tois-triageJs-nocentroid-dec2025
ENSEMBLE_NAME=$CONFIG_NAME
# N_TRIALS=30
N_TRIALS=150 # 150
N_RUNS=5 # Runs per trial (ensemble size) #5
SAMPLER=${SAMPLER:-qmc} # QMCSampler (Sobol quasi-random)
SEARCH_SPACE=${SEARCH_SPACE:-final}
CONFIG_FILE=""
USE_AUGMENTATION=${USE_AUGMENTATION:-false}

# PRETRAIN_MODEL_DIR=/pdo/users/cshallue/astronet/models/triage/20250420/cshallue-h5/AstroCNNModel_cshallue_20250420_174804
PRETRAIN_MODEL_DIR=/pdo/users/pablomer/mnt/tess/models/triage/20250520/pablomer-h5/AstroCNNModel_pablomer_20250520_181651

#–– Paths ––#
CODE_DIR=/pdo/users/pablomer/Astronet-Triage
# DATA_DIR=/pdo/users/pablomer/mnt/tess/astronet/
# DATA_DIR=/pdo/astronet-data/data/tfrecords/oct2025_30minbin/
# DATA_DIR=/pdo/astronet-data/data/tfrecords/oct2025_cadencebin/
# DATA_DIR=/pdo/astronet-data/data/tfrecords/oct2025_original/
# DATA_DIR=/pdo/astronet-data/data/tfrecords/dec2025_cad_scat_v2/
DATA_DIR_AUG=/pdo/astronet-data/data/tfrecords/dec2025_cad_scat_v5_aug/10x_0p1/
DATA_DIR=/pdo/astronet-data/data/tfrecords/dec2025_cad_scat_v5_duration24/
# DATA_DIR=/pdo/astronet-data/data/tfrecords/oct2025_30minbin_v2/s

# OUTPUT_DIR=/pdo/users/pablomer/mnt/tess/models/vetting/${DATE}/${ENSEMBLE_NAME}_optuna_${DATE}
# OUTPUT_DIR=/pdo/astronet-data/models/vetting/experimental/pablomer/dec2025_cad_scat_v2/${DATE}-optuna/${ENSEMBLE_NAME}-phase1/
OUTPUT_DIR=/pdo/users/pablomer/mnt/tess/models/vetting/${DATE}-final-study_noaug/${ENSEMBLE_NAME}_optuna_${DATE}

# CONFIG_OVERRIDES (Phase 1: quick sweep over high-impact knobs)
# CONFIG_OVERRIDES="train_steps=2000,init_from_pretrained_model=true"
CONFIG_OVERRIDES=""
# CONFIG_OVERRIDES="train_steps=30,init_from_pretrained_model=false"
# CONFIG_OVERRIDES="train_steps=3000"


#–– Launch Optuna tuning ––#
echo "Starting Optuna tuning for ${ENSEMBLE_NAME} (n_trials=${N_TRIALS})"
# PYTHONNOUSERSITE=1 \
# LD_LIBRARY_PATH=/pdo/users/pablomer/miniconda3/lib:$LD_LIBRARY_PATH \
# /pdo/users/pablomer/miniconda3/envs/astronet-gpu/bin/python $CODE_DIR/astronet/tune_with_optuna.py \
/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python $CODE_DIR/astronet/tune_with_optuna.py \
    --model=AstroCNNModelVetting \
    --config_name=$CONFIG_NAME \
    --config_file=$CONFIG_FILE \
    --config_overrides=$CONFIG_OVERRIDES \
    --pretrain_model_dir=$PRETRAIN_MODEL_DIR \
    --train_files="${DATA_DIR}${TFRECORD_PREFIX}-train/*" \
    --train_files_aug="${DATA_DIR_AUG}${TFRECORD_PREFIX}-train/*" \
    --eval_files="${DATA_DIR}${TFRECORD_PREFIX}-val/*" \
    --model_dir="${OUTPUT_DIR}" \
    --n_trials=${N_TRIALS} \
    --n_runs=${N_RUNS} \
    --sampler=${SAMPLER} \
    --search_space=${SEARCH_SPACE} \
    --use_augmentation=${USE_AUGMENTATION}

echo "Optuna tuning finished. Results at ${OUTPUT_DIR}"
