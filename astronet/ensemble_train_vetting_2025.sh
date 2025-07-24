#!/bin/bash

set -e

DATE=$(date +%Y%m%d)
CONFIG_NAME=pablomer
#CONFIG_OVERRIDES="inputs.random_reverse_time_series=true"
CONFIG_OVERRIDES="train_steps=1000"
ENSEMBLE_NAME=$CONFIG_NAME


# PRETRAIN_MODEL_DIR=/pdo/users/cshallue/astronet/models/triage/20250420/cshallue-h5/AstroCNNModel_cshallue_20250420_174804
PRETRAIN_MODEL_DIR=/pdo/users/pablomer/mnt/tess/models/triage/20250520/pablomer-h5/AstroCNNModel_pablomer_20250520_181651

# CODE_DIR=/pdo/users/cshallue/git/astronet
CODE_DIR=/pdo/users/pablomer/Astronet-Triage
DATA_DIR=/pdo/users/pablomer/mnt/tess/astronet/

# TFRECORD_PREFIX=tfrecords-vetting-v01-tois-triageJs-nocentroid-newTOIs2025-v3-noreshufle

TFRECORD_PREFIX=tfrecords-vetting-v01-tois-triageJs-nocentroid-april2025


OUTPUT_DIR=/pdo/users/pablomer/mnt/tess/models/vetting/$DATE/$ENSEMBLE_NAME/

# CONFIG_OVERRIDES="init_from_pretrained_model=true,\
# freeze_pretrained_params=true"

for i in {1..1}
do
    echo "Training model ${i}"
    # PYTHONPATH=$CODE_DIR LD_LIBRARY_PATH=/pdo/users/cshallue/miniconda3/lib python $CODE_DIR/astronet/train.py \
    # PYTHONPATH=$CODE_DIR_chris LD_LIBRARY_PATH=/pdo/users/cshallue/miniconda3/lib python $CODE_DIR/astronet/train.py \
    # /pdo/users/dmuth/miniconda3/envs/tf/bin/python $CODE_DIR/astronet/train.py \
    /pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python $CODE_DIR/astronet/train.py \
        --model=AstroCNNModelVetting \
        --config_name=$CONFIG_NAME \
        --config_file=$CONFIG_FILE \
        --config_overrides=$CONFIG_OVERRIDES \
        --pretrain_model_dir=$PRETRAIN_MODEL_DIR \
        --model_dir="$OUTPUT_DIR" \
        --train_files="$DATA_DIR/$TFRECORD_PREFIX-train/*" \
        --eval_files="val:$DATA_DIR/$TFRECORD_PREFIX-val/*" \
        --eval_files="test:$DATA_DIR/$TFRECORD_PREFIX-test/*" \
        --dump_block_weights=false
done
# /pdo/users/pablomer/miniconda3/envs/tf-env/bin/python $CODE_DIR/astronet/train.py \
# --astro_ids_file="/pdo/users/pablomer/Astronet-Triage/testing_live_sectors/astro_ids_s86_tois.csv"
