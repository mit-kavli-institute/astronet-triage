#!/bin/bash

set -e

DATE=20250423
CONFIG_NAME=cshallue
#CONFIG_OVERRIDES="inputs.random_reverse_time_series=true"
# CONFIG_OVERRIDES="train_steps=2500"
ENSEMBLE_NAME=$CONFIG_NAME


PRETRAIN_MODEL_DIR=/pdo/users/cshallue/astronet/models/triage/20250420/cshallue-h5/AstroCNNModel_cshallue_20250420_174804


CODE_DIR=/pdo/users/cshallue/git/astronet
# CODE_DIR=/pdo/users/pablomer/Astronet-Triage/
DATA_DIR=/pdo/users/pablomer/mnt/tess/astronet/

TFRECORD_PREFIX=tfrecords-vetting-v01-tois-triageJs-nocentroid-newTOIs2025-v3-noreshufle

OUTPUT_DIR=/pdo/users/pablomer/mnt/tess/models/vetting/$DATE/$ENSEMBLE_NAME/

for i in {1..1}
do
    echo "Training model ${i}"
    # PYTHONPATH=$CODE_DIR LD_LIBRARY_PATH=/pdo/users/cshallue/miniconda3/lib python $CODE_DIR/astronet/train.py \
    # PYTHONPATH=$CODE_DIR_chris LD_LIBRARY_PATH=/pdo/users/cshallue/miniconda3/lib python $CODE_DIR/astronet/train.py \
    /pdo/users/dmuth/miniconda3/envs/tf/bin/python $CODE_DIR/astronet/train.py \
        --model=AstroCNNModelVetting \
        --config_name=$CONFIG_NAME \
        --config_file=$CONFIG_FILE \
        --config_overrides=$CONFIG_OVERRIDES \
        --pretrain_model_dir=$PRETRAIN_MODEL_DIR \
        --model_dir="$OUTPUT_DIR" \
        --train_files="$DATA_DIR/$TFRECORD_PREFIX-train/*" \
        --eval_files="val:$DATA_DIR/$TFRECORD_PREFIX-val/*" \
        --eval_files="test:$DATA_DIR/$TFRECORD_PREFIX-test/*"
done
