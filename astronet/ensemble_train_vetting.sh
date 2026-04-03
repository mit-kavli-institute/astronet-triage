#!/bin/bash

set -e

DATE=20260325
CONFIG_NAME=cshallue
ENSEMBLE_NAME=dimond
NAME=vetting-v01-tois-triageJs-nocentroid-april2025
PRETRAIN_MODEL_DIR=/pdo/users/cshallue/astronet/models/triage/20250420/cshallue-h5/AstroCNNModel_cshallue_20250420_174804

CODE_DIR=/pdo/users/dimond/astronet
DATA_DIR=/pdo/astronet-data/data/tfrecords/vetting-aug-2025-new-features/
OUTPUT_DIR=/pdo/astronet-data/models/vetting/experimental/dimond/$DATE/$ENSEMBLE_NAME

for i in {1..1}
do
    echo "Training model ${i}"
    PYTHONPATH=$CODE_DIR python $CODE_DIR/astronet/train.py \
        --model=AstroCNNModelVetting \
        --config_name=$CONFIG_NAME \
        --config_file=$CONFIG_FILE \
        --config_overrides=$CONFIG_OVERRIDES \
        --pretrain_model_dir=$PRETRAIN_MODEL_DIR \
        --model_dir="$OUTPUT_DIR" \
        --train_files="$DATA_DIR/train/*" \
        --eval_files="val:$DATA_DIR/val/*" \
        --eval_files="test:$DATA_DIR/test/*"
done