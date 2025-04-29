#!/bin/bash

set -e

DATE=20250428
CONFIG_NAME=cshallue
ENSEMBLE_NAME=dimond
NAME=vetting-v01-tois-triageJs-nocentroid-april2025
PRETRAIN_MODEL_DIR=/pdo/users/cshallue/astronet/models/triage/20250420/cshallue-h5/AstroCNNModel_cshallue_20250420_174804

CODE_DIR=/pdo/users/dimond/astronet
DATA_DIR=/pdo/users/pablomer/mnt/tess/astronet
TFRECORD_PREFIX=tfrecords-vetting-v01-tois-triageJs-nocentroid-newTOIs2025-v3-noreshufle
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
        --train_files="$DATA_DIR/$TFRECORD_PREFIX-train/*" \
        --eval_files="val:$DATA_DIR/$TFRECORD_PREFIX-val/*" \
        --eval_files="test:$DATA_DIR/$TFRECORD_PREFIX-test/*"
done

# for i in {1..10}
# do
#     echo "Training model ${i}"
#     python astronet/train.py \
#         --model=AstroCNNModelVetting \
#         --config_name=direct \
#         --train_files='../mnt/tess/astronet/tfrecords-vetting-7-train/*' \
#         --eval_files='../mnt/tess/astronet/tfrecords-vetting-7-toi-val/*' \
#         --pretrain_model_dir="../mnt/tess/astronet/checkpoints/revised_tuned_30_run_1/${i}" \
#         --train_steps=0 \
#         --train_epochs=1 \
#         --model_dir="../mnt/tess/astronet/checkpoints/direct_7_notoi_run_4/${i}"
# done
# # Try hyperparameter tuning between 500 to 2500 with 500 increments (maybe use an ensemble of 2 or 3 instead of 10 networks)
