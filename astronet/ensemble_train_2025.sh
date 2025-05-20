#!/bin/bash

set -e

#DATE=20250403
#STUDY_NAME=reverse-downweight2
#TRIAL_ID=84
#CONFIG_FILE=/pdo/users/cshallue/astronet/models/triage-tune/$DATE/$STUDY_NAME/$TRIAL_ID/config.json
#ENSEMBLE_NAME=$STUDY_NAME-$TRIAL_ID

DATE=20250420
CONFIG_NAME=cshallue
#CONFIG_OVERRIDES="inputs.random_reverse_time_series=true"
ENSEMBLE_NAME=$CONFIG_NAME-h5

CODE_DIR=/pdo/users/cshallue/git/astronet
DATA_DIR=/pdo/users/tey/astronet/triage-training
OUTPUT_DIR=/pdo/users/cshallue/astronet/models/triage/$DATE/$ENSEMBLE_NAME/

for i in {1..10}
do
    echo "Training model ${i}"
    PYTHONPATH=$CODE_DIR LD_LIBRARY_PATH=/pdo/users/cshallue/miniconda3/lib python $CODE_DIR/astronet/train.py \
        --model=AstroCNNModel \
        --config_name=$CONFIG_NAME \
        --config_file=$CONFIG_FILE \
        --config_overrides=$CONFIG_OVERRIDES \
        --model_dir="$OUTPUT_DIR" \
        --train_files="$DATA_DIR/tfrecords-train/*" \
        --eval_files="val:$DATA_DIR/tfrecords-val/*" \
        --eval_files="test:$DATA_DIR/tfrecords-test/*"
done
