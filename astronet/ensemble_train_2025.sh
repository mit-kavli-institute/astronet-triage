#!/bin/bash
# THIS IS A SCRIPT FOR TRAIING THE TRIAGE MODEL
set -e

#DATE=20250403
#STUDY_NAME=reverse-downweight2
#TRIAL_ID=84
#CONFIG_FILE=/pdo/users/cshallue/astronet/models/triage-tune/$DATE/$STUDY_NAME/$TRIAL_ID/config.json
#ENSEMBLE_NAME=$STUDY_NAME-$TRIAL_ID

DATE=20250520
CONFIG_NAME=pablomer
#CONFIG_OVERRIDES="inputs.random_reverse_time_series=true"
ENSEMBLE_NAME=$CONFIG_NAME-h5

# CODE_DIR=/pdo/users/cshallue/git/astronet
CODE_DIR=/pdo/users/pablomer/Astronet-Triage
DATA_DIR=/pdo/users/tey/astronet/triage-training
OUTPUT_DIR=/pdo/users/pablomer/mnt/tess/models/triage/$DATE/$ENSEMBLE_NAME/

#Change to 10 or whatever number of models you want to train for the ensemble
for i in {1..5}
do
    echo "Training model ${i}"
    /pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python $CODE_DIR/astronet/train.py \
        --model=AstroCNNModel \
        --config_name=$CONFIG_NAME \
        --config_file=$CONFIG_FILE \
        --config_overrides=$CONFIG_OVERRIDES \
        --model_dir="$OUTPUT_DIR" \
        --train_files="$DATA_DIR/tfrecords-train/*" \
        --eval_files="val:$DATA_DIR/tfrecords-val/*" \
        --eval_files="test:$DATA_DIR/tfrecords-test/*"
done
