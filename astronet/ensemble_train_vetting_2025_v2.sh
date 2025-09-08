#!/bin/bash

set -e

DATE=$(date +%Y%m%d)
TIME=$(date +%H%M)
CONFIG_NAME=cshallue
#CONFIG_OVERRIDES="inputs.random_reverse_time_series=true"
CONFIG_OVERRIDES="train_steps=1000"
ENSEMBLE_NAME=$CONFIG_NAME


PRETRAIN_MODEL_DIR=/pdo/users/cshallue/astronet/models/triage/20250420/cshallue-h5/AstroCNNModel_cshallue_20250420_174804


# CODE_DIR=/pdo/users/cshallue/git/astronet
CODE_DIR=/pdo/users/pablomer/Astronet-Triage/
DATA_DIR=/pdo/users/pablomer/mnt/tess/astronet/


TFRECORD_PREFIX=tfrecords-vetting-v01-tois-triageJs-nocentroid-april2025


OUTPUT_DIR=/pdo/users/pablomer/mnt/tess/models/vetting/$DATE/$ENSEMBLE_NAME/$TIME/


#Best run of the 300 trials training run on May 1st, 2025
# CONFIG_OVERRIDES="hparams.learning_rate=0.0012738895510624943,\
# hparams.weight_decay=2.213220594759495e-06,\
# hparams.pre_logits_dropout_rate=0.4394965848842629,\
# hparams.num_pre_logits_hidden_layers=1,\
# hparams.pre_logits_hidden_layer_size=256,\
# init_from_pretrained_model=false,\
# freeze_pretrained_params=true"



for i in {1..5}
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
        --eval_files="test:$DATA_DIR/$TFRECORD_PREFIX-test/*"
done
# /pdo/users/pablomer/miniconda3/envs/tf-env/bin/python $CODE_DIR/astronet/train.py \

# After training loop, generate a csv with the combined predictions
echo "All models trained. Now generating combined predictions..."
/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python \
    /pdo/users/pablomer/Astronet-Triage/astronet/combine_model_results.py \
    --base_path="$OUTPUT_DIR"
