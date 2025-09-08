#!/bin/bash
set -e

DATE=$(date +%Y%m%d)
TIME=$(date +%H%M)
CONFIG_NAME=pablomer
ENSEMBLE_NAME=$CONFIG_NAME

# where your pretrained model lives
PRETRAIN_MODEL_DIR=/pdo/users/pablomer/mnt/tess/models/triage/20250520/pablomer-h5/AstroCNNModel_pablomer_20250520_181651

# code root and TFRecord root
CODE_DIR=/pdo/users/pablomer/Astronet-Triage/
TFRECORD_DIR=/pdo/users/pablomer/mnt/tess/astronet/crossvaltfrecords

# base name of your TFRecord files
TFRECORD_PREFIX=tfrecords-vetting-v01-tois-triageJs-nocentroid-april2025

# where to dump all folds
BASE_OUTPUT_DIR=/pdo/users/pablomer/mnt/tess/models/vetting/$DATE-$TIME/${ENSEMBLE_NAME}-cv

# any overrides you need
# CONFIG_OVERRIDES="init_from_pretrained_model=true,freeze_pretrained_params=false"

#Best run of the 20 trials training run on May 21st, 2025
CONFIG_OVERRIDES="hparams.learning_rate=0.0017782794100389254,\
hparams.weight_decay=0.0027384196342643626,\
hparams.pre_logits_dropout_rate=0.40625,\
hparams.num_pre_logits_hidden_layers=1,\
hparams.pre_logits_hidden_layer_size=512"

mkdir -p "$BASE_OUTPUT_DIR"

for i in {1..5}; do
  echo "=== Fold $i ==="
  TRAIN_PATTERN="$TFRECORD_DIR/${TFRECORD_PREFIX}-train_fold${i}/*"
  VAL_PATTERN="$TFRECORD_DIR/${TFRECORD_PREFIX}-val_fold${i}/*"

  FOLD_DIR="$BASE_OUTPUT_DIR/fold${i}"
  mkdir -p "$FOLD_DIR"

  /pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python $CODE_DIR/astronet/train.py \
    --model=AstroCNNModelVetting \
    --config_name=$CONFIG_NAME \
    --config_overrides=$CONFIG_OVERRIDES \
    --pretrain_model_dir=$PRETRAIN_MODEL_DIR \
    --model_dir="$FOLD_DIR" \
    --train_files="$TRAIN_PATTERN" \
    --eval_files="val:$VAL_PATTERN" \
    --dump_block_weights=false
done
