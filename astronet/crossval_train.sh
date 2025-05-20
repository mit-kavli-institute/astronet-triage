#!/bin/bash
set -e

DATE=$(date +%Y%m%d)
TIME=$(date +%H%M)
CONFIG_NAME=cshallue
ENSEMBLE_NAME=$CONFIG_NAME

# where your pretrained model lives
PRETRAIN_MODEL_DIR=/pdo/users/cshallue/astronet/models/triage/20250420/cshallue-h5/AstroCNNModel_cshallue_20250420_174804

# code root and TFRecord root
CODE_DIR=/pdo/users/pablomer/Astronet-Triage/
TFRECORD_DIR=/pdo/users/pablomer/mnt/tess/astronet/crossvaltfrecords

# base name of your TFRecord files
TFRECORD_PREFIX=tfrecords-vetting-v01-tois-triageJs-nocentroid-april2025

# where to dump all folds
BASE_OUTPUT_DIR=/pdo/users/pablomer/mnt/tess/models/vetting/$DATE-$TIME/${ENSEMBLE_NAME}-cv

# any overrides you need
# CONFIG_OVERRIDES="init_from_pretrained_model=true,freeze_pretrained_params=false"

mkdir -p "$BASE_OUTPUT_DIR"

for i in {1..4}; do
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
