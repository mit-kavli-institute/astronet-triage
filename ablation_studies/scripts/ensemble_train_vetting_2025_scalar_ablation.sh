#!/bin/bash

set -euo pipefail

DATE=$(date +%Y%m%d)
CONFIG_NAME=pablomer
CONFIG_OVERRIDES="train_steps=2000,init_from_pretrained_model=false,hparams.pre_logits_hidden_layer_size=512"
ENSEMBLE_NAME="${CONFIG_NAME}-scalar-ablation"

CODE_DIR=/pdo/users/pablomer/Astronet-Triage
PYTHON_BIN=/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python
DATA_DIR=/pdo/astronet-data/data/tfrecords/dec2025_cad_scat_v5_duration24/
TFRECORD_PREFIX=tfrecords-vetting-v01-tois-triageJs-nocentroid-dec2025

OUTPUT_DIR=/pdo/astronet-data/models/vetting/experimental/pablomer/dec2025_cad_scat_v5_duration24/$DATE/$ENSEMBLE_NAME-2k-nopretrained-z_dim512/

for i in {1..2}
do
  echo "Training scalar-ablation model ${i}"
  PYTHONPATH="$CODE_DIR" "$PYTHON_BIN" "$CODE_DIR/astronet/train.py" \
    --model=AstroCNNModelVettingScalarAblation \
    --config_name="$CONFIG_NAME" \
    --config_overrides="$CONFIG_OVERRIDES" \
    --model_dir="$OUTPUT_DIR" \
    --train_files="$DATA_DIR/$TFRECORD_PREFIX-train/*" \
    --eval_files="val:$DATA_DIR/$TFRECORD_PREFIX-val/*" \
    --eval_files="test:$DATA_DIR/$TFRECORD_PREFIX-test/*" \
    --dump_block_weights=false \
    --early_stopping_patience=20
done

echo "All scalar-ablation models trained. Generating combined predictions..."
PYTHONPATH="$CODE_DIR" "$PYTHON_BIN" "$CODE_DIR/astronet/combine_model_results.py" \
  --base_path="$OUTPUT_DIR"
