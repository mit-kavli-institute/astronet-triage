#!/bin/bash

set -euo pipefail

DATE=$(date +%Y%m%d)
CONFIG_NAME=dimond_final
CONFIG_OVERRIDES=""
ENSEMBLE_NAME="${CONFIG_NAME}"

PRETRAIN_MODEL_DIR=/pdo/users/pablomer/mnt/tess/models/triage/20250520/pablomer-h5/AstroCNNModel_pablomer_20250520_181651

CODE_DIR=/pdo/users/dimond/astronet_secondary/astronet

DATA_DIR=/pdo/astronet-data/data/tfrecords/dec2025_cad_scat_v5_aug/10x_0p1

TFRECORD_PREFIX=tfrecords-vetting-v01-tois-triageJs-nocentroid-dec2025

# OUTPUT_DIR=/pdo/users/pablomer/mnt/tess/models/vetting/$DATE/$ENSEMBLE_NAME/
OUTPUT_DIR=/pdo/astronet-data/models/vetting/experimental/dimond/$DATE/$ENSEMBLE_NAME-baseline/

for i in {1..10}
do
    echo "Training model ${i}"
    # PYTHONPATH=$CODE_DIR LD_LIBRARY_PATH=/pdo/users/cshallue/miniconda3/lib python $CODE_DIR/astronet/train.py \
    # PYTHONPATH=$CODE_DIR_chris LD_LIBRARY_PATH=/pdo/users/cshallue/miniconda3/lib python $CODE_DIR/astronet/train.py \
    # /pdo/users/dmuth/miniconda3/envs/tf/bin/python $CODE_DIR/astronet/train.py \
    PYTHONPATH=$CODE_DIR python $CODE_DIR/astronet/train.py \
        --model=AstroCNNModelVetting \
        --config_name=$CONFIG_NAME \
        --pretrain_model_dir="$PRETRAIN_MODEL_DIR" \
        --config_overrides=$CONFIG_OVERRIDES \
        --model_dir="$OUTPUT_DIR" \
        --train_files="$DATA_DIR/tfrecords-vetting-v01-tois-triageJs-nocentroid-dec2025-train/*" \
        --eval_files="val:$DATA_DIR/tfrecords-vetting-v01-tois-triageJs-nocentroid-dec2025-val/*" \
        --eval_files="test:$DATA_DIR/tfrecords-vetting-v01-tois-triageJs-nocentroid-dec2025-test/*" \
        --dump_block_weights=false
done

# After training loop, generate a csv with the combined predictions
echo "All models trained. Now generating combined predictions..."
PYTHONPATH="$CODE_DIR" python \
    /pdo/users/pablomer/Astronet-Triage/astronet/combine_model_results.py \
    --base_path="$OUTPUT_DIR"
