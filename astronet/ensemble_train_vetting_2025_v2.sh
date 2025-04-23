#!/bin/bash
set -e

# 1) Point to your conda install
CONDA_BASE=/pdo/users/cshallue/miniconda3

# 2) Source conda.sh so you can run `conda activate`
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 3) Activate the GPU-enabled Astronet env
conda activate astronet-gpu

DATE=20250420
CONFIG_NAME=cshallue
#CONFIG_OVERRIDES="inputs.random_reverse_time_series=true"
ENSEMBLE_NAME=$CONFIG_NAME
#PRETRAIN_MODEL_DIR=/pdo/users/cshallue/astronet/models/triage/20250418/cshallue/AstroCNNModel_cshallue_20250418_115544
PRETRAIN_MODEL_DIR=/pdo/users/cshallue/astronet/models/triage/20250420/cshallue-h5/AstroCNNModel_cshallue_20250420_174804

CODE_DIR=/pdo/users/cshallue/git/astronet
DATA_DIR=/pdo/users/pablomer/mnt/tess/astronet
TFRECORD_PREFIX=tfrecords-vetting-v01-tois-triageJs-nocentroid-newTOIs2025-v3-noreshufle

OUTPUT_DIR=/pdo/users/pablomer/mnt/tess/models/vetting/$DATE/$ENSEMBLE_NAME/ # Changed

# 5) Exports
export PYTHONPATH="$CODE_DIR"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"


for i in {1..1}
do
    echo "Training model ${i}"
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
