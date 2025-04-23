#!/bin/bash

set -e

NAME=vetting-v01-tois-triageJs-nocentroid-newTOIs2025-v3-noreshufle-crossval #Pablo Feb 25 run 2

MODEL_DIR_BASE="../mnt/tess/astronet/checkpoints/${NAME}"
DATA_DIR_BASE="../mnt/tess/astronet/aggregated_by_cp"

PYTHON=/pdo/users/dmuth/miniconda3/envs/tf/bin/python

# First, check if the source data exists
echo "Checking if data directories exist..."
ls -la "${DATA_DIR_BASE}" || { echo "ERROR: Data directory doesn't exist!"; exit 1; }

# For testing with a single fold only
ENSEMBLE=1
FOLD=0

echo "=== Starting ensemble ${ENSEMBLE} ==="
echo "Training fold ${FOLD} of ensemble ${ENSEMBLE}"

# Let's try using one specific fold for testing to isolate the issue
TEST_DIR="${DATA_DIR_BASE}/0"
echo "Looking for files in test directory: ${TEST_DIR}"
ls -la "${TEST_DIR}" || { echo "ERROR: Test directory doesn't exist!"; exit 1; }

# Find actual TFRecord files
TFRECORD_FILES=$(find "${TEST_DIR}" -name "*.tfrecord" | head -10)
echo "Found TFRecord files:"
echo "${TFRECORD_FILES}"

# Check if any TFRecord files were found
if [ -z "${TFRECORD_FILES}" ]; then
    echo "ERROR: No TFRecord files found in ${TEST_DIR}"
    echo "Checking for files with any extension:"
    find "${TEST_DIR}" -type f | head -10
    exit 1
fi

# For testing, use a single specific file that we know exists
TRAIN_FILES="${TFRECORD_FILES}"
echo "Using train files: ${TRAIN_FILES}"

# Now try with a single file that we know exists
${PYTHON} astronet/train.py \
  --model=AstroCNNModelVetting \
  --config_name=base_new \
  --train_files="${TRAIN_FILES}" \
  --eval_files="" \
  --pretrain_model_dir="/pdo/users/dmuth/mnt/tess/fa1t_38_run_1/10" \
  --train_steps=10 \
  --train_epochs=1 \
  --model_dir="${MODEL_DIR_BASE}/single_file_test"

echo "Test completed"
