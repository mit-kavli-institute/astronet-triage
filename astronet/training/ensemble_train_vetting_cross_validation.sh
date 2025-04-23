#!/bin/bash

set -e


NAME=vetting-v01-tois-triageJs-nocentroid-newTOIs2025-v3-noreshufle-crossval #Pablo Feb 25 run 2

MODEL_DIR_BASE="../mnt/tess/astronet/checkpoints/${NAME}"
DATA_DIR_BASE="../mnt/tess/astronet/aggregated_by_cp"

PYTHON=/pdo/users/dmuth/miniconda3/envs/tf/bin/python
# python3 or /pdo/users/dmuth/miniconda3/envs/tf/bin/python


#we will create 2 ensembles
for ENSEMBLE in 1 2
do
  echo "=== Starting ensemble ${ENSEMBLE} ==="

  # 5-fold cross-validation
  for FOLD in 0 1 2 3 4
  do
    echo "Training fold ${FOLD} of ensemble ${ENSEMBLE}"

    # Build comma-separated list of training files
    TRAIN_FILES=""
    for OTHER_FOLD in 0 1 2 3 4
    do
      if [ "${OTHER_FOLD}" != "${FOLD}" ]; then
        # For each fold that is not the current test fold
        if [ -z "$TRAIN_FILES" ]; then
          TRAIN_FILES="${DATA_DIR_BASE}/${OTHER_FOLD}/*"
        else
          TRAIN_FILES="${TRAIN_FILES},${DATA_DIR_BASE}/${OTHER_FOLD}/*"
        fi
      fi
    done

    # print train files to debug
    echo "Train files: ${TRAIN_FILES}"

    ${PYTHON} astronet/train.py \
      --model=AstroCNNModelVetting \
      --config_name=base_new \
      --train_files="${TRAIN_FILES}" \
      --eval_files="" \
      --pretrain_model_dir="/pdo/users/dmuth/mnt/tess/fa1t_38_run_1/10" \
      --train_steps=2500 \
      --train_epochs=1 \
      --model_dir="${MODEL_DIR_BASE}/fold_${FOLD}_ensemble_${ENSEMBLE}"

    echo "Finished fold ${FOLD} of ensemble ${ENSEMBLE}"
  done
done
