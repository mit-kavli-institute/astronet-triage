#!/bin/bash

NAME=vetting-v01-tois-triageJs-nocentroid-newTOIs2025-v3-noreshufle
BASE_PATH="../mnt/tess/astronet/tfrecords-${NAME}-all"
DEST_PATH="../mnt/tess/astronet"

mkdir -p ${DEST_PATH}

for i in {0..4}
do
  echo "Creating fold_${i}_val and fold_${i}_train"

  mkdir -p ${DEST_PATH}/fold_${i}_val
  mkdir -p ${DEST_PATH}/fold_${i}_train

  VAL_FILE="0000${i}-of-00005"

  # Move validation file to the validation folder
  cp ${BASE_PATH}/${VAL_FILE} ${DEST_PATH}/fold_${i}_val/

  # Move training files (all except the validation file) to the training folder
  for file in ${BASE_PATH}/0000*-of-00005; do
    if [ $(basename "$file") != "$VAL_FILE" ]; then
      cp "$file" ${DEST_PATH}/fold_${i}_train/
    fi
  done

done

echo "Folders created and files distributed successfully."
