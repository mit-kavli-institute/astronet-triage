#!/bin/bash

set -e

export OPENBLAS_NUM_THREADS=2

# LCDIR=../mnt/tess/lc_vetting_and_triage_symbolic_links
# LCDIR=/pdo/users/pablomer/mnt/tess/new_fit_files_combined
# LCDIR=../mnt/tess/lc_vetting_and_triage_symbolic_links
# LCDIR=../mnt/tess/april2025_dataset_fits_files/
LCDIR=/pdo/astronet-data/data/fits/oct2025_dataset_all_fits_files/
# LCDIR=/pdo/users/pablomer/mnt/tess

# NAME=vetting-v01-tois-triageJs-nocentroid-april2025dataset
NAME=vetting-v01-tois-triageJs-nocentroid-dec2025

# Augmentation config
AUGMENT_TIMES=3       # number of augmented copies per TCE
TAG=10x_0p1               # base tag: 10x augmentation, 10% random drop


# Redirect all output (stdout + stderr) to a log file
exec > >(tee "generate_records_${NAME}_aug_${TAG}.log") 2>&1

OUT_DIR=/pdo/astronet-data/data/tfrecords/dec2025_cad_scat_v5_aug/${TAG}/tfrecords-${NAME}-train

mkdir -p "${OUT_DIR}"

OFFSET=7

# Run the TFRecord generation script AUGMENT_TIMES times with different file suffixes
for i in $(seq 1 "${AUGMENT_TIMES}"); do
  REP_IDX=$((i + OFFSET))
  FILE_SUFFIX="_rep${REP_IDX}"
  FIRST_SHARD=$(printf "%s/%05d-of-%05d_aug%s" "${OUT_DIR}" 0 50 "${FILE_SUFFIX}")

  # Avoid overwriting an existing augmentation replica
  if [ -f "${FIRST_SHARD}" ]; then
    echo "Skipping augmentation run ${REP_IDX}/${AUGMENT_TIMES}: first shard ${FIRST_SHARD} already exists."
    continue
  fi

  echo "Starting augmentation run ${REP_IDX}/${AUGMENT_TIMES} with suffix ${FILE_SUFFIX}"

  /pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python3 \
    astronet/preprocess/generate_input_records_3.py \
    --input_tce_csv_file=../mnt/tess/astronet/tces-${NAME}-train.csv \
    --tess_data_dir="${LCDIR}" \
    --output_dir="${OUT_DIR}" \
    --mode=vetting \
    --num_shards=50 \
    --remove_random_points \
    --file_suffix="${FILE_SUFFIX}"
done
