#!/bin/bash

set -e

export OPENBLAS_NUM_THREADS=2

# LCDIR=../mnt/tess/lc_vetting_and_triage_symbolic_links
# LCDIR=/pdo/users/pablomer/mnt/tess/new_fit_files_combined
# LCDIR=../mnt/tess/lc_vetting_and_triage_symbolic_links
# LCDIR=../mnt/tess/april2025_dataset_fits_files/
LCDIR=/pdo/astronet-data/data/fits/oct2025_dataset_all_fits_files/

# NAME=vetting-v01-tois-triageJs-nocentroid-april2025dataset
NAME=vetting-v01-tois-triageJs-nocentroid-april2025

# Redirect all output (stdout + stderr) to log.txt
exec > >(tee "generate_records_${NAME}_2.log") 2>&1

/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python3 astronet/preprocess/generate_input_records_2.py --input_tce_csv_file=../mnt/tess/astronet/tces-${NAME}-train.csv --tess_data_dir=${LCDIR} --output_dir=/pdo/astronet-data/data/tfrecords/oct2025_original_aug/tfrecords-${NAME}-train --mode=vetting --num_shards=50 --remove_random_points
