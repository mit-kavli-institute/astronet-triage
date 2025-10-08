#!/bin/bash

set -e

# LCDIR=../mnt/tess/lc_vetting_and_triage_symbolic_links
# LCDIR=/pdo/users/pablomer/mnt/tess/new_fit_files_combined
# LCDIR=../mnt/tess/lc_vetting_and_triage_symbolic_links
LCDIR=/pdo/astronet-data/data/fits/sector-87-p-v2
# CSVDIR=/pdo/users/pablomer/mnt/tess/debug-h5fits/tois_sector_85_to_87_all_renamed_withfilenames_renamed.csv

CSVDIR=/pdo/astronet-data/data/properties/tces-sector87-p.csv
# NAME=vetting-v01-tois-triageJs-nocentroid-april2025dataset
# NAME=vetting-v01-tois-triageJs-nocentroid-april2025
NAME=sector87

OUTPUTDIR=/pdo/astronet-data/data/tfrecords/sector-87-p-v4

/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python3 astronet/preprocess/generate_input_records.py --input_tce_csv_file=${CSVDIR} --tess_data_dir=${LCDIR} --output_dir=${OUTPUTDIR} --mode=vetting --num_shards=20 --training=false

# /pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python3 astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-${NAME}-val.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-${NAME}-val --mode=vetting --num_shards=5

# /pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python3 astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-${NAME}-test.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-${NAME}-test --mode=vetting --num_shards=5
