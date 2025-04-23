#!/bin/bash

set -e

#modified from gen_tfrecords_vetting to generate the tf records for a subset of the test set
#which consists of only long period events

# LCDIR=../mnt/tess/lc_vetting_and_triage_symbolic_links
LCDIR=../../dmuth/mnt/tess/lc_vetting_and_triage_symbolic_links

NAME=vetting-v01-tois-triageJs-nocentroid-long

#python3 astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-${NAME}-train.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-${NAME}-train --mode=vetting --num_shards=5

#python3 astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-${NAME}-val.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-${NAME}-val --mode=vetting --num_shards=5


# source ~/miniconda3/etc/profile.d/conda.sh  # Ensure Conda is set up
# conda activate tf  # Activate the environment

/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python3 astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-${NAME}-test.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-${NAME}-test --mode=vetting --num_shards=5




# comment out toi
# python astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-vetting-v7-toi-train.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-vetting-8-toi-train --vetting_features=y --num_shards=2

# python astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-vetting-v7-toi-val.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-vetting-8-toi-val --vetting_features=y --num_shards=2

# # python astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-vetting-v7-toi-test.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-vetting-8-toi-test --vetting_features=y --num_shards=2
