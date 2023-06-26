#!/bin/bash

set -e

LCDIR=../mnt/tess/lc-v

python astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-vetting-v01-train.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-vetting-01-train --vetting_features=y --num_shards=2

python astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-vetting-v01-val.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-vetting-01-val --vetting_features=y --num_shards=2

python astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-vetting-v01-test.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-vetting-01-test --vetting_features=y --num_shards=2




# comment out toi 
# python astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-vetting-v7-toi-train.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-vetting-8-toi-train --vetting_features=y --num_shards=2

# python astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-vetting-v7-toi-val.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-vetting-8-toi-val --vetting_features=y --num_shards=2

# # python astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-vetting-v7-toi-test.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-vetting-8-toi-test --vetting_features=y --num_shards=2
