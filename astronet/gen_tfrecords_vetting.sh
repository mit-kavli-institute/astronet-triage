#!/bin/bash

set -e

LCDIR=../mnt/tess/lc

python astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-vetting-v02-tois_as_planets-train.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-vetting-v02-tois_as_planets-train --mode=vetting --num_shards=5

python astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-vetting-v02-tois_as_planets-val.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-vetting-v02-tois_as_planets-val --mode=vetting --num_shards=5

python astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-vetting-v02-tois_as_planets-test.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-vetting-v02-tois_as_planets-test --mode=vetting --num_shards=5




# comment out toi 
# python astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-vetting-v7-toi-train.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-vetting-8-toi-train --vetting_features=y --num_shards=2

# python astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-vetting-v7-toi-val.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-vetting-8-toi-val --vetting_features=y --num_shards=2

# # python astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-vetting-v7-toi-test.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-vetting-8-toi-test --vetting_features=y --num_shards=2
4