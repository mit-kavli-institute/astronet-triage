#!/bin/bash

set -e

LCDIR=/pdo/users/dimond/tfrecord_gen/fits
NAME=vetting-sector82-all

#python preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-${NAME}-train.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-${NAME}-train --mode=vetting --num_shards=5

#python preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-${NAME}-val.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-${NAME}-val --mode=vetting --num_shards=5

python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/pdo/astronet-data/data/properties/tces-sector82.csv --tess_data_dir=${LCDIR} --output_dir=/pdo/astronet-data/data/tfrecords/sector-82 --mode=vetting --num_shards=1




# comment out toi 
# python astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-vetting-v7-toi-train.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-vetting-8-toi-train --vetting_features=y --num_shards=2

# python astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-vetting-v7-toi-val.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-vetting-8-toi-val --vetting_features=y --num_shards=2

# # python astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-vetting-v7-toi-test.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-vetting-8-toi-test --vetting_features=y --num_shards=2
