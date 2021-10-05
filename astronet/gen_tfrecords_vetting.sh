#!/bin/bash

set -e

LCDIR=/mnt/tess/lc-v

python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/mnt/tess/astronet/tces-vetting-v3-train.csv --tess_data_dir=${LCDIR} --output_dir=/mnt/tess/astronet/tfrecords-vetting-4-train --vetting_features=y --num_shards=2

python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/mnt/tess/astronet/tces-vetting-v3-val.csv --tess_data_dir=${LCDIR} --output_dir=/mnt/tess/astronet/tfrecords-vetting-4-val --vetting_features=y --num_shards=2

# python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/mnt/tess/astronet/tces-vetting-v3-test.csv --tess_data_dir=${LCDIR} --output_dir=/mnt/tess/astronet/tfrecords-vetting-4-test --vetting_features=y --num_shards=2

python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/mnt/tess/astronet/tces-vetting-v3-notoi-train.csv --tess_data_dir=${LCDIR} --output_dir=/mnt/tess/astronet/tfrecords-vetting-4-notoi-train --vetting_features=y --num_shards=2

python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/mnt/tess/astronet/tces-vetting-v3-notoi-val.csv --tess_data_dir=${LCDIR} --output_dir=/mnt/tess/astronet/tfrecords-vetting-4-notoi-val --vetting_features=y --num_shards=2

# python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/mnt/tess/astronet/tces-vetting-v3-test.csv --tess_data_dir=${LCDIR} --output_dir=/mnt/tess/astronet/tfrecords-vetting-4-test --vetting_features=y --num_shards=2
