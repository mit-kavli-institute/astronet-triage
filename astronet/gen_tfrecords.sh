#!/bin/bash

set -e

LCDIR=/mnt/tess/lc

python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/mnt/tess/astronet/tces-v8-train.csv --tess_data_dir=${LCDIR} --output_dir=/mnt/tess/astronet/tfrecords-27-train --num_shards=50

python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/mnt/tess/astronet/tces-v8-val.csv --tess_data_dir=${LCDIR} --output_dir=/mnt/tess/astronet/tfrecords-27-val --num_shards=5

# python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/mnt/tess/astronet/tces-v5-test.csv --tess_data_dir=${LCDIR} --output_dir=/mnt/tess/astronet/tfrecords-25-test --num_shards=5
