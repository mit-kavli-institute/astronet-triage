#!/bin/bash

set -e

LCDIR=/mnt/tess/lc-v

python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/mnt/tess/astronet/tces-vetting-v2-train.csv --tess_data_dir=${LCDIR} --output_dir=/mnt/tess/astronet/tfrecords-vetting-3-train --vetting_features=y --num_shards=2

python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/mnt/tess/astronet/tces-vetting-v2-val.csv --tess_data_dir=${LCDIR} --output_dir=/mnt/tess/astronet/tfrecords-vetting-3-val --vetting_features=y --num_shards=2

python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/mnt/tess/astronet/tces-vetting-v2-test.csv --tess_data_dir=${LCDIR} --output_dir=/mnt/tess/astronet/tfrecords-vetting-3-test --vetting_features=y --num_shards=2
