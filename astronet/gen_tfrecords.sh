#!/bin/bash

set -e

TEMPDIR=tmp/tfrecords
LCDIR=~/lc

rm -Rf ${TEMPDIR}
mkdir -p ${TEMPDIR}

python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/mnt/tess/astronet/tces-v4-train.csv --tess_data_dir=${LCDIR} --output_dir=${TEMPDIR}/tfrecords-19-train --num_shards=20

python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/mnt/tess/astronet/tces-v4-val.csv --tess_data_dir=${LCDIR} --output_dir=${TEMPDIR}/tfrecords-19-val --num_shards=2

# python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/mnt/tess/astronet/tces-v4-test.csv --tess_data_dir=${LCDIR} --output_dir=${TEMPDIR}/tfrecords-16-test --num_shards=2

cp -R ${TEMPDIR}/* /mnt/tess/astronet
rm -Rf tmp
