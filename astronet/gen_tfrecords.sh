#!/bin/bash

set -e

TEMPDIR=tmp/tfrecords
LCDIR=/mnt/tess/lc

rm -Rf ${TEMPDIR}
mkdir -p ${TEMPDIR}

python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/mnt/tess/astronet/tces-v8-train.csv --tess_data_dir=${LCDIR} --output_dir=${TEMPDIR}/tfrecords-27-train --num_shards=20

python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/mnt/tess/astronet/tces-v8-val.csv --tess_data_dir=${LCDIR} --output_dir=${TEMPDIR}/tfrecords-27-val --num_shards=2

# python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/mnt/tess/astronet/tces-v5-test.csv --tess_data_dir=${LCDIR} --output_dir=${TEMPDIR}/tfrecords-25-test --num_shards=2

cp -R ${TEMPDIR}/* /mnt/tess/astronet
rm -Rf tmp
