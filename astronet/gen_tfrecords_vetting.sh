#!/bin/bash

set -e

TEMPDIR=tmp-vetting/tfrecords
LCDIR=/mnt/tess/lc-v

rm -Rf ${TEMPDIR}
mkdir -p ${TEMPDIR}

python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/mnt/tess/astronet/tces-vetting-v2-train.csv --tess_data_dir=${LCDIR} --output_dir=${TEMPDIR}/tfrecords-vetting-2-train --vetting_features=y --num_shards=2

python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/mnt/tess/astronet/tces-vetting-v2-val.csv --tess_data_dir=${LCDIR} --output_dir=${TEMPDIR}/tfrecords-vetting-2-val --vetting_features=y --num_shards=2

python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/mnt/tess/astronet/tces-vetting-v2-test.csv --tess_data_dir=${LCDIR} --output_dir=${TEMPDIR}/tfrecords-vetting-2-test --vetting_features=y --num_shards=2

cp -R ${TEMPDIR}/* /mnt/tess/astronet
rm -Rf tmp-vetting
