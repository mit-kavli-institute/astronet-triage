#!/bin/bash

set -e

TEMPDIR=tmp-vetting/tfrecords
LCDIR=/mnt/tess/lc-v

rm -Rf ${TEMPDIR}
mkdir -p ${TEMPDIR}

python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/mnt/tess/astronet/tces-vetting-v1-train.csv --tess_data_dir=${LCDIR} --output_dir=${TEMPDIR}/tfrecords-vetting-1-train --vetting_features=y --num_shards=2

python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/mnt/tess/astronet/tces-vetting-v1-val.csv --tess_data_dir=${LCDIR} --output_dir=${TEMPDIR}/tfrecords-vetting-1-val --include_vetting_features=y --num_shards=2

cp -R ${TEMPDIR}/* /mnt/tess/astronet
rm -Rf tmp-vetting
