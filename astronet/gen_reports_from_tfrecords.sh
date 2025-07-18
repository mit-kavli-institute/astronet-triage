#!/bin/bash

TFRECORD_DIR=/pdo/users/dimond/mnt/tess/astronet/tfrecords-vetting-sector86-all-test
#OUTPUT_DIR=/pdo/users/dimond/astronet_secondary/astronet/astronet/reports
OUTPUT_DIR=/pdo/astronet-data/data/tfrecord_reports/

echo "Generating TFRecord Reports For $TFRECORD_DIR -- Saving to $OUTPUT_DIR"
python astronet/gen_reports_from_tfrecords.py \
    --tfrecord_dir="$TFRECORD_DIR" \
    --output_dir="$OUTPUT_DIR"