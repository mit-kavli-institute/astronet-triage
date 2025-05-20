#!/bin/bash
set -euo pipefail

LCDIR="../mnt/tess/april2025_dataset_fits_files"
NAME="vetting-v01-tois-triageJs-nocentroid-april2025"

for i in {1..5}; do
    echo "Generating tfrecords for fold ${i}"

    TRAIN_CSV="../mnt/tess/astronet/crossvalcsvs/tces-${NAME}-train_fold${i}.csv"
    VAL_CSV="../mnt/tess/astronet/crossvalcsvs/tces-${NAME}-val_fold${i}.csv"

    for CSV in "$TRAIN_CSV" "$VAL_CSV"; do
        if [[ -f "$CSV" ]]; then
            echo "✅ Found $CSV"
        else
            echo "❌ Missing $CSV" >&2
            exit 1
        fi
    done

    /pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python3 \
      astronet/preprocess/generate_input_records.py \
      --input_tce_csv_file="$TRAIN_CSV" \
      --tess_data_dir="$LCDIR" \
      --output_dir="../mnt/tess/astronet/crossvaltfrecords/tfrecords-${NAME}-train_fold${i}" \
      --mode=vetting \
      --num_shards=5

    /pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python3 \
      astronet/preprocess/generate_input_records.py \
      --input_tce_csv_file="$VAL_CSV" \
      --tess_data_dir="$LCDIR" \
      --output_dir="../mnt/tess/astronet/crossvaltfrecords/tfrecords-${NAME}-val_fold${i}" \
      --mode=vetting \
      --num_shards=5
done
