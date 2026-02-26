#!/bin/bash

set -euo pipefail

DATE=$(date +%Y%m%d)
CONFIG_NAME=pablomer
CONFIG_OVERRIDES="train_steps=2000,init_from_pretrained_model=false"
ENSEMBLE_NAME=$CONFIG_NAME

PRETRAIN_MODEL_DIR=/pdo/users/pablomer/mnt/tess/models/triage/20250520/pablomer-h5/AstroCNNModel_pablomer_20250520_181651
CODE_DIR=/pdo/users/pablomer/Astronet-Triage
PYTHON_BIN=/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python

DATA_DIR=/pdo/astronet-data/data/tfrecords/dec2025_cad_scat_v5_duration24/
TFRECORD_PREFIX=tfrecords-vetting-v01-tois-triageJs-nocentroid-dec2025

OUTPUT_DIR=/pdo/astronet-data/models/vetting/experimental/pablomer/dec2025_cad_scat_v5_duration24/$DATE/$ENSEMBLE_NAME-2k-nopretrained-rp/

RP_METADATA_CSV=/pdo/users/pablomer/mnt/tess/astronet/tces-vetting-v01-tois-triageJs-nocentroid-dec2025-test.csv
RP_FILTER_DIR=/pdo/users/pablomer/Astronet-Triage/rp_filter
RP_THRESHOLD=0.9
RP_OUTPUT_CSV=all_preds_with_rp.csv
RP_PLOT_PNG=$RP_FILTER_DIR/rp_distribution_${DATE}_${ENSEMBLE_NAME}_threshold_0p9.png

for i in {1..2}
do
    echo "Training model ${i}"
    "$PYTHON_BIN" "$CODE_DIR/astronet/train.py" \
        --model=AstroCNNModelVetting \
        --config_name=$CONFIG_NAME \
        --config_overrides=$CONFIG_OVERRIDES \
        --pretrain_model_dir=$PRETRAIN_MODEL_DIR \
        --model_dir="$OUTPUT_DIR" \
        --train_files="$DATA_DIR/$TFRECORD_PREFIX-train/*" \
        --eval_files="val:$DATA_DIR/$TFRECORD_PREFIX-val/*" \
        --eval_files="test:$DATA_DIR/$TFRECORD_PREFIX-test/*" \
        --dump_block_weights=false
done

echo "All models trained. Generating combined predictions and r_p output..."
"$PYTHON_BIN" "$CODE_DIR/astronet/combine_model_results.py" \
    --base_path="$OUTPUT_DIR" \
    --output_rp=true \
    --rp_metadata_csv="$RP_METADATA_CSV" \
    --rp_output_filename="$RP_OUTPUT_CSV"

echo "Generating r_p distribution plot..."
"$PYTHON_BIN" "$RP_FILTER_DIR/plot_rp_distribution.py" \
    --input_csv="$OUTPUT_DIR/$RP_OUTPUT_CSV" \
    --threshold="$RP_THRESHOLD" \
    --output_png="$RP_PLOT_PNG"

echo "Done."
echo "Combined CSV: $OUTPUT_DIR/all_preds.csv"
echo "r_p CSV: $OUTPUT_DIR/$RP_OUTPUT_CSV"
echo "Plot: $RP_PLOT_PNG"
