#!/bin/bash

set -euo pipefail

DATE=$(date +%Y%m%d)
CONFIG_NAME=pablomer_final
# CONFIG_OVERRIDES="train_steps=1000"
# CONFIG_OVERRIDES="train_steps=2000"
CONFIG_OVERRIDES=""

# CODE_DIR=/pdo/users/cshallue/git/astronet
CODE_DIR=/pdo/users/pablomer/Astronet-Triage

# Hardcoded teacher ensemble run (for matching softlabel pipeline outputs)
ENSEMBLE_DIR='/pdo/astronet-data/models/vetting/experimental/pablomer/march2026/20260305/pablomer_final-final-3k-ensemble/'

# Softlabel TFRecords run directories (produced by run_softlabels_pipeline.sh)
SOFTLABEL_TFRECORD_ROOT=/pdo/astronet-data/data/tfrecords/softlabels_runs/march2026

TFRECORD_PREFIX=tfrecords-vetting-v01-tois-triageJs-nocentroid-dec2025

# Output directory for trained student model
RUN_TIMESTAMP=$(date +%H%M%S)
OUTPUT_DIR=/pdo/astronet-data/models/vetting/experimental/pablomer/march2026/$DATE/student-distill-${RUN_TIMESTAMP}-s1-h0p05-t2/

# Distillation hyperparameterss
SOFT_LABEL_WEIGHT=1.0
HARD_LABEL_WEIGHT=0.05
TEMPERATURE=2.0

# Evaluation datasets (using original TFRecords, not softlabels)
EVAL_DATA_DIR=/pdo/astronet-data/data/tfrecords/dec2025_cad_scat_v5_aug/10x_0p1
EVAL_TFRECORD_PREFIX=tfrecords-vetting-v01-tois-triageJs-nocentroid-dec2025

resolve_softlabel_run_dir() {
    # Allow manual override if needed.
    if [ -n "${SOFTLABEL_RUN_DIR:-}" ]; then
        echo "${SOFTLABEL_RUN_DIR}"
        return
    fi

    local ensemble_date ensemble_name latest_softlabel_dir
    ensemble_date="$(basename "$(dirname "${ENSEMBLE_DIR}")")"
    ensemble_name="$(basename "${ENSEMBLE_DIR}")"

    latest_softlabel_dir=$(find "${SOFTLABEL_TFRECORD_ROOT}" -mindepth 1 -maxdepth 1 -type d \
        -name "${ensemble_date}_${ensemble_name}_*" \
        -printf '%T@ %p\n' \
        | sort -n \
        | tail -1 \
        | cut -d' ' -f2-)

    if [ -z "${latest_softlabel_dir}" ]; then
        echo "ERROR: No softlabel run found under ${SOFTLABEL_TFRECORD_ROOT} for ${ensemble_date}/${ensemble_name}" >&2
        echo "Run distillation/run_softlabels_pipeline.sh first, or set SOFTLABEL_RUN_DIR manually." >&2
        exit 1
    fi

    echo "${latest_softlabel_dir}"
}

SOFTLABEL_RUN_DIR="$(resolve_softlabel_run_dir)"
TRAIN_SOFTLABEL_DIR="${SOFTLABEL_RUN_DIR}/${TFRECORD_PREFIX}-train"

if [ ! -d "${TRAIN_SOFTLABEL_DIR}" ]; then
    echo "ERROR: Expected train softlabel directory not found: ${TRAIN_SOFTLABEL_DIR}" >&2
    exit 1
fi

echo "Training student model with soft labels"
echo "========================================"
echo "Config: $CONFIG_NAME"
echo "Softlabel run dir: $SOFTLABEL_RUN_DIR"
echo "Train files: $TRAIN_SOFTLABEL_DIR/*"
echo "Output dir: $OUTPUT_DIR"
echo "Soft label weight: $SOFT_LABEL_WEIGHT"
echo "Hard label weight: $HARD_LABEL_WEIGHT"
echo "Temperature: $TEMPERATURE"
echo "========================================"

PYTHONNOUSERSITE=1 \
LD_LIBRARY_PATH=/pdo/users/cshallue/miniconda3/lib:$LD_LIBRARY_PATH \
/pdo/users/cshallue/miniconda3/envs/astronet-gpu/bin/python $CODE_DIR/distillation/train_student.py \
    --model=AstroCNNModelVetting \
    --config_name=$CONFIG_NAME \
    --config_overrides="$CONFIG_OVERRIDES" \
    --model_dir="$OUTPUT_DIR" \
    --train_files="$TRAIN_SOFTLABEL_DIR/*" \
    --soft_label_weight=$SOFT_LABEL_WEIGHT \
    --hard_label_weight=$HARD_LABEL_WEIGHT \
    --temperature=$TEMPERATURE \
    --use_binary_loss_for_hard_labels \
    --eval_files="val:$EVAL_DATA_DIR/${EVAL_TFRECORD_PREFIX}-val/*" \
    --eval_files="test:$EVAL_DATA_DIR/${EVAL_TFRECORD_PREFIX}-test/*" \
    --shuffle_buffer_size=25000 \
    --save_format=h5

echo "Student model training complete!"
echo "Model saved to: $OUTPUT_DIR"
