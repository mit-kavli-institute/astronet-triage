#!/bin/bash
# Script to generate soft labels for both train and validation datasets

set -euo pipefail

ENSEMBLE_DIR='/pdo/astronet-data/models/vetting/experimental/pablomer/march2026/20260305/pablomer_final-final-3k-ensemble/'

TRAIN_DATA_DIR='/pdo/astronet-data/data/tfrecords/dec2025_cad_scat_v5_aug/10x_0p1/tfrecords-vetting-v01-tois-triageJs-nocentroid-dec2025-train/*'
VAL_DATA_DIR='/pdo/astronet-data/data/tfrecords/dec2025_cad_scat_v5_aug/10x_0p1/tfrecords-vetting-v01-tois-triageJs-nocentroid-dec2025-val/*'

PREDICTIONS_OUTPUT_ROOT='/pdo/astronet-data/data/labels/softlabels/march2026'
SOFTLABEL_TFRECORD_ROOT='/pdo/astronet-data/data/tfrecords/softlabels_runs/march2026'

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to process a dataset
process_dataset() {
    local dataset_name=$1
    local data_dir=$2
    local predictions_output_dir=$3
    local softlabel_output_dir=$4

    echo ""
    echo "================================================================================"
    echo "Processing ${dataset_name} dataset"
    echo "================================================================================"
    echo "Data directory: ${data_dir}"
    echo "Predictions output directory: ${predictions_output_dir}"
    echo "Softlabel TFRecord output directory: ${softlabel_output_dir}"
    echo "================================================================================"

    # Step 1: Generate ensemble predictions
    echo ""
    echo "Step 1: Generating ensemble predictions for ${dataset_name}..."
    python "${SCRIPT_DIR}/generate_ensemble_predictions.py" \
        --ensemble_dir "${ENSEMBLE_DIR}" \
        --data_dir "${data_dir}" \
        --output_dir "${predictions_output_dir}"

    # Derive input TFRecord directory from data_dir (remove /* pattern)
    local input_tfrecord_dir
    input_tfrecord_dir=$(echo "${data_dir}" | sed 's|/\*$||')
    if [ ! -d "${input_tfrecord_dir}" ]; then
        input_tfrecord_dir=$(dirname "${input_tfrecord_dir}")
    fi

    # Step 2: Update TFRecords with soft labels
    echo ""
    echo "Step 2: Updating TFRecords with soft labels for ${dataset_name}..."
    python "${SCRIPT_DIR}/update_tfrecords_with_predictions.py" \
        --input_tfrecord_dir "${input_tfrecord_dir}" \
        --predictions_csv "${predictions_output_dir}/ensemble_predictions_averaged.csv" \
        --output_tfrecord_dir "${softlabel_output_dir}"

    echo ""
    echo "✅ Completed processing ${dataset_name} dataset"
    echo ""
}

# Main execution
main() {
    ENSEMBLE_DATE="$(basename "$(dirname "${ENSEMBLE_DIR}")")"
    ENSEMBLE_NAME="$(basename "${ENSEMBLE_DIR}")"
    RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    RUN_TAG="${ENSEMBLE_DATE}_${ENSEMBLE_NAME}_${RUN_TIMESTAMP}"

    BASE_OUTPUT_DIR="${PREDICTIONS_OUTPUT_ROOT}/${RUN_TAG}"
    SOFTLABEL_RUN_DIR="${SOFTLABEL_TFRECORD_ROOT}/${RUN_TAG}"

    echo "================================================================================"
    echo "Soft Labels Generation Pipeline"
    echo "================================================================================"
    echo "Ensemble directory: ${ENSEMBLE_DIR}"
    echo "Predictions base output directory: ${BASE_OUTPUT_DIR}"
    echo "Softlabel TFRecords run directory: ${SOFTLABEL_RUN_DIR}"
    echo "================================================================================"

    # Create run-specific output directories so previous runs are not overwritten.
    mkdir -p "${BASE_OUTPUT_DIR}"
    mkdir -p "${SOFTLABEL_RUN_DIR}"

    # Process train dataset
    TRAIN_OUTPUT_DIR="${BASE_OUTPUT_DIR}/train"
    TRAIN_SOFTLABEL_DIR="${SOFTLABEL_RUN_DIR}/$(basename "${TRAIN_DATA_DIR%/*}")"
    process_dataset "train" "${TRAIN_DATA_DIR}" "${TRAIN_OUTPUT_DIR}" "${TRAIN_SOFTLABEL_DIR}"

    # Process validation dataset
    VAL_OUTPUT_DIR="${BASE_OUTPUT_DIR}/val"
    VAL_SOFTLABEL_DIR="${SOFTLABEL_RUN_DIR}/$(basename "${VAL_DATA_DIR%/*}")"
    process_dataset "val" "${VAL_DATA_DIR}" "${VAL_OUTPUT_DIR}" "${VAL_SOFTLABEL_DIR}"

    {
        echo "ENSEMBLE_DIR=${ENSEMBLE_DIR}"
        echo "RUN_TAG=${RUN_TAG}"
        echo "PREDICTIONS_BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR}"
        echo "SOFTLABEL_RUN_DIR=${SOFTLABEL_RUN_DIR}"
        echo "TRAIN_SOFTLABEL_DIR=${TRAIN_SOFTLABEL_DIR}"
        echo "VAL_SOFTLABEL_DIR=${VAL_SOFTLABEL_DIR}"
    } > "${BASE_OUTPUT_DIR}/pipeline_outputs.env"

    echo ""
    echo "================================================================================"
    echo "✅ All datasets processed successfully!"
    echo "================================================================================"
    echo ""
    echo "Output directories:"
    echo "  Train predictions: ${TRAIN_OUTPUT_DIR}"
    echo "  Val predictions: ${VAL_OUTPUT_DIR}"
    echo ""
    echo "Updated TFRecord directories (run-specific):"
    echo "  Train: ${TRAIN_SOFTLABEL_DIR}"
    echo "  Val: ${VAL_SOFTLABEL_DIR}"
    echo ""
    echo "Run metadata:"
    echo "  ${BASE_OUTPUT_DIR}/pipeline_outputs.env"
    echo ""
}

# Run main function
main
