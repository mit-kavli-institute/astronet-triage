#!/bin/bash
# Script to generate soft labels for both train and validation datasets

set -e  # Exit on error

# Configuration - EDIT THESE PATHS
# ENSEMBLE_DIR='/pdo/astronet-data/models/vetting/experimental/pablomer/oct2025_30minbin/20251104/pablomer-2k-pretrained'
ENSEMBLE_DIR='/pdo/astronet-data/models/vetting/experimental/pablomer/oct2025_cadencebin/20251028/pablomer-2k-nopretrained/'
TRAIN_DATA_DIR='/pdo/astronet-data/data/tfrecords/oct2025_cadencebin/tfrecords-vetting-v01-tois-triageJs-nocentroid-april2025-train/*'
VAL_DATA_DIR='/pdo/astronet-data/data/tfrecords/oct2025_cadencebin/tfrecords-vetting-v01-tois-triageJs-nocentroid-april2025-val/*'
BASE_OUTPUT_DIR='/pdo/astronet-data/data/labels/softlabels/oct2025_cadencebin_20251028_pablomer-2k-nopretrained'

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to process a dataset
process_dataset() {
    local dataset_name=$1
    local data_dir=$2
    local output_dir=$3

    echo ""
    echo "================================================================================"
    echo "Processing ${dataset_name} dataset"
    echo "================================================================================"
    echo "Data directory: ${data_dir}"
    echo "Output directory: ${output_dir}"
    echo "================================================================================"

    # Step 1: Generate ensemble predictions
    echo ""
    echo "Step 1: Generating ensemble predictions for ${dataset_name}..."
    python "${SCRIPT_DIR}/generate_ensemble_predictions.py" \
        --ensemble_dir "${ENSEMBLE_DIR}" \
        --data_dir "${data_dir}" \
        --output_dir "${output_dir}"

    # Derive input TFRecord directory from data_dir (remove /* pattern)
    local input_tfrecord_dir=$(echo "${data_dir}" | sed 's|/\*$||')
    if [ ! -d "${input_tfrecord_dir}" ]; then
        input_tfrecord_dir=$(dirname "${input_tfrecord_dir}")
    fi

    # Step 2: Update TFRecords with soft labels
    echo ""
    echo "Step 2: Updating TFRecords with soft labels for ${dataset_name}..."
    python "${SCRIPT_DIR}/update_tfrecords_with_predictions.py" \
        --input_tfrecord_dir "${input_tfrecord_dir}" \
        --predictions_csv "${output_dir}/ensemble_predictions_averaged.csv"

    echo ""
    echo "✅ Completed processing ${dataset_name} dataset"
    echo ""
}

# Main execution
main() {
    echo "================================================================================"
    echo "Soft Labels Generation Pipeline"
    echo "================================================================================"
    echo "Ensemble directory: ${ENSEMBLE_DIR}"
    echo "Base output directory: ${BASE_OUTPUT_DIR}"
    echo "================================================================================"

    # Create base output directory
    mkdir -p "${BASE_OUTPUT_DIR}"

    # Process train dataset
    TRAIN_OUTPUT_DIR="${BASE_OUTPUT_DIR}/train"
    process_dataset "train" "${TRAIN_DATA_DIR}" "${TRAIN_OUTPUT_DIR}"

    # Process validation dataset
    VAL_OUTPUT_DIR="${BASE_OUTPUT_DIR}/val"
    process_dataset "val" "${VAL_DATA_DIR}" "${VAL_OUTPUT_DIR}"

    echo ""
    echo "================================================================================"
    echo "✅ All datasets processed successfully!"
    echo "================================================================================"
    echo ""
    echo "Output directories:"
    echo "  Train predictions: ${TRAIN_OUTPUT_DIR}"
    echo "  Val predictions: ${VAL_OUTPUT_DIR}"
    echo ""
    echo "Updated TFRecord directories:"
    echo "  Train: $(dirname ${TRAIN_DATA_DIR})_softlabels"
    echo "  Val: $(dirname ${VAL_DATA_DIR})_softlabels"
    echo ""
}

# Run main function
main
