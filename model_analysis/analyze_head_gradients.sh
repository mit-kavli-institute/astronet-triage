#!/bin/bash
# Script to analyze head gradients with respect to backbone features

set -e  # Exit on error



# Configuration - EDIT THESE PATHS
MODEL_DIR='/pdo/astronet-data/models/vetting/experimental/pablomer/dec2025_cad_scat/20251204/pablomer-2k-nopretrained/AstroCNNModelVetting_pablomer_20251204_133625/'
FEATURES_FILE='/pdo/users/pablomer/Astronet-Triage/model_analysis/extracted_backbone_features/features.npz'
FEATURE_SLICES_FILE='/pdo/users/pablomer/Astronet-Triage/model_analysis/extracted_backbone_features/feature_slices.json'
OUTPUT_DIR='/pdo/users/pablomer/Astronet-Triage/model_analysis/gradient_analysis'
TARGET_CLASS=''  # Optional: leave empty to use predicted class, or set to a number like '0'
USE_GRAD_TIMES_INPUT=''  # Optional: leave empty for default (True), or set to 'false' to disable
BATCH_SIZE=''  # Optional: leave empty to use default (100), or set to a number like '200'

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Change to project root to ensure imports work
cd "${PROJECT_ROOT}"

echo ""
echo "================================================================================"
echo "Analyzing Head Gradients"
echo "================================================================================"
echo "Model directory: ${MODEL_DIR}"
echo "Features file: ${FEATURES_FILE}"
echo "Feature slices file: ${FEATURE_SLICES_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
if [ -n "${TARGET_CLASS}" ]; then
    echo "Target class: ${TARGET_CLASS}"
else
    echo "Target class: (using predicted class for each example)"
fi
if [ -n "${USE_GRAD_TIMES_INPUT}" ]; then
    echo "Use Grad × Input: ${USE_GRAD_TIMES_INPUT}"
else
    echo "Use Grad × Input: (default: true)"
fi
if [ -n "${BATCH_SIZE}" ]; then
    echo "Batch size: ${BATCH_SIZE}"
else
    echo "Batch size: (using default: 100)"
fi
echo "================================================================================"
echo ""

# Build the command
CMD="python ${SCRIPT_DIR}/analyze_head_gradients.py"
CMD="${CMD} --model_dir=\"${MODEL_DIR}\""
CMD="${CMD} --features_file=\"${FEATURES_FILE}\""
CMD="${CMD} --feature_slices_file=\"${FEATURE_SLICES_FILE}\""
CMD="${CMD} --output_dir=\"${OUTPUT_DIR}\""

if [ -n "${TARGET_CLASS}" ]; then
    CMD="${CMD} --target_class=${TARGET_CLASS}"
fi

if [ -n "${USE_GRAD_TIMES_INPUT}" ]; then
    CMD="${CMD} --use_grad_times_input=${USE_GRAD_TIMES_INPUT}"
fi

if [ -n "${BATCH_SIZE}" ]; then
    CMD="${CMD} --batch_size=${BATCH_SIZE}"
fi

# Execute the command
echo "Running: ${CMD}"
echo ""
eval ${CMD}

echo ""
echo "================================================================================"
echo "✅ Done! Gradient analysis saved to: ${OUTPUT_DIR}"
echo "================================================================================"
