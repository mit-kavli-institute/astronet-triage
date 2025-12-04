#!/bin/bash
# Script to extract backbone features from a trained model

set -e  # Exit on error



# Configuration - EDIT THESE PATHS
MODEL_DIR='/pdo/astronet-data/models/vetting/experimental/pablomer/dec2025_cad_scat/20251204/pablomer-2k-nopretrained/AstroCNNModelVetting_pablomer_20251204_133625/'
DATA_FILES='/pdo/astronet-data/data/tfrecords/dec2025_cad_scat/tfrecords-vetting-v01-tois-triageJs-nocentroid-april2025-val/*'
OUTPUT_DIR='/pdo/users/pablomer/Astronet-Triage/model_analysis/extracted_backbone_features'
BATCH_SIZE=''  # Optional: leave empty to use model config default, or set to a number like '100'

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Change to project root to ensure imports work
cd "${PROJECT_ROOT}"

echo ""
echo "================================================================================"
echo "Extracting Backbone Features"
echo "================================================================================"
echo "Model directory: ${MODEL_DIR}"
echo "Data files: ${DATA_FILES}"
echo "Output directory: ${OUTPUT_DIR}"
if [ -n "${BATCH_SIZE}" ]; then
    echo "Batch size: ${BATCH_SIZE}"
else
    echo "Batch size: (using model config default)"
fi
echo "================================================================================"
echo ""

# Build the command
CMD="python ${SCRIPT_DIR}/extract_backbone_features.py"
CMD="${CMD} --model_dir=\"${MODEL_DIR}\""
CMD="${CMD} --data_files=\"${DATA_FILES}\""
CMD="${CMD} --output_dir=\"${OUTPUT_DIR}\""

if [ -n "${BATCH_SIZE}" ]; then
    CMD="${CMD} --batch_size=${BATCH_SIZE}"
fi

# Execute the command
echo "Running: ${CMD}"
echo ""
eval ${CMD}

echo ""
echo "================================================================================"
echo "✅ Done! Features extracted to: ${OUTPUT_DIR}"
echo "================================================================================"
