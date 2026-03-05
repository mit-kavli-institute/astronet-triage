#!/bin/bash

set -e

DATE=$(date +%Y%m%d)
CONFIG_NAME=pablomer_cosine_aftertuning_
#CONFIG_OVERRIDES="inputs.random_reverse_time_series=true"
# CONFIG_OVERRIDES="train_steps=1000"
CONFIG_OVERRIDES="train_steps=3000,init_from_pretrained_model=false"
ENSEMBLE_NAME=$CONFIG_NAME


# PRETRAIN_MODEL_DIR=/pdo/users/cshallue/astronet/models/triage/20250420/cshallue-h5/AstroCNNModel_cshallue_20250420_174804
# PRETRAIN_MODEL_DIR=/pdo/users/pablomer/mnt/tess/models/triage/20250520/pablomer-h5/AstroCNNModel_pablomer_20250520_181651

# CODE_DIR=/pdo/users/cshallue/git/astronet
CODE_DIR=/pdo/users/pablomer/Astronet-Triage
# DATA_DIR=/pdo/users/pablomer/mnt/tess/astronet/
# DATA_DIR=/pdo/astronet-data/data/tfrecords/oct2025_30minbin/
# DATA_DIR=/pdo/astronet-data/data/tfrecords/oct2025_cadencebin/
DATA_DIR=/pdo/astronet-data/data/tfrecords/oct2025_original/
# DATA_DIR=/pdo/astronet-data/data/tfrecords/oct2025_30minbin_v2/
# TFRECORD_PREFIX=tfrecords-vetting-v01-tois-triageJs-nocentroid-newTOIs2025-v3-noreshufle

TFRECORD_PREFIX=tfrecords-vetting-v01-tois-triageJs-nocentroid-april2025


# OUTPUT_DIR=/pdo/users/pablomer/mnt/tess/models/vetting/$DATE/$ENSEMBLE_NAME/
OUTPUT_DIR=/pdo/astronet-data/models/vetting/experimental/pablomer/oct2025_original/$DATE-5/$ENSEMBLE_NAME-3k-no-pretrain/


# CONFIG_OVERRIDES="init_from_pretrained_model=true,\
# freeze_pretrained_params=true"

for i in {1..1}
do
    echo "Training model ${i}"
    PYTHONNOUSERSITE=1 \
    LD_LIBRARY_PATH=/pdo/users/cshallue/miniconda3/lib:$LD_LIBRARY_PATH \
    /pdo/users/cshallue/miniconda3/envs/astronet-gpu/bin/python $CODE_DIR/astronet/train.py \
        --model=AstroCNNModelVetting \
        --config_name=$CONFIG_NAME \
        --config_overrides=$CONFIG_OVERRIDES \
        --model_dir="$OUTPUT_DIR" \
        --train_files="$DATA_DIR/$TFRECORD_PREFIX-train/*" \
        --eval_files="val:$DATA_DIR/$TFRECORD_PREFIX-val/*" \
        --eval_files="test:$DATA_DIR/$TFRECORD_PREFIX-test/*" \
        --dump_block_weights=false \
        --log_training_history=true

    latest_model_dir=$(ls -td "$OUTPUT_DIR"/AstroCNNModelVetting_${CONFIG_NAME}_* 2>/dev/null | head -n 1)
    if [[ -n "$latest_model_dir" && -f "$latest_model_dir/training_history.json" ]]; then
        python - <<'PY'
import json
from pathlib import Path
import matplotlib.pyplot as plt

model_dir = Path(r"""'"$latest_model_dir"'""")
hist = json.loads((model_dir / "training_history.json").read_text())

plt.figure(figsize=(8, 5))
if "loss" in hist:
    plt.plot(hist["loss"], label="train_loss")
if "val_loss" in hist:
    plt.plot(hist["val_loss"], label="val_loss")
plt.xlabel("step")
plt.ylabel("loss")
plt.legend()
plt.tight_layout()
out_path = model_dir / "loss_plot.png"
plt.savefig(out_path)
print("Saved", out_path)
PY
    else
        echo "Skipping loss plot; training_history.json not found in $latest_model_dir"
    fi
done

# After training loop, generate a csv with the combined predictions
echo "All models trained. Now generating combined predictions..."
PYTHONNOUSERSITE=1 \
python \
    /pdo/users/pablomer/Astronet-Triage/astronet/combine_model_results.py \
    --base_path="$OUTPUT_DIR"
