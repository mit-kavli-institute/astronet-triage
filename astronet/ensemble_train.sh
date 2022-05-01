#!/bin/bash

set -e

for i in {1..10}
do
    echo "Training model ${i}"
    python astronet/train.py \
        --model=AstroCNNModel \
        --config_name=final_alpha_1 \
        --train_files='/mnt/tess/astronet/tfrecords-38-train/*' \
        --eval_files='/mnt/tess/astronet/tfrecords-38-val/*' \
        --train_steps=0 \
        --model_dir="/mnt/tess/astronet/checkpoints/fa1_38_run_1/${i}"
done

