#!/bin/bash

set -e

for i in {1..10}
do
    echo "Training model ${i}"
    python astronet/train.py \
        --model=AstroCNNModel \
        --config_name=extended \
        --train_files='/mnt/tess/astronet/tfrecords-35-y2-train/*' \
        --eval_files='/mnt/tess/astronet/tfrecords-35-y2-val/*' \
        --train_steps=0 \
        --model_dir="/mnt/tess/astronet/checkpoints/ext_35-y2_run_1/${i}"
done

