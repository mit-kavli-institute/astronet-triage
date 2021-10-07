#!/bin/bash

set -e

for i in {1..10}
do
    echo "Training model ${i}"
    python astronet/train.py \
        --model=AstroCNNModel \
        --config_name=revised_tuned \
        --train_files='/mnt/tess/astronet/tfrecords-30-train/*' \
        --eval_files='/mnt/tess/astronet/tfrecords-30-val/*' \
        --train_steps=0 \
        --model_dir="/mnt/tess/astronet/checkpoints/revised_tuned_30_run_1/${i}"
done

