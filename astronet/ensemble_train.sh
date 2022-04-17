#!/bin/bash

set -e

for i in {1..10}
do
    echo "Training model ${i}"
    python astronet/train.py \
        --model=AstroCNNModel \
        --config_name=revised_tuned \
        --train_files='/mnt/tess/astronet/tfrecords-35-train/*' \
        --eval_files='/mnt/tess/astronet/tfrecords-35-val/*' \
        --train_steps=0 \
        --model_dir="/mnt/tess/astronet/checkpoints/rev_tuned_35_run_2/${i}"
done

