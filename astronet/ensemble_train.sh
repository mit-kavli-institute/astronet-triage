#!/bin/bash

set -e

for i in {1..10}
do
    echo "Training model ${i}"
    python astronet/train.py \
        --model=AstroCNNModel \
        --config_name=extended \
        --train_files='/mnt/tess/astronet/tfrecords-19-train/*' \
        --eval_files='/mnt/tess/astronet/tfrecords-19-val/*' \
        --train_steps=20000 \
        --train_epochs=1 \
        --model_dir="/mnt/tess/astronet/checkpoints/extended_19_run_8/${i}"
done

