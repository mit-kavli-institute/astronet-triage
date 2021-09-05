#!/bin/bash

set -e

for i in {1..10}
do
    echo "Training model ${i}"
    python astronet/train.py \
        --model=AstroCNNModelVetting \
        --config_name=base \
        --train_files='/mnt/tess/astronet/tfrecords-vetting-1-train/*' \
        --eval_files='/mnt/tess/astronet/tfrecords-vetting-1-val/*' \
        --pretrain_model_dir="/mnt/tess/astronet/checkpoints/extended_26_run_18/${i}" \
        --train_steps=5000 \
        --train_epochs=1 \
        --model_dir="/mnt/tess/astronet/checkpoints/vetting_base_1_run_2/${i}"
done

