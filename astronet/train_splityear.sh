#!/bin/bash

set -e

for i in {1..10}
do
    echo "Training Y1 model ${i}"
    python astronet/train.py \
        --model=AstroCNNModel \
        --config_name=final_alpha_1 \
        --train_files='/mnt/tess/astronet/tfrecords-38-y1-train/*' \
        --eval_files='/mnt/tess/astronet/tfrecords-38-y1-val/*' \
        --train_steps=0 \
        --model_dir="/mnt/tess/astronet/checkpoints/fa1_38_y1_run_1/${i}"
done

for i in {1..10}
do
    echo "Training Y2 model ${i}"
    python astronet/train.py \
        --model=AstroCNNModel \
        --config_name=final_alpha_1 \
        --train_files='/mnt/tess/astronet/tfrecords-38-y2-train/*' \
        --eval_files='/mnt/tess/astronet/tfrecords-38-y2-val/*' \
        --train_steps=0 \
        --model_dir="/mnt/tess/astronet/checkpoints/fa1_38_y2_run_1/${i}"
done

for i in {1..10}
do
    echo "Training Y3 model ${i}"
    python astronet/train.py \
        --model=AstroCNNModel \
        --config_name=final_alpha_1 \
        --train_files='/mnt/tess/astronet/tfrecords-38-y3-train/*' \
        --eval_files='/mnt/tess/astronet/tfrecords-38-y3-val/*' \
        --train_steps=0 \
        --model_dir="/mnt/tess/astronet/checkpoints/fa1_38_y3_run_1/${i}"
done

