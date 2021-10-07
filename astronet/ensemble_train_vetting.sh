#!/bin/bash

set -e

for i in {1..10}
do
    echo "Training model ${i}"
    python astronet/train.py \
        --model=AstroCNNModelVetting \
        --config_name=vrevised_tuned \
        --train_files='/mnt/tess/astronet/tfrecords-vetting-4-toi-train/*' \
        --eval_files='/mnt/tess/astronet/tfrecords-vetting-4-toi-val/*' \
        --pretrain_model_dir="/mnt/tess/astronet/checkpoints/revised_tuned_27_run_1/${i}" \
        --train_steps=0 \
        --train_epochs=1 \
        --model_dir="/mnt/tess/astronet/checkpoints/vetting_vrevised_tuned_4_run_6/${i}"
done

