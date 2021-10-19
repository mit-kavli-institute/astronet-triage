#!/bin/bash

set -e

for i in {1..10}
do
    echo "Training model ${i}"
    python astronet/train.py \
        --model=AstroCNNModelVetting \
        --config_name=vrevised_tuned \
        --train_files='/mnt/tess/astronet/tfrecords-vetting-5-toi-train/*' \
        --eval_files='/mnt/tess/astronet/tfrecords-vetting-5-toi-val/*' \
        --pretrain_model_dir="/mnt/tess/astronet/checkpoints/revised_tuned_30_run_1/${i}" \
        --train_steps=0 \
        --train_epochs=1 \
        --model_dir="/mnt/tess/astronet/checkpoints/vrevised_tuned_5_run_1/${i}"
done

for i in {1..10}
do
    echo "Training model ${i}"
    python astronet/train.py \
        --model=AstroCNNModelVetting \
        --config_name=vrevised_tuned \
        --train_files='/mnt/tess/astronet/tfrecords-vetting-5-train/*' \
        --eval_files='/mnt/tess/astronet/tfrecords-vetting-5-val/*' \
        --pretrain_model_dir="/mnt/tess/astronet/checkpoints/revised_tuned_30_run_1/${i}" \
        --train_steps=0 \
        --train_epochs=1 \
        --model_dir="/mnt/tess/astronet/checkpoints/vrevised_tuned_5_notoi_run_1/${i}"
done
