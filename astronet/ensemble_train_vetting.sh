#!/bin/bash

set -e

for i in {1..10}
do
    echo "Training model ${i}"
    python astronet/train.py \
        --model=AstroCNNModelVetting \
        --config_name=base_new \
        --train_files='../mnt/tess/astronet/tfrecords-vetting-01-train/*' \
        --eval_files='../mnt/tess/astronet/tfrecords-vetting-01-val/*' \
        --pretrain_model_dir='/pdo/users/dmuth/mnt/tess/fa1t_38_run_1/10' \
        --train_steps=2500 \  # Try hyperparameter tuning between 500 to 2500 with 500 increments (maybe use an ensemble of 2 or 3 instead of 10 networks)
        --train_epochs=1 \
        --model_dir="../mnt/tess/astronet/checkpoints/new_test/${i}" #direct_7_run_4/${i}"
done

# for i in {1..10}
# do
#     echo "Training model ${i}"
#     python astronet/train.py \
#         --model=AstroCNNModelVetting \
#         --config_name=direct \
#         --train_files='../mnt/tess/astronet/tfrecords-vetting-7-train/*' \
#         --eval_files='../mnt/tess/astronet/tfrecords-vetting-7-toi-val/*' \
#         --pretrain_model_dir="../mnt/tess/astronet/checkpoints/revised_tuned_30_run_1/${i}" \
#         --train_steps=0 \
#         --train_epochs=1 \
#         --model_dir="../mnt/tess/astronet/checkpoints/direct_7_notoi_run_4/${i}"
# done
