#!/bin/bash

set -e

NAME=vetting-v02-tois-triageJs-nocentroid

for i in {1..10}
do
    echo "Training model ${i}"
    python astronet/train.py \
        --model=AstroCNNModelVetting \
        --config_name=base_new \
        --train_files="../mnt/tess/astronet/tfrecords-${NAME}-train/*" \
        --eval_files="../mnt/tess/astronet/tfrecords-${NAME}-notexist-val/*" \
        --pretrain_model_dir="/pdo/users/dmuth/mnt/tess/fa1t_38_run_1/10" \
        --train_steps=2500 \
        --train_epochs=1 \
        --model_dir="../mnt/tess/astronet/checkpoints/${NAME}_base_new_2500/${i}"
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
# # Try hyperparameter tuning between 500 to 2500 with 500 increments (maybe use an ensemble of 2 or 3 instead of 10 networks)
