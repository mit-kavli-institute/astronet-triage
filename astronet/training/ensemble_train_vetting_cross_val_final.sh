NAME=vetting-v01-tois-triageJs-nocentroid-newTOIs2025-v3-noreshufle #Pablo March 2025

PYTHON_BIN=/pdo/users/dmuth/miniconda3/envs/tf/bin/python

for i in {0..4}
do
  echo "Cross-validation Fold ${i}:"

  TRAIN_FILES="../mnt/tess/astronet/fold_${i}_train/*"
  VAL_FILES="../mnt/tess/astronet/fold_${i}_val/*"

  echo "Training on files: $TRAIN_FILES"
  echo "Validating on files: $VAL_FILES"

  $PYTHON_BIN astronet/train.py \
    --model=AstroCNNModelVetting \
    --config_name=base_new \
    --train_files=$TRAIN_FILES \
    --eval_files=$VAL_FILES \
    --pretrain_model_dir="/pdo/users/dmuth/mnt/tess/fa1t_38_run_1/10" \
    --train_steps=2500 \
    --train_epochs=1 \
    --model_dir="../mnt/tess/astronet/checkpoints/${NAME}_base_new_2500_crossval_fold${i}"

done
echo "Training completed"
