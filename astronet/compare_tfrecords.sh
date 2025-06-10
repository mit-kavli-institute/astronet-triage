
TFRECORD_DIR_A=/pdo/users/dimond/mnt/tess/astronet/tfrecords-vetting-sector86-all-test
MODEL_DIR=/pdo/users/pablomer/mnt/tess/models/vetting/20250502/cshallue/AstroCNNModelVetting_cshallue_20250502_000812
CODE_DIR=/pdo/users/dimond/astronet_secondary/astronet


echo "Evaluating model in $MODEL_DIR"
PYTHONPATH=$CODE_DIR python $CODE_DIR/astronet/compare_tfrecords.py \
    --model=AstroCNNModelVetting \
    --model_dir="$MODEL_DIR" \
    --tfrecord_dir_a="$TFRECORD_DIR_A/*" \
    --tfrecord_dir_b="/pdo/users/pablomer/mnt/tess/astronet/tfrecords-vetting-v01-tois-triageJs-nocentroid-newTOIs2025-v3-noreshufle-test/*"