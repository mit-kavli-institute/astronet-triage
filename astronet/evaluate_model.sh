#!/bin/bash

set -e

DATE=20250430
ENSEMBLE_NAME=dimond

CODE_DIR=/pdo/users/dimond/astronet_secondary/astronet
DATA_DIR=/pdo/users/dimond/mnt/tess/astronet
TFRECORD_PREFIX=tfrecords-vetting-sector86-all-test
OUTPUT_DIR=/pdo/users/dimond/eval_test/pablo_model/
MODEL_DIR=/pdo/users/pablomer/mnt/tess/models/vetting/20250502/cshallue/AstroCNNModelVetting_cshallue_20250502_000812

# Note: if you're looping over multiple models, you can pattern-match $MODEL_DIR with wildcards or use `find`

echo "Evaluating model in $MODEL_DIR"
PYTHONPATH=$CODE_DIR python $CODE_DIR/astronet/evaluate.py \
    --model=AstroCNNModelVetting \
    --model_dir="$MODEL_DIR" \
    --eval_files="test:$DATA_DIR/$TFRECORD_PREFIX/*" \
    --output_dir="$OUTPUT_DIR"