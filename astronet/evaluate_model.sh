#!/bin/bash

set -e

DATE=20250731
ENSEMBLE_NAME=dimond

CODE_DIR=/pdo/users/dimond/astronet_secondary/astronet
DATA_DIR=/pdo/astronet-data/data/tfrecords
TFRECORD_PREFIX=sector-87
OUTPUT_DIR=/pdo/astronet-data/models/vetting/experimental/dimond/sectors_85_to_87_with_embeddings
MODEL_DIR=/pdo/users/pablomer/mnt/tess/models/vetting/20250502/cshallue/AstroCNNModelVetting_cshallue_20250502_000812

# Note: if you're looping over multiple models, you can pattern-match $MODEL_DIR with wildcards or use `find`

echo "Evaluating model in $MODEL_DIR"
PYTHONPATH=$CODE_DIR python $CODE_DIR/astronet/evaluate.py \
    --model=AstroCNNModelVetting \
    --model_dir="$MODEL_DIR" \
    --eval_files="test:$DATA_DIR/sector-87/*" \
    --eval_files="test:$DATA_DIR/sector-85/*" \
    --eval_files="test:$DATA_DIR/sector-86/*" \
    --output_dir="$OUTPUT_DIR"
