#!/bin/bash

set -e

# Use specific Python interpreter from cshallue's environment
PYTHONNOUSERSITE=1 \
LD_LIBRARY_PATH=/pdo/users/cshallue/miniconda3/lib:$LD_LIBRARY_PATH \
/pdo/users/cshallue/miniconda3/envs/astronet-gpu/bin/python astronet/tune_local.py \
  --config_file=/pdo/users/pablomer/mnt/tess/astronet/studies/20251110-vetting-cosine.json \
  --config_overrides="base_param_overrides.hparams.learning_rate_decay_alpha=0.01" \
  --study_dir=/pdo/astronet-data/models/vetting/experimental/pablomer/tuning/20251110/cosinedecay-adam \
  --n_trials=100 \
  --gpu=2 \
  --overwrite
