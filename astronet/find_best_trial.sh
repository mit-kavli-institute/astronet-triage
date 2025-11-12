#!/bin/bash

# Script to find the best trial from a tuning study
# Usage: ./astronet/find_best_trial.sh --study_dir=<path> [--metric=<metric>]

python astronet/tuning/find_best_trial.py "$@"


# To call, do on terminal:
# ./astronet/find_best_trial.sh \
#      --study_dir=/pdo/users/cshallue/astronet/models/triage-tune/20250915/decay-adam-alpha0p01 \
#      --metric=val_loss


# ./astronet/find_best_trial.sh \
#      --study_dir=/pdo/astronet-data/models/vetting/experimental/pablomer/tuning/20251110/cosinedecay-adam \
#      --metric=val_average_precision
