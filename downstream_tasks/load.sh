#!/bin/bash

# Exit on any error
set -e

# --- Configuration ---
# Path to your specific environment's python
PYTHON_EXE=/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python

# Path to the Astronet-Triage code directory (to ensure imports work)
CODE_DIR=/pdo/users/pablomer/Astronet-Triage

# Path to the specific script you want to run
# SCRIPT_PATH=$CODE_DIR/downstream_tasks/load_model.py
# SCRIPT_PATH=$CODE_DIR/downstream_tasks/read_embeddings.py
# SCRIPT_PATH=$CODE_DIR/downstream_tasks/global_view.py
SCRIPT_PATH=$CODE_DIR/downstream_tasks/global_view_AE.py

# --- Set up Environment Variables ---
# This ensures that 'import astronet' etc. works correctly
export PYTHONPATH=$CODE_DIR:$PYTHONPATH

echo "Starting model load script using environment: daniel_env_cloned_v2"
echo "Python path: $PYTHON_EXE"


# --- Execute ---
$PYTHON_EXE $SCRIPT_PATH


# echo "Starting read embeddings script using environment: daniel_env_cloned_v2"
# SCRIPT_PATH=$CODE_DIR/downstream_tasks/read_embeddings.py

# $PYTHON_EXE $SCRIPT_PATH


echo "Done."
