#!/bin/bash

FIRST_INDEX=$1 # 1 arg1
LAST_INDEX=$2 # 9500 arg2
NPROC=$3 # 30 processes

# Check if GNU Parallel is installed
if ! command -v parallel &> /dev/null; then
    echo "GNU Parallel is not installed. Please install it to run this script."
    exit 1
fi

# Function to run the Python script
run_python_script() {
    arg1=$1
    arg2=$((arg1 + 1))
    echo "Running python generate_diffimages_and_compute_centroids.py $arg1 $arg2"
    nice python generate_diffimages_and_compute_centroids.py $arg1 $arg2
    sleep 1   
}

# Export the function so GNU Parallel can use itq
export -f run_python_script

# Use GNU Parallel to run the jobs
seq $FIRST_INDEX $LAST_INDEX | parallel -j $NPROC run_python_script
