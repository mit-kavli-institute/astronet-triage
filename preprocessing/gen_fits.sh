#!/bin/bash

# Run the Python script with the specified arguments
/sw/python-versions/python-3.11.9/bin/python3 /pdo/users/pablomer/FFITools/h5_to_fits_frommichelle.py \
    -i /pdo/users/pablomer/mnt/tess/astronet/junk_data_2025_TIC_list.txt \
    -o /pdo/users/pablomer/mnt/tess/junk_examples_fit_files \
    -s 85
