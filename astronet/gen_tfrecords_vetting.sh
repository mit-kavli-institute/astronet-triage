#!/bin/bash

set -e

<<<<<<< HEAD
LCDIR=/pdo/astronet-data/data/fits/sector-87/
NAME=vetting-sector87-all
=======
LCDIR=/pdo/users/dimond/mnt/tess/sector_86_fits/
NAME=vetting-sector86-all
>>>>>>> 561838703287c4f4aa00751c26bf9f05407a8d4a

#python preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-${NAME}-train.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-${NAME}-train --mode=vetting --num_shards=5

#python preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-${NAME}-val.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-${NAME}-val --mode=vetting --num_shards=5

<<<<<<< HEAD
python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/pdo/astronet-data/data/tfrecords/sector-87/tces-sector87.csv --tess_data_dir=${LCDIR} --output_dir=/pdo/astronet-data/data/tfrecords/sector-87 --mode=vetting --num_shards=25
=======
python astronet/preprocess/generate_input_records.py --input_tce_csv_file=/pdo/users/dimond/mnt/tess/astronet/tces-sector86-all2.csv --tess_data_dir=${LCDIR} --output_dir=/pdo/users/dimond/mnt/tess/astronet/tfrecords-${NAME}-test --mode=vetting
>>>>>>> 561838703287c4f4aa00751c26bf9f05407a8d4a




# comment out toi 
# python astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-vetting-v7-toi-train.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-vetting-8-toi-train --vetting_features=y --num_shards=2

# python astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-vetting-v7-toi-val.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-vetting-8-toi-val --vetting_features=y --num_shards=2

# # python astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-vetting-v7-toi-test.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-vetting-8-toi-test --vetting_features=y --num_shards=2
