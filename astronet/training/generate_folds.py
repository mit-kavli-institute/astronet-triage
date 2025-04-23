import os
import glob
import random
import shutil
import argparse

# Need to debug / check

def create_folds(input_dir, output_dir, num_folds=5, seed=42):
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(input_dir, "*.tfrecord"))
    random.seed(seed)
    random.shuffle(files)
    fold_size = len(files) // num_folds

    for i in range(num_folds):
        fold_dir = os.path.join(output_dir, f"fold_{i}")
        train_dir = os.path.join(fold_dir, "train")
        val_dir = os.path.join(fold_dir, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        val_files = files[i*fold_size : (i+1)*fold_size]
        train_files = [f for f in files if f not in val_files]

        for f in train_files:
            shutil.copy(f, train_dir)
        for f in val_files:
            shutil.copy(f, val_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory with .tfrecord files.")
    parser.add_argument("--output_dir", required=True, help="Directory to store folds.")
    parser.add_argument("--num_folds", type=int, default=5)
    args = parser.parse_args()

    create_folds(args.input_dir, args.output_dir, args.num_folds)