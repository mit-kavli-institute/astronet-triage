#!/usr/bin/env python3
"""
Generate ensemble predictions from all models in a directory.

This script:
1. Finds all model directories in the ensemble directory
2. Generates predictions for each model on the dataset
3. Creates a CSV with all predictions from all models
4. Creates a CSV with averaged predictions across all models
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from astronet import predict
from astronet.astro_cnn_model import input_ds
from astronet.util import config_util


def find_all_model_directories(ensemble_dir):
    """
    Find all model directories in the ensemble directory.

    Handles both flat and nested structures:
    - Flat: ensemble_dir/model1/, ensemble_dir/model2/, ...
    - Nested: ensemble_dir/1/model_name/, ensemble_dir/2/model_name/, ...

    Returns:
        List of paths to model directories
    """
    model_dirs = []
    ensemble_path = Path(ensemble_dir)

    if not ensemble_path.exists():
        raise ValueError(f"Ensemble directory does not exist: {ensemble_dir}")

    # Check if the directory itself is a model directory
    if (ensemble_path / "config.json").exists() and (ensemble_path / "train_flags.json").exists():
        return [str(ensemble_path)]

    # Check for nested structure (e.g., 1/, 2/, 3/, ...)
    subdirs = sorted([d for d in ensemble_path.iterdir() if d.is_dir()])

    # Check if subdirectories are numeric (indicating nested structure)
    numeric_subdirs = []
    for subdir in subdirs:
        try:
            int(subdir.name)
            numeric_subdirs.append(subdir)
        except ValueError:
            pass

    if numeric_subdirs:
        # Nested structure: ensemble_dir/1/model_name/, ensemble_dir/2/model_name/, ...
        print(f"Found nested structure with {len(numeric_subdirs)} numeric subdirectories")
        for subdir in numeric_subdirs:
            # Look for model directories inside each numeric subdirectory
            model_subdirs = [d for d in subdir.iterdir() if d.is_dir()]
            if len(model_subdirs) == 1:
                model_path = model_subdirs[0]
                if (model_path / "config.json").exists() and (model_path / "train_flags.json").exists():
                    model_dirs.append(str(model_path))
                    print(f"  Found model: {model_path}")
                else:
                    print(f"  Warning: {model_path} does not contain model files")
            elif len(model_subdirs) > 1:
                print(f"  Warning: Multiple subdirectories in {subdir}, checking all...")
                for model_subdir in model_subdirs:
                    if (model_subdir / "config.json").exists() and (model_subdir / "train_flags.json").exists():
                        model_dirs.append(str(model_subdir))
                        print(f"    Found model: {model_subdir}")
    else:
        # Flat structure: ensemble_dir/model1/, ensemble_dir/model2/, ...
        print(f"Found flat structure, checking {len(subdirs)} subdirectories")
        for subdir in subdirs:
            if (subdir / "config.json").exists() and (subdir / "train_flags.json").exists():
                model_dirs.append(str(subdir))
                print(f"  Found model: {subdir}")

    if not model_dirs:
        raise ValueError(
            f"No model directories found in {ensemble_dir}. "
            "Model directories must contain 'config.json' and 'train_flags.json' files."
        )

    print(f"\nTotal models found: {len(model_dirs)}")
    return sorted(model_dirs)


def load_true_labels(data_dir, config):
    """
    Load true labels from TFRecord files.

    Args:
        data_dir: Path or glob pattern to TFRecord files
        config: Model configuration

    Returns:
        Dictionary mapping astro_id to label array
    """
    print("\nLoading true labels from dataset...")
    batch_size = config.hparams.batch_size

    # Build dataset with labels
    ds_labels = input_ds.build_eval_dataset(
        file_pattern=data_dir,
        input_config=config.inputs,
        batch_size=batch_size,
        include_identifiers=True,
        include_labels=True
    )

    # Extract labels and IDs
    labels_dict = {}
    for features, labels_batch, weight_batch, id_batch in ds_labels:
        labels_np = labels_batch.numpy()
        ids_np = id_batch.numpy()

        for i, astro_id in enumerate(ids_np):
            labels_dict[astro_id] = labels_np[i]

    print(f"Loaded labels for {len(labels_dict)} astro_ids")
    return labels_dict


def generate_ensemble_predictions(ensemble_dir, data_dir, output_dir=None, output_all_csv=None, output_avg_csv=None):
    """
    Generate predictions from all models in the ensemble directory.

    Args:
        ensemble_dir: Directory containing model directories
        data_dir: Path or glob pattern to TFRecord files
        output_dir: Directory to save output CSV files (if None, uses script directory)
        output_all_csv: Path to save CSV with all predictions (default: output_dir/ensemble_predictions_all.csv)
        output_avg_csv: Path to save CSV with averaged predictions (default: output_dir/ensemble_predictions_averaged.csv)

    Returns:
        Tuple of (all_predictions_df, averaged_predictions_df)
    """
    # Find all model directories
    model_dirs = find_all_model_directories(ensemble_dir)

    # Load config from first model to get label columns (assumes all models have same config)
    first_model_dir = model_dirs[0]
    config = config_util.load_config(os.path.join(first_model_dir, 'config.json'))
    label_columns = config.inputs.label_columns

    # Load true labels
    true_labels_dict = load_true_labels(data_dir, config)

    # Generate predictions for each model
    print(f"\nGenerating predictions for {len(model_dirs)} models...")
    all_predictions = []

    for i, model_dir in enumerate(model_dirs, 1):
        print(f"\n[{i}/{len(model_dirs)}] Processing model: {os.path.basename(model_dir)}")
        try:
            pred_df, _ = predict.predict(model_dir, data_dir)
            pred_df["model_no"] = i - 1  # 0-indexed
            pred_df["model_dir"] = model_dir
            all_predictions.append(pred_df)
            print(f"  Generated {len(pred_df)} predictions")
        except Exception as e:
            print(f"  ERROR: Failed to generate predictions for {model_dir}: {e}")
            continue

    if not all_predictions:
        raise ValueError("No predictions were generated from any model!")

    # Concatenate all predictions
    print(f"\nConcatenating predictions from {len(all_predictions)} models...")
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    print(f"Total predictions: {len(all_predictions_df)}")
    print(f"Unique astro_ids: {all_predictions_df['astro_id'].nunique()}")

    # Get prediction column names (exclude astro_id, model_no, model_dir)
    pred_cols = [col for col in all_predictions_df.columns
                 if col not in ['astro_id', 'model_no', 'model_dir']]
    print(f"Prediction columns: {pred_cols}")

    # Create averaged predictions with true labels
    print("\nComputing averaged predictions...")
    avg_predictions = []

    for astro_id in sorted(all_predictions_df['astro_id'].unique()):
        astro_preds = all_predictions_df[all_predictions_df['astro_id'] == astro_id]

        # Compute mean across all models for this astro_id
        # Use 'avg_' prefix for predictions to distinguish from true labels
        avg_row = {'astro_id': astro_id}
        for col in pred_cols:
            avg_row[f'avg_{col}'] = astro_preds[col].mean()

        # Add true labels using exact names from TFRecords (disp_p, disp_e, disp_n, disp_j)
        if astro_id in true_labels_dict:
            true_label = true_labels_dict[astro_id]
            for i, label_col in enumerate(label_columns):
                avg_row[label_col] = true_label[i]
        else:
            # If label not found, set to NaN
            for label_col in label_columns:
                avg_row[label_col] = np.nan

        avg_predictions.append(avg_row)

    averaged_predictions_df = pd.DataFrame(avg_predictions)
    print(f"Averaged predictions: {len(averaged_predictions_df)}")

    # Count how many have true labels
    has_labels = averaged_predictions_df[label_columns].notna().all(axis=1).sum()
    print(f"  - Predictions with true labels: {has_labels}")
    print(f"  - Predictions without true labels: {len(averaged_predictions_df) - has_labels}")

    # Create version without true labels (only predictions)
    # Select columns: astro_id + all columns starting with 'avg_'
    avg_pred_cols = [col for col in averaged_predictions_df.columns if col.startswith('avg_')]
    averaged_predictions_no_labels_df = averaged_predictions_df[['astro_id'] + avg_pred_cols].copy()

    # Set default output paths if not provided
    if output_dir is None:
        output_dir = os.path.dirname(__file__)
    else:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    if output_all_csv is None:
        output_all_csv = os.path.join(output_dir, "ensemble_predictions_all.csv")
    if output_avg_csv is None:
        output_avg_csv = os.path.join(output_dir, "ensemble_predictions_averaged.csv")

    output_avg_no_labels_csv = os.path.join(output_dir, "ensemble_predictions_averaged_no_labels.csv")

    # Save to CSV files
    print(f"\nSaving all predictions to: {output_all_csv}")
    all_predictions_df.to_csv(output_all_csv, index=False)

    print(f"Saving averaged predictions (with true labels) to: {output_avg_csv}")
    averaged_predictions_df.to_csv(output_avg_csv, index=False)

    print(f"Saving averaged predictions (without true labels) to: {output_avg_no_labels_csv}")
    averaged_predictions_no_labels_df.to_csv(output_avg_no_labels_csv, index=False)

    print("\n✅ Done!")
    print(f"  - All predictions: {len(all_predictions_df)} rows")
    print(f"  - Averaged predictions (with labels): {len(averaged_predictions_df)} rows")
    print(f"  - Averaged predictions (without labels): {len(averaged_predictions_no_labels_df)} rows")
    print(f"  - Number of models: {len(model_dirs)}")

    return all_predictions_df, averaged_predictions_df


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Generate ensemble predictions from multiple models'
    )
    parser.add_argument(
        '--ensemble_dir',
        type=str,
        required=True,
        help='Directory containing model directories'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path or glob pattern to TFRecord files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save output CSV files (default: script directory)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Ensemble Prediction Generator")
    print("=" * 70)
    print(f"Ensemble directory: {args.ensemble_dir}")
    print(f"Data directory: {args.data_dir}")
    if args.output_dir:
        print(f"Output directory: {args.output_dir}")
    else:
        print("Output directory: (using default)")
    print("=" * 70)

    # Set default device to CPU to avoid GPU memory issues
    with tf.device('/CPU:0'):
        all_preds, avg_preds = generate_ensemble_predictions(
            ensemble_dir=args.ensemble_dir,
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )

    return all_preds, avg_preds


if __name__ == "__main__":
    main()
