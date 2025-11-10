#!/usr/bin/env python3
"""
Update TFRecord files with averaged predictions from ensemble models.

This script:
1. Loads averaged predictions from CSV
2. Reads all TFRecord shard files from input directory
3. Adds soft label columns (disp_p_soft, disp_e_soft, disp_n_soft, disp_j_soft)
   with averaged predictions, keeping original labels intact
4. Writes updated TFRecords to output directory
"""

import os
import sys
import argparse
import glob
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm


def load_predictions_mapping(csv_path):
    """
    Load averaged predictions and create a mapping from astro_id to label values.

    Args:
        csv_path: Path to ensemble_predictions_averaged.csv

    Returns:
        Dictionary mapping astro_id -> {disp_p, disp_e, disp_n, disp_j}
    """
    print(f"Loading predictions from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Create mapping: astro_id -> {disp_p, disp_e, disp_n, disp_j}
    # Use avg_disp_* columns as the new label values
    predictions_map = {}
    for _, row in df.iterrows():
        astro_id = int(row['astro_id'])
        predictions_map[astro_id] = {
            'disp_p': float(row['avg_disp_p']),
            'disp_e': float(row['avg_disp_e']),
            'disp_n': float(row['avg_disp_n']),
            'disp_j': float(row['avg_disp_j']),
        }

    print(f"Loaded predictions for {len(predictions_map)} astro_ids")
    return predictions_map


def update_tfrecord_file(input_file, output_file, predictions_map, label_cols):
    """
    Update a single TFRecord file by adding soft label columns.

    Args:
        input_file: Path to input TFRecord file
        output_file: Path to output TFRecord file
        predictions_map: Dictionary mapping astro_id to soft label values
        label_cols: List of label column names ['disp_p', 'disp_e', 'disp_n', 'disp_j']
                   (soft labels will be added as disp_p_soft, etc.)

    Returns:
        Tuple of (total_records, updated_records, missing_records)
    """
    total_records = 0
    updated_records = 0
    missing_records = 0

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with tf.io.TFRecordWriter(output_file) as writer:
        dataset = tf.data.TFRecordDataset([input_file])

        for raw_record in dataset:
            total_records += 1

            # Parse the Example proto
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())

            # Extract astro_id
            astro_id = None
            if 'astro_id' in example.features.feature:
                feature = example.features.feature['astro_id']
                if feature.int64_list.value:
                    astro_id = int(feature.int64_list.value[0])

            if astro_id is None:
                print(f"Warning: Could not extract astro_id from record {total_records} in {input_file}")
                # Write original record without modification
                writer.write(raw_record.numpy())
                continue

            # Check if we have predictions for this astro_id
            if astro_id in predictions_map:
                # Add soft label values (keep original labels intact)
                new_labels = predictions_map[astro_id]
                for label_col in label_cols:
                    soft_label_col = f"{label_col}_soft"
                    # Set the soft label value
                    example.features.feature[soft_label_col].float_list.value[:] = [new_labels[label_col]]

                updated_records += 1
            else:
                missing_records += 1
                # Write original record without modification
                # (or you could skip it, depending on your needs)

            # Write the (possibly modified) example
            writer.write(example.SerializeToString())

    return total_records, updated_records, missing_records


def update_tfrecords(input_dir, predictions_csv, output_dir, label_cols=None):
    """
    Update all TFRecord files in input_dir by adding soft label columns from CSV.

    Args:
        input_dir: Directory containing input TFRecord shard files
        predictions_csv: Path to ensemble_predictions_averaged.csv
        output_dir: Directory to write updated TFRecord files
        label_cols: List of label column names (default: ['disp_p', 'disp_e', 'disp_n', 'disp_j'])
                   Original labels are preserved, soft labels added as *_soft columns
    """
    if label_cols is None:
        label_cols = ['disp_p', 'disp_e', 'disp_n', 'disp_j']

    # Load predictions mapping
    predictions_map = load_predictions_mapping(predictions_csv)

    # Find all TFRecord files in input directory
    # TFRecord files typically have no extension
    input_pattern = os.path.join(input_dir, '*')
    input_files = [f for f in glob.glob(input_pattern) if os.path.isfile(f)]

    if not input_files:
        raise ValueError(f"No files found in {input_dir}")

    # Sort files for consistent processing
    input_files = sorted(input_files)
    print(f"\nFound {len(input_files)} TFRecord files to process")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each file
    total_all = 0
    updated_all = 0
    missing_all = 0

    for input_file in tqdm(input_files, desc="Processing TFRecord files"):
        # Get the filename (e.g., "00000-of-00050")
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename)

        total, updated, missing = update_tfrecord_file(
            input_file, output_file, predictions_map, label_cols
        )

        total_all += total
        updated_all += updated
        missing_all += missing

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Total records processed: {total_all}")
    print(f"Records updated with predictions: {updated_all}")
    print(f"Records without predictions (kept original): {missing_all}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Update TFRecord files with averaged predictions'
    )
    parser.add_argument(
        '--input_tfrecord_dir',
        type=str,
        required=True,
        help='Directory containing input TFRecord shard files'
    )
    parser.add_argument(
        '--predictions_csv',
        type=str,
        required=True,
        help='Path to ensemble_predictions_averaged.csv file'
    )
    parser.add_argument(
        '--output_tfrecord_dir',
        type=str,
        default=None,
        help='Directory to save updated TFRecord files (default: input_dir + _softlabels)'
    )

    args = parser.parse_args()

    # Derive output directory if not provided
    if args.output_tfrecord_dir is None:
        output_tfrecord_dir = args.input_tfrecord_dir.rstrip('/') + '_softlabels'
    else:
        output_tfrecord_dir = args.output_tfrecord_dir

    print("=" * 70)
    print("TFRecord Updater - Add soft labels (preserving original labels)")
    print("=" * 70)
    print(f"Input TFRecord directory: {args.input_tfrecord_dir}")
    print(f"Predictions CSV: {args.predictions_csv}")
    print(f"Output TFRecord directory: {output_tfrecord_dir}")
    print("=" * 70)

    if not os.path.exists(args.predictions_csv):
        raise FileNotFoundError(f"Predictions CSV not found: {args.predictions_csv}")

    if not os.path.exists(args.input_tfrecord_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_tfrecord_dir}")

    # Update TFRecords
    update_tfrecords(
        input_dir=args.input_tfrecord_dir,
        predictions_csv=args.predictions_csv,
        output_dir=output_tfrecord_dir,
        label_cols=['disp_p', 'disp_e', 'disp_n', 'disp_j']
    )

    print("\n✅ Done! Updated TFRecord files saved to:", output_tfrecord_dir)


if __name__ == "__main__":
    main()
