#!/usr/bin/env python3
"""
Extract backbone features from a trained model.

This script runs only the convolutional/time-series blocks (the "backbone") on a
dataset and stores the resulting feature vectors to disk, together with IDs.

The output includes:
- Feature vectors (concatenated from all time-series blocks + aux inputs)
- astro_id for each example
- A dictionary mapping block names to feature slice indices

Usage:
  python extract_backbone_features.py \
    --model_dir /path/to/model \
    --data_files /path/to/data/*.tfrecord \
    --output_dir /path/to/output

Output files:
  - features.npz: Contains 'features' (N x D array) and 'astro_ids' (N array)
  - feature_slices.json: Dictionary mapping block names to (start_idx, end_idx) tuples
"""

import json
import os

import numpy as np
import tensorflow as tf
from absl import app, flags, logging

from astronet import models
from astronet.astro_cnn_model import input_ds
from astronet.util import config_util

flags.DEFINE_string(
    "model_dir",
    None,
    "Directory containing a model checkpoint.",
    required=True)

flags.DEFINE_string(
    "data_files",
    None,
    "Path or glob pattern to TFRecord files.",
    required=True)

flags.DEFINE_string(
    "output_dir",
    None,
    "Directory to save output files.",
    required=True)

flags.DEFINE_integer(
    "batch_size",
    None,
    "Batch size for processing. If None, uses model config batch_size.")

FLAGS = flags.FLAGS


def compute_feature_slices(model, config):
  """Computes the slice indices for each block in the concatenated feature vector.

  Args:
    model: The AstroCNNModel instance.
    config: Model configuration.

  Returns:
    Tuple of (slices_dict, block_info_dict) where:
      - slices_dict: Dictionary mapping block names to (start_idx, end_idx) tuples
      - block_info_dict: Dictionary mapping block names to feature sizes
  """
  # Create dummy inputs to determine feature sizes
  dummy_inputs = {}
  for feature_name, feature_spec in config.inputs.features.items():
    shape = [1] + list(feature_spec.shape)
    dummy_inputs[feature_name] = tf.zeros(shape, dtype=tf.float32)

  # Run backbone to get feature sizes
  ts_outputs = model.apply_ts_blocks(dummy_inputs, training=False)
  aux_outputs = [dummy_inputs[key] for key in config.hparams.aux_inputs]

  # Compute slice indices and collect block info
  slices = {}
  block_info = {}
  current_idx = 0

  # Time-series blocks (in sorted order, matching backbone concatenation)
  for name in sorted(model.ts_blocks.keys()):
    block_output = ts_outputs[sorted(model.ts_blocks.keys()).index(name)]
    feature_size = int(block_output.shape[-1])
    slices[name] = (current_idx, current_idx + feature_size)
    block_info[name] = feature_size
    current_idx += feature_size

  # Auxiliary inputs
  aux_start = current_idx
  aux_total_size = 0
  for aux_key in config.hparams.aux_inputs:
    aux_value = dummy_inputs[aux_key]
    # Handle scalar features (shape [1, 1] or [1]) vs vector features
    # After batch dimension, aux inputs are typically scalars
    if len(aux_value.shape) == 1:
      # Shape [1] - scalar
      aux_size = 1
    elif len(aux_value.shape) == 2 and aux_value.shape[1] == 1:
      # Shape [1, 1] - scalar with batch dimension
      aux_size = 1
    else:
      # Vector feature
      aux_size = int(np.prod(aux_value.shape[1:]))

    # Store individual aux input slice
    aux_name = f"aux_{aux_key}"
    slices[aux_name] = (current_idx, current_idx + aux_size)
    block_info[aux_name] = aux_size
    aux_total_size += aux_size
    current_idx += aux_size

  # Also store combined aux_inputs slice for convenience
  if config.hparams.aux_inputs:
    slices["aux_inputs"] = (aux_start, current_idx)
    block_info["aux_inputs"] = aux_total_size

  return slices, block_info


def extract_backbone_features(model_dir, data_files, output_dir, batch_size=None):
  """Extracts backbone features from a model.

  Args:
    model_dir: Path to model directory.
    data_files: Path or glob pattern to TFRecord files.
    output_dir: Directory to save output files.
    batch_size: Batch size for processing. If None, uses model config.

  Returns:
    Tuple of (features array, astro_ids array, feature_slices dict).
  """
  # Load model
  logging.info(f"Loading model from {model_dir}")
  train_flags = config_util.load_config(
      os.path.join(model_dir, "train_flags.json"))
  config = config_util.load_config(os.path.join(model_dir, "config.json"))
  model_name = train_flags["model"]
  model = models.load_model(model_name, model_dir)

  # Set batch size
  if batch_size is None:
    batch_size = config.hparams.batch_size

  # Compute feature slices
  logging.info("Computing feature slice indices...")
  feature_slices, block_info = compute_feature_slices(model, config)

  # Print comprehensive block information
  logging.info("")
  logging.info("=" * 70)
  logging.info("Backbone Block Information")
  logging.info("=" * 70)
  for block_name in sorted(block_info.keys()):
    if block_name == "aux_inputs":
      continue  # Skip combined aux_inputs, we'll show it separately
    feature_size = block_info[block_name]
    start_idx, end_idx = feature_slices[block_name]
    logging.info(f"  {block_name:30s} | Features: {feature_size:6d} | Slice: [{start_idx:6d}, {end_idx:6d})")

  # Show combined aux_inputs if present
  if "aux_inputs" in block_info:
    feature_size = block_info["aux_inputs"]
    start_idx, end_idx = feature_slices["aux_inputs"]
    logging.info(f"  {'aux_inputs (combined)':30s} | Features: {feature_size:6d} | Slice: [{start_idx:6d}, {end_idx:6d})")
    total_features = end_idx  # Total is the end of aux_inputs
  else:
    # If no aux_inputs, find the maximum end_idx
    total_features = max(end_idx for _, (_, end_idx) in feature_slices.items())

  logging.info("=" * 70)
  logging.info(f"  {'TOTAL':30s} | Features: {total_features:6d}")
  logging.info("=" * 70)
  logging.info("")

  # Build dataset with identifiers
  logging.info(f"Building dataset from {data_files}")
  dataset = input_ds.build_eval_dataset(
      file_pattern=data_files,
      input_config=config.inputs,
      batch_size=batch_size,
      include_identifiers=True,
      include_labels=False)

  # Extract features and IDs
  logging.info("Extracting backbone features...")
  all_features = []
  all_astro_ids = []

  for batch_features, batch_ids in dataset:
    # Run backbone only
    batch_backbone_features = model.backbone(batch_features, training=False)
    all_features.append(batch_backbone_features.numpy())
    all_astro_ids.append(batch_ids.numpy())

  # Concatenate all batches
  features = np.concatenate(all_features, axis=0)
  astro_ids = np.concatenate(all_astro_ids, axis=0)

  logging.info(f"Extracted features for {len(astro_ids)} examples")
  logging.info(f"Feature vector shape: {features.shape}")
  logging.info(f"Number of features per example: {features.shape[1]}")

  # Create output directory
  os.makedirs(output_dir, exist_ok=True)

  # Save features and IDs
  features_path = os.path.join(output_dir, "features.npz")
  logging.info(f"Saving features to {features_path}")
  np.savez(features_path, features=features, astro_ids=astro_ids)

  # Save feature slices
  slices_path = os.path.join(output_dir, "feature_slices.json")
  logging.info(f"Saving feature slices to {slices_path}")
  # Convert numpy types to native Python types for JSON serialization
  slices_json = {
      name: [int(start), int(end)]
      for name, (start, end) in feature_slices.items()
  }
  with open(slices_path, "w") as f:
    json.dump(slices_json, f, indent=2)

  # Also save config for reference
  logging.info(f"Saving model config to {output_dir}")
  config_util.save_config(config, output_dir, basename="config")

  logging.info("✅ Done!")
  return features, astro_ids, feature_slices


def main(_):
  extract_backbone_features(
      model_dir=FLAGS.model_dir,
      data_files=FLAGS.data_files,
      output_dir=FLAGS.output_dir,
      batch_size=FLAGS.batch_size)


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  app.run(main)
