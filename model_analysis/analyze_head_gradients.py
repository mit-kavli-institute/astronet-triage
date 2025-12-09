#!/usr/bin/env python3
"""
Analyze gradients of the fully connected head with respect to backbone features.

This script loads saved feature vectors from extract_backbone_features.py, passes
them through the fully connected head, and computes gradients of the output with
respect to each input feature, grouped by block.

Usage:
  python analyze_head_gradients.py \
    --model_dir /path/to/model \
    --features_file /path/to/features.npz \
    --feature_slices_file /path/to/feature_slices.json \
    --output_dir /path/to/output \
    --target_class 0

Output files:
  - gradients.npz: Contains 'gradients' (N x D array), 'astro_ids' (N array),
    'predictions' (N x num_classes array), and 'target_class' (scalar)
  - block_importance.csv: Per-example block-level importance metrics
  - block_importance_summary.csv: Summary statistics per block
"""

import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from absl import app, flags, logging

from astronet import models
from astronet.util import config_util

flags.DEFINE_string(
    "model_dir",
    None,
    "Directory containing a model checkpoint.",
    required=True)

flags.DEFINE_string(
    "features_file",
    None,
    "Path to features.npz file from extract_backbone_features.py.",
    required=True)

flags.DEFINE_string(
    "feature_slices_file",
    None,
    "Path to feature_slices.json file from extract_backbone_features.py.",
    required=True)

flags.DEFINE_string(
    "output_dir",
    None,
    "Directory to save output files.",
    required=True)

flags.DEFINE_integer(
    "target_class",
    None,
    "Class index to compute gradients for. If None, uses predicted class for each example.")

flags.DEFINE_bool(
    "use_grad_times_input",
    True,
    "Whether to compute Grad × Input as importance measure (in addition to raw gradients).")

flags.DEFINE_integer(
    "batch_size",
    100,
    "Batch size for processing gradients.")

FLAGS = flags.FLAGS


def build_head_model(model_dir, config):
  """Builds a model containing only the head (dense_block + output_layer).

  Args:
    model_dir: Path to model directory.
    config: Model configuration.

  Returns:
    Keras model with only the head layers.
  """
  from astronet.astro_cnn_model import base

  # Create a temporary full model to get the head layers
  train_flags = config_util.load_config(
      os.path.join(model_dir, "train_flags.json"))
  model_name = train_flags["model"]
  full_model = models.load_model(model_name, model_dir)

  # Extract head components
  head_model = tf.keras.Sequential([
      full_model.dense_block,
      full_model.output_layer
  ])

  return head_model, full_model


def compute_gradients(head_model, features, target_class=None):
  """Computes gradients of head output with respect to input features.

  Args:
    head_model: Keras model containing only the head.
    features: Feature vectors (N x D array).
    target_class: Class index to compute gradients for. If None, uses predicted class.

  Returns:
    Tuple of (gradients, predictions, target_classes).
      - gradients: N x D array of gradients
      - predictions: N x num_classes array of predictions
      - target_classes: N array of target class indices used
  """
  # Compute gradients per example to avoid unconnected gradient issues when the
  # chosen class depends on the prediction (argmax). This yields an N x D
  # matrix of gradients aligned with the provided features.
  gradients = []
  predictions = []
  target_classes = []

  for feat in features:
    feat_tf = tf.convert_to_tensor(feat[None, :], dtype=tf.float32)
    with tf.GradientTape() as tape:
      tape.watch(feat_tf)
      preds = head_model(feat_tf, training=False)[0]

      if target_class is None:
        cls = tf.argmax(preds, axis=0, output_type=tf.int32)
      else:
        cls = tf.cast(target_class, tf.int32)

      # Scalar target for this example
      target_output = preds[cls]

    grad = tape.gradient(target_output, feat_tf)[0]

    gradients.append(grad.numpy())
    predictions.append(preds.numpy())
    target_classes.append(int(cls.numpy()))

  return (np.stack(gradients, axis=0),
          np.stack(predictions, axis=0),
          np.array(target_classes, dtype=np.int32))


def aggregate_by_block(gradients, feature_slices, use_grad_times_input=True, features=None):
  """Aggregates gradients by block.

  Args:
    gradients: N x D array of gradients.
    feature_slices: Dictionary mapping block names to (start_idx, end_idx) tuples.
    use_grad_times_input: Whether to compute Grad × Input.
    features: Feature vectors (N x D array) for computing Grad × Input.

  Returns:
    Dictionary with per-block aggregated metrics.
  """
  n_examples = len(gradients)
  block_metrics = {}

  for block_name, (start_idx, end_idx) in feature_slices.items():
    block_grads = gradients[:, start_idx:end_idx]

    # Compute various aggregation metrics
    metrics = {
        "grad_sum": np.sum(block_grads, axis=1),  # Sum of gradients
        "grad_mean": np.mean(block_grads, axis=1),  # Mean of gradients
        "grad_l2": np.linalg.norm(block_grads, axis=1),  # L2 norm
        "grad_l1": np.sum(np.abs(block_grads), axis=1),  # L1 norm
    }

    if use_grad_times_input and features is not None:
      block_features = features[:, start_idx:end_idx]
      grad_times_input = block_grads * block_features
      metrics["grad_times_input_sum"] = np.sum(grad_times_input, axis=1)
      metrics["grad_times_input_mean"] = np.mean(grad_times_input, axis=1)
      metrics["grad_times_input_l2"] = np.linalg.norm(grad_times_input, axis=1)
      metrics["grad_times_input_l1"] = np.sum(np.abs(grad_times_input), axis=1)

    block_metrics[block_name] = metrics

  return block_metrics


def analyze_gradients(model_dir, features_file, feature_slices_file, output_dir,
                      target_class=None, use_grad_times_input=True, batch_size=100):
  """Analyzes gradients of head with respect to backbone features.

  Args:
    model_dir: Path to model directory.
    features_file: Path to features.npz file.
    feature_slices_file: Path to feature_slices.json file.
    output_dir: Directory to save output files.
    target_class: Class index to compute gradients for. If None, uses predicted class.
    use_grad_times_input: Whether to compute Grad × Input.
    batch_size: Batch size for processing.

  Returns:
    Dictionary with results.
  """
  # Load features and slices
  logging.info(f"Loading features from {features_file}")
  data = np.load(features_file)
  features = data["features"]
  astro_ids = data["astro_ids"]

  logging.info(f"Loaded {len(astro_ids)} examples with feature shape {features.shape}")

  logging.info(f"Loading feature slices from {feature_slices_file}")
  with open(feature_slices_file, "r") as f:
    feature_slices = json.load(f)
  # Convert list pairs back to tuples
  feature_slices = {k: tuple(v) for k, v in feature_slices.items()}
  logging.info(f"Feature slices: {feature_slices}")

  # Load model config
  config = config_util.load_config(os.path.join(model_dir, "config.json"))

  # Build head model
  logging.info("Building head model...")
  head_model, full_model = build_head_model(model_dir, config)

  # Compute gradients in batches
  logging.info("Computing gradients...")
  all_gradients = []
  all_predictions = []
  all_target_classes = []

  n_batches = (len(features) + batch_size - 1) // batch_size
  for i in range(n_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(features))
    batch_features = features[start_idx:end_idx]

    batch_grads, batch_preds, batch_targets = compute_gradients(
        head_model, batch_features, target_class)

    all_gradients.append(batch_grads)
    all_predictions.append(batch_preds)
    all_target_classes.append(batch_targets)

    if (i + 1) % 10 == 0:
      logging.info(f"Processed {i + 1}/{n_batches} batches")

  gradients = np.concatenate(all_gradients, axis=0)
  predictions = np.concatenate(all_predictions, axis=0)
  target_classes = np.concatenate(all_target_classes, axis=0)

  logging.info(f"Computed gradients with shape {gradients.shape}")

  # Aggregate by block
  logging.info("Aggregating gradients by block...")
  block_metrics = aggregate_by_block(
      gradients, feature_slices, use_grad_times_input, features)

  # Create output directory
  os.makedirs(output_dir, exist_ok=True)

  # Save raw gradients
  gradients_path = os.path.join(output_dir, "gradients.npz")
  logging.info(f"Saving gradients to {gradients_path}")
  np.savez(
      gradients_path,
      gradients=gradients,
      astro_ids=astro_ids,
      predictions=predictions,
      target_class=target_class if target_class is not None else -1,
      target_classes=target_classes)

  # Create DataFrame with block-level importance
  block_data = {"astro_id": astro_ids}
  for block_name, metrics in block_metrics.items():
    for metric_name, values in metrics.items():
      block_data[f"{block_name}_{metric_name}"] = values

  block_df = pd.DataFrame(block_data)
  block_csv_path = os.path.join(output_dir, "block_importance.csv")
  logging.info(f"Saving block importance to {block_csv_path}")
  block_df.to_csv(block_csv_path, index=False)

  # Create summary statistics
  summary_data = []
  for block_name, metrics in block_metrics.items():
    for metric_name, values in metrics.items():
      summary_data.append({
          "block": block_name,
          "metric": metric_name,
          "mean": np.mean(values),
          "std": np.std(values),
          "min": np.min(values),
          "max": np.max(values),
          "median": np.median(values),
      })

  summary_df = pd.DataFrame(summary_data)
  summary_csv_path = os.path.join(output_dir, "block_importance_summary.csv")
  logging.info(f"Saving summary statistics to {summary_csv_path}")
  summary_df.to_csv(summary_csv_path, index=False)

  logging.info("✅ Done!")
  return {
      "gradients": gradients,
      "predictions": predictions,
      "target_classes": target_classes,
      "block_metrics": block_metrics,
  }


def main(_):
  analyze_gradients(
      model_dir=FLAGS.model_dir,
      features_file=FLAGS.features_file,
      feature_slices_file=FLAGS.feature_slices_file,
      output_dir=FLAGS.output_dir,
      target_class=FLAGS.target_class,
      use_grad_times_input=FLAGS.use_grad_times_input,
      batch_size=FLAGS.batch_size)


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  app.run(main)
