"""Saves predictions and labels from a trained model on a dataset."""

import os

import numpy as np
import tensorflow as tf
from absl import app, flags, logging

from astronet import evaluation
from astronet.astro_cnn_model import input_ds
from astronet.util import config_util

flags.DEFINE_string(
    "model_dir",
    None,
    "Directory containing a model checkpoint.",
    required=True)

flags.DEFINE_string(
    "eval_files",
    None,
    "File patterns matching the TFRecord files in the evaluation dataset.",
    required=True,
)

flags.DEFINE_string(
    "output_dir", None, "Directory to save the output.", required=True)

flags.DEFINE_string("output_basename", "y",
                    "Base name of output arrays (e.g. dataset name).")

flags.DEFINE_integer("batch_size", 100,
                     "Batch size for generating predictions.")

flags.DEFINE_bool("overwrite", False, "Whether to overwrite existing files.")

FLAGS = flags.FLAGS


def _save_array(arr, name):
  filename = os.path.join(FLAGS.output_dir, f"{name}.npy")
  if os.path.exists(filename) and not FLAGS.overwrite:
    logging.warn(f"Output file already exists, skipping: {filename}")
  else:
    np.save(filename, arr)


def main(_):
  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  # Build model and dataset.
  config = config_util.load_config(FLAGS.model_dir)
  model = tf.keras.models.load_model(FLAGS.model_dir)
  dataset = input_ds.build_dataset(
      file_pattern=FLAGS.eval_files,
      input_config=config.inputs,
      batch_size=FLAGS.batch_size)
  y_label, y_pred = evaluation.generate_labels_and_predictions(model, dataset)
  # Save the arrays.
  _save_array(y_pred, f"{FLAGS.output_basename}_pred")
  _save_array(y_label, f"{FLAGS.output_basename}_label")


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  app.run(main)
