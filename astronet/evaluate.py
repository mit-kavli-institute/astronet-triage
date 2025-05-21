"""Script for evaluating a trained AstroNet model."""

import os
import numpy as np
import tensorflow as tf
from absl import app, flags, logging
import pandas as pd

from astronet import evaluation, models
from astronet.util import config_util

from astronet.astro_cnn_model import input_ds

flags.DEFINE_string("model", None, "Name of the model class.", required=True)

flags.DEFINE_string(
    "model_dir", None,
    "Directory of the trained model to evaluate (must contain config.yaml).",
    required=True)

flags.DEFINE_string(
    "output_dir", None,
    "Directory of where to store results.",
    required=True)

flags.DEFINE_multi_string(
    "eval_files", None,
    "File patterns matching the TFRecord files in the evaluation dataset(s). "
    "Each dataset can be named with the format name:file_patterns. If a single "
    "pattern is passed, it defaults to the name 'eval'.",
    required=True)

flags.DEFINE_float("threshold", 0.215,
                   "Threshold for binary classification evaluation.")

FLAGS = flags.FLAGS


def main(_):
  # Load config from model_dir
  config = config_util.load_config(FLAGS.model_dir)
  model_class = models.get_model_class(FLAGS.model)
  model = model_class(config)

  # Load the trained model weights
  model = models.load_model(FLAGS.model, FLAGS.model_dir)

  output_dir = FLAGS.output_dir

  # Set up evaluation datasets
  eval_datasets = []
  for file_pattern in FLAGS.eval_files:
    if ":" in file_pattern:
      name, pattern = file_pattern.split(":", 1)
    elif len(FLAGS.eval_files) == 1:
      name, pattern = "eval", file_pattern
    else:
      raise ValueError("Multiple datasets must be named as name:file_pattern")
    eval_datasets.append((name, pattern))

  # Evaluation results directory
  output_dir = os.path.join(output_dir)
  os.makedirs(output_dir, exist_ok=True)

  # Run evaluation
  all_metrics = {}
  for name, file_pattern in eval_datasets:
    dataset = input_ds.build_eval_dataset(
      file_pattern=file_pattern,
      input_config=config.inputs,
      batch_size=config.hparams.batch_size,
      include_identifiers=True,
      include_labels=False)
    astro_ids = []
    tic_ids = []
    planet_nos = []
    model_nos = []

    for batch in dataset:
        inputs, identifiers = batch
        astro_ids.extend([x for x in identifiers.numpy()])
        tic_ids = [int(str(x)[:-2]) for x in astro_ids]
        planet_nos = [int(str(x)[-2:]) for x in astro_ids]
        model_nos = [0 for x in astro_ids]

    # Now create a DataFrame
    y_pred = model.predict(dataset)
    df = pd.DataFrame(y_pred, columns=["disp_p", "disp_e", "disp_n", "disp_j"])
    df.insert(0,"model_no", model_nos)
    df.insert(0,"planetno", planet_nos)
    df.insert(0, "tic_id", tic_ids)
    df.insert(0, "Astro ID", astro_ids)
    csv_path = os.path.join(output_dir, f"{name}_predictions.csv")
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  app.run(main)