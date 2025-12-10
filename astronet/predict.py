# Copyright 2018 The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generates predictions using a trained model."""

import multiprocessing
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from absl import app, flags
from tqdm import tqdm

from astronet import models
from astronet.astro_cnn_model import input_ds
from astronet.util import config_util
from astronet.util.configdict import ConfigDict

flags.DEFINE_string(
    "model_dir",
    None,
    "Directory containing a model checkpoint.",
    required=True)

flags.DEFINE_string(
    "data_files",
    None,
    "Comma-separated list of file patterns matching the TFRecord files.",
    required=True)

flags.DEFINE_string("output_file", None,
                    "Name of file in which predictions will be saved.")

FLAGS = flags.FLAGS


def predict(model_dir: str, data_files: str) -> tuple[pd.DataFrame, ConfigDict]:
  """
  Run model predictions.

  Args:
    model_dir: Path to directory containing model config and weights.
    data_files: Path or glob pattern to match files containing records to run
      predictions on.

  Returns:
    A tuple `(predictions, config)` where
    - `predictions` is a dataframe containing the model predictions, in which
      the first column is `astro_id`.
    - `config` is a `ConfigDict` containing the model configuration.
  """
  train_flags = config_util.load_config(
      os.path.join(model_dir, "train_flags.json"))
  config = config_util.load_config(os.path.join(model_dir, "config.json"))
  model_name = train_flags["model"]
  model = models.load_model(model_name, model_dir)
  model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=tf.keras.losses.BinaryCrossentropy(),
      metrics=[
          tf.keras.metrics.BinaryAccuracy(name="accuracy"),
          tf.keras.metrics.AUC(name="auc")
      ])

  prediction_dataset = input_ds.build_eval_dataset(
      data_files,
      input_config=config.inputs,
      batch_size=config.hparams.batch_size,
      include_identifiers=True,
      include_labels=False,
  )
  predictions = model.predict(prediction_dataset)
  # batch[1] is 'astro_id'
  ids = np.concatenate([batch[1].numpy() for batch in prediction_dataset])
  prediction_df = pd.DataFrame(predictions, columns=config.inputs.label_columns)
  prediction_df.insert(0, "astro_id", ids)

  return prediction_df, config


def _is_model_directory(dir: str) -> bool:
  """
  Check if a directory contains a model compatible with `predict`.

  This requires a `config.json` file and a `train_flags.json` file.
  """
  return (os.path.isfile(os.path.join(dir, "config.json")) and
          os.path.isfile(os.path.join(dir, "train_flags.json")))


def get_model_directories(models_dir: str) -> list[str]:
  """
  Get a list containing model directories within `models_dir`.

  Supports either passing a model directory directly or a directory containing
  many model directories for an ensemble:

  ```
  /model_dir/
    config.json
    train_flags.json
    model_weights.h5

  /ensemble_dir/
    model_1/
      config.json
      train_flags.json
      model_weights.h5
    model_2/
      config.json
      train_flags.json
      model_weights.h5
    ...
  ```

  Returns:
    A list of paths, each of which is a model directory. Sorted alphabetically.
  """
  if _is_model_directory(models_dir):
    return [models_dir]
  return sorted([
      os.path.join(models_dir, subdirectory)
      for subdirectory in os.listdir(models_dir)
      if _is_model_directory(os.path.join(models_dir, subdirectory))
  ])


def batch_predict(models_dir: str, data_files: str) -> pd.DataFrame:
  """
  Run predictions from a model or ensemble.

  Args:
    models_dir: Directory which either directly contains a model directory
      or contains many model subdirectories for an ensemble.
    data_files: Path or glob pattern of paths to files with records to run
      predictions on.

  Returns:
    A dataframe with model predictions from each model. Has an `astro_id` column
    with identifiers for the records and a `model_no` column identifying which
    model in the ensemble produced each prediction. For ensembles, model numbers
    correspond to an alphabetical listing of the model directories.
  """
  model_dirs = get_model_directories(models_dir)
  if not model_dirs:
    raise ValueError(
        f"No models found in {os.path.abspath(models_dir)}. Model directories "
        "must contain 'config.json' and 'train_flags.json' files.")
  ensemble_preds = [
      predict(model_dir, data_files)[0] for model_dir in model_dirs
  ]

  for i, predictions in enumerate(ensemble_preds):
    predictions["model_no"] = i

  return pd.concat(ensemble_preds, ignore_index=True)


def main(_):
  model = tf.keras.models.load_model(FLAGS.model_dir)
  config = config_util.load_config(FLAGS.model_dir)

  ds = input_ds.build_dataset(
      file_pattern=FLAGS.data_files,
      input_config=config.inputs,
      batch_size=1,
      include_labels=False,
      shuffle_filenames=False,
      repeat=1,
      include_identifiers=True)

  label_columns = config.inputs.label_columns
  label_index = {i: k.lower() for i, k in enumerate(label_columns)}

  series = []
  for features, identifiers in tqdm(ds, unit="records"):
    preds = model(features, training=False)

    row = {}
    row["astro_id"] = identifiers.numpy().item()
    for i, p in enumerate(preds.numpy()[0]):
      row[label_index[i]] = p

    series.append(row)

  results = pd.DataFrame.from_dict(series)

  if FLAGS.output_file:
    with tf.io.gfile.GFile(FLAGS.output_file, "w") as f:
      results.to_csv(f)

  return results, config


if __name__ == "__main__":
  app.run(main)
