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
"""Script for training an AstroNet model."""

import datetime
import os

import numpy as np
import tensorflow as tf
from absl import app, flags, logging

from astronet import evaluation, models, training

flags.DEFINE_string("model", None, "Name of the model class.", required=True)

flags.DEFINE_string(
    "config_name",
    None,
    "Name of the model and training configuration.",
    required=True,
)

flags.DEFINE_string(
    "train_files",
    None,
    "Comma-separated list of file patterns matching the TFRecord files in "
    "the training dataset.",
    required=True,
)

flags.DEFINE_multi_string(
    "eval_files", None,
    "File patterns matching the TFRecord files in the evaluation dataset(s). "
    "Multiple evaluation datasets are allowed. The training set does not need "
    "to be specified. Each evaluation dataset can be named with the format "
    "name:file_patterns.")

flags.DEFINE_string(
    "model_dir",
    None,
    "Directory for model checkpoints and summaries.",
    required=True)

flags.DEFINE_string("pretrain_model_dir", None,
                    "Directory for pretrained model checkpoints.")

flags.DEFINE_integer("train_steps", None,
                     "Total number of steps to train the model for.")

flags.DEFINE_integer("shuffle_buffer_size", 25000,
                     "Size of the shuffle buffer for the training dataset.")

FLAGS = flags.FLAGS


def main(_):
  config = models.get_model_config(FLAGS.model, FLAGS.config_name)
  model_class = models.get_model_class(FLAGS.model)
  model_dir = (f"{FLAGS.model_dir}/{FLAGS.model}_{FLAGS.config_name}_"
               f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")

  if FLAGS.pretrain_model_dir:
    pretrain_model = tf.keras.models.load_model(
        os.path.join(FLAGS.pretrain_model_dir,
                     os.listdir(FLAGS.pretrain_model_dir + "/")[0]))
    model = model_class(config, pretrain_model)
  else:
    model = model_class(config)

  # Set the number of training steps.
  config["train_steps"] = FLAGS.train_steps or config["train_steps"]
  if not config["train_steps"]:
    raise ValueError(
        "train_steps must be set in the config or via --train_steps")

  training.train(
      model,
      config,
      train_files=FLAGS.train_files,
      model_dir=model_dir,
      shuffle_buffer_size=FLAGS.shuffle_buffer_size)

  # Construct evaluation datasets.
  # This includes the training set and possibly additional datasets.
  eval_datasets = [("train", FLAGS.train_files)]
  for file_pattern in FLAGS.eval_files:
    if ":" in file_pattern:
      name, file_pattern = file_pattern.split(":")
    elif len(FLAGS.eval_files) == 1:
      # If there is only a single evaluation dataset, default name is 'eval'.
      name = "eval"
    else:
      raise ValueError("Multiple evaluation datasets must be named with format "
                       "name:file_patterns")
    eval_datasets.append((name, file_pattern))

  # Generate predictions on the evaluation datasets and save the output files.
  eval_dir = os.path.join(model_dir, "evaluation")
  if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)
  all_metrics = {}
  for name, file_pattern in eval_datasets:
    metrics, labels, predictions = evaluation.evaluate_model(
        model, config.inputs, file_pattern, config.hparams.batch_size)
    all_metrics[name] = metrics
    np.save(os.path.join(eval_dir, f"{name}_label.npy"), labels)
    np.save(os.path.join(eval_dir, f"{name}_pred.npy"), predictions)
  evaluation.save_metrics(all_metrics, eval_dir)


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  app.run(main)
