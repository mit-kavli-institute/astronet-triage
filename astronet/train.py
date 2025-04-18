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
from astronet.util import config_util

flags.DEFINE_string("model", None, "Name of the model class.", required=True)

flags.DEFINE_string("config_name", None,
                    "Name of the model and training configuration.")

flags.DEFINE_string("config_file", None,
                    "File containing the model and training configuration.")

flags.DEFINE_string("config_overrides", None,
                    "Overrides to the base configuration.")

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
  # Keep track of training flags for record-keeping purposes.
  train_flags = {
      "model": FLAGS.model,
      "train_files": FLAGS.train_files,
      "eval_files": FLAGS.eval_files,
      "shuffle_buffer_size": FLAGS.shuffle_buffer_size,
  }

  # Load the config.
  if bool(FLAGS.config_name) == bool(FLAGS.config_file):
    raise ValueError("Exactly one of config_name and config_file is required")
  if FLAGS.config_name:
    config = models.get_model_config(FLAGS.model, FLAGS.config_name)
    train_flags["config_name"] = FLAGS.config_name
    expt_name = f"{FLAGS.model}_{FLAGS.config_name}"
  else:
    config = config_util.load_config(FLAGS.config_file)
    train_flags["config_file"] = FLAGS.config_file
    logging.info(f"Loaded config from {FLAGS.config_file}")
    expt_name = FLAGS.model
  if FLAGS.config_overrides:
    overrides = config_util.parse_config_str(FLAGS.config_overrides)
    train_flags["config_overrides"] = overrides
    config_util.update(config, overrides)
    logging.info(f"Updated config with overrides {overrides}")

  # Set the number of training steps.
  if FLAGS.train_steps:
    config["train_steps"] = FLAGS.train_steps
    logging.info(f"Set config.train_steps to {FLAGS.train_steps}")
  if not config["train_steps"]:
    raise ValueError(
        "train_steps must be set in the config or via --train_steps")

  # Build the model.
  model_class = models.get_model_class(FLAGS.model)
  model = model_class(config)
  init_from_pretrained_model = config.get("init_from_pretrained_model")
  if bool(init_from_pretrained_model) != bool(FLAGS.pretrain_model_dir):
    raise ValueError(
        f"{init_from_pretrained_model=} but {FLAGS.pretrain_model_dir=}")
  if init_from_pretrained_model:
    pretrain_model = tf.keras.models.load_model(FLAGS.pretrain_model_dir)
    train_flags["pretrain_model_dir"] = FLAGS.pretrain_model_dir
    for name, block in model.ts_blocks.items():
      pretrain_block = pretrain_model.ts_blocks.get(name)
      if pretrain_block is not None:
        block.set_weights(pretrain_block.get_weights())
        logging.info("Block '{name}': set params from pretrained model")
        if config.freeze_pretrained_params:
          block.trainable = False
      else:
        logging.info("Block '{name}': no such block in pretrained model")

  # Make model directory and save the configs.
  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  model_dir = os.path.join(FLAGS.model_dir, f"{expt_name}_{timestamp}")
  os.makedirs(model_dir)
  if FLAGS.pretrain_model_dir:
    train_flags["pretrain_model_dir"] = FLAGS.pretrain_model_dir
  config_util.save_config(train_flags, model_dir, basename="train_flags")
  config_util.save_config(config, model_dir)

  # Train and save model.
  training.train(
      model,
      config,
      train_files=FLAGS.train_files,
      shuffle_buffer_size=FLAGS.shuffle_buffer_size)

  # Save the model in Keras format.
  model.save(model_dir)
  # Uncomment to save the weights in h5 format.
  # model.save_weights(os.path.join(model_dir, "model.weights.h5"))

  # Construct evaluation datasets.
  # This includes the training set and possibly additional datasets.
  eval_datasets = [("train", FLAGS.train_files)]
  for file_pattern in FLAGS.eval_files:
    if ":" in file_pattern:
      name, file_pattern = file_pattern.split(":")
    elif len(FLAGS.eval_files) == 1:
      # If there is only a single evaluation dataset, default name is "eval".
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
