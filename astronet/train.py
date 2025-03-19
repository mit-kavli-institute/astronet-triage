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

import tensorflow as tf
from absl import app, flags, logging

from astronet import models
from astronet.astro_cnn_model import input_ds
from astronet.util import config_util

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

flags.DEFINE_string(
    "eval_files", None,
    "Comma-separated list of file patterns matching the TFRecord files in "
    "the validation dataset.")

flags.DEFINE_string("model_dir", None,
                    "Directory for model checkpoints and summaries.")

flags.DEFINE_string("pretrain_model_dir", None,
                    "Directory for pretrained model checkpoints.")

flags.DEFINE_integer("train_steps", None,
                     "Total number of steps to train the model for.")

flags.DEFINE_integer("shuffle_buffer_size", 25000,
                     "Size of the shuffle buffer for the training dataset.")

FLAGS = flags.FLAGS


def compile(model, config):
  """Compiles a model for training."""
  if config.hparams.optimizer != "adam":
    raise ValueError(config.hparams.optimizer)

  lr = config.hparams.learning_rate
  beta_1 = 1.0 - config.hparams.one_minus_adam_beta_1
  beta_2 = 1.0 - config.hparams.one_minus_adam_beta_2
  epsilon = config.hparams.adam_epsilon
  optimizer = tf.keras.optimizers.Adam(
      learning_rate=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

  if config.inputs.get("exclusive_labels", False):
    loss = tf.keras.losses.CategoricalCrossentropy()
  else:
    loss = tf.keras.losses.BinaryCrossentropy()

  metrics = [
      tf.keras.metrics.Recall(
          name="r",
          class_id=config.inputs.primary_class,
          thresholds=0.2,
      ),
      tf.keras.metrics.Precision(
          name="p",
          class_id=config.inputs.primary_class,
          thresholds=0.2,
      ),
  ]

  model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


def train(model, config):
  """Trains a model."""
  if FLAGS.model_dir:
    dir_name = (f"{FLAGS.model_dir}/{FLAGS.model}_{FLAGS.config_name}_"
                f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    config_util.log_and_save_config(config, dir_name)

  ds = input_ds.build_dataset(
      file_pattern=FLAGS.train_files,
      input_config=config.inputs,
      batch_size=config.hparams.batch_size,
      shuffle_values_buffer=FLAGS.shuffle_buffer_size,
      repeat=None)

  if FLAGS.eval_files:
    eval_ds = input_ds.build_dataset(
        file_pattern=FLAGS.eval_files,
        input_config=config.inputs,
        batch_size=config.hparams.batch_size)
  else:
    eval_ds = None

  compile(model, config)
  history = model.fit(
      ds, steps_per_epoch=config["train_steps"], validation_data=eval_ds)

  if FLAGS.model_dir:
    model.save(dir_name)

  return history


def main(_):
  config = models.get_model_config(FLAGS.model, FLAGS.config_name)
  model_class = models.get_model_class(FLAGS.model)

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

  train(model, config)


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  app.run(main)
