"""Functions used for training an AstroNet model."""

import tensorflow as tf
from absl import logging

from astronet.astro_cnn_model import input_ds
from astronet.util import config_util


def compile_model(model, config):
  """Compiles a model for training."""
  if config.hparams.optimizer != "adam":
    raise ValueError(config.hparams.optimizer)

  hparams = {
      "learning_rate": config.hparams.learning_rate,
      "beta_1": 1.0 - config.hparams.one_minus_adam_beta_1,
      "beta_2": 1.0 - config.hparams.one_minus_adam_beta_2,
      "epsilon": config.hparams.adam_epsilon,
      "weight_decay": config.hparams.get("weight_decay")
  }
  optimizer = tf.keras.optimizers.Adam(**hparams)
  logging.info(f"Using '{optimizer.name}' optimizer with parameters {hparams}")

  n_labels = len(config.inputs.label_columns)
  if n_labels > 1 and config.inputs.get("exclusive_labels", False):
    loss = tf.keras.losses.CategoricalCrossentropy()
  else:
    loss = tf.keras.losses.BinaryCrossentropy()
  logging.info(f"Using '{loss.name}' loss")

  model.compile(optimizer=optimizer, loss=loss)


def train(model, config, train_files, model_dir=None, shuffle_buffer_size=2500):
  """Trains a model."""
  ds = input_ds.build_dataset(
      file_pattern=train_files,
      input_config=config.inputs,
      batch_size=config.hparams.batch_size,
      shuffle_values_buffer=shuffle_buffer_size,
      repeat=None)

  compile_model(model, config)
  history = model.fit(ds, steps_per_epoch=config["train_steps"])

  if model_dir:
    # Save the training and model configs along with the model.
    config_util.save_config(
        {
            # TODO(cshallue): add model name and base config name here too.
            "train_files": train_files,
            "shuffle_buffer_size": shuffle_buffer_size,
        },
        model_dir,
        basename="train_config")
    config_util.save_config(config, model_dir)
    model.save(model_dir)

  return history
