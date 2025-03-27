"""Functions used for training an AstroNet model."""

import tensorflow as tf
from absl import logging

from astronet.astro_cnn_model import input_ds


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


def train(model, config, train_files, shuffle_buffer_size=2500):
  """Trains a model."""
  ds = input_ds.build_train_dataset(
      file_pattern=train_files,
      input_config=config.inputs,
      batch_size=config.hparams.batch_size,
      shuffle_values_buffer=shuffle_buffer_size)

  compile_model(model, config)
  history = model.fit(ds, steps_per_epoch=config["train_steps"])

  return history
