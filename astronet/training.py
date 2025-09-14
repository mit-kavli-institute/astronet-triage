"""Functions used for training an AstroNet model."""

import tensorflow as tf
from absl import logging

from astronet.astro_cnn_model import input_ds


def compile_model(model, config):
  """Compiles a model for training."""
  # Set up the learning rate schedule.
  if config.hparams.learning_rate_schedule == "constant":
    if config.hparams.learning_rate_warmup_frac:
      raise ValueError(
          "Learning rate warmup is not supported with constant schedule")
    learning_rate = config.hparams.learning_rate
  elif config.hparams.learning_rate_schedule == "cosine":
    train_steps = config.train_steps
    warmup_frac = config.hparams.learning_rate_warmup_frac
    warmup_steps = int(warmup_frac * train_steps)
    peak_learning_rate = float(config.hparams.learning_rate)
    initial_learning_rate = peak_learning_rate / 1000
    decay_hparams = dict(
        initial_learning_rate=initial_learning_rate,
        warmup_target=peak_learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=train_steps - warmup_steps,
        alpha=float(config.hparams.learning_rate_decay_alpha),
    )
    logging.info(
        f"Using cosine learning rate decay with parameters {decay_hparams}")
    learning_rate = tf.keras.optimizers.schedules.CosineDecay(**decay_hparams)
  else:
    raise ValueError(config.hparams.learning_rate_schedule)

  # Set up the optimizer.
  opt_hparams = dict(
      learning_rate=learning_rate,
      weight_decay=config.hparams.get("weight_decay"))
  if config.hparams.optimizer == "sgd":
    opt_hparams.update(momentum=1.0 - config.hparams.one_minus_momentum,)
    optimizer = tf.keras.optimizers.SGD(**opt_hparams)
  elif config.hparams.optimizer == "adam":
    opt_hparams.update(
        beta_1=1.0 - config.hparams.one_minus_adam_beta_1,
        beta_2=1.0 - config.hparams.one_minus_adam_beta_2,
        epsilon=config.hparams.adam_epsilon,
    )
    optimizer = tf.keras.optimizers.Adam(**opt_hparams)
  else:
    raise ValueError(config.hparams.optimizer)
  logging.info(
      f"Using '{optimizer.name}' optimizer with parameters {opt_hparams}")

  n_labels = len(config.inputs.label_columns)
  if n_labels > 1 and config.inputs.get("exclusive_labels", False):
    loss = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=config.hparams.get("label_smoothing", 0.0))
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
