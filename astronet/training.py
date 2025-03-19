"""Functions used for training an AstroNet model."""

import tensorflow as tf

from astronet.astro_cnn_model import input_ds
from astronet.util import config_util


def compile_model(model, config):
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


def train(model,
          config,
          train_files,
          eval_files=None,
          model_dir=None,
          shuffle_buffer_size=2500):
  """Trains a model."""
  if model_dir:
    config_util.save_config(config, model_dir)

  ds = input_ds.build_dataset(
      file_pattern=train_files,
      input_config=config.inputs,
      batch_size=config.hparams.batch_size,
      shuffle_values_buffer=shuffle_buffer_size,
      repeat=None)

  eval_ds = None
  if eval_files:
    eval_ds = input_ds.build_dataset(
        file_pattern=eval_files,
        input_config=config.inputs,
        batch_size=config.hparams.batch_size)

  compile_model(model, config)
  history = model.fit(
      ds, steps_per_epoch=config["train_steps"], validation_data=eval_ds)

  if model_dir:
    model.save(model_dir)

  return history
