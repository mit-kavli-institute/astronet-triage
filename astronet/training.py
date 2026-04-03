"""Functions used for training an AstroNet model."""

import tensorflow as tf
from absl import logging
import pandas as pd

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
    loss = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=config.hparams.get("label_smoothing", 0.0))
  else:
    loss = tf.keras.losses.BinaryCrossentropy()
  logging.info(f"Using '{loss.name}' loss")

  model.compile(optimizer=optimizer, loss=loss)


def make_weight_table(spec_df: pd.DataFrame,
                      astro_col: str = "astro_id",
                      weight_col: str = "sample_weight",
                      default_weight: float = 1.0):
    """
    Build a StaticHashTable for per-example sample weights.
    Missing astro_ids fall back to `default_weight`.
    
    spec_df: pd.DataFrame that may be sparse (only override rows).
    """

    # Only keep rows where weight_col is present & non-null
    df = spec_df[[astro_col, weight_col]].dropna()

    if len(df) == 0:
        # Return a table that always returns default
        return tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant([], tf.int64),
                values=tf.constant([], tf.float32),
            ),
            default_value=tf.constant(default_weight, tf.float32),
        )

    # Cast to TensorFlow-friendly dtypes
    keys = tf.constant(df[astro_col].astype("int64").values)
    vals = tf.constant(df[weight_col].astype("float32").values)

    table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(keys, vals),
        default_value=tf.constant(default_weight, tf.float32),
    )
    return table

def make_upsample_table(spec_df: pd.DataFrame,
                        astro_col: str = "astro_id",
                        up_col: str = "upsample_factor",
                        default_factor: int = 1):
    """
    Build a StaticHashTable for per-example upsampling factors.
    Missing astro_ids fall back to `default_factor`.
    """

    df = spec_df[[astro_col, up_col]].dropna()

    if len(df) == 0:
        # Always return default_factor
        return tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant([], tf.int64),
                values=tf.constant([], tf.int32),
            ),
            default_value=tf.constant(default_factor, tf.int32),
        )

    keys = tf.constant(df[astro_col].astype("int64").values)
    vals = tf.constant(df[up_col].astype("int32").values)

    table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(keys, vals),
        default_value=tf.constant(default_factor, tf.int32),
    )
    return table

def train(model, config, train_files, shuffle_buffer_size=2500):
  """Trains a model."""

  spec = pd.read_parquet("/pdo/users/dimond/train_spec.parquet")

  train_df = spec[spec.split=="train"]
  val_df   = spec[spec.split=="val"]
  test_df  = spec[spec.split=="test"]
  weight_table = make_weight_table(spec)
  upsample_table = make_upsample_table(spec)
  print("Weight table contents:")
  print(spec[["astro_id", "sample_weight"]])

  print("\nUpsample table contents:")
  print(spec[["astro_id", "upsample_factor"]])
  #1/0

  ds = input_ds.build_train_dataset(
      file_pattern=train_files,
      input_config=config.inputs,
      batch_size=config.hparams.batch_size,
      shuffle_values_buffer=shuffle_buffer_size,
      #weight_table=weight_table,
      #upsample_table=upsample_table,
  )

  compile_model(model, config)
  history = model.fit(ds, steps_per_epoch=config["train_steps"])

  return history
