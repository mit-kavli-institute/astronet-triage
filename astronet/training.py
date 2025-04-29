"""Functions used for training an AstroNet model."""

import tensorflow as tf
from collections import Counter
from absl import logging

from astronet.astro_cnn_model import input_ds

class ThresholdPrecision(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.3, name='precision_thresh', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.precision = tf.keras.metrics.Precision()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_thresh = tf.cast(y_pred > self.threshold, tf.float32)
        self.precision.update_state(y_true, y_pred_thresh, sample_weight)

    def result(self):
        return self.precision.result()

    def reset_states(self):
        self.precision.reset_states()

class ThresholdRecall(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.3, name='recall_thresh', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_thresh = tf.cast(y_pred > self.threshold, tf.float32)
        self.recall.update_state(y_true, y_pred_thresh, sample_weight)

    def result(self):
        return self.recall.result()

    def reset_states(self):
        self.recall.reset_states()

def compute_class_weights(dataset, num_classes, sample_size=10000, primary_class=0, primary_boost=1.0):
    """
    Weight classes by inverse of frequency and apply an optional boost to primary class
    """

    class_counts = Counter()

    for i, batch in enumerate(dataset.take(sample_size)):
        labels = tf.argmax(batch[1], axis=-1).numpy()
        class_counts.update(labels)

    total = sum(class_counts.values())

    class_weights = {
        i: total / (num_classes * class_counts.get(i, 1))
        for i in range(num_classes)
    }

    avg_weight = sum(class_weights.values()) / num_classes
    class_weights = {k: v / avg_weight for k, v in class_weights.items()}

    class_weights[primary_class] *= primary_boost

    return class_weights


def focal_loss(gamma=2., alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal_weight = alpha_factor * tf.pow((1 - p_t), gamma)
        return tf.reduce_mean(-tf.math.log(p_t) * focal_weight)
    return loss_fn


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
    if config.use_focal_loss:
       loss = focal_loss(gamma=config.focal_loss_gamma, alpha=config.focal_loss_alpha)
    else:
        loss = tf.keras.losses.BinaryCrossentropy()
  logging.info(f"Using '{loss.name}' loss")

  model.compile(optimizer=optimizer, loss=loss, metrics=[
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
        ThresholdPrecision(threshold=0.3),
        ThresholdRecall(threshold=0.3)
  ])

def train(model, config, train_files, shuffle_buffer_size=2500):
  """Trains a model."""
  ds = input_ds.build_train_dataset(
      file_pattern=train_files,
      input_config=config.inputs,
      batch_size=config.hparams.batch_size,
      shuffle_values_buffer=shuffle_buffer_size)

  compile_model(model, config)
  class_weights = None
  class_weights = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
  if config.hparams.use_class_weights:
    class_weights = compute_class_weights(ds, num_classes=4)
    print("Class weights:", class_weights)

  history = model.fit(ds, steps_per_epoch=config["train_steps"], class_weight=class_weights)

  return history
