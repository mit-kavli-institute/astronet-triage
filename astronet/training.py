"""Functions used for training an AstroNet model."""

import numpy as np
import tensorflow as tf
from collections import Counter
from absl import logging
import pandas as pd

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

     def reset_state(self):
         self.precision.reset_state()

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

    def reset_state(self):
        self.recall.reset_state()


class PerStepMetricsCallback(tf.keras.callbacks.Callback):
    """Callback to collect metrics at each training step and validation."""
    def __init__(self, validation_data=None, validation_steps=None, enabled=True):
        super().__init__()
        self.step_metrics = {}
        self.val_step_metrics = {}  # Validation metrics per training step
        self.validation_data = validation_data
        self.validation_steps = validation_steps
        self.enabled = enabled  # Whether to collect metrics

    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of each training batch."""
        if not self.enabled:
            return

        if logs is None:
            logs = {}
        # Store training metrics for this step
        for metric_name, value in logs.items():
            if metric_name not in self.step_metrics:
                self.step_metrics[metric_name] = []
            # Convert tensor/numpy scalar to float if needed
            if hasattr(value, 'numpy'):
                value = float(value.numpy())
            elif isinstance(value, (np.integer, np.floating)):
                value = float(value)
            else:
                value = float(value)
            self.step_metrics[metric_name].append(value)

        # Evaluate on validation set after each training step
        if self.validation_data is not None:
            # Strip sample weights from validation dataset (like evaluation.py does)
            # This avoids the warning about weighted_metrics
            val_ds_no_weights = self.validation_data.map(lambda x, y, w: (x, y))

            # Run validation evaluation
            val_results = self.model.evaluate(
                val_ds_no_weights,
                steps=self.validation_steps,
                verbose=0,  # Don't print during evaluation
                return_dict=True
            )

            # Store validation metrics
            for metric_name, value in val_results.items():
                # Remove 'val_' prefix if present (model.evaluate doesn't add it)
                clean_name = metric_name[4:] if metric_name.startswith('val_') else metric_name
                if clean_name not in self.val_step_metrics:
                    self.val_step_metrics[clean_name] = []
                # Convert to float
                if hasattr(value, 'numpy'):
                    value = float(value.numpy())
                elif isinstance(value, (np.integer, np.floating)):
                    value = float(value)
                else:
                    value = float(value)
                self.val_step_metrics[clean_name].append(value)

def compute_class_weights(dataset, num_classes, sample_size=10000):
    logging.warning("WARNING: This is an old version of compute_class_weights() and behavior might not be as expected. See David's development branch if you want to use class weighting.")
    class_counts = Counter()
    for i, batch in enumerate(dataset.take(sample_size)):
        # batch[1] is the one-hot encoded labels
        labels = tf.argmax(batch[1], axis=-1)  # shape: (batch_size,)
        label_values = labels.numpy()

        class_counts.update(label_values)

    total = sum(class_counts.values())
    class_weights = {
        i: total / (num_classes * class_counts.get(i, 1))  # Avoid division by zero
        for i in range(num_classes)
    }

    return class_weights

def compile_model(model, config):
  """Compiles a model for training."""
  # Set up the learning rate schedule.
  if config.hparams.learning_rate_schedule == "constant":
    if config.hparams.learning_rate_warmup_frac:
      raise ValueError(
          "Learning rate warmup is not supported with constant schedule")
    learning_rate = config.hparams.learning_rate
  elif config.hparams.learning_rate_schedule == "cosine":
    logging.info(f"Using cosine learning rate schedule")
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

  model.compile(
      optimizer=optimizer,
      loss=loss,
      metrics=[
         tf.keras.metrics.Precision(name='precision'),
         tf.keras.metrics.Recall(name='recall'),
         tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
         ThresholdPrecision(threshold=0.3),
         ThresholdRecall(threshold=0.3)
      ],
      weighted_metrics=[]  # Silence warning about sample_weight in evaluate()
   )


def train(model, config, train_files, shuffle_buffer_size=2500, exclude_astro_ids=None,
          validation_data=None, validation_steps=None, log_training_history=False):
  """Trains a model.

  Args:
    model: The model to train
    config: Configuration dictionary
    train_files: Training data file pattern
    shuffle_buffer_size: Size of shuffle buffer
    exclude_astro_ids: Set of Astro IDs to exclude from training
    validation_data: Optional validation dataset
    validation_steps: Optional number of validation steps per epoch

  Returns:
    history: Keras History object (contains per-epoch metrics)
    step_metrics: Dictionary of per-step training metrics (collected via callback)
    val_step_metrics: Dictionary of per-step validation metrics (evaluated after each training step)
  """
  ds = input_ds.build_train_dataset(
      file_pattern=train_files,
      input_config=config.inputs,
      batch_size=config.hparams.batch_size,
      shuffle_values_buffer=shuffle_buffer_size,
      exclude_astro_ids=exclude_astro_ids  # <-- pass the actual set here
  )

  compile_model(model, config)

  # Create callback to collect per-step metrics (only if enabled)
  # Pass validation_data to callback so it can evaluate after each step
  per_step_callback = PerStepMetricsCallback(
      validation_data=validation_data,
      validation_steps=validation_steps,
      enabled=log_training_history
  )

  fit_kwargs = {
      'x': ds,
      'steps_per_epoch': config["train_steps"],
      'callbacks': [per_step_callback]
  }

  # Note: We don't pass validation_data to model.fit() because we're evaluating
  # manually in the callback after each step. This gives us per-step validation metrics.
  # If you want epoch-end validation too, you can add validation_data here as well.

  history = model.fit(**fit_kwargs)

  return history, per_step_callback.step_metrics, per_step_callback.val_step_metrics
