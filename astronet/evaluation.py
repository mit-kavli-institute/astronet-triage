"""Functions for evaluating a trained model."""

import numpy as np

from astronet.astro_cnn_model import input_ds


def evaluate_model(model, input_config, file_pattern, batch_size):
  """Calculates a trained model's metrics over a dataset."""
  ds = input_ds.build_dataset(
      file_pattern=file_pattern,
      input_config=input_config,
      batch_size=batch_size)
  results = model.evaluate(ds)
  return dict(zip(model.metrics_names, results))


def generate_labels_and_predictions(model, input_config, file_pattern,
                                    batch_size):
  """Generates labels and predictions from a trained model on a dataset."""
  ds = input_ds.build_dataset(
      file_pattern=file_pattern,
      input_config=input_config,
      batch_size=batch_size)
  # Generate predictions.
  y_pred = model.predict(ds)
  # Get the true labels.
  y_label = []
  for _, labels, _ in ds:
    y_label.append(labels)
  y_label = np.concatenate(y_label).astype(np.int32)
  return y_label, y_pred
