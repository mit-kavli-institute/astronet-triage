"""Functions for evaluating a trained model."""

import numpy as np

from astronet.astro_cnn_model import input_ds


def calc_keras_metrics(model, dataset):
  """Calculates a trained model's metrics over a dataset."""
  results = model.evaluate(dataset)
  if len(model.metrics_names) == 1:
    results = [results]
  return dict(zip(model.metrics_names, results))


def generate_labels_and_predictions(model, dataset):
  """Generates labels and predictions from a trained model on a dataset."""
  # Generate predictions.
  y_pred = model.predict(dataset)
  # Get the true labels.
  y_label = []
  for _, labels, _ in dataset:
    y_label.append(labels)
  y_label = np.concatenate(y_label).astype(np.int32)
  return y_label, y_pred


def calc_auc_scores(y_label, y_pred, primary_class):
  """Calculates AUC scores for predictions and labels."""
  # Lazily import sklearn as it is currently only used within this function.
  # This makes the code backwards-compatible with environments that don't have
  # sklearn installed.
  import sklearn

  p_label = y_label[:, primary_class]
  p_pred = y_pred[:, primary_class]
  auc = sklearn.metrics.roc_auc_score(p_label, p_pred)
  ap = sklearn.metrics.average_precision_score(p_label, p_pred)
  return {"roc_auc": auc, "average_precision": ap}


def evaluate_model(model, input_config, file_pattern, batch_size):
  """Evaluates a model over a dataset."""
  dataset = input_ds.build_dataset(
      file_pattern=file_pattern,
      input_config=input_config,
      batch_size=batch_size)
  y_label, y_pred = generate_labels_and_predictions(model, dataset)
  metrics = calc_auc_scores(y_label, y_pred, input_config.primary_class)
  keras_metrics = calc_keras_metrics(model, dataset)
  metrics.update(keras_metrics)
  return metrics, y_label, y_pred
