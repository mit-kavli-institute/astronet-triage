"""Functions for evaluating a trained model."""

import copy
import json
import os

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

  if np.ndim(y_label) == 2:
    y_label = y_label[:, primary_class]
    y_pred = y_pred[:, primary_class]
  elif np.ndim(y_label) != 1 or primary_class != 0:
    raise ValueError(
        f"y_label has shape {y_label.shape}, primary_class={primary_class}")
  auc = sklearn.metrics.roc_auc_score(y_label, y_pred)
  ap = sklearn.metrics.average_precision_score(y_label, y_pred)
  return {"roc_auc": auc, "average_precision": ap}


def evaluate_model(model, input_config, file_pattern, batch_size):
  """Evaluates a model over a dataset."""
  dataset = input_ds.build_eval_dataset(
      file_pattern=file_pattern,
      input_config=input_config,
      batch_size=batch_size)
  y_label, y_pred = generate_labels_and_predictions(model, dataset)
  metrics = calc_auc_scores(y_label, y_pred, input_config.primary_class)
  keras_metrics = calc_keras_metrics(model, dataset)
  metrics.update(keras_metrics)
  return metrics, y_label, y_pred


def save_metrics(metrics, eval_dir):
  """Saves the metrics dictionary as a json file."""
  filename = os.path.join(eval_dir, "metrics.json")
  with open(filename, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)


def calc_ensemble_mean(all_results):
  """Calculates the mean of the results of an ensemble."""
  ensembled_results = {}
  for trial_results in all_results:
    for dataset_name, results in trial_results.items():
      if dataset_name not in ensembled_results:
        ensembled_results[dataset_name] = copy.deepcopy(results)
      else:
        cumulative_results = ensembled_results[dataset_name]
        for key, value in results.items():
          if key == "y_label":
            if not np.all(cumulative_results[key] == value):
              raise ValueError(f"Inconsistent labels")
          elif key in ["loss", "y_pred"]:
            cumulative_results[key] += value
          else:
            raise KeyError(key)

  n_models = len(all_results)
  for dataset_name, cumulative_results in ensembled_results.items():
    for key in cumulative_results:
      if key in ["loss", "y_pred"]:
        cumulative_results[key] /= n_models
      elif key != "y_label":
        raise KeyError(key)

  return ensembled_results


def calc_ensemble_metrics(all_results, primary_class):
  """Calculates metrics given the results from an ensemble."""
  all_ensembled_results = calc_ensemble_mean(all_results)
  all_metrics = {}
  for dataset_name, ensembled_results in all_ensembled_results.items():
    metrics = calc_auc_scores(
        y_label=ensembled_results["y_label"],
        y_pred=ensembled_results["y_pred"],
        primary_class=primary_class)
    metrics["loss"] = ensembled_results["loss"]
    all_metrics[dataset_name] = metrics

  return all_metrics
