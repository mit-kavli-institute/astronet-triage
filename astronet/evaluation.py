import json
import os

import numpy as np
import pandas as pd

from astronet.astro_cnn_model import input_ds


def calc_keras_metrics(model, dataset):
  """Calculates a trained model's metrics over a dataset."""
  results = model.evaluate(dataset)
  if len(model.metrics_names) == 1:
    results = [results]
  return dict(zip(model.metrics_names, results))

def export_dash_file(labels, predictions, astro_ids, results_path):
  model_no = 1

  # Step 1: Verify one-hot encoding
  print(labels)

  # Step 2: Map label indices to class names
  class_names = np.array(['p', 'e', 'n', 'j'])

  # Get the index of the true label for each row
  true_label_indices = labels.argmax(axis=1)
  true_labels = class_names[true_label_indices]

  # Step 3: Build DataFrame
  df = pd.DataFrame({
      'astro_id': astro_ids,
      'model_no': model_no,
      'disp_p': predictions[:, 0],
      'disp_e': predictions[:, 1],
      'disp_n': predictions[:, 2],
      'disp_j': predictions[:, 3],
      'true_label': true_labels
  })

  # Step 4: Export to CSV
  df.to_csv(results_path, index=False)
  print(f"✅ Exported to {results_path}")


def generate_labels_and_predictions(model, dataset, threshold=None):
  """Generates labels and predictions from a trained model on a dataset.

  If threshold is provided, returns binary predictions using that threshold.
  """
  y_pred = model.predict(dataset)
  y_label = []
  # for features, labels, weights in dataset:
  #   y_label.append(labels)
  #y_label = np.concatenate(y_label).astype(np.int32)
  print(y_pred)
  1/0

  if threshold is not None: # <-- this should use argmax
    y_pred_binary = (y_pred > threshold).astype(np.int32)
    return y_label, y_pred, y_pred_binary

  return y_label, y_pred


def calc_auc_scores(y_label, y_pred, primary_class):
  """Calculates AUC scores for predictions and labels."""
  from sklearn import metrics

  if np.ndim(y_label) == 2:
    y_label = y_label[:, primary_class]
    y_pred = y_pred[:, primary_class]
  elif np.ndim(y_label) != 1 or primary_class != 0:
    raise ValueError(
        f"y_label has shape {y_label.shape}, primary_class={primary_class}")
  auc = metrics.roc_auc_score(y_label, y_pred)
  ap = metrics.average_precision_score(y_label, y_pred)
  return {"roc_auc": auc, "average_precision": ap}


def calc_precision_recall_f1(y_label, y_pred_binary, primary_class):
  """Calculates precision, recall, and F1 score for binary predictions."""
  from sklearn import metrics

  if np.ndim(y_label) == 2:
    y_label = y_label[:, primary_class]
    y_pred_binary = y_pred_binary[:, primary_class]
  elif np.ndim(y_label) != 1 or primary_class != 0:
    raise ValueError(
        f"y_label has shape {y_label.shape}, primary_class={primary_class}")

  precision = metrics.precision_score(y_label, y_pred_binary)
  recall = metrics.recall_score(y_label, y_pred_binary)
  f1 = metrics.f1_score(y_label, y_pred_binary)
  return {"precision_thresh": precision, "recall_thresh": recall, "f1_thresh": f1}


def evaluate_model(model, input_config, file_pattern, batch_size, threshold=None):
  """Evaluates a model over a dataset with optional threshold."""
  dataset = input_ds.build_eval_dataset(
      file_pattern=file_pattern,
      input_config=input_config,
      batch_size=batch_size,
      include_identifiers=False,
      include_labels=False)

  if threshold is not None:
    y_label, y_pred, y_pred_binary = generate_labels_and_predictions(model, dataset, threshold=threshold)
  else:
    y_label, y_pred = generate_labels_and_predictions(model, dataset)
    y_pred_binary = None

  metrics = calc_auc_scores(y_label, y_pred, input_config.primary_class)
  keras_metrics = calc_keras_metrics(model, dataset)
  metrics.update(keras_metrics)

  if y_pred_binary is not None:
    threshold_metrics = calc_precision_recall_f1(y_label, y_pred_binary, input_config.primary_class)
    metrics.update(threshold_metrics)

  dataset_with_astro_ids = input_ds.build_eval_dataset(
    file_pattern=file_pattern,
    input_config=input_config,
    batch_size=batch_size,
    include_identifiers=True,
    include_labels=False)
  astro_ids = []
  for _, astro_id in dataset_with_astro_ids:
    astro_ids.append(astro_id)
  astro_ids = np.concatenate(astro_ids).astype(np.int32)

  return metrics, y_label, y_pred, astro_ids


def save_metrics(metrics, eval_dir):
  """Saves the metrics dictionary as a json file."""
  filename = os.path.join(eval_dir, "metrics.json")
  with open(filename, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)