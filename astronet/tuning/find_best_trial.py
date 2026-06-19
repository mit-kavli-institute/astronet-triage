"""Script to find the best trial from a tuning study."""

import json
import os
import sys
from absl import app, flags, logging

flags.DEFINE_string(
    "study_dir",
    None,
    "Directory containing the tuning study results.",
    required=True)

flags.DEFINE_string(
    "metric",
    "val_loss",
    "Metric to optimize (e.g., 'val_loss', 'val_roc_auc', 'val_average_precision'). "
    "For loss, lower is better. For other metrics, higher is better.")

flags.DEFINE_bool(
    "minimize",
    None,
    "Whether to minimize the metric (True) or maximize (False). "
    "If None, will auto-detect based on metric name (loss -> minimize).")

FLAGS = flags.FLAGS


def load_trial_metrics(study_dir):
  """Loads metrics from all completed trials."""
  trials = {}

  for trial_id in sorted(os.listdir(study_dir)):
    trial_dir = os.path.join(study_dir, trial_id)
    if not os.path.isdir(trial_dir):
      continue

    metrics_file = os.path.join(trial_dir, "metrics.json")
    if not os.path.exists(metrics_file):
      continue

    try:
      with open(metrics_file, "r") as f:
        metrics = json.load(f)

      # Also load trial params
      params_file = os.path.join(trial_dir, "trial_params.json")
      params = {}
      if os.path.exists(params_file):
        with open(params_file, "r") as f:
          params = json.load(f)

      trials[int(trial_id)] = {
          "metrics": metrics,
          "params": params,
          "dir": trial_dir
      }
    except Exception as e:
      logging.warning(f"Failed to load trial {trial_id}: {e}")
      continue

  return trials


def get_metric_value(metrics, metric_name):
  """Extracts a metric value from the metrics dict."""
  # Handle format like "val_loss" or "train_roc_auc"
  parts = metric_name.split("_", 1)
  if len(parts) == 2:
    dataset, metric = parts
    if dataset in metrics and metric in metrics[dataset]:
      return metrics[dataset][metric]

  # Try direct lookup
  if metric_name in metrics:
    return metrics[metric_name]

  # Try nested lookup
  for dataset in ["train", "val", "eval"]:
    if dataset in metrics and metric_name in metrics[dataset]:
      return metrics[dataset][metric_name]

  return None


def main(_):
  study_dir = FLAGS.study_dir
  if not os.path.exists(study_dir):
    logging.error(f"Study directory does not exist: {study_dir}")
    sys.exit(1)

  # Determine if we should minimize or maximize
  minimize = FLAGS.minimize
  if minimize is None:
    # Auto-detect: loss metrics should be minimized
    minimize = "loss" in FLAGS.metric.lower()

  logging.info(f"Loading trials from: {study_dir}")
  logging.info(f"Optimizing metric: {FLAGS.metric} ({'minimize' if minimize else 'maximize'})")

  trials = load_trial_metrics(study_dir)
  if not trials:
    logging.error("No completed trials found!")
    sys.exit(1)

  logging.info(f"Found {len(trials)} completed trials")

  # Find best trial
  best_trial_id = None
  best_value = None

  valid_trials = []
  for trial_id, trial_data in trials.items():
    value = get_metric_value(trial_data["metrics"], FLAGS.metric)
    if value is None:
      logging.warning(f"Trial {trial_id} does not have metric {FLAGS.metric}")
      continue

    valid_trials.append((trial_id, value, trial_data))

    if best_value is None:
      best_trial_id = trial_id
      best_value = value
    elif minimize and value < best_value:
      best_trial_id = trial_id
      best_value = value
    elif not minimize and value > best_value:
      best_trial_id = trial_id
      best_value = value

  if best_trial_id is None:
    logging.error(f"No trials found with metric {FLAGS.metric}")
    sys.exit(1)

  best_trial = trials[best_trial_id]

  # Print results
  print("\n" + "=" * 80)
  print(f"BEST TRIAL: {best_trial_id}")
  print("=" * 80)
  print(f"\nMetric ({FLAGS.metric}): {best_value}")
  print(f"\nTrial directory: {best_trial['dir']}")

  print("\nHyperparameters:")
  print(json.dumps(best_trial["params"], indent=2))

  print("\nAll metrics:")
  print(json.dumps(best_trial["metrics"], indent=2))

  # Show top 5 trials
  valid_trials.sort(key=lambda x: x[1], reverse=not minimize)
  print(f"\n{'=' * 80}")
  print("TOP 5 TRIALS:")
  print("=" * 80)
  for i, (trial_id, value, _) in enumerate(valid_trials[:5], 1):
    marker = " <-- BEST" if trial_id == best_trial_id else ""
    print(f"{i}. Trial {trial_id}: {FLAGS.metric} = {value}{marker}")


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  app.run(main)
