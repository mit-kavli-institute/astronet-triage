import copy
import os
import shutil

import tensorflow as tf
from absl import logging
from tensorboard.plugins.hparams import api as hp

from astronet import evaluation, models, training
from astronet.tuning import search_space
from astronet.util import config_util

# Concise metric labels for TensorBoard.
_METRIC_LABELS = {
    "loss": "loss",
    "roc_auc": "auc",
    "average_precision": "ap",
}


def run_trial(model_class, config, train_files, val_files, shuffle_buffer_size):
  """Runs a single tuning trial."""
  model = model_class(config)
  training.train(
      model,
      config,
      train_files=train_files,
      shuffle_buffer_size=shuffle_buffer_size)
  results = {}
  for dataset, file_pattern in [("train", train_files), ("val", val_files)]:
    batch_size = config.hparams.batch_size
    metrics, y_label, y_pred = evaluation.evaluate_model(
        model, config.inputs, file_pattern, batch_size=batch_size)
    results[dataset] = dict(
        loss=metrics["loss"], y_label=y_label, y_pred=y_pred)
  return results


def run_tuning_study(study_config, study_dir, n_trials=None, overwrite=False):
  """Runs a tuning study."""
  if os.path.exists(study_dir):
    if overwrite:
      logging.info(f"Removing existing output directory: {study_dir}")
      shutil.rmtree(study_dir)
    else:
      raise ValueError(f"Output directory exists: {study_dir}")

  # Save the study config.
  os.makedirs(study_dir)
  config_util.save_config(study_config, study_dir, basename="study_config")

  model_class = models.get_model_class(study_config.model)
  base_config = models.get_model_config(study_config.model,
                                        study_config.config_name)
  ss = search_space.from_config(study_config.search_space)

  # Log information for TensorBoard in the top level directory.
  metric_specs = []
  for dataset in ["train", "val"]:
    for name, label in _METRIC_LABELS.items():
      metric_specs.append(
          hp.Metric(f"{dataset}_{name}", display_name=f"{dataset}_{label}"))
  with tf.summary.create_file_writer(study_dir).as_default():
    hp.hparams_config(hparams=ss.get_tensorboard_specs(), metrics=metric_specs)

  # Run trials.
  for n, search_params in enumerate(ss.search()):
    trial_id = str(n)
    trial_dir = os.path.join(study_dir, trial_id)
    if os.path.exists(trial_dir):
      continue  # Already done.

    logging.info(f"Trial {n}")

    # Override the base config with trial parameters.
    trial_params = copy.deepcopy(study_config.base_param_overrides)
    trial_params.update(search_params)
    trial_params = config_util.unflatten(trial_params)
    config_util.save_config(trial_params, trial_dir, basename="trial_params")
    logging.info(f"Params: {trial_params}")
    trial_config = copy.deepcopy(base_config)
    config_util.update(trial_config, trial_params)
    config_util.save_config(trial_config, trial_dir)

    # Run the trial.
    all_results = []
    n_ensemble = study_config.n_ensemble
    for i in range(n_ensemble):
      logging.info(f"Model {i + 1}/{n_ensemble} in ensemble")
      results = run_trial(
          model_class,
          trial_config,
          train_files=study_config.train_files,
          val_files=study_config.val_files,
          shuffle_buffer_size=study_config.shuffle_buffer_size)
      logging.info(f"Train loss: {results['train'][0]:.4g}, "
                   f"val loss: {results['val'][0]:.4g}")
      all_results.append(results)
    final_metrics = evaluation.calc_ensemble_metrics(
        all_results, primary_class=trial_config.inputs.primary_class)
    logging.info(f"Metrics over {n_ensemble}-model ensemble: {final_metrics}")
    evaluation.save_metrics(final_metrics, trial_dir)

    # Log to Tensorboard.
    with tf.summary.create_file_writer(trial_dir).as_default():
      hp.hparams(search_params, trial_id=trial_id)
      for dataset, metrics in final_metrics.items():
        for metric, value in metrics.items():
          tf.summary.scalar(
              f"{dataset}_{metric}", value, step=trial_config.train_steps)

    if n_trials and n >= n_trials - 1:
      break
