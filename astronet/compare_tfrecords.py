"""Script for comparing TFRecords."""

import os
import numpy as np
import tensorflow as tf
from absl import app, flags, logging
import pandas as pd
from scipy.stats import gaussian_kde


from astronet import evaluation, models
from astronet.util import config_util
import matplotlib.pyplot as plt
import seaborn as sns

from astronet.astro_cnn_model import input_ds

flags.DEFINE_string("model", None, "Name of the model class.", required=True)

flags.DEFINE_string("tfrecord_dir_a", None, "Name of the model class.", required=True)
flags.DEFINE_string("tfrecord_dir_b", None, "Name of the model class.", required=True)


flags.DEFINE_string(
    "model_dir", None,
    "Directory of the trained model to evaluate (must contain config.yaml).",
    required=True)

FLAGS = flags.FLAGS

def main(_):

  print('starting analysis')
  config = config_util.load_config(FLAGS.model_dir)
 
  scalar_keys = [
    'secondary_phase', 'secondary_phase_0.3', 'secondary_phase_5.0',
    'local_scale', 'local_scale_present', 'local_scale_0.3', 'local_scale_present_0.3',
    'local_scale_5.0', 'local_scale_present_5.0',
    'secondary_scale', 'secondary_scale_present', 'secondary_scale_0.3',
    'secondary_scale_present_0.3', 'secondary_scale_5.0', 'secondary_scale_present_5.0',
    'global_std', 'global_mask', 'global_transit_mask',
    'local_std', 'local_mask',
    'local_std_odd', 'local_std_even', 'local_view_half_period_std',
    'secondary_std', 'secondary_mask',
    'Period', 'Duration', 'Transit_Depth', 'Tmag',
    'star_mass', 'star_mass_present', 'star_rad', 'star_rad_present',
    'n_folds',
    'local_aperture_s', 'local_aperture_m', 'local_aperture_l'
  ]
  scalar_data_a = {key: [] for key in scalar_keys}
  scalar_data_b = {key: [] for key in scalar_keys}

  ts_keys = [
      'global_view', 'global_view_0.3', 'global_view_5.0',
      'local_view', 'local_view_0.3', 'local_view_5.0',
      'local_view_odd', 'local_view_even',
      'secondary_view', 'secondary_view_0.3', 'secondary_view_5.0',
      'global_view_double_period', 'global_view_double_period_0.3', 'global_view_double_period_5.0',
  ] # removed sample_segments_view
  ts_data_a = {key: [] for key in ts_keys}
  ts_data_b = {key: [] for key in ts_keys}

  
  print('Generating A records')
  # get tfrecord A inputs
  dataset = input_ds.build_eval_dataset(
      file_pattern=FLAGS.tfrecord_dir_a,
      input_config=config.inputs,
      batch_size=config.hparams.batch_size,
      include_identifiers=True,
      include_labels=False)
  for batch in dataset:
      inputs, identifiers = batch
      for key in scalar_keys:
          scalar_data_a[key].append(inputs[key].numpy())
      for key in ts_keys:
          ts_data_a[key].append(inputs[key].numpy())
  # get tfrecord B inputs
  print('Generating B records')
  dataset = input_ds.build_eval_dataset(
      file_pattern=FLAGS.tfrecord_dir_b,
      input_config=config.inputs,
      batch_size=config.hparams.batch_size,
      include_identifiers=True,
      include_labels=False)
  for batch in dataset:
      inputs, identifiers = batch
      for key in scalar_keys:
          scalar_data_b[key].append(inputs[key].numpy())
      for key in ts_keys:
          ts_data_b[key].append(inputs[key].numpy())

  # post process scalars
  for key in scalar_keys:
    scalar_data_a[key] = np.concatenate(scalar_data_a[key], axis=0)
    scalar_data_b[key] = np.concatenate(scalar_data_b[key], axis=0)

  # # plot scalars
  for key in scalar_keys:
    print(key)
    a_data = scalar_data_a[key]
    b_data = scalar_data_b[key]

    # Clean data (remove NaNs if needed)
    a_data = a_data[~np.isnan(a_data)]
    b_data = b_data[~np.isnan(b_data)]

    if a_data.size == 0 or b_data.size == 0:
        print(f"Skipping {key} due to empty data")
        continue
    # Build KDEs
    kde_a = gaussian_kde(a_data, bw_method=0.2)  # bw_method can be tuned
    kde_b = gaussian_kde(b_data, bw_method=0.2)

    # Define x range (shared for both to compare meaningfully)
    x_min = min(a_data.min(), b_data.min())
    x_max = max(a_data.max(), b_data.max())
    x_vals = np.linspace(x_min, x_max, 1000)

    plt.figure()
    plt.fill_between(x_vals, kde_a(x_vals), alpha=0.5, label='Real Sector', color='red')
    plt.fill_between(x_vals, kde_b(x_vals), alpha=0.5, label='Test Data', color='blue')
    plt.title(f"Distribution of {key}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("/pdo/users/dimond/astronet_secondary/astronet/astronet/tfrecord_viz", f"{key}_distribution.png"))
    plt.close()

  # plot ts
  for key in ts_keys:
    print(key)
    data_a = np.concatenate(ts_data_a[key], axis=0)  # shape: [N, time]
    data_b = np.concatenate(ts_data_b[key], axis=0)  # shape: [N, time]

    mean_curve_a = np.mean(data_a, axis=0)
    mean_curve_b = np.mean(data_b, axis=0)
    std_curve_a = np.std(data_a, axis=0)
    std_curve_b = np.std(data_b, axis=0)

    plt.figure()
    plt.plot(mean_curve_a, label='Real Sector Mean')
    plt.fill_between(np.arange(len(mean_curve_a)), mean_curve_a - std_curve_a, mean_curve_a + std_curve_a, alpha=0.3, label='±1 STD')

    plt.plot(mean_curve_b, label='Test Data Mean')
    plt.fill_between(np.arange(len(mean_curve_b)), mean_curve_b - std_curve_b, mean_curve_b + std_curve_b, alpha=0.3, label='±1 STD')
    plt.title(f"Mean ± STD of {key}")
    plt.legend()
    plt.savefig(os.path.join("/pdo/users/dimond/astronet_secondary/astronet/astronet/tfrecord_viz", f"{key}_distribution.png"))

if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  app.run(main)