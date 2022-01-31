#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import multiprocessing
import os
import sys

from absl import logging
from absl import app
import numpy as np
import pandas as pd
import tensorflow as tf

from astronet.preprocess import preprocess


parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_tce_csv_file",
    type=str,
    required=True)

parser.add_argument(
    "--tess_data_dir",
    type=str,
    required=True)

parser.add_argument(
    "--output_dir",
    type=str,
    required=True)

parser.add_argument(
    "--num_shards",
    type=int,
    default=20)

parser.add_argument(
    "--vetting_features",
    type=str,
    default='n')


def _set_float_feature(ex, tic_id, name, value):
  """Sets the value of a float feature in a tensorflow.train.Example proto."""
  assert name not in ex.features.feature, "Duplicate feature: %s" % name
  if isinstance(value, np.ndarray):
    value = value.reshape((-1,))
  values = [float(v) for v in value]
  if any(np.isnan(values)):
    raise ValueError(f'NaNs in {name} for {tic_id}')
  ex.features.feature[name].float_list.value.extend(values)


def _set_bytes_feature(ex, name, value):
  """Sets the value of a bytes feature in a tensorflow.train.Example proto."""
  assert name not in ex.features.feature, "Duplicate feature: %s" % name
  ex.features.feature[name].bytes_list.value.extend([str(v).encode("latin-1") for v in value])


def _set_int64_feature(ex, name, value):
  """Sets the value of an int64 feature in a tensorflow.train.Example proto."""
  assert name not in ex.features.feature, "Duplicate feature: %s" % name
  ex.features.feature[name].int64_list.value.extend([int(v) for v in value])


def _standard_views(ex, tic, time, flux, period, epoc, duration, bkspace, aperture_fluxes):
  if bkspace is None:
    tag = ''
  else:
    tag = f'_{bkspace}'

  detrended_time, detrended_flux, transit_mask = preprocess.detrend_and_filter(tic, time, flux, period, epoc, duration, bkspace)

  time, flux, fold_num, tr_mask = preprocess.phase_fold_and_sort_light_curve(
      detrended_time, detrended_flux, transit_mask, period, epoc)
  odds = ((fold_num % 2) == 1)
  evens = ((fold_num % 2) == 0)

  view, std, mask, _, _ = preprocess.global_view(tic, time, flux, period)
  tr_mask, _, _, _, _ = preprocess.tr_mask_view(tic, time, tr_mask, period)
  _set_float_feature(ex, tic, f'global_view{tag}', view)
  _set_float_feature(ex, tic, f'global_std{tag}', std)
  _set_float_feature(ex, tic, f'global_mask{tag}', mask)
  _set_float_feature(ex, tic, f'global_transit_mask{tag}', tr_mask)

  view, std, mask, scale, depth = preprocess.local_view(tic, time, flux, period, duration)
  _set_float_feature(ex, tic, f'local_view{tag}', view)
  _set_float_feature(ex, tic, f'local_std{tag}', std)
  _set_float_feature(ex, tic, f'local_mask{tag}', mask)
  _set_float_feature(ex, tic, f'local_scale{tag}', [scale])
  _set_float_feature(ex, tic, f'local_scale_present{tag}', [1.0 if scale > 0 else 0.0])
  for k, (t, f) in aperture_fluxes.items():
    t, f, m = preprocess.detrend_and_filter(tic, t, f, period, epoc, duration, bkspace)
    t, f, _, _ = preprocess.phase_fold_and_sort_light_curve(t, f, m, period, epoc)
    view, std, _, _, _ = preprocess.local_view(tic, t, f, period, duration, scale=scale, depth=depth)
    _set_float_feature(ex, tic, f'local_aperture_{k}{tag}', view)

  view, std, mask, _, _ = preprocess.local_view(tic, time[odds], flux[odds], period, duration, scale=scale, depth=depth)
  _set_float_feature(ex, tic, f'local_view_odd{tag}', view)
  _set_float_feature(ex, tic, f'local_std_odd{tag}', std)
  _set_float_feature(ex, tic, f'local_mask_odd{tag}', mask)

  view, std, mask, _, _ = preprocess.local_view(tic, time[evens], flux[evens], period, duration, scale=scale, depth=depth)
  _set_float_feature(ex, tic, f'local_view_even{tag}', view)
  _set_float_feature(ex, tic, f'local_std_even{tag}', std)
  _set_float_feature(ex, tic, f'local_mask_even{tag}', mask)

  (_, _, _, sec_scale, sec_depth), t0 = preprocess.secondary_view(tic, time, flux, period, duration)
  (view, std, mask, scale, _), t0 = preprocess.secondary_view(tic, time, flux, period, duration, scale=scale, depth=depth)
  _set_float_feature(ex, tic, f'secondary_view{tag}', view)
  _set_float_feature(ex, tic, f'secondary_std{tag}', std)
  _set_float_feature(ex, tic, f'secondary_mask{tag}', mask)
  _set_float_feature(ex, tic, f'secondary_phase{tag}', [t0 / period])
  _set_float_feature(ex, tic, f'secondary_scale{tag}', [sec_scale])
  _set_float_feature(ex, tic, f'secondary_scale_present{tag}', [1.0 if scale > 0 else 0.0])

  full_view = preprocess.sample_segments_view(tic, time, flux, fold_num, period, duration)
  _set_float_feature(ex, tic, f'sample_segments_view{tag}', full_view)

  odd_view = preprocess.sample_segments_view(
      tic, time[odds], flux[odds], fold_num[odds], period, duration, num_bins=61, num_transits=4, local=True)
  even_view = preprocess.sample_segments_view(
      tic, time[evens], flux[evens], fold_num[evens], period, duration, num_bins=61, num_transits=4, local=True)
  full_view = np.concatenate([odd_view, even_view], axis=-1)
  _set_float_feature(ex, tic, f'sample_segments_local_view{tag}', full_view)
  
  time, flux, fold_num, _ = preprocess.phase_fold_and_sort_light_curve(
      detrended_time, detrended_flux, transit_mask, period * 2, epoc - period / 2)
  view, std, _, scale, _ = preprocess.global_view(tic, time, flux, period * 2)
  _set_float_feature(ex, tic, f'global_view_double_period{tag}', view)
  _set_float_feature(ex, tic, f'global_view_double_period_std{tag}', std)

  time, flux, fold_num, _ = preprocess.phase_fold_and_sort_light_curve(
      detrended_time, detrended_flux, transit_mask, period / 2, epoc)
  view, std, _, scale, _ = preprocess.global_view(tic, time, flux, period / 2)
  _set_float_feature(ex, tic, f'global_view_half_period{tag}', view)
  _set_float_feature(ex, tic, f'global_view_half_period_std{tag}', std)
  
  view, std, _, scale, _ = preprocess.local_view(tic, time, flux, period / 2, duration)
  _set_float_feature(ex, tic, f'local_view_half_period{tag}', view)
  _set_float_feature(ex, tic, f'local_view_half_period_std{tag}', std)
    
  return fold_num, sec_scale - sec_depth


def _process_tce(tce, bkspace=None):
  if FLAGS.vetting_features == 'y':
    extra_fluxes = {
        's': 'SAP_FLUX_SML',
        'm': 'SAP_FLUX_MID',
        'l': 'SAP_FLUX_LAG',
    }
  else:
    extra_fluxes = {}
  
  time, flux, apertures = preprocess.read_and_process_light_curve(
      tce.tic_id, FLAGS.tess_data_dir, 'SAP_FLUX', extra_fluxes, tce.filename)
  ex = tf.train.Example()

  for bkspace in [0.3, 5.0, None]:
    fold_num, sec_depth = _standard_views(ex, tce.tic_id, time, flux, tce.Period, tce.Epoc, tce.Duration, bkspace, apertures)

  _set_float_feature(ex, tce, 'n_folds', [len(set(fold_num))])
  _set_float_feature(ex, tce, 'n_points', [len(fold_num)])
    
  for col_name, value in tce.items():
    if col_name.lower() in ('tic id',) or col_name.startswith('disp_'):
        _set_int64_feature(ex, col_name, [int(value)])
    elif col_name.lower() in ('filename',):
        _set_bytes_feature(ex, col_name, [value])
    else:
        f_val = float(value)
        if np.isnan(f_val):
            _set_int64_feature(ex, f'{col_name}_present', [0])
            f_val = 0
        else:
            _set_int64_feature(ex, f'{col_name}_present', [1])
        _set_float_feature(ex, tce, col_name, [f_val])

  return ex, sec_depth


def _process_file_shard(tce_table, file_name):
  process_name = multiprocessing.current_process().name
  shard_name = os.path.basename(file_name)
  shard_size = len(tce_table)
    
  existing = {}
  try:
    tfr = tf.data.TFRecordDataset(file_name)
    for record in tfr:
      ex_str = record.numpy()
      ex = tf.train.Example.FromString(ex_str)
      existing[ex.features.feature["tic_id"].int64_list.value[0]] = ex_str
  except:
    pass

  with tf.io.TFRecordWriter(file_name) as writer:
    num_processed = 0
    num_skipped = 0
    num_existing = 0
    print("", end='')
    for _, tce in tce_table.iterrows():
      num_processed += 1
      print("\r                                      ", end="")
      print(f"\r[{num_processed}/{shard_size}] {tce['tic_id']}", end="")

      if int(tce["tic_id"]) in existing:
        print(" exists", end="")
        sys.stdout.flush()
        writer.write(existing[int(tce["tic_id"])])
        num_existing += 1
        continue

      examples = []
      try:
        print(" processing", end="")
        sys.stdout.flush()
        ex, sec_depth = _process_tce(tce)
        examples.append(ex)
      except Exception as e:
        print(f" *** error: {e}")
        num_skipped += 1
        continue

      print(" writing                   ", end="")
      sys.stdout.flush()
      for example in examples:
        writer.write(example.SerializeToString())
        

  num_new = num_processed - num_skipped - num_existing
  print(f"\r{shard_name}: {num_processed}/{shard_size} {num_new} new {num_skipped} bad            ")


def main(_):
    tf.io.gfile.makedirs(FLAGS.output_dir)

    tce_table = pd.read_csv(
        FLAGS.input_tce_csv_file,
        header=0,
        dtype={'tic_id': str})

    num_tces = len(tce_table)
    logging.info("Read %d TCEs", num_tces)

    # Further split training TCEs into file shards.
    file_shards = []  # List of (tce_table_shard, file_name).
    boundaries = np.linspace(
        0, len(tce_table), FLAGS.num_shards + 1).astype(np.int)
    for i in range(FLAGS.num_shards):
      start = boundaries[i]
      end = boundaries[i + 1]
      file_shards.append((
          start,
          end,
          os.path.join(FLAGS.output_dir, "%.5d-of-%.5d" % (i, FLAGS.num_shards))
      ))

    logging.info("Processing %d total file shards", len(file_shards))
    for start, end, file_shard in file_shards:
        _process_file_shard(tce_table[start:end], file_shard)
    logging.info("Finished processing %d total file shards", len(file_shards))


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
