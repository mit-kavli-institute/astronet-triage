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
from typing import Literal, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from absl import app, logging
from typing_extensions import Protocol
import traceback

from astronet.preprocess import preprocess

SKIPPED_TICS=[]

class LCGetter(Protocol):
   def __call__(self, astro_id: int, aperture: Optional[Literal['s', 'm', 'l']] = None): ...
AstronetMode = Literal["triage", "vetting"]


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
    "--mode",
    type=str,
    choices=["triage", "vetting"],
    required=True)

parser.add_argument(
   "--not-training",
   action="store_true")

parser.add_argument(
  "--remove_random_points",
  action="store_true"
)

def _set_float_feature(ex, name, value):
  """Sets the value of a float feature in a tensorflow.train.Example proto."""
  assert name not in ex.features.feature, "Duplicate feature: %s" % name
  if isinstance(value, np.ndarray):
    value = value.reshape((-1,))
  values = [float(v) for v in value]
  if any(np.isnan(values)):
    raise ValueError(f'NaNs in {name}')
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
  if time is None or len(time) == 0 or flux is None or len(flux) == 0:
    logging.warning(f"Skipping TIC {tic}: empty time/flux array.")
    SKIPPED_TICS.append(tic)
    return None

  # # Randomly drop a subset of the data is the flag is active
  # if FLAGS.remove_random_points:
  #   time, flux = preprocess.remove_random_datapoints(time, flux, 0.1)


  if bkspace is None:
    tag = ''
  else:
    tag = f'_{bkspace}'

  detrended_time, detrended_flux, transit_mask = preprocess.detrend_and_filter(
      tic, time, flux, period, epoc, duration, bkspace)

  # Randomly drop a subset of the data if the flag is active (after detrending)
  if FLAGS.remove_random_points:
    detrended_time, detrended_flux, mask_removal = preprocess.remove_random_datapoints(
        detrended_time, detrended_flux, 0.1)
    transit_mask = transit_mask[mask_removal]

  time, flux, fold_num, tr_mask = preprocess.phase_fold_and_sort_light_curve(
      detrended_time, detrended_flux, transit_mask, period, epoc)
  odds = ((fold_num % 2) == 1)
  evens = ((fold_num % 2) == 0)

  view, std, mask, _, _ = preprocess.global_view(tic, time, flux, period)
  tr_mask, _, _, _, _ = preprocess.tr_mask_view(tic, time, tr_mask, period)
  _set_float_feature(ex, f'global_view{tag}', view)
  _set_float_feature(ex, f'global_std{tag}', std)
  _set_float_feature(ex, f'global_mask{tag}', mask)
  _set_float_feature(ex, f'global_transit_mask{tag}', tr_mask)

  view, std, mask, scale, depth = preprocess.local_view(tic, time, flux, period, duration)
  _set_float_feature(ex, f'local_view{tag}', view)
  _set_float_feature(ex, f'local_std{tag}', std)
  _set_float_feature(ex, f'local_mask{tag}', mask)
  if scale is not None:
    _set_float_feature(ex, f'local_scale{tag}', [scale])
    _set_float_feature(ex, f'local_scale_present{tag}', [1.0])
  else:
    _set_float_feature(ex, f'local_scale{tag}', [0.0])
    _set_float_feature(ex, f'local_scale_present{tag}', [0.0])
  for k, (t, f) in aperture_fluxes.items():
    t, f, m = preprocess.detrend_and_filter(tic, t, f, period, epoc, duration, bkspace)
    t, f, _, _ = preprocess.phase_fold_and_sort_light_curve(t, f, m, period, epoc)
    view, std, _, _, _ = preprocess.local_view(tic, t, f, period, duration, scale=scale, depth=depth)
    _set_float_feature(ex, f'local_aperture_{k}{tag}', view)

  view, std, mask, _, _ = preprocess.local_view(tic, time[odds], flux[odds], period, duration, scale=scale, depth=depth)
  _set_float_feature(ex, f'local_view_odd{tag}', view)
  _set_float_feature(ex, f'local_std_odd{tag}', std)
  _set_float_feature(ex, f'local_mask_odd{tag}', mask)

  view, std, mask, _, _ = preprocess.local_view(tic, time[evens], flux[evens], period, duration, scale=scale, depth=depth)
  _set_float_feature(ex, f'local_view_even{tag}', view)
  _set_float_feature(ex, f'local_std_even{tag}', std)
  _set_float_feature(ex, f'local_mask_even{tag}', mask)

  try:
      (_, _, _, sec_scale, _), t0 = preprocess.secondary_view(tic, time, flux, period, duration)
      (view, std, mask, scale, _), t0 = preprocess.secondary_view(tic, time, flux, period, duration, scale=scale, depth=depth)
  except IndexError as e:
      logging.warning(f"Skipping TIC {tic}: preprocess.secondary_view failed (IndexError).", exc_info=False)
      time_info = f"shape {time.shape}" if hasattr(time, 'shape') else f"length {len(time)}"
      flux_info = f"shape {flux.shape}" if hasattr(flux, 'shape') else f"length {len(flux)}"
      logging.debug(f"  Details for failed TIC {tic}: Error='{e}', Input time info='{time_info}', Input flux info='{flux_info}'")
      SKIPPED_TICS.append(tic)
      return None

  _set_float_feature(ex, f'secondary_view{tag}', view)
  _set_float_feature(ex, f'secondary_std{tag}', std)
  _set_float_feature(ex, f'secondary_mask{tag}', mask)
  _set_float_feature(ex, f'secondary_phase{tag}', [t0 / period])
  if sec_scale is not None:
    _set_float_feature(ex, f'secondary_scale{tag}', [sec_scale])
    _set_float_feature(ex, f'secondary_scale_present{tag}', [1.0])
  else:
    _set_float_feature(ex, f'secondary_scale{tag}', [0.0])
    _set_float_feature(ex, f'secondary_scale_present{tag}', [0.0])

  full_view = preprocess.sample_segments_view(tic, time, flux, fold_num, period, duration)
  _set_float_feature(ex, f'sample_segments_view{tag}', full_view)

  odd_view = preprocess.sample_segments_view(
      tic, time[odds], flux[odds], fold_num[odds], period, duration, num_bins=61, num_transits=4, local=True)
  even_view = preprocess.sample_segments_view(
      tic, time[evens], flux[evens], fold_num[evens], period, duration, num_bins=61, num_transits=4, local=True)
  full_view = np.concatenate([odd_view, even_view], axis=-1)
  _set_float_feature(ex, f'sample_segments_local_view{tag}', full_view)

  time, flux, fold_num, _ = preprocess.phase_fold_and_sort_light_curve(
      detrended_time, detrended_flux, transit_mask, period * 2, epoc - period / 2)
  view, std, mask, scale, _ = preprocess.global_view(tic, time, flux, period * 2)
  _set_float_feature(ex, f'global_view_double_period{tag}', view)
  _set_float_feature(ex, f'global_view_double_period_std{tag}', std)
  _set_float_feature(ex, f'global_view_double_period_mask{tag}', mask)

  time, flux, fold_num, _ = preprocess.phase_fold_and_sort_light_curve(
      detrended_time, detrended_flux, transit_mask, period / 2, epoc)
  view, std, mask, scale, _ = preprocess.global_view(tic, time, flux, period / 2)
  _set_float_feature(ex, f'global_view_half_period{tag}', view)
  _set_float_feature(ex, f'global_view_half_period_std{tag}', std)
  _set_float_feature(ex, f'global_view_half_period_mask{tag}', mask)

  view, std, mask, scale, _ = preprocess.local_view(tic, time, flux, period / 2, duration)
  _set_float_feature(ex, f'local_view_half_period{tag}', view)
  _set_float_feature(ex, f'local_view_half_period_std{tag}', std)
  _set_float_feature(ex, f'local_view_half_period_mask{tag}', mask)

  return fold_num


def _process_tce(
    tce,
    get_lightcurve: LCGetter,
    mode: AstronetMode,
    training: bool
):
  import time
  start_time = time.time()
  astro_id = tce['Astro ID']
  logging.debug(f'Starting to process {astro_id}')
  tme, flux = get_lightcurve(tce['Astro ID'])
  if mode == 'vetting':
    apertures = {
      's': get_lightcurve(tce['Astro ID'], aperture='s'),
      'm': get_lightcurve(tce['Astro ID'], aperture='m'),
      'l': get_lightcurve(tce['Astro ID'], aperture='l'),
    }
  else:
    apertures = {}

  ex = tf.train.Example()

  for bkspace in [0.3, 5.0, None]:
    fold_num = _standard_views(ex, tce['TIC ID'], tme, flux, tce.Per, tce.Epoc, tce.Dur, bkspace, apertures)
    if fold_num is None:
      # We are skipping this TCE entirely
      return None

  _set_int64_feature(ex, 'astro_id', [tce['Astro ID']])

  if training:
    if mode == "vetting":
      _set_int64_feature(ex, 'disp_e', [tce['disp_e']])
      _set_int64_feature(ex, 'disp_p', [tce['disp_p']])
      _set_int64_feature(ex, 'disp_n', [tce['disp_n']])
      _set_int64_feature(ex, 'disp_b', [tce['disp_b']])
      _set_int64_feature(ex, 'disp_t', [tce['disp_t']])
      _set_int64_feature(ex, 'disp_u', [tce['disp_u']])
      _set_int64_feature(ex, 'disp_j', [tce['disp_j']])
    elif mode == "triage":
      _set_int64_feature(ex, 'disp_E', [tce['disp_E']])
      _set_int64_feature(ex, 'disp_N', [tce['disp_N']])
      _set_int64_feature(ex, 'disp_J', [tce['disp_J']])
      _set_int64_feature(ex, 'disp_S', [tce['disp_S']])
      _set_int64_feature(ex, 'disp_B', [tce['disp_B']])
    else:
      raise ValueError(f'Mode "{mode}" not supported.')

  assert not np.isnan(tce.Per)
  _set_float_feature(ex, 'Period', [tce.Per])

  assert not np.isnan(tce.Dur)
  _set_float_feature(ex, 'Duration', [tce.Dur])

  assert not np.isnan(tce.Depth)
  _set_float_feature(ex, 'Transit_Depth', [tce.Depth])

  assert not np.isnan(tce.Tmag)
  _set_float_feature(ex, 'Tmag', [tce.Tmag])

  # _set_float_feature(ex, 'centroid_dist', [tce.centroid_dist])

  if np.isnan(tce.SMass):
    _set_float_feature(ex, 'star_mass', [0])
    _set_float_feature(ex, 'star_mass_present', [0])
  else:
    _set_float_feature(ex, 'star_mass', [tce.SMass])
    _set_float_feature(ex, 'star_mass_present', [1])

  if np.isnan(tce.SRad):
    _set_float_feature(ex, 'star_rad', [0])
    _set_float_feature(ex, 'star_rad_present', [0])
  else:
    _set_float_feature(ex, 'star_rad', [tce.SRad])
    _set_float_feature(ex, 'star_rad_present', [1])

  if np.isnan(tce.SRadEst):
    _set_float_feature(ex, 'star_rad_est', [0])
    _set_float_feature(ex, 'star_rad_est_present', [0])
  else:
    _set_float_feature(ex, 'star_rad_est', [tce.SRadEst])
    _set_float_feature(ex, 'star_rad_est_present', [1])

  _set_float_feature(ex, 'n_folds', [len(set(fold_num))])
  _set_float_feature(ex, 'n_points', [len(fold_num)])

  end_time = time.time()
  logging.debug(f'Finished processing {astro_id} in {end_time - start_time} seconds')
  return ex

tce_table = None

def get_lightcurve(astro_id: int, aperture: Optional[str] = None):
    aperture_key_map = {
        "s": "SAP_FLUX_SML",
        "m": "SAP_FLUX_MID",
        "l": "SAP_FLUX_LAG",
        None: "SAP_FLUX",
    }
    global tce_table
    matching_tces = tce_table[tce_table["Astro ID"] == astro_id]
    try:
        _, tce = next(matching_tces.iterrows())
    except StopIteration as e:
        raise ValueError(f"Astro ID not found: {astro_id}") from e
    if "MinT" not in tce:
        tce["MinT"] = -np.inf
    if "MaxT" not in tce:
        tce["MaxT"] = np.inf
    return preprocess.read_and_process_light_curve(
        FLAGS.tess_data_dir,
        aperture_key_map[aperture],
        tce.File,
        tce.MinT,
        tce.MaxT,
    )

class ProcessRecordWorker:
    def __init__(self, existing, get_lightcurve, mode, training, _process_tce, tce_table, output_dir, augment_times):
        self.existing = existing
        self.get_lightcurve = get_lightcurve
        self.mode = mode
        self.training = training
        self._process_tce = _process_tce
        self.tce_table = tce_table
        self.augment_times = augment_times
        self.output_dir = output_dir

    def __call__(self, tce_row_dict):
        tce = pd.Series(tce_row_dict)
        recid = int(tce['Astro ID'])
        examples = []

        try:
            if recid in self.existing and self.augment_times == 0:
                return [(self.existing[recid], 'reused', recid)]

            # Original example: simply pass through the light curve
            def passthrough_lc_getter(astro_id, aperture=None):
                return self.get_lightcurve(astro_id, aperture)

            ex = self._process_tce(tce, passthrough_lc_getter, self.mode, self.training)
            if ex is not None:
                examples.append((ex.SerializeToString(), 'new', recid))
            else:
                # TCE was skipped (e.g., due to secondary_view IndexError)
                return [(None, 'skipped', recid)]

            # Augmented examples
            # TODO: add augmentation support which modifies the lc_getter
        except KeyboardInterrupt:
            logging.debug("ProcessRecordWorker propagating keyboard interrupt")
            raise
        except Exception as e:
            logging.warning(f"Skipping Astro ID {recid} due to error: {''.join(traceback.format_exception_only(type(e), e))}".strip())
            logging.debug(f"Full traceback for Astro ID {recid} (SAFELY CAUGHT):\n{traceback.format_exc()}")
            return [(None, 'skipped', recid)]

        return examples


def create(
    tce_table: pd.DataFrame,
    file_name: str,
    get_lightcurve,
    mode,
    training: bool,
    output_dir: str,
    num_processes: int = None,
):
  shard_name = os.path.basename(file_name)
  shard_size = len(tce_table)
  num_processes = num_processes or 1

  # Read existing TFRecords
  existing = {}
  try:
    tfr = tf.data.TFRecordDataset(file_name)
    for record in tfr:
      ex_str = record.numpy()
      ex = tf.train.Example.FromString(ex_str)
      astro_id = ex.features.feature['astro_id'].int64_list.value[0]
      existing[astro_id] = ex_str
  except KeyboardInterrupt:
    logging.debug("Propagating keyboard interrupt")
    raise
  except Exception as e:
    logging.debug(f"Warning: could not read existing records from {file_name}: {e}")
  tce_dicts = tce_table.to_dict(orient='records')
  logging.info(f"[{shard_name}] Starting processing with {num_processes} processes on {len(tce_dicts)} TCEs")

  worker = ProcessRecordWorker(existing, get_lightcurve, mode, training, _process_tce, tce_table, output_dir, 5)

  with multiprocessing.Pool(processes=num_processes) as pool:
    results_nested  = pool.map(worker, tce_dicts)

  results = [item for sublist in results_nested for item in sublist]  # Flatten

  serialized = []
  stats = {"new": 0, "reused": 0, "augmented": 0, "skipped": 0}

  for ex_bytes, status, recid in results:
    logging.debug(f"[{shard_name}] Processed Astro ID {recid} -> {status}")
    if ex_bytes:
      serialized.append(ex_bytes)
    stats[status] += 1

  with tf.io.TFRecordWriter(file_name) as writer:
    for ex in serialized:
      writer.write(ex)

  logging.info(f"[{shard_name}] Done. Total: {shard_size}, Stats: {stats}")

def main(_):
  if FLAGS.remove_random_points:
    print("\n=======================================")
    print('Generating augmentation tfrecords.')
    print("\n=======================================")

  tf.io.gfile.makedirs(FLAGS.output_dir)

  global tce_table
  tce_table = pd.read_csv(FLAGS.input_tce_csv_file, header=0, low_memory=False)

  num_tces = len(tce_table)
  logging.info("Read %d TCEs", num_tces)

  # Further split training TCEs into file shards.
  file_shards = []  # List of (tce_table_shard, file_name).
  boundaries = np.linspace(
      0, len(tce_table), FLAGS.num_shards + 1).astype(np.int)
  suffix = "_aug" if FLAGS.remove_random_points else ""
  for i in range(FLAGS.num_shards):
    start = boundaries[i]
    end = boundaries[i + 1]
    file_shards.append((
        start,
        end,
        os.path.join(FLAGS.output_dir, "%.5d-of-%.5d%s" % (i, FLAGS.num_shards, suffix))
    ))

  logging.info("Processing %d total file shards", len(file_shards))
  for start, end, file_shard in file_shards:
    logging.info(f'Starting shard {file_shard}')
    logging.info(f'{FLAGS.output_dir}')
    create(tce_table[start:end], file_shard, get_lightcurve, FLAGS.mode, not FLAGS.not_training, output_dir=FLAGS.output_dir, num_processes=35)
  logging.info("Finished processing %d total file shards", len(file_shards))

  ### [ADDED] At the very end, write the problematic TICs to a file
  if SKIPPED_TICS:
    problematic_path = os.path.join(FLAGS.output_dir, "problematic_tics.txt")
    with open(problematic_path, "a") as f:
      for tic in SKIPPED_TICS:
        f.write(f"{tic}\n")

    # Print final summary
    print("\n=======================================")
    print("The following TICs had empty time arrays:")
    print(SKIPPED_TICS)
    print(f"\nA list of these TICs is also saved to:\n {problematic_path}")
    print("=======================================\n")
  else:
    print("No problematic TICs found.")


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
