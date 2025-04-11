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
"""Script for generating TFrecord files from TESS TCEs."""

import multiprocessing
import os
import sys
from typing import Literal, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from absl import app, flags, logging
from typing_extensions import Protocol

from astronet.preprocess import preprocess
from astronet.util.example_util import set_float_feature, set_int64_feature

SKIPPED_TICS=[]

class LCGetter(Protocol):

  def __call__(self,
               astro_id: int,
               aperture: Optional[Literal["s", "m", "l"]] = None):
    ...


AstronetMode = Literal["triage", "vetting"]

flags.DEFINE_string(
    "input_tce_csv_file", None, "Filename of input TCE file.", required=True)

flags.DEFINE_string(
    "tess_data_dir", None, "TESS data directory.", required=True)

flags.DEFINE_string("output_dir", None, "Output directory.", required=True)

flags.DEFINE_integer("num_shards", 20, "Number of output shards.")

flags.DEFINE_enum(
    "mode",
    None, ["triage", "vetting"],
    "Whether to generate for triage or vetting.",
    required=True)

flags.DEFINE_bool("training", True, "Whether to generate for training.")

FLAGS = flags.FLAGS


def _standard_views(ex, tic, time, flux, period, epoc, duration, bkspace,
                    aperture_fluxes):

  if time is None or len(time) == 0 or flux is None or len(flux) == 0:
    logging.warning(f"Skipping TIC {tic}: empty time/flux array.") # <-- Changed line
    SKIPPED_TICS.append(tic)  # global list
    return None

  if bkspace is None:
    tag = ""
  else:
    tag = f"_{bkspace}"

  detrended_time, detrended_flux, transit_mask = preprocess.detrend_and_filter(
      tic, time, flux, period, epoc, duration, bkspace)

  time, flux, fold_num, tr_mask = preprocess.phase_fold_and_sort_light_curve(
      detrended_time, detrended_flux, transit_mask, period, epoc)
  odds = (fold_num % 2) == 1
  evens = (fold_num % 2) == 0

  view, std, mask, _, _ = preprocess.global_view(tic, time, flux, period)
  tr_mask, _, _, _, _ = preprocess.tr_mask_view(tic, time, tr_mask, period)
  set_float_feature(ex, f"global_view{tag}", view)
  set_float_feature(ex, f"global_std{tag}", std)
  set_float_feature(ex, f"global_mask{tag}", mask)
  set_float_feature(ex, f"global_transit_mask{tag}", tr_mask)

  view, std, mask, scale, depth = preprocess.local_view(tic, time, flux, period,
                                                        duration)
  set_float_feature(ex, f"local_view{tag}", view)
  set_float_feature(ex, f"local_std{tag}", std)
  set_float_feature(ex, f"local_mask{tag}", mask)
  if scale is not None:
    set_float_feature(ex, f"local_scale{tag}", [scale])
    set_float_feature(ex, f"local_scale_present{tag}", [1.0])
  else:
    set_float_feature(ex, f"local_scale{tag}", [0.0])
    set_float_feature(ex, f"local_scale_present{tag}", [0.0])
  for k, (t, f) in aperture_fluxes.items():
    t, f, m = preprocess.detrend_and_filter(tic, t, f, period, epoc, duration,
                                            bkspace)
    t, f, _, _ = preprocess.phase_fold_and_sort_light_curve(
        t, f, m, period, epoc)
    view, std, _, _, _ = preprocess.local_view(
        tic, t, f, period, duration, scale=scale, depth=depth)
    set_float_feature(ex, f"local_aperture_{k}{tag}", view)

  view, std, mask, _, _ = preprocess.local_view(
      tic, time[odds], flux[odds], period, duration, scale=scale, depth=depth)
  set_float_feature(ex, f"local_view_odd{tag}", view)
  set_float_feature(ex, f"local_std_odd{tag}", std)
  set_float_feature(ex, f"local_mask_odd{tag}", mask)

  view, std, mask, _, _ = preprocess.local_view(
      tic, time[evens], flux[evens], period, duration, scale=scale, depth=depth)
  set_float_feature(ex, f"local_view_even{tag}", view)
  set_float_feature(ex, f"local_std_even{tag}", std)
  set_float_feature(ex, f"local_mask_even{tag}", mask)

  # print("Debugging: tic",tic)
  # (_, _, _, sec_scale,
  #  _), t0 = preprocess.secondary_view(tic, time, flux, period, duration)
  # (view, std, mask, scale, _), t0 = preprocess.secondary_view(
  #     tic, time, flux, period, duration, scale=scale, depth=depth)

  try:
      (_, _, _, sec_scale, _), t0 = preprocess.secondary_view(tic, time, flux, period, duration)
      (view, std, mask, scale, _), t0 = preprocess.secondary_view(
          tic, time, flux, period, duration, scale=scale, depth=depth)
  except IndexError as e:
      logging.warning(f"Skipping TIC {tic}: preprocess.secondary_view failed (IndexError).", exc_info=False) # Set exc_info=True to include traceback

      # Log details at a lower level (e.g., DEBUG) if desired
      time_info = f"shape {time.shape}" if hasattr(time, 'shape') else f"length {len(time)}"
      flux_info = f"shape {flux.shape}" if hasattr(flux, 'shape') else f"length {len(flux)}"
      logging.debug(f"  Details for failed TIC {tic}: Error='{e}', Input time info='{time_info}', Input flux info='{flux_info}'")

      SKIPPED_TICS.append(tic)
      return None




  set_float_feature(ex, f"secondary_view{tag}", view)
  set_float_feature(ex, f"secondary_std{tag}", std)
  set_float_feature(ex, f"secondary_mask{tag}", mask)
  set_float_feature(ex, f"secondary_phase{tag}", [t0 / period])
  if sec_scale is not None:
    set_float_feature(ex, f"secondary_scale{tag}", [sec_scale])
    set_float_feature(ex, f"secondary_scale_present{tag}", [1.0])
  else:
    set_float_feature(ex, f"secondary_scale{tag}", [0.0])
    set_float_feature(ex, f"secondary_scale_present{tag}", [0.0])

  full_view = preprocess.sample_segments_view(tic, time, flux, fold_num, period,
                                              duration)
  set_float_feature(ex, f"sample_segments_view{tag}", full_view)

  odd_view = preprocess.sample_segments_view(
      tic,
      time[odds],
      flux[odds],
      fold_num[odds],
      period,
      duration,
      num_bins=61,
      num_transits=4,
      local=True)
  even_view = preprocess.sample_segments_view(
      tic,
      time[evens],
      flux[evens],
      fold_num[evens],
      period,
      duration,
      num_bins=61,
      num_transits=4,
      local=True)
  full_view = np.concatenate([odd_view, even_view], axis=-1)
  set_float_feature(ex, f"sample_segments_local_view{tag}", full_view)

  time, flux, fold_num, _ = preprocess.phase_fold_and_sort_light_curve(
      detrended_time, detrended_flux, transit_mask, period * 2,
      epoc - period / 2)
  view, std, mask, scale, _ = preprocess.global_view(tic, time, flux,
                                                     period * 2)
  set_float_feature(ex, f"global_view_double_period{tag}", view)
  set_float_feature(ex, f"global_view_double_period_std{tag}", std)
  set_float_feature(ex, f"global_view_double_period_mask{tag}", mask)

  time, flux, fold_num, _ = preprocess.phase_fold_and_sort_light_curve(
      detrended_time, detrended_flux, transit_mask, period / 2, epoc)
  view, std, mask, scale, _ = preprocess.global_view(tic, time, flux,
                                                     period / 2)
  set_float_feature(ex, f"global_view_half_period{tag}", view)
  set_float_feature(ex, f"global_view_half_period_std{tag}", std)
  set_float_feature(ex, f"global_view_half_period_mask{tag}", mask)

  view, std, mask, scale, _ = preprocess.local_view(tic, time, flux, period / 2,
                                                    duration)
  set_float_feature(ex, f"local_view_half_period{tag}", view)
  set_float_feature(ex, f"local_view_half_period_std{tag}", std)
  set_float_feature(ex, f"local_view_half_period_mask{tag}", mask)

  return fold_num


def _process_tce(tce, get_lightcurve: LCGetter, mode: AstronetMode,
                 training: bool):
  time, flux = get_lightcurve(tce["Astro ID"])
  if mode == "vetting":
    apertures = {
        "s": get_lightcurve(tce["Astro ID"], aperture="s"),
        "m": get_lightcurve(tce["Astro ID"], aperture="m"),
        "l": get_lightcurve(tce["Astro ID"], aperture="l"),
    }
  else:
    apertures = {}

  ex = tf.train.Example()

  for bkspace in [0.3, 5.0, None]:
    fold_num = _standard_views(ex, tce["TIC ID"], time, flux, tce.Per, tce.Epoc,
                               tce.Dur, bkspace, apertures)
    if fold_num is None:
      # We are skipping this TCE entirely
      return None

  set_int64_feature(ex, "astro_id", [tce["Astro ID"]])

  if training:
    if mode == "vetting":
      set_int64_feature(ex, "disp_e", [tce["disp_e"]])
      set_int64_feature(ex, "disp_p", [tce["disp_p"]])
      set_int64_feature(ex, "disp_n", [tce["disp_n"]])
      set_int64_feature(ex, "disp_b", [tce["disp_b"]])
      set_int64_feature(ex, "disp_t", [tce["disp_t"]])
      set_int64_feature(ex, "disp_u", [tce["disp_u"]])
      set_int64_feature(ex, "disp_j", [tce["disp_j"]])
    elif mode == "triage":
      set_int64_feature(ex, "disp_E", [tce["disp_E"]])
      set_int64_feature(ex, "disp_N", [tce["disp_N"]])
      set_int64_feature(ex, "disp_J", [tce["disp_J"]])
      set_int64_feature(ex, "disp_S", [tce["disp_S"]])
      set_int64_feature(ex, "disp_B", [tce["disp_B"]])
    else:
      raise ValueError(f"Mode '{mode}' not supported.")

  assert not np.isnan(tce.Per)
  set_float_feature(ex, "Period", [tce.Per])

  assert not np.isnan(tce.Dur)
  set_float_feature(ex, "Duration", [tce.Dur])

  assert not np.isnan(tce.Depth)
  set_float_feature(ex, "Transit_Depth", [tce.Depth])

  assert not np.isnan(tce.Tmag)
  set_float_feature(ex, "Tmag", [tce.Tmag])

  # set_float_feature(ex, "centroid_dist", [tce.centroid_dist])

  if np.isnan(tce.SMass):
    set_float_feature(ex, "star_mass", [0])
    set_float_feature(ex, "star_mass_present", [0])
  else:
    set_float_feature(ex, "star_mass", [tce.SMass])
    set_float_feature(ex, "star_mass_present", [1])

  if np.isnan(tce.SRad):
    set_float_feature(ex, "star_rad", [0])
    set_float_feature(ex, "star_rad_present", [0])
  else:
    set_float_feature(ex, "star_rad", [tce.SRad])
    set_float_feature(ex, "star_rad_present", [1])

  if np.isnan(tce.SRadEst):
    set_float_feature(ex, "star_rad_est", [0])
    set_float_feature(ex, "star_rad_est_present", [0])
  else:
    set_float_feature(ex, "star_rad_est", [tce.SRadEst])
    set_float_feature(ex, "star_rad_est_present", [1])

  set_float_feature(ex, "n_folds", [len(set(fold_num))])
  set_float_feature(ex, "n_points", [len(fold_num)])

  return ex


def _process_file_shard(
    tce_table: pd.DataFrame,
    file_name: str,
    get_lightcurve: LCGetter,
    mode: AstronetMode,
    training: bool,
):
  shard_name = os.path.basename(file_name)
  shard_size = len(tce_table)

  existing = {}
  # tfr = tf.data.TFRecordDataset(file_name)
  # for record in tfr:
  #   ex_str = record.numpy()
  #   ex = tf.train.Example.FromString(ex_str)
  #   existing[ex.features.feature["astro_id"].int64_list.value[0]] = ex_str

  if tf.io.gfile.exists(file_name):
    tfr = tf.data.TFRecordDataset(file_name)
    for record in tfr:
        ex_str = record.numpy()
        ex = tf.train.Example.FromString(ex_str)
        existing[ex.features.feature["astro_id"].int64_list.value[0]] = ex_str

  with tf.io.TFRecordWriter(file_name) as writer:
    num_processed = 0
    num_skipped = 0
    num_existing = 0
    print("", end="")
    for _, tce in tce_table.iterrows():
      num_processed += 1
      recid = int(tce["Astro ID"])
      print("\r                                      ", end="")
      print(f"\r[{num_processed}/{shard_size}] {recid}", end="")

      if recid in existing:
        print(" exists", end="")
        sys.stdout.flush()
        writer.write(existing[recid])
        num_existing += 1
        continue

      examples = []
      print(" processing", end="")
      sys.stdout.flush()

      ### OLD version:
      # ex = _process_tce(tce, get_lightcurve, mode, training)
      # examples.append(ex)

      # print(" writing                   ", end="")
      # sys.stdout.flush()
      # for example in examples:
      #   writer.write(example.SerializeToString())

      ### New version:
      ex = _process_tce(tce, get_lightcurve, mode, training)

      # If ex is None, we skip this TCE:
      if ex is None:
        num_skipped += 1
        print(" -> Skipped", end="")
      else:
        # Otherwise, we append it to 'examples'
        examples.append(ex)
      print(" writing                   ", end="")
      sys.stdout.flush()
      # ### [MODIFIED] We only write out the non-None examples:
      for example in examples:
        writer.write(example.SerializeToString())

  num_new = num_processed - num_skipped - num_existing
  print(f"\r{shard_name}: {num_processed}/{shard_size} {num_new} new "
        "{num_skipped} bad            ")


def create(
    tce_table: pd.DataFrame,
    output_dir: str,
    num_shards: int,
    num_processes: int,
    mode: AstronetMode,
    training: bool,
    get_lightcurve: LCGetter,
):
  tf.io.gfile.makedirs(output_dir)
  logging.info(f"Processing {len(tce_table)} TCEs")

  tce_shards: tuple[pd.DataFrame,
                    str] = []  # List of (tce_table_shard, file_name)
  boundaries = np.linspace(0, len(tce_table), num_shards + 1).astype(int)
  for i in range(num_shards):
    start, end = boundaries[i:i + 2]
    tce_shards.append(
        (start, end, os.path.join(output_dir, f"{i+1:05d}-of-{num_shards:05f}")))
  logging.info(f"Processing {len(tce_table)} TCEs in {len(tce_shards)} shards.")

  if num_processes == 1:
    for start, end, file in tce_shards:
      _process_file_shard(tce_table[start:end], file, get_lightcurve, mode,
                          training)
  else:
    with multiprocessing.Pool(num_processes) as pool:
      pool.starmap(
          _process_file_shard,
          [(
              tce_table[start:end],
              file,
              get_lightcurve,
              mode,
              training,
          ) for start, end, file in tce_shards],
      )
  logging.info("Finished processing")


def main(_):
  tf.io.gfile.makedirs(FLAGS.output_dir)

  tce_table = pd.read_csv(FLAGS.input_tce_csv_file, header=0, low_memory=False)

  def get_lightcurve(
      astro_id: int,
      aperture: Optional[str] = None) -> tuple[np.ndarray, np.ndarray]:
    aperture_key_map = {
        "s": "SAP_FLUX_SML",
        "m": "SAP_FLUX_MID",
        "l": "SAP_FLUX_LAG",
        None: "SAP_FLUX",
    }
    matching_tces = tce_table[
        tce_table["Astro ID"] ==
        astro_id]  # tce_table.where(tce_table["Astro ID"] == astro_id)
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

  num_tces = len(tce_table)
  logging.info("Read %d TCEs", num_tces)

  # Further split training TCEs into file shards.
  file_shards = []  # List of (tce_table_shard, file_name).
  boundaries = np.linspace(
      0,
      len(tce_table),
      FLAGS.num_shards + 1,
  ).astype(int)
  for i in range(FLAGS.num_shards):
    start = boundaries[i]
    end = boundaries[i + 1]
    filename = f"{i:05}-of-{FLAGS.num_shards:05}"
    file_shards.append((start, end, os.path.join(FLAGS.output_dir, filename)))

  logging.info("Processing %d total file shards", len(file_shards))
  for start, end, file_shard in file_shards:
    _process_file_shard(tce_table[start:end], file_shard, get_lightcurve,
                        FLAGS.mode, FLAGS.training)
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
  app.run(main)
