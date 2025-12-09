# Copyright 2018 The TensorFlow Authors.
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
"""Functions for reading and preprocessing light curves."""

import os

import warnings

import numpy as np

from light_curve_util import keplersplinev2, median_filter2, tess_io, util

def robust_std(flux):
    ''' Calculates an estimate of the standard deviation robust to outliers'''
    return np.median(np.abs(flux[1:]-flux[:-1]))*1.48 / np.sqrt(2)

def split_and_calculate_weights(time, flux, gap_width=2):
    """Split the time and flux whenever there is a gap in the time array.
    For each segment, calculate the scatter of the flux values and return
    an array of weights where the higher scatter values have lower weights.

    Args:
        time: 1D array of time values
        flux: 1D array of flux values
        period: The period of the event (in days)
        num_bins: The number of intervals to divide the time axis into
        t_min: The inclusive leftmost value to consider on the time axis
        t_max: The exclusive rightmost value to consider on the time axis
        gap_width: Minimum gap size (in time units) for a split

    Returns:
        weights: Array of weights for each data point
    """
    import numpy as np
    from light_curve_util.keplersplinev2 import split

    # # Split the data into segments based on gaps
    split_times, split_fluxes = split(time, flux, gap_width)
    #instead, split every 10 points
    # split_times = []
    # split_fluxes = []
    # for i in range(0, len(time), 500):
    #     split_times.append(time[i:i+500])
    #     split_fluxes.append(flux[i:i+500])

    # Initialize weights array
    weights = np.ones(len(time))

    # Calculate scatter for each segment
    segment_scatters = []
    for seg_flux in split_fluxes:
        if len(seg_flux) > 1:
            segment_scatters.append(robust_std(seg_flux))
        else:
            segment_scatters.append(0.0)  # Single points get scatter 0

    # Calculate weights based on inverse scatter
    if len(segment_scatters) > 1 and np.max(segment_scatters) > 0:
        # Normalize scatters to [0, 1] range
        max_scatter = np.max(segment_scatters)
        normalized_scatters = np.array(segment_scatters) / max_scatter

        # Weight is inversely proportional to normalized scatter
        # Add small value to avoid division by zero
        segment_weights = 1.0 / (normalized_scatters**2 + 1e-2)

        # Apply weights to each segment
        start_idx = 0
        for i, (seg_time, seg_flux) in enumerate(zip(split_times, split_fluxes)):
            end_idx = start_idx + len(seg_time)
            weights[start_idx:end_idx] = segment_weights[i]
            start_idx = end_idx

    return weights

def read_and_process_light_curve(tess_data_dir, flux_key, filename, min_t,
                                 max_t):
  filename = os.path.join(tess_data_dir, filename)
  all_time, all_mag = tess_io.read_tess_light_curve(filename, flux_key)

  mask = np.logical_and(all_time >= min_t, all_time <= max_t)
  all_time = all_time[mask]
  all_mag = all_mag[mask]

  assert len(all_time)
  return all_time, all_mag

def remove_random_datapoints(time,flux,fraction_to_remove,seed=None):
    """
    Randomly select a fraction of the data points to remove.
    """
    rng = np.random.default_rng(seed)
    num_to_remove = int(fraction_to_remove * len(time))
    indices_to_remove = rng.choice(len(time), size=num_to_remove, replace=False)
    mask = np.ones(len(time), dtype=bool)
    mask[indices_to_remove] = False
    return time[mask], flux[mask]

def get_spline_mask(time, period, t0, tdur):
  phase, _ = util.phase_fold_time(time, period, t0)
  outtran = (np.abs(phase) > (tdur / 2))
  return outtran


def filter_outliers(time, flux, mask):
  valid = ~np.isnan(flux)
  return time[valid], flux[valid], mask[valid]


def detrend_and_filter(tic_id, time, flux, period, epoch, duration,
                       fixed_bkspace):
  del tic_id  # Unused.
  input_mask = get_spline_mask(time, period, epoch, duration)
  spline_flux = keplersplinev2.choosekeplersplinev2(
      time, flux, input_mask=input_mask, fixed_bkspace=fixed_bkspace)
  detrended_flux = flux / spline_flux
  return filter_outliers(time, detrended_flux, input_mask)


def phase_fold_and_sort_light_curve(time, flux, mask, period, t0):
  if not np.size(time):
    return np.array([]), np.array([]), np.array([]), np.array([])

  # Phase fold time.
  time, fold_num = util.phase_fold_time(time, period, t0)

  # Sort by ascending time.
  sorted_i = np.argsort(time)
  time = time[sorted_i]
  flux = flux[sorted_i]
  mask = mask[sorted_i]
  fold_num = fold_num[sorted_i]

  return time, flux, fold_num, mask

def align_raw_time(detr_t, detr_f, period, epoch):
  folded_abs_time, _ = util.phase_fold_time(detr_t, period, epoch)
  sort_idx = np.argsort(folded_abs_time)
  raw_time_aligned = detr_t[sort_idx]
  raw_flux_aligned = detr_f[sort_idx]  # optional, if you also want raw_flux
  return raw_time_aligned, raw_flux_aligned

def align_scatter_weights(detr_t, period, epoch, scatter_weights):
  folded_abs_time,_ = util.phase_fold_time(detr_t, period, epoch)
  sort_idx = np.argsort(folded_abs_time)
  weights_aligned = scatter_weights[sort_idx]
  return weights_aligned


def generate_view(
    tic_id,
    time,
    flux,
    period,
    num_bins,
    t_min,
    t_max,
    normalize=True,
    binning=None,
    trim_edges=False,
    scale=None,
    depth=None,
    raw_time=None,
    raw_flux=None,
    all_30min=True,
    scatter_weights=None,
):
  """Generates a view of a phase-folded light curve using a median filter.

  Args:
    time: 1D array of time values, phase folded and sorted in ascending order.
    flux: 1D array of flux values.
    num_bins: The number of intervals to divide the time axis into.
    t_min: The inclusive leftmost value to consider on the time axis.
    t_max: The exclusive rightmost value to consider on the time axis.
    normalize: Whether to center the median at 1 and minimum value at 0.

  Returns:
    1D NumPy array of size num_bins containing the median flux values of
    uniformly spaced bins on the phase-folded time axis.
  """
  del tic_id  # Unused.
  if binning is None:
    view, mask, std = median_filter2.new_binning(
        time, flux, period, num_bins, t_min, t_max, trim_edges=trim_edges, raw_time=raw_time, raw_flux=raw_flux, all_30min=all_30min, scatter_weights=scatter_weights)
  else:
    view, mask, std = median_filter2.new_binning(
        time,
        flux,
        period,
        num_bins,
        t_min,
        t_max,
        method=binning,
        trim_edges=trim_edges,
        raw_time=raw_time,
        raw_flux=raw_flux,
        all_30min=all_30min,
        scatter_weights=scatter_weights)

  if normalize:
    # Normalization places:
    #  * the minimum value at -1.0
    #  * the median at 0.0
    # This assumes the median holds the out-of-transit average value, so that
    # negative values are transit-like and positive values overshoots.
    # TODO: Use mean(50%ile) instead?
    bool_mask = mask > 0
    if any(bool_mask):
      if depth is None:
        depth = np.min(view[bool_mask])
      view = np.where(bool_mask, view - depth, view)
      if scale is None:
        scale = np.abs(np.median(view[bool_mask]))
      if scale > 0:
        view /= scale
        std /= scale
      view -= 1.0
      view = np.where(bool_mask, view, 0.0)
      std = np.where(bool_mask, std, 0.0)
    else:
      scale = None

  return view, std, mask, scale, depth


def global_view(tic_id, time, flux, period, num_bins=201, raw_time=None, raw_flux=None, all_30min=True, scatter_weights=None):
  """Generates a 'global view' of a phase folded light curve.

  See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
  http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta

  Args:
    time: 1D array of time values, sorted in ascending order.
    flux: 1D array of flux values.
    period: The period of the event (in days).
    num_bins: The number of intervals to divide the time axis into.

  Returns:
    1D NumPy array of size num_bins containing the median flux values of
    uniformly spaced bins on the phase-folded time axis.
  """



  return generate_view(
      tic_id,
      time,
      flux,
      period,
      num_bins=num_bins,
      t_min=-period / 2,
      t_max=period / 2,
      raw_time=raw_time,
      raw_flux=raw_flux,
      all_30min=all_30min,
      scatter_weights=scatter_weights)


def tr_mask_view(tic_id, time, tr_mask, period, num_bins=201, all_30min=True, raw_time=None, raw_flux=None, scatter_weights=None):
  return generate_view(
      tic_id,
      time,
      1 - tr_mask,
      period,
      num_bins=num_bins,
      t_min=-period / 2,
      t_max=period / 2,
      normalize=False,
      binning='max',
      raw_time=raw_time,
      raw_flux=raw_flux,
      all_30min=all_30min,
      scatter_weights=scatter_weights)


def local_view(tic_id,
               time,
               flux,
               period,
               duration,
               num_bins=61,
               num_durations=2,
               scale=None,
               depth=None,
               all_30min=True,
               raw_time=None,
               raw_flux=None,
               scatter_weights=None):
  """Generates a 'local view' of a phase folded light curve.
  See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
  http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta
  Args:
    time: 1D array of time values, sorted in ascending order.
    flux: 1D array of flux values.
    period: The period of the event (in days).
    duration: The duration of the event (in days).
    num_bins: The number of intervals to divide the time axis into.
    num_durations: The number of durations to consider on either side of 0 (the
        event is assumed to be centered at 0).
  Returns:
    1D NumPy array of size num_bins containing the median flux values of
    uniformly spaced bins on the phase-folded time axis.
  """
  return generate_view(
      tic_id,
      time,
      flux,
      period,
      num_bins=num_bins,
      t_min=max(-period / 2, -duration * num_durations),
      t_max=min(period / 2, duration * num_durations),
      scale=scale,
      depth=depth,
      all_30min=all_30min,
      raw_time=raw_time,
      raw_flux=raw_flux,
      scatter_weights=scatter_weights
  )


def mask_transit(time, duration, period, mask_width=2, phase_limit=0.1):
  mask = [(abs(t) > duration * mask_width / 2) and
          (abs(t) > period * phase_limit) for t in time]
  return np.array(mask)


def find_secondary(time, flux, duration, period, mask_width=2, phase_limit=0.1, raw_time=None):
  """Mask out transits, rearrange LC such that time goes from 0 to period. Then
  perform grid search for most likely secondary eclipse. To be called after
  preprocess.phase_fold_and_sort_light_curve. OOT flux should be 1.
    :param time: 1D array of time values, folded and sorted in ascending order,
        with the transit located at time 0.
    :param flux: 1D array of fluxes.
    :param duration: The duration of the event (in days).
    :param period: the period of the event (in days).
    :param mask_width: number of durations to mask out.
    :param phase_limit: minimum phase to search for secondary eclipse.
    :param raw_time: 1D array of raw time values corresponding to the folded time.
    :return: time of centre of most likely secondary.
  """
  if period < 1:
    mask_width = 1

  mask = mask_transit(time, duration, period, mask_width, phase_limit)
  if not any(mask):
    mask = mask_transit(time, duration, period, mask_width / 2, phase_limit)
    if not any(mask):
      mask = mask_transit(time, duration, period, mask_width / 2,
                          phase_limit / 10)

  new_time = time[mask]
  new_flux = flux[mask]
  new_raw_time = raw_time[mask] if raw_time is not None else None

  # rearrange so that time goes from 0 to period
  new_time[new_time < 0] += period
  new_index = np.argsort(new_time)
  new_time = new_time[new_index]
  new_flux = new_flux[new_index]
  if new_raw_time is not None:
    new_raw_time = new_raw_time[new_index]
  new_flux -= 1.  # centre flux at zero

  # grid search for secondary. Fix duration to duration of primary.
  time_grid = np.arange(new_time[0] + duration, new_time[-1] - duration,
                        duration * 0.1)
  min_index = 0
  max_index = min_index
  best_t0 = period / 2
  best_sr = 0

  for t0 in time_grid:
    while new_time[min_index] < (t0 - duration):
      min_index += 1
    min_in_transit = min_index
    max_in_transit = min_in_transit
    while (new_time[max_index] < (t0 + duration)) and (max_index
                                                       < len(new_time)):
      max_index += 1
    while new_time[min_in_transit] < (t0 - duration / 2):
      min_in_transit += 1
    while new_time[max_in_transit] < (t0 + duration / 2):
      max_in_transit += 1
    if max_index - min_index < 5:
      continue
    r = float(max_in_transit - min_in_transit + 1) / len(
        new_time)  # assuming identical uniform weights
    s = sum(new_flux[min_in_transit:max_in_transit] / float(len(new_time)))

    sr = s**2 / (r * (1 - r))
    if sr > best_sr:
      best_t0 = t0
      best_sr = sr
  return best_t0, new_time, new_flux + 1., new_raw_time


def secondary_view(tic_id,
                   time,
                   flux,
                   period,
                   duration,
                   num_bins=61,
                   num_durations=2,
                   scale=None,
                   depth=None,
                   all_30min=True,
                   raw_time=None,
                   raw_flux=None):
  """Generates a 'local view' of a phase folded light curve, centered on phase
  0.5. See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
  http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta
  Args:
    time: 1D array of time values, sorted in ascending order, with the transit
        located at time 0.
    flux: 1D array of flux values.
    period: The period of the event (in days).
    duration: The duration of the event (in days).
    num_bins: The number of intervals to divide the time axis into.
    num_durations: The number of durations to consider on either side of 0 (the
        event is assumed to be centered at 0).
  Returns:
    1D NumPy array of size num_bins containing the median flux values of
    uniformly spaced bins on the phase-folded time axis.
  """

  # Only use raw_time when all_30min=False
  if all_30min is True:
        # Force 30-minute cadence, don't use raw_time
        if raw_time is not None:
            raise ValueError("Cannot use raw_time when all_30min=True. Set all_30min=False to use raw_time for cadence selection.")
        raw_time = None
        raw_flux = None

  if len(time):
    t0, new_time, new_flux, new_raw_time = find_secondary(time, flux, duration, period, raw_time=raw_time)
    t_min = max(t0 - period / 2, t0 - duration * num_durations, new_time[0])
    t_max = min(t0 + period / 2, t0 + duration * num_durations, new_time[-1])
  else:
    t0, new_time, new_flux, new_raw_time = 0.0, time, flux, raw_time
    t_min = 0.0
    t_max = 0.0

  return (
      generate_view(
          tic_id,
          new_time,
          new_flux,
          period,
          num_bins=num_bins,
          t_min=t_min,
          t_max=t_max,
          scale=scale,
          depth=depth,
          all_30min=all_30min,
          raw_time=new_raw_time,
          raw_flux=raw_flux),
      t0,
  )


def sample_segments(time, flux, fold_num, num_transits):
  if not np.size(time):
    return [], [], []

  n_folds = max(fold_num) + 1
  fold_size = [np.count_nonzero(fold_num == i) for i in range(n_folds)]
  # Add a small amount of noise to break ties between equally sized folds.
  sort_indicator = [fs + np.random.uniform(0.5) for fs in fold_size]
  sorted_fold_num = np.flip(np.argsort(sort_indicator))
  fold_nums = sorted_fold_num[:num_transits]

  times = []
  fluxes = []
  for i in fold_nums:
    times.append(time[fold_num == i])
    fluxes.append(flux[fold_num == i])
  return times, fluxes, fold_nums


def sample_segments_view(tic_id,
                         time,
                         flux,
                         fold_num,
                         period,
                         duration,
                         num_bins=201,
                         num_transits=7,
                         local=False,
                         all_30min=True,
                         raw_time=None,
                         raw_flux=None):

  if all_30min is False:
    warnings.warn(
            "sample_segments_view: all_30min=False is not implemented yet; "
            "falling back to all_30min=True (30-min cadence). In fact, this shouldn't change behavior at all",
            category=UserWarning,
            stacklevel=2,
        )
    all_30min = True
    raw_time = None
    raw_flux = None

  times, fluxes, nums = sample_segments(
      time, flux, fold_num, num_transits=num_transits)
  full_view = []
  for t, f, n in zip(times, fluxes, nums):
    t_min = period / 2
    t_max = period / 2
    if local:
      t_min = max(t_min, 2 * duration)
      t_max = min(t_max, 2 * duration)
    view, _, mask, _, _ = generate_view(
        tic_id,
        t,
        f,
        period,
        num_bins=num_bins,
        t_min=period * n - t_min,
        t_max=period * n + t_min,
        normalize=False,
        trim_edges=True,
        all_30min=all_30min,
        raw_time=None,
        raw_flux=None
    )
    full_view.append(view)
    full_view.append(mask)

  for _ in range(num_transits - len(times)):
    full_view.append(np.zeros([num_bins], dtype=float))
    full_view.append(np.zeros([num_bins], dtype=float))

  # values in channel i, mask in channel i + 1
  return np.stack(full_view, axis=-1)
