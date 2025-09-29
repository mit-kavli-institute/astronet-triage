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

"""Utility function for smoothing data using a median filter."""
import numpy as np
import os

from light_curve_util import keplersplinev2


def tmod(t, p, e):
    tmodn = (t % p) - (e % p)
    tmodn = tmodn + p * (tmodn <= -0.5 * p) - p * (tmodn >= 0.5 * p)
    return(tmodn)


PHASE2_T = 2036.2
HC_PHASE1 = 30.0 / 60.0 / 24 / 2
HC_PHASE2 = 10.0 / 60.0 / 24 / 2
# TODO(pablomer): This chooses between 30min cadence and 10 min cadence, need to add also the check for EM2 where the cadence is 200s


def get_overlap(hbw, cadence_hw):
    bin_overlap = max(0.0, min(hbw, cadence_hw) - max(-hbw, -cadence_hw))
    return bin_overlap


def new_binning(time, flux, period, num_bins, t_min, t_max, method='weighted_mean', trim_edges=False, raw_time=None, raw_flux=None, scatter_weights=None):
  t = time.copy()
  # Use raw_time for cadence selection if provided, otherwise use folded time
  time_for_cadence = raw_time if raw_time is not None else t

  # Debug controls (non-intrusive): set ASTRONET_DEBUG_BINNING=1 to enable
  DEBUG = os.getenv("ASTRONET_DEBUG_BINNING") == "1"
  if DEBUG:
    total_points_phase1 = 0
    total_points_phase2 = 0
    bins_all_phase1 = 0
    bins_all_phase2 = 0
    bins_mixed = 0
    print('raw_time',raw_time)
    print('t',t)
    print('time for cadence',time_for_cadence)

  bins_left_edge, step = np.linspace(
      t_min, t_max, num=num_bins, endpoint=False, retstep=True)

  bin_width = step
  hbw = bin_width / 2

  bins_center = bins_left_edge + 0.5 * bin_width

  f = np.zeros(num_bins)
  s = np.zeros(num_bins)
  m = np.ones(num_bins)

  for i, b in enumerate(bins_center):
    # time from bin center (use folded time for binning)
    t_c = tmod(t, period, b)

    # find which points are within the bin
    # Use raw_time for cadence selection, but align with folded time indices
    if raw_time is not None:
        cadence_hw = np.where(time_for_cadence > PHASE2_T, HC_PHASE2, HC_PHASE1)
    else:
        cadence_hw = np.where(time_for_cadence > PHASE2_T, HC_PHASE2, HC_PHASE1)

    bin_mask = abs(t_c) <= hbw + cadence_hw

    if DEBUG:
      if np.any(bin_mask):
        used = cadence_hw[bin_mask]
        c1 = np.sum(used == HC_PHASE1)
        c2 = np.sum(used == HC_PHASE2)
        total_points_phase1 += int(c1)
        total_points_phase2 += int(c2)
        if c1 > 0 and c2 == 0:
          bins_all_phase1 += 1
        elif c2 > 0 and c1 == 0:
          bins_all_phase2 += 1
        elif c1 > 0 and c2 > 0:
          bins_mixed += 1

    if not any(bin_mask):
        m[i] = 0.0
        continue

    in_bin = t_c[bin_mask]
    f_x = flux[bin_mask]
    cadence_in_bin = cadence_hw[bin_mask]

    # Extract scatter weights for points in this bin
    scatter_weights_in_bin = None
    if scatter_weights is not None:
        scatter_weights_in_bin = scatter_weights[bin_mask]

    if not len(f_x):
        m[i] = 0.0
        continue

    if len(f_x) == 1:
        f[i] = f_x[0]
        continue

    if method == 'weighted_mean':
        # calculate the robust mean to remove outliers
        mask = keplersplinev2.robust_mean_mask(f_x)

        # remove outliers
        f_x = f_x[mask]
        in_bin = in_bin[mask]
        cadence_in_bin = cadence_in_bin[mask]

        # Also apply the same mask to scatter weights if available
        if scatter_weights_in_bin is not None:
            scatter_weights_in_bin = scatter_weights_in_bin[mask]

    if not len(f_x):
        m[i] = 0.0
        continue

    if method == 'weighted_mean':
        if len(in_bin) > 1:
            # Calculate base weights from cadence overlap
            base_weights = [get_overlap(hbw, cadence_in_bin[j]) / bin_width
                          for j in range(len(in_bin))]

            # Combine with scatter weights if available
            if scatter_weights_in_bin is not None:
                # print('DEBUG: scatter_weights_in_bin',scatter_weights_in_bin)
                # Combine cadence weights with scatter weights
                combined_weights = [base_weights[j] * scatter_weights_in_bin[j]
                                  for j in range(len(in_bin))]
            else:
                combined_weights = base_weights

            f[i] = np.average(f_x, weights=combined_weights)
        else:
            f[i], = f_x
    elif method == 'max':
        f[i] = np.max(f_x)

    s[i] = np.std(f_x)

  if trim_edges:
      clear_bins = set()
      for i in range(len(m)):
        if m[i] < 1:
            if i > 0:
                clear_bins.add(i - 1)
            if i < len(m) - 1:
                clear_bins.add(i + 1)
      for i in list(clear_bins):
        m[i] = 0.0

  if DEBUG:
      print("[new_binning DEBUG] cadence selection summary:")
      print(f"  bins all phase1: {bins_all_phase1}")
      print(f"  bins all phase2: {bins_all_phase2}")
      print(f"  bins mixed:      {bins_mixed}")
      print(f"  points phase1:   {total_points_phase1}")
      print(f"  points phase2:   {total_points_phase2}")

  return f, m, s
