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
PHASE3_T = 2825.25
HC_PHASE1 = 30.0 / 60.0 / 24 / 2
HC_PHASE2 = 10.0 / 60.0 / 24 / 2
HC_PHASE3 = 3.33 / 60.0 / 24 / 2


def get_overlap(hbw, t_centered, half_cadence):
    """
    Overlap between a bin of half-width hbw centered at 0 and a point with
    'reach' given by half_cadence, located at t_centered (already folded
    relative to the bin center).
    """
    return max(0, min(hbw, t_centered + half_cadence) - max(-hbw, t_centered - half_cadence))


def new_binning(time, flux, period, num_bins, t_min, t_max, method='weighted_mean', trim_edges=False, raw_time=None, raw_flux=None, all_30min=True, scatter_weights=None):
  t = time.copy()
  # Use raw_time for cadence selection if provided, otherwise use folded time
  time_for_cadence = raw_time if raw_time is not None else t

  # Debug controls (non-intrusive): set ASTRONET_DEBUG_BINNING=1 to enable
  DEBUG = os.getenv("ASTRONET_DEBUG_BINNING") == "1"
  if DEBUG:
    total_points_phase1 = total_points_phase2 = total_points_phase3 = 0
    bins_all_phase1 = bins_all_phase2 = bins_all_phase3 = 0
    bins_mixed = 0
    print('Shape of t: ', t.shape)
    if raw_time is not None:
      print('Shape of raw_time: ', raw_time.shape)
    else:
      print('raw_time is None')
    # print the number of points between tmin and tmax
    print('Number of points between tmin and tmax: ', len(t[(t >= t_min) & (t <= t_max)]))


  bins_left_edge, step = np.linspace(
      t_min, t_max, num=num_bins, endpoint=False, retstep=True)

  bin_width = step
  hbw = bin_width / 2

  bins_center = bins_left_edge + 0.5 * bin_width

  f = np.zeros(num_bins)
  s = np.zeros(num_bins)
  m = np.ones(num_bins)

# Use raw_time for cadence selection, but align with folded time indices
  if all_30min:
        # Force 30-minute cadence for all points
        cadence_hw = np.full_like(t, HC_PHASE1, dtype=float)
  elif raw_time is not None:
        cadence_hw = np.select(
            [time_for_cadence <= PHASE2_T,
            (time_for_cadence > PHASE2_T) & (time_for_cadence <= PHASE3_T),
            time_for_cadence > PHASE3_T],
            [HC_PHASE1, HC_PHASE2, HC_PHASE3]
        )
  else:
        raise ValueError("Cannot determine cadence: all_30min=False and raw_time=None.")

  point_bin_hits_pre  = np.zeros(len(t), dtype=int)
  point_bin_hits_post = np.zeros(len(t), dtype=int)
  for i, b in enumerate(bins_center):
    # time from bin center (use folded time for binning)
    t_c = tmod(t, period, b)

    # find which points are within the bin


    bin_mask = abs(t_c) <= hbw + cadence_hw

    point_bin_hits_pre[bin_mask] += 1

    if DEBUG:
      if np.any(bin_mask):
        used = cadence_hw[bin_mask]
        c1 = np.sum(used == HC_PHASE1)
        c2 = np.sum(used == HC_PHASE2)
        c3 = np.sum(used == HC_PHASE3)
        total_points_phase1 += int(c1)
        total_points_phase2 += int(c2)
        total_points_phase3 += int(c3)
        if c1 > 0 and c2 == 0 and c3 == 0:
          bins_all_phase1 += 1
        elif c2 > 0 and c1 == 0 and c3 == 0:
          bins_all_phase2 += 1
        elif c3 > 0 and c1 == 0 and c2 == 0:
          bins_all_phase3 += 1
        else:
          bins_mixed += 1

    if not any(bin_mask):
        m[i] = 0.0
        continue

    in_bin = t_c[bin_mask]
    f_x = flux[bin_mask]

    if not len(f_x):
        m[i] = 0.0
        continue

    if len(f_x) == 1:
        f[i] = f_x[0]
        continue

    if method == 'weighted_mean':
        # calculate the robust mean to remove outliers
        mask = keplersplinev2.robust_mean_mask(f_x)

        # indices of original points in this bin
        idx_in_bin = np.where(bin_mask)[0]

        # apply outlier mask to original indices, then count post-outlier hits
        if len(idx_in_bin):
          point_bin_hits_post[idx_in_bin[mask]] += 1

        # remove outliers
        f_x = f_x[mask]
        in_bin = in_bin[mask]
        cad_in = cadence_hw[bin_mask][mask]

    if not len(f_x):
        m[i] = 0.0
        continue

    if method == 'weighted_mean':
        if len(in_bin) > 1:
            # Use raw_time for cadence in get_overlap too
            # weight = [get_overlap(hbw, in_bin[j], b) / bin_width
            overlap_weight = [get_overlap(hbw, in_bin[j], cad_in[j]) / bin_width
                      for j in range(len(in_bin))]

            # Combine overlap weights with scatter weights if provided
            if scatter_weights is not None:
                # Get scatter weights for points in this bin (after outlier removal)
                scatter_w = scatter_weights[bin_mask][mask]
                # Combine weights: multiply overlap weight by scatter weight
                weight = [overlap_weight[j] * scatter_w[j] for j in range(len(in_bin))]
            else:
                weight = overlap_weight

            f[i] = np.average(f_x, weights=weight)
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
    print(f"  bins all phase3: {bins_all_phase3}")
    print(f"  bins mixed:      {bins_mixed}")
    print(f"  points phase1:   {total_points_phase1}")
    print(f"  points phase2:   {total_points_phase2}")
    print(f"  points phase3:   {total_points_phase3}")


    two_plus_pre  = int(np.sum(point_bin_hits_pre  >= 2))
    two_plus_post = int(np.sum(point_bin_hits_post >= 2))
    print(f"[new_binning DEBUG] points in ≥2 bins (pre-outlier):  {two_plus_pre}")
    print(f"[new_binning DEBUG] points in ≥2 bins (post-outlier): {two_plus_post}")

    three_plus_pre  = int(np.sum(point_bin_hits_pre  >= 3))
    three_plus_post = int(np.sum(point_bin_hits_post >= 3))
    print(f"[new_binning DEBUG] points in ≥3 bins (pre-outlier):  {three_plus_pre}")
    print(f"[new_binning DEBUG] points in ≥3 bins (post-outlier): {three_plus_post}")

    # (Optional) show full histogram: how many points hit 0,1,2,3,... bins
    hist_pre  = np.bincount(point_bin_hits_pre)
    hist_post = np.bincount(point_bin_hits_post)
    print(f"[new_binning DEBUG] hits histogram pre : {dict(enumerate(hist_pre))}")
    print(f"[new_binning DEBUG] hits histogram post: {dict(enumerate(hist_post))}")

  return f, m, s
