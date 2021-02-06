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
from light_curve_util import keplersplinev2


def tmod(t, p, e):
    tmodn = (t % p) - (e % p)
    tmodn = tmodn + p * (tmodn <= -0.5 * p) - p * (tmodn >= 0.5 * p)
    return(tmodn)


PHASE2_T = 2036.2
HC_PHASE1 = 30.0 / 60.0 / 24 / 2
HC_PHASE2 = 10.0 / 60.0 / 24 / 2


def get_overlap(hbw, t):
    hc = HC_PHASE1 if t < PHASE2_T else HC_PHASE2
    return max(0, min(hbw, t + hc) - max(-hbw, t - hc))


def new_binning(time, flux, period, num_bins, t_min, t_max, method='weighted_mean'):
  t = time.copy()
  
  bins_left_edge, step = np.linspace(
      t_min, t_max, num=num_bins, endpoint=False, retstep=True)

  bin_width = step
  hbw = bin_width / 2
  
  bins_center = bins_left_edge + 0.5 * bin_width

  f = np.zeros(num_bins)
  s = np.zeros(num_bins)
  m = np.ones(num_bins)
  for i, b in enumerate(bins_center):
    # time from bin center
    t_c = tmod(t, period, b)
    
    # find which points are within the bin
    bin_mask = abs(t_c) <= hbw + np.where(t_c > PHASE2_T, HC_PHASE2, HC_PHASE1)

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

        # remove outliers
        f_x = f_x[mask]
        in_bin = in_bin[mask]
    
    if not len(f_x):
        m[i] = 0.0
        continue

    if method == 'weighted_mean':
        # get the weight of each time point within the bin
        weight = [get_overlap(hbw, in_bin[j]) / bin_width
                  for j in range(len(in_bin))]
        bin_flux = np.sum(weight * f_x) / np.sum(weight)
        f[i] = bin_flux
    elif method == 'max':
        f[i] = np.max(f_x)

    s[i] = np.std(f_x)

  return f, m, s
