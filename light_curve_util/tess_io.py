# Copyright 2018 Liang Yu.
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

"""Functions for reading TESS data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import glob
import os
import re

import numpy as np
import astropy 
from astropy.io import fits

SECTOR_RE = re.compile('.*-s([^-]+)-.*')


def _sector(f):
    m = SECTOR_RE.match(f);
    if m is None:
        return ''
    return m.group(1)


def tess_filenames(tic, base_dir):
    """Returns the light curve filename for a TESS target star.

    Args:
      tic: TIC of the target star. May be an int or a possibly zero-
          padded string.
      base_dir: Base directory containing Kepler data.
      sector: Int, sector number of data.
      cam: Int, camera number of data.
      ccd: Int, CCD number of data.
      injected: Bool, whether target also has a light curve with injected planets.
      injected_dir: Directory containing light curves with injected transits.
      check_existence: If True, only return filenames corresponding to files that
          exist.

    Returns:
      filename for given TIC.
    """
    fitsfile = "tess*-%.16d-*-cr_llc.fits.gz" % int(tic)
    file_names = glob.glob(os.path.join(base_dir, fitsfile))
    if not file_names:
        fitsfile2 = "tess*-%d-cr_llc.fits.gz" % int(tic)
        file_names = glob.glob(os.path.join(base_dir, fitsfile2))
    if not file_names:
        fitsfile3 = "*tess*-%.16d_*.fits" % int(tic)
        file_names = glob.glob(os.path.join(base_dir, fitsfile3))
    if len(file_names) > 1:
        file_names = sorted(file_names, reverse=True, key=_sector)[:1]
        print(f'multiple matches, selected {file_names}')
    if len(file_names) != 1:
        raise ValueError(f'found {len(file_names)} files for {tic}: {file_names}')
    filename, = file_names

    return filename


def read_tess_light_curve(filename, flux_key):
    """Reads time and flux measurements for a Kepler target star.

    Args:
      filename: str name of fits file containing light curve.
      flux_key: Key of fits column containing flux.

    Returns:
      time: Numpy array; the time values of the light curve.
      flux: Numpy array corresponding to the time array.
    """
    f = fits.open(filename)
    time = (f[1].data["TIME"]).astype(float)
    flux = (f[1].data[flux_key]).astype(float)
    
    if np.max(time) > 1354: 
        quality = f[1].data["QUALITY"]
        quality_flag = quality==0
        
        # Remove outliers
        time = time[quality_flag]
        flux = flux[quality_flag]

        # Remove NaN flux values.
        valid_indices = np.where(np.isfinite(flux))
        time = time[valid_indices]
        flux = flux[valid_indices]
    else:
        # manually remove sector 1 outliers
        bad = np.array([0, 1, 2, 31, 49, 88, 121, 152, 186, 188, 199,
               224, 225, 228, 241, 340, 359, 361, 463, 464, 465, 481,
               482, 483, 546, 547, 583, 584, 598, 599, 600, 601, 602,
               631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641,
               642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652,
               653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663,
               664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674,
               675, 676, 677, 678, 723, 726, 727, 730, 748, 749, 752,
               753, 754, 755, 756, 817, 819, 839, 853, 854, 855, 866,
               872, 873, 874, 875, 969, 971, 977, 987, 992, 993, 994,
               995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1005, 1006,
               1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017,
               1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028,
               1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039,
               1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050,
               1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061,
               1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072,
               1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083,
               1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094,
               1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1108,
               1112, 1113, 1114, 1141, 1175, 1180, 1183, 1191, 1193, 1195, 1196,
               1208, 1209, 1210, 1214, 1225, 1226, 1231, 1232, 1233, 1235, 1258,
               1278, 1279, 1280])

        bad = bad[bad < len(time)]
        mask = np.ones(len(time))
        mask[bad] = 0
        mask = mask.astype(bool)
        time = time[mask]
        flux = flux[mask]

        valid_indices = np.where(np.isfinite(flux))
        time = time[valid_indices]
        flux = flux[valid_indices]


    return time, flux 


