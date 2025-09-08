"""Astronet H5 -> FITS tool

This command line tool produces FITS files for astronet consumption.

For an input file with TIC IDs, we look for orbit LCs on pdo (up to a given sector limit), merge the
light curves, and write them to a FITS file in the provided output directory.

The FITS files primarily expose the following columns (which are read by astronet):
    * TIME
    * QUALITY
    * SAP_FLUX
    * SAP_FLUX_SML
    * SAP_FLUX_MID
    * SAP_FLUX_LAG

Other data / header information is just auxiliary and may potentially be useful for tracking.

The original code was adapted from lctools-lctomaster-h5.py from FFITools 0.9.2

Example: python3 h5_to_fits.py -i tic.ls -o output/ -s 27
"""

import argparse
import itertools as it
import logging
import os
from collections import namedtuple
from datetime import datetime
from multiprocessing import Pool
from typing import Any, Dict, List

# Hack to stop numpy multithreading on top of multiprocessing
os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
    NUMEXPR_MAX_THREADS="1",
    MKL_NUM_THREADS="1",
)

import numpy as np
import pandas as pd
import pyticdb

from astropy.io import fits
from lightcurvedb import db
from qlp.lctools.hdf5lc import HDFLightCurve
from qlp.util.util import orbits_from_sector, time_correction

N_APERTURES = 5


def cmd_parse():
    p = argparse.ArgumentParser()
    p.add_argument("--inlist", "-i", help="inlist. file with 1 TIC per line")
    p.add_argument("--outdir", "-o", help="output directory to place files")
    p.add_argument("--sector", "-s", type=int, help="last sector to search")
    p.add_argument("--nprocs", "-n", type=int, default=3, help="# of processors to use, default=3")
    p.add_argument(
        "--tree",
        "-t",
        action="store_true",
        help="store in directory tree to avoid too many files in one directory",
    )
    p.add_argument("--overwrite", "-r", action="store_true", help="replace existing files")
    p.add_argument("--debug", action="store_true", help="log in debug mode")

    return p.parse_args()


def construct_fitsfile(outdir: str, sector: int, tic: int, tree: bool = False) -> str:
    fitsfile = "astronet_hlsp_qlp_tess_ffi-s%.4d-%.16d_tess_v01_llc.fits" % (sector, tic)
    if tree:
        # If we're making lots of fits files, make a directory tree using bits of the TIC ID so we
        # don't have to put too many files in one directory.
        dirs = [f"{tic:016d}"[i : i + 4] for i in range(0, 16, 4)]
        full_dir = os.path.join(outdir, *dirs)
        if not os.path.exists(full_dir):
            os.makedirs(full_dir)
        return os.path.join(full_dir, fitsfile)
    else:
        return os.path.join(outdir, fitsfile)


OrbitLCLocation = namedtuple("OrbitLCLocation", ("sector", "orbit", "cam", "ccd", "lcfile"))


def get_orbit_lc_locations(tic: int, last_sector: int) -> List[OrbitLCLocation]:
    orbit_lc_locations: List[OrbitLCLocation] = []
    for sector in range(1, last_sector + 1):
        for orbit in orbits_from_sector(sector):
            for cam, ccd in it.product([1, 2, 3, 4], repeat=2):
                lcfile = f"/pdo/qlp-data/orbit-{orbit}/ffi/cam{cam}/ccd{ccd}/LC/{tic}.h5"
                if os.path.exists(lcfile):
                    orbit_lc_locations.append(OrbitLCLocation(sector, orbit, cam, ccd, lcfile))
    return orbit_lc_locations


def merge_lcs(
    tic: int, orbit_lc_locations: List[OrbitLCLocation], tmag: float, ra: float, dec: float
) -> Dict[str, Any]:
    bjd_tdb = []
    cadence = []
    qflag = []
    raw_mags: Dict[str, List[float]] = {f"Aperture_{i:03d}": [] for i in range(N_APERTURES)}
    for orbit_lc_location in orbit_lc_locations:
        sector, orbit, cam, ccd, lcfile = orbit_lc_location
        qflag_file = np.loadtxt(f"/pdo/qlp-data/qflagpath/orbit{orbit}cam{cam}ccd{ccd}_qflag.txt")
        orbit_qflags = qflag_file[:, 1]
        # qflag cadences are assumed to match orbit_lc.data["cadence"]
        # TODO: we should verify this assumption

        orbit_lc = HDFLightCurve(name=lcfile)
        orbit_lc.load_basic_info()
        # N.B. Year 1 LCs have improperly corrected jd arrays, so until they get fixed, let's
        # reconstruct the jd array on the fly
        if orbit <= 34:
            # HACK: in multiprocessing, this opens a connection for every worker. As long as these
            # die quickly, things should be okay.
            with db:
                mid_tjd = db.query_frames_by_orbit(orbit, cam)["mid_tjd"]
            bjd = time_correction(orbit, mid_tjd, ra=ra, dec=dec)
            orbit_lc.data["jd"] = bjd
        bjd_tdb += list(orbit_lc.data["jd"])
        cadence += list(orbit_lc.data["cadence"])
        qflag += list(orbit_qflags)

        for i in range(N_APERTURES):
            orbit_lc.load_from_file(ap=i, label="all")
            raw_mags[f"Aperture_{i:03d}"] += list(
                orbit_lc.data["rlc"]
                - (np.nanmedian(orbit_lc.data["rlc"][np.array(orbit_qflags) == 0]) - tmag)
            )

    data: Dict[str, Any] = {}
    # Make sure everything is sorted chronologically by cadence. We know O61 (S27) is not properly
    # sorted, but haven't fully checked everything.
    idxsort = np.argsort(np.array(cadence))
    data["bjd"] = np.array(bjd_tdb)[idxsort]
    data["cadence"] = np.array(cadence)[idxsort]
    data["flag"] = np.array(qflag)[idxsort]
    for i in range(N_APERTURES):
        data[f"Aperture_{i:03d}"] = np.array(raw_mags[f"Aperture_{i:03d}"])[idxsort]
    return data


def get_bestap(tmag: float) -> int:
    magbins = np.array([6, 7, 8, 9, 10, 11, 12])
    bestaps = np.array([4, 3, 3, 2, 2, 2, 1])
    index = np.searchsorted(magbins, tmag)
    if index == 0:
        bestap = bestaps[0]
    elif index >= len(magbins):
        bestap = bestaps[-1]
    else:
        if tmag > magbins[index] - 0.5:
            bestap = bestaps[index]
        else:
            bestap = bestaps[index - 1]
    return bestap


def flux_from_mag(mag, ref_mag):
    return 10 ** (-0.4 * (mag - ref_mag))


def h5tofits(tic: int, job_no: int) -> None:
    logger.info("Job #{}/{}: Converting TIC-{} from h5 to fits".format(job_no, ntics, tic))

    fitspath = construct_fitsfile(args.outdir, args.sector, tic, args.tree)
    logger.debug(fitspath)
    if not args.overwrite and os.path.exists(fitspath):
        logger.warning(f"{tic} fits already exists. Use -r to overwrite.")
        return

    orbit_lc_locations = get_orbit_lc_locations(tic, args.sector)
    logger.info(f"Found {tic} in orbits: {[o.orbit for o in orbit_lc_locations]}")
    logger.debug(orbit_lc_locations)

    # This makes a query for each TIC. If it becomes limiting, we can make one bulk query
    # The original HLSP code uses ra_orig and dec_orig. For high PM stars, this is more accurate,
    # but the TIC doesn't always have these, so we'll just default to ra, dec (this might be an
    # existing problem with the HLSP delivery code?)
    field_list = ["tmag","ra","dec"]
    db_result = pyticdb.query_by_id(tic, *field_list)[0]
    tic_info = dict(zip(field_list, db_result))
    logger.debug(tic_info)

    # lcdata = merge_lcs(tic, orbit_lc_locations, tic_info["tmag"], tic_info["ra"], tic_info["dec"])
    # try:
    #     lcdata = merge_lcs(tic, orbit_lc_locations, tic_info["tmag"], tic_info["ra"], tic_info["dec"])
    # except FileNotFoundError as e:
    #     logger.error(f"FileNotFoundError for TIC={tic}. Could not load qflag file. Original error: {e}")
    #     raise  # re-raise so the Pool sees it and stops, but your log will show which TIC caused it
    try:
        lcdata = merge_lcs(tic, orbit_lc_locations, tic_info["tmag"], tic_info["ra"], tic_info["dec"])
    except FileNotFoundError as e:
        logger.error(f"[SKIPPING] TIC={tic} because file was not found: {e}")
        return  # Stop here, skip this TIC, but don't blow up the entire job


    logger.debug(lcdata)

    logger.info("Writing %s to %s" % (tic, fitspath))
    hdu1 = fits.PrimaryHDU()
    hdu1.header.set("NEXTEND", value=1, comment="number of standard extensions")
    hdu1.header.set("EXTNAME", value="PRIMARY", comment="name of extension")
    hdu1.header.set("TICID", value=int(tic), comment="unique TESS target identifier")
    hdu1.header.set("SECTOR", value=int(args.sector), comment="last observed sector")
    hdu1.header.set("DATE", value=datetime.today().strftime("%Y-%m-%d"), comment="Date created")

    cols = []
    cols.append(fits.Column(name="TIME", format="D", array=lcdata["bjd"], unit="BJD-2457000, days"))
    cols.append(fits.Column(name="CADENCENO", format="J", array=lcdata["cadence"]))

    bestap = get_bestap(tic_info["tmag"])
    bestap_flux = flux_from_mag(lcdata[f"Aperture_{bestap:03d}"], tic_info["tmag"])
    cols.append(fits.Column(name="SAP_FLUX", format="E", array=bestap_flux))

    # Based on old LCs generated by Chelsea, they don't appear to have SPOC flags, so we only use
    # QLP flags (unlike the typical HLSP prepration)
    cols.append(fits.Column(name="QUALITY", format="J", array=lcdata["flag"]))

    # While the best aperture is chosen per tmag, small, medium, and large aps are fixed to 1, 2, 3
    for i, size in zip([1, 2, 3], ["SML", "MID", "LAG"]):
        flux = flux_from_mag(lcdata[f"Aperture_00{i}"], tic_info["tmag"])
        cols.append(fits.Column(name=f"SAP_FLUX_{size}", format="E", array=flux))

    hdu2 = fits.BinTableHDU().from_columns(fits.ColDefs(cols))
    hdu2.header.set("INHERIT", value="T", comment="inherit the primary header")
    hdu2.header.set("EXTNAME", value="LIGHTCURVE", comment="name of extension")
    hdu2.header.set("BESTAP", value=bestap, comment="the best aperture index (0 to 4)")
    new_hdul = fits.HDUList([hdu1, hdu2])
    new_hdul.writeto(fitspath, overwrite=True)


if __name__ == "__main__":
    args = cmd_parse()
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO" if not args.debug else "DEBUG")

    tics = np.loadtxt(args.inlist, dtype=int, ndmin=1)
    logger.info("Read {} TICs from {}".format(len(tics), args.inlist))

    ntics = len(tics)
    tasks = [[tic, i + 1] for i, tic in enumerate(tics)]
    with Pool(args.nprocs) as pool:
        pool.starmap(h5tofits, tasks)
