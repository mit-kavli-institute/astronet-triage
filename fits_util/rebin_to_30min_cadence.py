import argparse
import os
from pathlib import Path
from astropy.io import fits
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


STANDARD_MIN = np.array([0.3333, 2.0, 3.3333, 10.0, 30.0])  # 20s, 2m, 200s, 10m, 30m
class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

def _nearest_standard(min_val, tol_frac=0.08):
    """Map a cadence (min) to nearest standard if within tol_frac (e.g., 8%)."""
    idx = np.argmin(np.abs(STANDARD_MIN - min_val))
    nearest = float(STANDARD_MIN[idx])
    return (nearest if abs(nearest - min_val) <= tol_frac * nearest else None), nearest

def _robust_stats(x):
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sig = 1.4826 * mad if mad > 0 else (np.std(x) if x.size > 1 else 0.0)
    return med, sig

def rebin_cadence(cad, bin_indices):
    """
    cad: original CADENCENO array
    bin_indices: integer array mapping each original point to its rebinned bin
    Returns array of rebinned CADENCENO (median per bin)
    """
    uniq_bins = np.unique(bin_indices)
    cad_rebinned = np.empty(len(uniq_bins), dtype=int)
    for i, b in enumerate(uniq_bins):
        cad_rebinned[i] = int(np.median(cad[bin_indices == b]))
    return cad_rebinned

def rebin_quality(quality, bin_indices):
    uniq_bins = np.unique(bin_indices)
    quality_rebinned = np.empty(len(uniq_bins), dtype=int)
    for i, b in enumerate(uniq_bins):
        quality_rebinned[i] = np.max(quality[bin_indices == b])
    return quality_rebinned

def _segment_by_cadence(bjd, gap_minutes=180.0, tol_frac=0.08, min_run=3):
    """
    Segment by mapping each delta to nearest standard cadence; long gaps split segments.
    Returns list of (i_start, i_end_inclusive, detected_min, err_min, classified_min, n_deltas).
    Indices refer to the *point indices* in bjd (not the delta indices).
    """
    t = np.asarray(bjd, float)
    order = np.argsort(t)
    t = t[order]
    dt_min = np.diff(t) * 1440.0
    n = t.size

    if n < 3:
        return []

    # classify each delta as a cadence label or 'gap'
    labels = []
    for d in dt_min:
        if not np.isfinite(d) or d <= 0:
            labels.append(("bad", None))
            continue
        if d > gap_minutes:
            labels.append(("gap", None))
            continue
        classified, nearest = _nearest_standard(d, tol_frac=tol_frac)
        if classified is None:
            # non-standard small gap => mark as 'other' so we can split
            labels.append(("other", nearest))
        else:
            labels.append((f"cad_{classified:.4g}", classified))

    # walk through labels to form segments when label changes or gap occurs
    segs = []
    i0 = 0
    current = labels[0][0]

    for k in range(1, len(labels)):
        if labels[k][0] != current or current in ("gap", "bad"):
            # close segment over deltas [i0..k-1] => points [i0 .. k]
            if current.startswith("cad_"):
                i_start = i0
                i_end = k  # inclusive point index
                dts = dt_min[i_start:i_end]  # these are the deltas inside segment
                if dts.size >= max(1, min_run - 1):
                    med, sig = _robust_stats(dts)
                    classified, _ = _nearest_standard(med, tol_frac=tol_frac)
                    segs.append((i_start, i_end, med, sig, classified, dts.size))
            # start new
            i0 = k
            current = labels[k][0]

    # close final segment (if cadence)
    if current.startswith("cad_"):
        i_start = i0
        i_end = len(labels)  # last delta index = n-2, end point index = n-1
        dts = dt_min[i_start:i_end]
        if dts.size >= max(1, min_run - 1):
            med, sig = _robust_stats(dts)
            classified, _ = _nearest_standard(med, tol_frac=tol_frac)
            segs.append((i_start, i_end, med, sig, classified, dts.size))

    # Convert back to original unsorted indices for users if needed (we’ll keep sorted for simplicity)
    # Also coalesce adjacent segments that classify to the same cadence and touch.
    merged = []
    for s in segs:
        if not merged:
            merged.append(list(s))
        else:
            prev = merged[-1]
            if (prev[1] == s[0]) and (prev[4] == s[4]):  # contiguous & same classified cadence
                # merge: extend end and recompute stats over combined deltas
                new_i0 = prev[0]
                new_i1 = s[1]
                dts = dt_min[new_i0:new_i1]
                med, sig = _robust_stats(dts)
                merged[-1] = [new_i0, new_i1, med, sig, prev[4], dts.size]
            else:
                merged.append(list(s))
    # build report
    report = []
    for (i_start, i_end, med, sig, classified, n_d) in merged:
        # points covered are indices [i_start .. i_end] inclusive
        p_start = i_start
        p_end = i_end
        report.append(dict(
            point_start_index=int(p_start),
            point_end_index=int(p_end),
            start_bjd=float(t[p_start]),
            end_bjd=float(t[p_end]),
            n_points=int(p_end - p_start + 1),
            n_deltas=int(n_d),
            cadence_median_min=float(med),
            cadence_err_min=float(sig),
            cadence_classified_min=(None if classified is None else float(classified))
        ))
    return report

def _rebin_fixed_minutes(t_days, y, yerr=None, bin_minutes=30.0, min_points_per_bin=1, original_quality=None, original_cad=None):
    t = np.asarray(t_days, float)
    f = np.asarray(y, float)
    fe = None if yerr is None else np.asarray(yerr, float)
    good = np.isfinite(t) & np.isfinite(f)
    if fe is not None: good &= np.isfinite(fe) & (fe > 0)
    if original_quality is not None: good &= (np.asarray(original_quality) == 0)
    if original_cad is not None: cad = original_cad[good]
    if original_quality is not None: quality = original_quality[good]
    t, f = t[good], f[good]
    fe = None if fe is None else fe[good]
    if t.size == 0:
        return np.array([]), np.array([]), (None if fe is None else np.array([])), np.array([])

    order = np.argsort(t); t, f = t[order], f[order]
    if fe is not None: fe = fe[order]

    dt_days = bin_minutes / 1440.0
    t0 = np.min(t)
    bins = np.floor((t - t0) / dt_days).astype(int)
    cad_rebinned = rebin_cadence(cad, bins)
    quality_rebinned = rebin_quality(quality, bins)
    uniq = np.unique(bins)
    t_mid = t0 + (uniq + 0.5) * dt_days

    yb = np.empty(uniq.size); yeb = None if fe is None else np.empty(uniq.size)
    nbin = np.zeros(uniq.size, dtype=int)

    for i, b in enumerate(uniq):
        idx = (bins == b)
        fi = f[idx]
        if fi.size < min_points_per_bin:
            yb[i] = np.nan
            if yeb is not None: yeb[i] = np.nan
            continue
        if fe is not None:
            wi = 1.0 / (fe[idx] ** 2)
            mu = np.sum(wi * fi) / np.sum(wi)
            yb[i] = mu
            yeb[i] = np.sqrt(1.0 / np.sum(wi))
        else:
            yb[i] = np.mean(fi)
        nbin[i] = fi.size

    keep = np.isfinite(yb)
    t_mid, yb = t_mid[keep], yb[keep]
    if yeb is not None: yeb = yeb[keep]
    nbin = nbin[keep]
    return t_mid, yb, yeb, cad_rebinned, quality_rebinned, nbin

def analyze_and_rebin_to_binned(bjd, mag, bin_minutes, mag_err=None, quality=None, cadences=None,
                             gap_minutes=180.0, classify_tol_frac=0.08):
    """
    - Segments the LC into cadence-homogeneous chunks and reports cadence stats per chunk.
    - Independently rebins the *entire* LC to 30-minute cadence (flux-space weighting if errors provided).

    Returns:
      {
        'segments': [ {point_start_index, point_end_index, start_bjd, end_bjd,
                       n_points, n_deltas, cadence_median_min, cadence_err_min,
                       cadence_classified_min}, ... ],
        'time_binned': array(days),
        'mag_binned': array,
        'mag_err_binned': array or None,
        'per_bin_counts': array(int)
      }
    """
    bjd = np.asarray(bjd, float)
    mag = np.asarray(mag, float)
    order = np.argsort(bjd)
    bjd, mag = bjd[order], mag[order]
    mag_err = None if mag_err is None else np.asarray(mag_err, float)[order]
    quality = None if quality is None else np.asarray(quality)[order]

    segments = _segment_by_cadence(bjd, gap_minutes=gap_minutes, tol_frac=classify_tol_frac)

    # Re-bin globally to specified min (safe even with mixed cadences)
    flux = 10.0 ** (-0.4 * mag)
    flux_err = None
    if mag_err is not None:
        flux_err = (np.log(10) * 0.4) * flux * mag_err

    t_binned, f_binned, fe_binned, c_binned, q_binned, nbin = _rebin_fixed_minutes(
        bjd, flux, yerr=flux_err, bin_minutes=bin_minutes, min_points_per_bin=1, original_quality=quality, original_cad=cadences,
    )
    if t_binned.size > 0:
        mag_binned = -2.5 * np.log10(f_binned)
        mag_err_binned = None if fe_binned is None else (2.5 / np.log(10)) * (fe_binned / f_binned)
    else:
        mag_binned, mag_err_binned = np.array([]), (None if fe_binned is None else np.array([]))

    return dict(
        segments=segments,
        time_binned=t_binned,
        quality_binned=q_binned,
        cadences_binned=c_binned,
        mag_binned=mag_binned,
        mag_err_binned=mag_err_binned,
        per_bin_counts=nbin
    )

def get_fits_data(fits_path):
    with fits.open(fits_path) as hdul:
        data = hdul["LIGHTCURVE"].data
        bjd = data["TIME"] + 2457000
        fluxes = {
            "SAP_FLUX": data["SAP_FLUX"],
            "SAP_FLUX_SML": data["SAP_FLUX_SML"],
            "SAP_FLUX_MID": data["SAP_FLUX_MID"],
            "SAP_FLUX_LAG": data["SAP_FLUX_LAG"]
        }
        cadences = data["CADENCENO"]
        quality = data["QUALITY"]
        return bjd, fluxes, quality, cadences

def write_fits_rebinned(outpath: str, rebinned_fluxes: dict, cad_rebinned=None, quality_rebinned=None):
    """
    Write rebinned TESS light curve to FITS.

    Parameters
    ----------
    outpath : str
        Path to output FITS.
    rebinned_fluxes : dict
        Dictionary of rebinned flux/mag results for each aperture, e.g.,
        {"SAP_FLUX": {...}, "SAP_FLUX_SML": {...}, ...}
        Each entry must have "time_binned" and "mag_binned" (or flux_binned).
    cad_rebinned : np.ndarray, optional
        Rebinned CADENCENO array (int), same length as time_binned
    quality_rebinned : np.ndarray, optional
        Rebinned QUALITY array (int), same length as time_binned
    """
    # Reference time array (assumes all apertures use same rebinned time)
    ref = rebinned_fluxes["SAP_FLUX"]
    time = np.asarray(ref["time_binned"]) - 2457000

    if cad_rebinned is None:
        cad_rebinned = np.arange(len(time), dtype=int)
    if quality_rebinned is None:
        quality_rebinned = np.zeros(len(time), dtype=int)

    cols = [
        fits.Column(name="TIME", format="D", array=time, unit="BJD-2457000, days"),
        fits.Column(name="CADENCENO", format="J", array=cad_rebinned),
        fits.Column(name="QUALITY", format="J", array=quality_rebinned),
    ]

    # Add all rebinned apertures
    for key, data in rebinned_fluxes.items():
        if "flux_binned" in data:
            flux = np.asarray(data["flux_binned"], dtype=np.float32)
        else:
            flux = 10 ** (-0.4 * np.asarray(data["mag_binned"], dtype=np.float32))
        cols.append(fits.Column(name=key, format="E", array=flux))

    # Optional: per-bin counts
    perbin_counts = ref.get("perbin_counts", np.zeros_like(time, dtype=int))
    cols.append(fits.Column(name="PERBIN_COUNTS", format="J", array=np.asarray(perbin_counts, dtype=int)))

    # Construct FITS
    hdu_primary = fits.PrimaryHDU()
    hdu_table = fits.BinTableHDU.from_columns(cols)
    hdu_table.header["EXTNAME"] = "LIGHTCURVE"
    hdu_table.header["REBIN"] = "30 min"

    fits.HDUList([hdu_primary, hdu_table]).writeto(outpath, overwrite=True)
    print(f"Saved rebinned FITS: {outpath}")

def process_fits_file(fits_file, output_dir, bin_minutes):
    print(f"Processing {fits_file} ...")
    
    # Read data
    bjd, fluxes, quality, cadences = get_fits_data(fits_file)

    # Rebin all fluxes
    rebinned_fluxes = {}
    for name, f in fluxes.items():
        mag = -2.5 * np.log10(f, where=(f > 0))
        rebinned_fluxes[name] = analyze_and_rebin_to_binned(
            bjd, mag, bin_minutes=bin_minutes, quality=quality, cadences=cadences
        )

    # Write JSON stats
    stats = {'original_fits_location': str(fits_file)}
    for flux in rebinned_fluxes:
        stats[flux] = {
            'segments': rebinned_fluxes[flux]['segments'],
            'per_bin_counts': list(rebinned_fluxes[flux]['per_bin_counts'])
        }
    stats_output_path = str(Path(output_dir) / Path(fits_file).stem) + ".json"
    with open(stats_output_path, "w") as f:
        json.dump(stats, f, indent=2, cls=NumpyEncoder)

    # Write rebinned FITS
    rebinned_output_path = str(Path(output_dir) / Path(fits_file).name)
    write_fits_rebinned(str(rebinned_output_path), rebinned_fluxes)

def main(args):
    print(f'Rebinning all FITS files in {args.input_dir} to {args.bin_minutes} min, output: {args.output_dir}')
    
    os.makedirs(args.output_dir, exist_ok=True)

    fits_files = list(Path(args.input_dir).glob("*.fits"))

    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = {executor.submit(process_fits_file, f, args.output_dir, args.bin_minutes): f for f in fits_files}

        total = len(futures)
        for i, future in enumerate(as_completed(futures), start=1):
            try:
                future.result()
                print(f"[{i}/{total}] Completed: {futures[future].name}")
            except Exception as e:
                print(f"[{i}/{total}] Error processing {futures[future].name}: {e}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebin all FITS files in a given directory to 30 mins and store them in the output directory."
    )
    parser.add_argument('--input_dir', '-i', type=str, required=True, help='Input FITS files directory')
    parser.add_argument('--output_dir', '-o', type=str, required=True, help='Output (rebinned) FITS files directory')
    parser.add_argument('--bin_minutes', '-b', type=int, required=True, help='Cadence to bin to, in minutes')
    args = parser.parse_args()
    args = parser.parse_args()
    main(args)