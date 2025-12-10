import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from tqdm import tqdm

# Adjust paths as needed
input_tce_csv_file = "/pdo/users/pablomer/mnt/tess/astronet/tces-vetting-v01-tois-triageJs-nocentroid-april2025-all.csv"
tess_data_dir = '/pdo/users/pablomer/mnt/tess/april2025_dataset_fits_files'

# Change to Astronet-Triage directory if needed
os.chdir('/pdo/users/pablomer/Astronet-Triage')
sys.path.insert(0, "/pdo/users/pablomer/Astronet-Triage")

from astronet.preprocess import preprocess


def get_lightcurve(astro_id: int, tce_table: pd.DataFrame, tess_data_dir: str,
                   aperture: Optional[str] = None) -> tuple[np.ndarray, np.ndarray]:
    """Load light curve data for a given astro ID."""
    aperture_key_map = {
        "s": "SAP_FLUX_SML",
        "m": "SAP_FLUX_MID",
        "l": "SAP_FLUX_LAG",
        None: "SAP_FLUX",
    }

    matching_tces = tce_table[tce_table["Astro ID"] == astro_id]
    try:
        _, tce = next(matching_tces.iterrows())
    except StopIteration as e:
        raise ValueError(f"Astro ID not found: {astro_id}") from e

    if "MinT" not in tce:
        tce["MinT"] = -np.inf
    if "MaxT" not in tce:
        tce["MaxT"] = np.inf

    # Use the File column from the TCE table if available, otherwise construct filename
    if "File" in tce and pd.notna(tce["File"]):
        filename = tce["File"]
    else:
        # Fallback: construct filename (adjust format as needed for your data)
        astro_id_str = str(int(astro_id))[:-2].zfill(16)
        filename = (
        f"{tess_data_dir}/astronet_hlsp_qlp_tess_ffi-s0087-{astro_id_str}_tess_v01_llc.fits"
        )

    return preprocess.read_and_process_light_curve(
        tess_data_dir,
        aperture_key_map[aperture],
        filename,
        tce.MinT,
        tce.MaxT,
    )


def process_example(tce: pd.Series, tce_table: pd.DataFrame, tess_data_dir: str) -> Optional[dict]:
    """
    Process a single TCE example and generate both global views.
    Returns a dictionary with results or None if processing fails.
    """
    try:
        # Load light curve
        time, flux = get_lightcurve(tce['Astro ID'], tce_table, tess_data_dir)

        if len(time) == 0 or len(flux) == 0:
            return None

        # Detrend and filter
        detrended_time, detrended_flux, transit_mask = preprocess.detrend_and_filter(
            tce['TIC ID'], time, flux, tce.Per, tce.Epoc, tce.Dur, fixed_bkspace=None
        )

        if len(detrended_time) == 0:
            return None

        # Ensure epoch is within detrended time range
        epoch = tce.Epoc
        while epoch < detrended_time[0]:
            epoch += tce.Per

        # Calculate scatter weights
        scatter_weights = preprocess.split_and_calculate_weights(
            detrended_time, detrended_flux, gap_width=2
        )

        # Phase fold and sort
        folded_time, folded_flux, fold_num, tr_mask = preprocess.phase_fold_and_sort_light_curve(
            detrended_time, detrended_flux, transit_mask, tce.Per, epoch
        )

        # Align raw time for cadence selection
        raw_time_aligned, raw_flux_aligned = preprocess.align_raw_time(
            detrended_time, detrended_flux, tce.Per, epoch
        )

        # Align the weights
        weights_aligned = preprocess.align_scatter_weights(
            detrended_time, tce.Per, epoch, scatter_weights
        )

        # Generate global view WITHOUT scatter weights
        view_no_weights, std_no_weights, mask_no_weights, _, _ = preprocess.global_view(
            tce['TIC ID'],
            folded_time,
            folded_flux,
            tce.Per,
            all_30min=True,
            raw_time=raw_time_aligned,
            raw_flux=raw_flux_aligned,
            scatter_weights=None  # No scatter weights
        )

        # Generate global view WITH scatter weights
        view_with_weights, std_with_weights, mask_with_weights, _, _ = preprocess.global_view(
            tce['TIC ID'],
            folded_time,
            folded_flux,
            tce.Per,
            all_30min=True,
            raw_time=raw_time_aligned,
            raw_flux=raw_flux_aligned,
            scatter_weights=weights_aligned  # With scatter weights
        )

        return {
            'astro_id': tce['Astro ID'],
            'tic_id': tce['TIC ID'],
            'period': tce.Per,
            'epoch': tce.Epoc,
            'duration': tce.Dur,
            'view_no_weights': view_no_weights,
            'view_with_weights': view_with_weights,
            'folded_time': folded_time,
            'folded_flux': folded_flux,
        }

    except Exception as e:
        print(f"Error processing Astro ID {tce['Astro ID']}: {e}")
        return None



# Load TCE table
tce_table = pd.read_csv(input_tce_csv_file, header=0, low_memory=False)

# Filter to only examples with disp_e=1

if 'disp_e' in tce_table.columns:
    filtered_table = tce_table[tce_table['disp_e'] == 1]
    print(f"Filtered to {len(filtered_table)} examples with disp_e=1 (out of {len(tce_table)} total)")
else:
    print("Warning: 'disp_e' column not found in table. Proceeding without filter.")
    filtered_table = tce_table

# Get first 50 unique astro IDs from filtered table
unique_astro_ids = filtered_table["Astro ID"].unique()[:5823]
print(f"Processing {len(unique_astro_ids)} examples...")

# Get TCE data for each astro ID (take first occurrence)
examples = []
for astro_id in unique_astro_ids:
    tce = filtered_table[filtered_table["Astro ID"] == astro_id].iloc[0]
    examples.append(tce)

print(f"Loaded {len(examples)} examples to process")


# Process all examples
results = []
for tce in tqdm(examples, desc="Processing examples"):
    result = process_example(tce, tce_table, tess_data_dir)
    if result is not None:
        results.append(result)

print(f"\nSuccessfully processed {len(results)} out of {len(examples)} examples")


np.save('/pdo/users/pablomer/Astronet-Triage/astronet/preprocess/view_differences_disp_e.npy', results, allow_pickle=True)
