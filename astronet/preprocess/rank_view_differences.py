import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from datetime import datetime
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


def process_example(tce: pd.Series, tce_table: pd.DataFrame, tess_data_dir: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Process a single TCE example and generate both global views.
    Returns a tuple of (result_dict, error_message):
    - If successful: (dict with results, None)
    - If failed: (None, error_message string with full traceback)
    """
    try:
        # Load light curve
        time, flux = get_lightcurve(tce['Astro ID'], tce_table, tess_data_dir)

        if len(time) == 0 or len(flux) == 0:
            error_msg = f"Empty light curve data for Astro ID {tce['Astro ID']}, TIC ID {tce['TIC ID']}"
            return None, ("empty_data", error_msg)

        # Detrend and filter
        detrended_time, detrended_flux, transit_mask = preprocess.detrend_and_filter(
            tce['TIC ID'], time, flux, tce.Per, tce.Epoc, tce.Dur, fixed_bkspace=None
        )

        if len(detrended_time) == 0:
            error_msg = f"Empty detrended data for Astro ID {tce['Astro ID']}, TIC ID {tce['TIC ID']}"
            return None, ("empty_data", error_msg)

        # Ensure epoch is within detrended time range
        epoch = tce.Epoc
        while epoch < detrended_time[0]:
            epoch += tce.Per

        # Calculate scatter weights with error handling
        try:
            scatter_weights = preprocess.split_and_calculate_weights(
                detrended_time, detrended_flux, gap_width=2
            )
            scatter_weights_error = None
        except Exception as e:
            # If scatter weights calculation fails, we'll use None and continue
            # This allows us to still generate the view without weights
            scatter_weights = None
            scatter_weights_error = f"split_and_calculate_weights failed: {type(e).__name__}: {str(e)}"
            # Store the traceback for logging
            scatter_weights_traceback = traceback.format_exc()

        # Phase fold and sort
        folded_time, folded_flux, fold_num, tr_mask = preprocess.phase_fold_and_sort_light_curve(
            detrended_time, detrended_flux, transit_mask, tce.Per, epoch
        )

        # Align raw time for cadence selection
        raw_time_aligned, raw_flux_aligned = preprocess.align_raw_time(
            detrended_time, detrended_flux, tce.Per, epoch
        )

        # Align the weights (only if scatter_weights was successfully calculated)
        if scatter_weights is not None:
            weights_aligned = preprocess.align_scatter_weights(
                detrended_time, tce.Per, epoch, scatter_weights
            )
        else:
            weights_aligned = None

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
        # If scatter weights failed, we'll use None (same as without weights)
        if weights_aligned is not None:
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
        else:
            # If scatter weights calculation failed, use None (same as without weights)
            view_with_weights, std_with_weights, mask_with_weights, _, _ = preprocess.global_view(
                tce['TIC ID'],
                folded_time,
                folded_flux,
                tce.Per,
                all_30min=True,
                raw_time=raw_time_aligned,
                raw_flux=raw_flux_aligned,
                scatter_weights=None  # No scatter weights (fallback)
            )

        # If scatter weights failed but we still processed successfully, return with warning
        if scatter_weights_error is not None:
            error_msg = f"Warning: Scatter weights calculation failed but processing continued:\n"
            error_msg += f"{scatter_weights_error}\n"
            error_msg += f"Traceback:\n{scatter_weights_traceback}"
            # Return success but with error info
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
            }, ("scatter_weights_warning", error_msg)

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
        }, None

    except Exception as e:
        # Check if the error is from split_and_calculate_weights by examining traceback
        tb_str = traceback.format_exc()
        error_source = "other"
        if "split_and_calculate_weights" in tb_str:
            error_source = "split_and_calculate_weights"

        error_msg = f"Error processing Astro ID {tce['Astro ID']}, TIC ID {tce['TIC ID']}:\n"
        error_msg += f"Error source: {error_source}\n"
        error_msg += f"Exception: {type(e).__name__}: {str(e)}\n"
        error_msg += f"Traceback:\n{tb_str}"
        return None, (error_source, error_msg)



# Load TCE table
tce_table = pd.read_csv(input_tce_csv_file, header=0, low_memory=False)

# Filter to only examples with disp_e=1
disp_col = 'disp_p'

if disp_col in tce_table.columns:
    filtered_table = tce_table[tce_table[disp_col] == 1]
    print(f"Filtered to {len(filtered_table)} examples with {disp_col}=1 (out of {len(tce_table)} total)")
else:
    print(f"Warning: '{disp_col}' column not found in table. Proceeding without filter.")
    filtered_table = tce_table

# Get all unique astro IDs from filtered table
unique_astro_ids = filtered_table["Astro ID"].unique()
print(f"Processing {len(unique_astro_ids)} examples...")

# Get TCE data for each astro ID (take first occurrence)
examples = []
for astro_id in unique_astro_ids:
    tce = filtered_table[filtered_table["Astro ID"] == astro_id].iloc[0]
    examples.append(tce)

print(f"Loaded {len(examples)} examples to process")


# Process all examples
results = []
problematic_ids = []  # List of dicts with astro_id, tic_id, and error info
error_log = []  # List of error messages with timestamps
error_source_counts = {
    'split_and_calculate_weights': 0,
    'scatter_weights_warning': 0,
    'other': 0,
    'empty_data': 0
}  # Count errors by source

for tce in tqdm(examples, desc="Processing examples"):
    result, error_info = process_example(tce, tce_table, tess_data_dir)

    if result is not None:
        # Success case
        if error_info is not None:
            # Success but with warning (scatter weights failed but continued)
            error_source, error_msg = error_info
            error_source_counts['scatter_weights_warning'] += 1

            # Log the warning but don't mark as problematic (still processed successfully)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            error_log.append(f"\n{'='*80}\n")
            error_log.append(f"WARNING (still processed): {timestamp}\n")
            error_log.append(f"Astro ID: {tce['Astro ID']}\n")
            error_log.append(f"TIC ID: {tce['TIC ID']}\n")
            error_log.append(f"{error_msg}\n")
            print(f"Warning for Astro ID {tce['Astro ID']}, TIC ID {tce['TIC ID']}: Scatter weights failed but processing continued")

        results.append(result)
    else:
        # Failure case
        error_source, error_msg = error_info

        # Track error source
        if error_source in error_source_counts:
            error_source_counts[error_source] += 1
        else:
            error_source_counts['other'] += 1

        # Track problematic ID
        problematic_ids.append({
            'astro_id': tce['Astro ID'],
            'tic_id': tce['TIC ID'],
            'period': tce.Per if 'Per' in tce else None,
            'epoch': tce.Epoc if 'Epoc' in tce else None,
            'duration': tce.Dur if 'Dur' in tce else None,
            'error_source': error_source,
        })
        # Log error with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_log.append(f"\n{'='*80}\n")
        error_log.append(f"ERROR: {timestamp}\n")
        error_log.append(f"Error Source: {error_source}\n")
        error_log.append(f"Astro ID: {tce['Astro ID']}\n")
        error_log.append(f"TIC ID: {tce['TIC ID']}\n")
        error_log.append(f"{error_msg}\n")
        # Also print to console
        print(f"Error processing Astro ID {tce['Astro ID']}, TIC ID {tce['TIC ID']} ({error_source}): {error_msg.split(chr(10))[0]}")

print(f"\nSuccessfully processed {len(results)} out of {len(examples)} examples")
print(f"Failed to process {len(problematic_ids)} examples")
print(f"\nError source breakdown:")
print(f"  split_and_calculate_weights errors: {error_source_counts['split_and_calculate_weights']}")
print(f"  scatter_weights warnings (still processed): {error_source_counts['scatter_weights_warning']}")
print(f"  empty_data errors: {error_source_counts['empty_data']}")
print(f"  other errors: {error_source_counts['other']}")

# Save results
np.save(f'/pdo/users/pablomer/Astronet-Triage/astronet/preprocess/view_differences_{disp_col}.npy', results, allow_pickle=True)

# Save problematic IDs to CSV
if problematic_ids:
    problematic_df = pd.DataFrame(problematic_ids)
    problematic_csv_path = f'/pdo/users/pablomer/Astronet-Triage/astronet/preprocess/problematic_ids_{disp_col}.csv'
    problematic_df.to_csv(problematic_csv_path, index=False)
    print(f"\nSaved {len(problematic_ids)} problematic IDs to: {problematic_csv_path}")
else:
    print("\nNo problematic IDs to save.")

# Save error log to text file
if error_log:
    error_log_path = f'/pdo/users/pablomer/Astronet-Triage/astronet/preprocess/error_log_{disp_col}.txt'
    with open(error_log_path, 'w') as f:
        f.write(f"Error Log for View Differences Processing\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total errors: {len(problematic_ids)}\n")
        f.write(f"Total warnings (still processed): {error_source_counts['scatter_weights_warning']}\n")
        f.write(f"\nError source breakdown:\n")
        f.write(f"  split_and_calculate_weights errors: {error_source_counts['split_and_calculate_weights']}\n")
        f.write(f"  scatter_weights warnings (still processed): {error_source_counts['scatter_weights_warning']}\n")
        f.write(f"  empty_data errors: {error_source_counts['empty_data']}\n")
        f.write(f"  other errors: {error_source_counts['other']}\n")
        f.write(f"{'='*80}\n")
        f.writelines(error_log)
    print(f"Saved error log to: {error_log_path}")
else:
    print("\nNo errors to log.")

# Print summary of problematic IDs
if problematic_ids:
    print(f"\n{'='*80}")
    print("SUMMARY OF PROBLEMATIC IDs")
    print(f"{'='*80}")
    print(f"Total problematic IDs: {len(problematic_ids)}")

    # Count by error source
    error_source_summary = {}
    for pid in problematic_ids:
        source = pid.get('error_source', 'unknown')
        error_source_summary[source] = error_source_summary.get(source, 0) + 1

    print(f"\nError source breakdown for problematic IDs:")
    for source, count in sorted(error_source_summary.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source}: {count}")

    print(f"\nFirst 10 problematic IDs:")
    for i, pid in enumerate(problematic_ids[:10], 1):
        error_source = pid.get('error_source', 'unknown')
        print(f"  {i}. Astro ID: {pid['astro_id']}, TIC ID: {pid['tic_id']}, Error: {error_source}")
    if len(problematic_ids) > 10:
        print(f"  ... and {len(problematic_ids) - 10} more (see CSV file for full list)")
    print(f"{'='*80}\n")
