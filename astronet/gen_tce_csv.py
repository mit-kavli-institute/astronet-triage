import argparse
import glob
import os
import re

import pandas as pd

parser = argparse.ArgumentParser(
    description="Generate a CSV with file paths and properties for generating TFRecords for a given sector. Assumes the FITS files are in /pdo/astronet-data/data/data/fits."
)
parser.add_argument('--sector', type=int, required=True, help='Sector number')
args = parser.parse_args()
sector = args.sector

fits_pattern = f"/pdo/astronet-data/data/fits/sector-{sector}/*.fits"
tce_properties_csv = f"/pdo/qlp-data/sector-{sector}/ffi/run/astronet-vetting-tce-catalog.csv"
output_csv = f"/pdo/astronet-data/data/tfrecords/sector-{sector}/tces-sector{sector}.csv"

# Step 1: Find all .fits files
fits_files = glob.glob(fits_pattern)

# Step 2: Extract TIC ID from filenames and store mapping
fits_df = pd.DataFrame({
    "File":
        fits_files,
    "TIC ID": [
        int(
            re.search(r'-(\d{16})_tess',
                      os.path.basename(f)).group(1).lstrip('0'))
        for f in fits_files
        if re.search(r'-(\d{16})_tess', os.path.basename(f))
    ]
})

# Step 3: Load the TCE properties CSV
tce_df = pd.read_csv(tce_properties_csv)

# Step 4: Merge on TIC ID
merged = pd.merge(tce_df, fits_df, on="TIC ID", how="left")
merged = merged.drop(columns=["Unnamed: 0"], errors="ignore")
merged = merged.fillna("")

# Step 5: Save result
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
merged.to_csv(output_csv, index=False)
print(f"Saved merged CSV to {output_csv}")
