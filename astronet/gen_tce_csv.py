
import argparse
import pandas as pd
import glob
import os
import re

parser = argparse.ArgumentParser(description="Generate a CSV with file paths and properties for generating TFRecords.")
parser.add_argument('--sector', type=int, required=True, help='Sector number')
args = parser.parse_args()
sector = args.sector

fits_pattern = f"/pdo/astronet-data/data/fits/sector-{sector}/*.fits"
tce_properties_csv = f"/pdo/qlp-data/sector-{sector}/ffi/run/astronet-vetting-tce-catalog.csv"
output_csv = f"/pdo/astronet-data/data/properties/tces-sector{sector}.csv"

# Step 1: Find all .fits files
fits_files = glob.glob(fits_pattern)

# Step 2: Extract TIC ID from filenames and store mapping
fits_df = pd.DataFrame({
    "File": fits_files,
    "TIC ID": [
        int(re.search(r'-(\d{16})_tess', os.path.basename(f)).group(1).lstrip('0'))
        for f in fits_files
        if re.search(r'-(\d{16})_tess', os.path.basename(f))
    ]
})

# Step 3: Load the TCE properties CSV
tce_df = pd.read_csv(tce_properties_csv)
tce_df = tce_df.drop(columns=["Unnamed: 0"], errors="ignore")

print(len(tce_df))
dupes = tce_df[tce_df.duplicated(subset=['Astro ID'], keep=False)]

# Check if the rows with same Astro ID are identical across *all* other columns
dupes_equal = (
    dupes
    .groupby('Astro ID')
    .apply(lambda g: g.drop(columns=['Astro ID']).nunique().max() == 1)
)

print("True duplicates:", len(dupes_equal[dupes_equal].index.tolist()))
print("Non-identical repeats:", len(dupes_equal[~dupes_equal].index.tolist()))
tce_df['Astro ID'] = tce_df['Astro ID'].astype('int64')
tce_df = tce_df.drop_duplicates(subset=['Astro ID'], keep='first').reset_index(drop=True)
print(len(tce_df))

# Step 4: Merge on TIC ID
merged = pd.merge(tce_df, fits_df, on="TIC ID", how="left")
merged = merged.drop(columns=["Unnamed: 0"], errors="ignore")
merged = merged.fillna("")
# merged['Sector'] = sector
# merged['Dur'] = merged['Dur'] * 24
# merged['Phase Width'] = (merged['Dur'] / 24) / merged['Per']

# Step 5: Save result
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
merged.to_csv(output_csv, index=False)
print(f"Saved merged CSV to {output_csv}")