import pandas as pd

# Load both files
df_2026 = pd.read_csv("/pdo/astronet-data/data/toi-catalog_2026-04-07.csv")
df_2025 = pd.read_csv("/pdo/astronet-data/data/toi-plus-2025-07-16.csv")

# Ensure TIC is consistent type (important!)
df_2026["TIC"] = df_2026["TIC"].astype(str)
df_2025["TIC"] = df_2025["TIC"].astype(str)

# Create a set of TICs from 2025 file
tic_2025_set = set(df_2025["TIC"])

# Add boolean column
df_2026["in_2025_csv"] = df_2026["TIC"].isin(tic_2025_set)

# Save result
df_2026.to_csv("/pdo/astronet-data/data/toi-catalog_2026-04-07_with_flag.csv", index=False)