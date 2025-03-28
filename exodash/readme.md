# ExoDash

**ExoDash** is a Streamlit-based dashboard for exploring exoplanet datasets, analyzing Astronet model performance, and visualizing key statistics and distributions from the TESS Input Catalog (TIC).

## Features

- Dataset summary with row/column counts
- Train/validation/test split distribution
- Summary statistics and label distributions
- Missing data overview
- Sidebar navigation to additional pages (TIC exploration, model performance, etc.)

## How to run
1. Ensure your data management is working properly using the DataManagementExample.ipynb
2. Run the streamlit dashboard locally: python -m streamlit run exodash/app.py (from the main astronet folder)