# ExoDash

**ExoDash** is a Streamlit-based dashboard for exploring exoplanet datasets, analyzing Astronet model performance, and visualizing key statistics and distributions from the TESS Input Catalog (TIC).

## Features

- Dataset summary with row/column counts
- Train/validation/test split distribution
- Summary statistics and label distributions
- Missing data overview
- Sidebar navigation to additional pages (TIC exploration, model performance, etc.)

## First time configuration
1. pip install -r requirements.txt
2. Ensure you have set up the Google Service Account credentials
- Download the credentials from <link>
- Set the OS environment variable GOOGLE_APPLICATION_CREDENTIALS to point to the downloaded file.
3. Ensure your data management is working properly using the DataManagementExample.ipynb

## How to run
1. Ensure your config.yaml is up-to-
2. ssh connection to PDO (to be able to query for TIC reports from QLP)
- ssh -L 5001:localhost:5001 pdo6
3. Run the streamlit dashboard locally: python -m streamlit run exodash/app.py (from the main astronet folder)