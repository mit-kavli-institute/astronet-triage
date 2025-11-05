from pathlib import Path
import pandas as pd
from astropy.table import join, Table, vstack
import numpy as np
import scipy
from typing import List, Dict
import os
from exodash.utils.live_evaluation import LiveEvaluation


ASTRONET_BASE_PATH = Path("/pdo/astronet-data/data/")
LATEST_TOI_DATA = Path("/pdo/astronet-data/data/toi-plus-2025-07-16.csv")

class ProductionSector:
    sector: int
    
    def __init__(self, sector: int):
        self.sector = sector
        self.fits_path = ASTRONET_BASE_PATH / "fits"
        self.tfrecords_path = ASTRONET_BASE_PATH / "tfrecords" / f"sector-{self.sector}"
        self.properties_path = ASTRONET_BASE_PATH / "properties" / f"tces-sector{sector}_with_labels.csv"

    @property
    def eval_files(self) -> List[str]:
        return [f"test:{self.tfrecords_path}/*"]
    
    @property
    def properties_df(self) -> pd.DataFrame:
        properties_df = pd.read_csv(self.properties_path, index_col=False)
        properties_df = properties_df.rename(columns={'per': 'period', 'dur': 'duration', 'epoc': 'epoch'})
        return properties_df.loc[:, ~properties_df.columns.str.startswith('Unnamed')]



def get_ephemeris_matches(astronet_rows, toi_rows):
    cartesian_table = join(
        astronet_rows, toi_rows, join_type="cartesian", table_names=["astronet", "toi"]
    )
    p_astronet = cartesian_table["period_astronet"]
    t_astronet = cartesian_table["epoch_astronet"]
    p_toi = cartesian_table["period_toi"]
    t_toi = cartesian_table["epoch_toi"]

    p_min = np.minimum(p_astronet, p_toi)
    delta_p = (p_astronet - p_toi) / p_min
    delta_p_prime = np.abs(delta_p - np.round(delta_p))
    sigma_p = np.sqrt(2) * scipy.special.erfcinv(delta_p_prime)
    cartesian_table["sigma_p"] = sigma_p

    delta_t = (t_astronet - t_toi) / p_min
    delta_t_prime = np.abs(delta_t - np.round(delta_t))
    sigma_t = np.sqrt(2) * scipy.special.erfcinv(delta_t_prime)
    cartesian_table["sigma_t"] = sigma_t

    cartesian_table["match_strength"] = sigma_p ** 2 + sigma_t ** 2
    best_match = cartesian_table[np.argmax(cartesian_table["match_strength"])]
    return best_match

def get_qlp_astronet_scores(sector: int) -> pd.DataFrame:
    dfs = []

    for cam in [1, 2, 3, 4]:
        try:
            csv_file = f"/pdo/qlp-data/sector-{sector}/ffi/run/astronet_vetting_scores_cam{cam}.csv"
        except Exception:
            raise Exception(f"QLP Astronet Vetting scores not available for cam {cam}")
        df = pd.read_csv(csv_file, index_col=False)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)

    combined_df.drop_duplicates(subset=['astro_id'], inplace=True)
    combined_df = combined_df.loc[:, ~combined_df.columns.str.startswith('Unnamed')]
    return combined_df

def read_toi_data() -> pd.DataFrame:
    return pd.read_csv(LATEST_TOI_DATA, comment="#")[[
        "TIC", "Full TOI ID", "TOI Disposition", "TMag Value", "Orbital Epoch Value", "Orbital Period (days) Value",
        "Transit Duration (hours) Value", "Transit Depth Value", "Planet Number", "Detection Pipeline(s)"
    ]].rename(columns={
            "TIC": "tic_id",
            "Full TOI ID": "toi_id",
            "TOI Disposition": "toi_disposition",
            "TMag Value": "tmag",
            "Orbital Epoch Value": "epoch",
            "Orbital Period (days) Value": "period",
            "Transit Duration (hours) Value": "duration",
            "Transit Depth Value": "depth",
            "Planet Number": "planetno",
            "Detection Pipeline(s)": "detection_pipeline",
        }
    )

def get_production_sector_df(sectors: List[int], custom_model, sector_to_astronet_scores_override: Dict[int, Path]) -> pd.DataFrame:
    all_production_sectors = []

    for sector in sectors:
        production_sector = ProductionSector(sector)
        all_production_sectors.append(production_sector)
    
    all_astronet_scores = None
    if custom_model:
        live_evaluation = LiveEvaluation(model_dir=custom_model)
        all_eval_files = [f for x in all_production_sectors for f in x.eval_files]
        all_astronet_scores = live_evaluation.evaluate(all_eval_files)
    else:
        per_sector_astronet_scores = []
        for sector in all_production_sectors:
            if sector.sector in sector_to_astronet_scores_override:
                astronet_scores = pd.read_csv(sector_to_astronet_scores_override[sector.sector], index_col=False)
                astronet_scores = astronet_scores.rename(columns={'Sector': 'sector', 'Astro ID': 'astro_id'})
                per_sector_astronet_scores.append(astronet_scores)
            else:    
                per_sector_astronet_scores.append(get_qlp_astronet_scores(sector.sector))
        all_astronet_scores = pd.concat(per_sector_astronet_scores, ignore_index=True)

    disp_cols = ['disp_p', 'disp_e', 'disp_n', 'disp_j']
    all_astronet_scores = (
        all_astronet_scores.groupby(['astro_id', 'tic_id', 'planetno'])[disp_cols]
        .mean()
        .reset_index()
    )

    all_properties = pd.concat([x.properties_df for x in all_production_sectors], ignore_index=True)
    merged_df = pd.merge(
        all_astronet_scores,
        all_properties,
        on=['astro_id', 'tic_id', 'planetno'],          # column to join on
        how='inner'             # only rows present in both DataFrames
    )

    merged_df['operator_passed'] = merged_df['true_label'] == 'p'
    merged_df.drop(columns=['true_label'], inplace=True)

    #combined_df = pd.concat(per_sector_dfs, ignore_index=True)
    merged_df.drop_duplicates(subset=['astro_id'], inplace=True)

    toi_data = read_toi_data()
    toi_data = Table.from_pandas(toi_data)

    astronet_data = Table.from_pandas(merged_df)

    matched_astronet_data = vstack(
        [
            get_ephemeris_matches(row, toi_data[toi_data["tic_id"] == row["tic_id"]])
            for row in astronet_data
            if row["tic_id"] in toi_data["tic_id"]# and row["centroid_distance_arcsec"] < 21
        ]
    )

    # Here, `matched_astronet_data` contains the best match for each signal, but we want to throw out
    # matches that don't meet the criterion \sigma_P, \sigma_T > 3.
    matched_astronet_data = matched_astronet_data[
        (matched_astronet_data["sigma_p"] > 3) & (matched_astronet_data["sigma_t"] > 3)
    ]


    toi_data = matched_astronet_data.to_pandas()
    tce_data = merged_df

    tce_data['has_toi'] = tce_data['tic_id'].isin(toi_data['tic_id_astronet'])

    toi_subset = toi_data[['tic_id_astronet', 'planetno_astronet', 'toi_id', 'toi_disposition',
                        'detection_pipeline', 'match_strength']].copy()
    toi_subset['tic_id_astronet'] = toi_subset['tic_id_astronet'].astype(tce_data['tic_id'].dtype)
    toi_subset['planetno_astronet'] = toi_subset['planetno_astronet'].astype(tce_data['planetno'].dtype)

    tce_data['has_toi'] = tce_data.set_index(['tic_id', 'planetno']).index.isin(
        toi_subset.set_index(['tic_id_astronet', 'planetno_astronet']).index
    )

    # Step 3: Merge selected TOI info into tce_properties
    tce_data = tce_data.merge(
        toi_subset,
        left_on=['tic_id', 'planetno'],
        right_on=['tic_id_astronet', 'planetno_astronet'],
        how='left'
    ).drop(columns=['tic_id_astronet', 'planetno_astronet'])
    return tce_data
