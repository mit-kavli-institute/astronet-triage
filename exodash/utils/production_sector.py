from pathlib import Path
import pandas as pd
from astropy.table import join, Table, vstack
import numpy as np
import scipy
from typing import List, Dict, Tuple
import re
import os
from exodash.utils.live_evaluation import LiveEvaluation
import streamlit as st
from collections import defaultdict


ASTRONET_BASE_PATH = Path("/pdo/astronet-data/data/")
LATEST_TOI_DATA = Path("/pdo/astronet-data/data/toi-plus-2025-07-16.csv")

if "selected_sectors" not in st.session_state:
    st.session_state.selected_sectors = set()
def get_production_sector_selector():
    sectors = list(range(85, 95))
    available = {}
    errors = defaultdict(list)
    for sector in sectors:
        if sector == 85:
            available[sector] = True
            continue
        tfrecords_exist = os.path.isdir(f'/pdo/astronet-data/data/tfrecords/sector-{sector}') and os.listdir(f'/pdo/astronet-data/data/tfrecords/sector-{sector}')
        properties_exist = os.path.isfile(f'/pdo/astronet-data/data/properties/tces-sector{sector}.csv')
        astronet_scores_exist = os.path.isfile(f'/pdo/qlp-data/sector-{sector}/ffi/run/astronet_vetting_scores_cam1.alltriage.csv')
        qlp_delivery_exists = os.path.isfile(f'/pdo/qlp-data/tev/qlp-delivery/sector-{sector}/batch3/cand.ls')
        if properties_exist and tfrecords_exist and astronet_scores_exist and qlp_delivery_exists:
            available[sector] = True
        else:
            available[sector] = False
        if not tfrecords_exist:
            errors[sector].append('Missing TFRecords in /pdo/astronet-data/data/tfrecords/')
        if not properties_exist:
            errors[sector].append('Missing properties CSV in /pdo/astronet-data/data/properties/')
        if not astronet_scores_exist:
            errors[sector].append(f'Missing Astronet scores in /pdo/qlp-data/sector-{sector}/ffi/run/')
        if not qlp_delivery_exists:
            errors[sector].append(f'Missing QLP delivery cand.ls in /pdo/qlp-data/tev/qlp-delivery/sector-{sector}/batch3/')
    
    cols = st.columns(len(sectors))

    for i, astro_id in enumerate(sectors):
        with cols[i]:
            disabled = not available[astro_id]
            selected = astro_id in st.session_state.selected_sectors

            if st.button(
                f"{astro_id}",
                key=f"btn_{astro_id}",
                disabled=disabled,
                type="primary" if selected else "secondary",
                use_container_width=True,
            ):
                if selected:
                    st.session_state.selected_sectors.remove(astro_id)
                else:
                    st.session_state.selected_sectors.add(astro_id)
                st.rerun()
    for sector in errors:
        st.warning(f'Sector {sector} failed with errors: {errors[sector]}')

class ProductionSector:
    sector: int
    
    def __init__(self, sector: int):
        self.sector = sector
        self.fits_path = ASTRONET_BASE_PATH / "fits"
        self.tfrecords_path = ASTRONET_BASE_PATH / "tfrecords" / f"sector-{self.sector}"
        self.properties_path = ASTRONET_BASE_PATH / "properties" / f"tces-sector{sector}.csv"

    @property
    def eval_files(self) -> List[str]:
        return [f"sector_{self.sector}:{self.tfrecords_path}/*"]
    
    def get_delivered_candidates(self) -> List[Tuple[int, int]]:
        try:
            delivery_directory = Path(f"/pdo/qlp-data/tev/qlp-delivery/sector-{self.sector}/batch3")
            if self.sector == 85:
                delivery_directory = Path(f"/pdo/qlp-data/tev/qlp-delivery/sector-{self.sector}")
            cand_data = pd.read_csv(delivery_directory / "cand.ls", sep=r"\s+", header=None, names=["star_tic", "planet_planetno"]).set_index(["star_tic", "planet_planetno"])
            if len(cand_data) == 0:
                raise Exception("No candidates found in ls file.")
            return list(cand_data.index.values)
        except Exception:
            ls_file = f"/pdo/qlp-data/sector-{self.sector}/ffi/run/cand.ls"
            df = pd.read_csv(ls_file, sep=r"\s+", names=["tic_id", "planetno"], on_bad_lines="skip")
            return list(df.itertuples(index=False, name=None))

    def clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        def to_snake(name: str) -> str:
            name = re.sub(r"[ -]+", "_", name)
            name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
            name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
            name = name.lower()
            name = re.sub(r"[^a-z0-9_]", "", name)
            name = re.sub(r"__+", "_", name)
            return name.strip("_")
        df.columns = [to_snake(col) for col in df.columns]
        return df

    @property
    def properties_df(self, include_labels: bool = True) -> pd.DataFrame:
        properties_df = pd.read_csv(self.properties_path, index_col=False)
        properties_df = self.clean_columns(properties_df)
        properties_df = properties_df.rename(columns={'per': 'period', 'dur': 'duration', 'epoc': 'epoch'})

        delivered_candidates = self.get_delivered_candidates()
        were_results_delivered = properties_df.apply(lambda row: (row["tic_id"], row["planetno"]) in delivered_candidates, axis=1)
        properties_df["true_label"] = np.where(were_results_delivered, "p", "j")
        properties_df["sector"] = f"sector_{self.sector}"
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
            csv_file = f"/pdo/qlp-data/sector-{sector}/ffi/run/astronet_vetting_scores_cam{cam}.alltriage.csv"
            if sector == 85:
                csv_file = f"/pdo/qlp-data/sector-{sector}/ffi/run/astronet_vetting_scores_cam{cam}_alltriage.csv"
            df = pd.read_csv(csv_file, index_col=False)
        except Exception:
            print(f"All triage QLP Astronet Vetting scores not available for cam {cam}")
            try:
                csv_file = f"/pdo/qlp-data/sector-{sector}/ffi/run/astronet_vetting_scores_cam{cam}.csv"
                df = pd.read_csv(csv_file, index_col=False)
            except Exception:
                raise Exception(f"Could not locate Astronet scores for sector {sector}")
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
