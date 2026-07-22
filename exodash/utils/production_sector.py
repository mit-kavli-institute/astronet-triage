from pathlib import Path
import pandas as pd
from astropy.table import join, Table, vstack
import numpy as np
import scipy
from typing import List, Dict, Optional, Tuple
import os
from exodash.utils.live_evaluation import LiveEvaluation
import re
from itertools import chain
from exodash.utils.mast import fetch_tic_rows_by_id

ASTRONET_BASE_PATH = Path("/pdo/astronet-data/data/")
LATEST_TOI_DATA = Path("/pdo/astronet-data/data/toi-catalog_2026-04-07_with_flag.csv")

class ProductionSector:
    sector: int
    
    def __init__(self, sector: int, tfrecord_postfix: Optional[str] = None):
        self.sector = sector
        self.fits_path = ASTRONET_BASE_PATH / "fits"
        self.tfrecords_path = ASTRONET_BASE_PATH / "tfrecords" / f"sector-{self.sector}"
        if tfrecord_postfix:
            self.tfrecords_path = Path(str(self.tfrecords_path) + f'-{tfrecord_postfix}')
        self.properties_path = ASTRONET_BASE_PATH / "properties" / f"tces-sector{sector}.csv"
        self.qlp_properties_path = ASTRONET_BASE_PATH / "properties" / f"tces-sector{sector}-qlp.csv"

    @property
    def eval_files(self) -> List[str]:
        return [f"sector_{self.sector}:{self.tfrecords_path}/*"]

    @property
    def astro_ids_passing_centroid_filter(self) -> List[int]:
        tic_ids = []

        for cam in [1, 2, 3, 4]:
            try:
                centroid_ls_file = f"/pdo/qlp-data/sector-{self.sector}/ffi/run/centroid_cam{cam}.ls"
                df = pd.read_csv(centroid_ls_file, sep=r"\s+", header=None, names=["tic_id", "planetno"])
                passing_centroid_ids = (
                    df["tic_id"].astype(str)
                    + df["planetno"].astype(str).str.zfill(2)
                ).astype(int)
            except Exception as e:
                #st.error(f"Could not locate all centroid filter files: {e}")
                passing_centroid_ids = []
            tic_ids.extend(passing_centroid_ids)
        return tic_ids

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
    def individual_vetting_df(self) -> pd.DataFrame:
        individual_vetting_path = Path(f"/pdo/astronet-data/data/individual_vetting_labels/sector_{self.sector}.csv")
        if not individual_vetting_path.exists():
            return pd.DataFrame()
        vetting_df = pd.read_csv(individual_vetting_path)
        vetting_df = vetting_df.rename(columns={
            "TIC": "tic_id",
            "Planet Number": "planetno",
            "Vetting Disposition": "vetting_disposition",
            "eclipsing_score": "triage_eclipsing_score",
            "single_score": "triage_single_score",
            "binary_score": "triage_binary_score",
            "junk_score": "triage_junk_score",
            "not_sure_score": "triage_not_sure_score",
        })
        vetting_cols = ["tic_id", "planetno", "vetting_disposition",
                        "triage_eclipsing_score", "triage_single_score", "triage_binary_score",
                        "triage_junk_score", "triage_not_sure_score", "human_triage", "planet_equilibrium_temperature"]
        return vetting_df[[c for c in vetting_cols if c in vetting_df.columns]]
    
    @property
    def properties_df(self, include_labels: bool = True) -> pd.DataFrame:
        properties_df = pd.read_csv(self.properties_path, index_col=False)
        properties_df = self.clean_columns(properties_df)
        properties_df = properties_df.rename(columns={'per': 'period', 'dur': 'duration', 'epoc': 'epoch'})

        qlp_properties_df = pd.read_csv(self.qlp_properties_path, index_col=False)
        qlp_properties_df = self.clean_columns(qlp_properties_df)

        properties_df = properties_df.merge(
            qlp_properties_df,
            on='astro_id',
            how='left',
            suffixes=('', '_qlp')
        )

        delivered_candidates = self.get_delivered_candidates()
        were_results_delivered = properties_df.apply(lambda row: (row["tic_id"], row["planetno"]) in delivered_candidates, axis=1)
        properties_df["true_label"] = np.where(were_results_delivered, "p", "j")
        properties_df["sector"] = f"sector_{self.sector}"

        vetting_df = self.individual_vetting_df

        print(vetting_df)
        print(properties_df)
        if not vetting_df.empty:
            vetting_df["tic_id"] = vetting_df["tic_id"].astype(properties_df["tic_id"].dtype)
            vetting_df["planetno"] = vetting_df["planetno"].astype(properties_df["planetno"].dtype)
            properties_df = properties_df.merge(vetting_df, on=["tic_id", "planetno"], how="left")
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

def get_qlp_astronet_scores(sector: ProductionSector) -> pd.DataFrame:
    dfs = []

    found_astronet_scores = True
    sector_num = sector.sector
    for cam in [1, 2, 3, 4]:
        try:
            csv_file = f"/pdo/qlp-data/sector-{sector_num}/ffi/run/astronet_vetting_scores_cam{cam}.alltriage.csv"
            if sector_num == 85:
                csv_file = f"/pdo/qlp-data/sector-{sector_num}/ffi/run/astronet_vetting_scores_cam{cam}_alltriage.csv"
            df = pd.read_csv(csv_file, index_col=False)
        except Exception:
            print(f"All triage QLP Astronet Vetting scores not available for cam {cam}")
        try:
            csv_file = f"/pdo/qlp-data/sector-{sector_num}/ffi/run/astronet_vetting_scores_cam{cam}.csv"
            df = pd.read_csv(csv_file, index_col=False)
        except Exception:
            print("Failed attempt 2")
            found_astronet_scores = False
        
    if found_astronet_scores:
        dfs.append(df)
        combined_df = pd.concat(dfs, ignore_index=True)
    else:
        live_evaluation = LiveEvaluation(model_dir="/pdo/astronet-data/models/vetting/experimental/pablomer/dec2025_cad_scat_v5_duration24/20251217/pablomer-2k-nopretrained/")
        all_eval_files = sector.eval_files
        combined_df = live_evaluation.evaluate(all_eval_files)
                    

    combined_df.drop_duplicates(subset=['astro_id'], inplace=True)
    combined_df = combined_df.loc[:, ~combined_df.columns.str.startswith('Unnamed')]
    return combined_df

def read_toi_data() -> pd.DataFrame:
    return pd.read_csv(LATEST_TOI_DATA, comment="#")[[
        "TIC", "Full TOI ID", "TOI Disposition", "TMag Value", "Orbital Epoch Value", "Orbital Period (days) Value",
        "Transit Duration (hours) Value", "Transit Depth Value", "Planet Number", "Detection Pipeline(s)", "in_2025_csv"
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

def get_production_sector_df(sectors: List[int], custom_model, sector_to_astronet_scores_override: Dict[int, Path], tfrecord_postfix: Optional[str] = None) -> pd.DataFrame:
    all_production_sectors = []

    for sector in sectors:
        production_sector = ProductionSector(sector, tfrecord_postfix=tfrecord_postfix)
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
                per_sector_astronet_scores.append(get_qlp_astronet_scores(sector))
        all_astronet_scores = pd.concat(per_sector_astronet_scores, ignore_index=True)

    disp_cols = ['disp_p', 'disp_e', 'disp_n', 'disp_j']
    fc_cols = [c for c in all_astronet_scores.columns if c.startswith('fc_')]
    disp_cols.extend(fc_cols)
    all_astronet_scores = (
        all_astronet_scores.groupby(['astro_id', 'tic_id', 'planetno'])[disp_cols]
        .mean()
        .reset_index()
    )

    all_properties = pd.concat([x.properties_df for x in all_production_sectors], ignore_index=True)
    all_astro_ids_passing_centroid_filter = [x.astro_ids_passing_centroid_filter for x in all_production_sectors]
    all_astro_ids_passing_centroid_filter = list(chain.from_iterable(all_astro_ids_passing_centroid_filter))
    all_properties["passed_qlp_centroid_filter"] = all_properties['astro_id'].isin(all_astro_ids_passing_centroid_filter)
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

    if False:
        tic_ids = merged_df["tic_id"].dropna().astype(int).unique().tolist()
        print('Fetching from mast...')
        tic_df = fetch_tic_rows_by_id(
            tic_ids,
        )
        tic_df = tic_df.rename(columns={"ID": "tic_id"})
        tic_df["tic_id"] = tic_df["tic_id"].astype(int)
        merged_df["tic_id"] = merged_df["tic_id"].astype(int)
        merged_df = pd.merge(
            merged_df,
            tic_df,
            on="tic_id",
            how="left", 
        )

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
                        'detection_pipeline', 'match_strength', 'in_2025_csv']].copy()
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