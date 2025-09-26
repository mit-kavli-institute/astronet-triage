
import pandas as pd
import numpy as np
from astropy.table import join, Table, vstack
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special
from tqdm.notebook import tqdm


uses_sectors = True
#model_predictions = "/pdo/users/dimond/astronet/astronet/20250429_181612_predictions_sector86.csv"
#model_predictions = "/pdo/users/dimond/astronet/astronet/20250723_modelwithtoisremoved_predictions_sector86.csv"
#model_predictions = "/pdo/astronet-data/models/vetting/experimental/dimond/entire_vetting_09_06_2025/test_predictions.csv" # combined sectors 85, 86, and 87
model_predictions = '/pdo/astronet-data/models/vetting/experimental/dimond/sector_87_reprocessed_with_embeddings/test_predictions.csv'
#tce_catalog = '/pdo/astronet-data/data/properties/tces-sector87_with_labels.csv'
#tce_catalog = "/pdo/astronet-data/data/labels/tces-vetting-v01-tois-triageJs-nocentroid-april2025-all.csv"
#tce_catalog = "/pdo/users/dimond/astronet/astronet/astronet-vetting-tce-catalog-with-offsets-CORRECTED.csv"
#qlp_labels = "/pdo/users/dimond/astronet/astronet/pcs.ls"
tce_catalog = "/pdo/astronet-data/data/labels/sector_85_to_87_analysis.csv"
vetter_labels = "/pdo/users/dimond/sectors_85_to_87_candidates.ls"
toi_info = "/pdo/users/dimond/astronet/astronet/toi-plus-2025-07-16.csv"
toi_notes = "/pdo/users/dimond/astronet/astronet/Astronet Testing TOIs - s86.csv"
output_file = '/pdo/astronet-data/data/labels/sector-87-reprocessed-with-embeddings.csv'



# model_predictions_df = pd.read_csv(model_predictions)
# tce_catalog_df = pd.read_csv(tce_catalog)
# qlp_labels_df = pd.read_csv(qlp_labels)
# toi_info_df = pd.read_csv(toi_info)

# print(model_predictions_df)
# print(tce_catalog_df)
# print(qlp_labels_df)
# print(toi_info_df)

astronet_results = pd.read_csv(
    model_predictions,
    #names=["astro_id", "tic_id", "planetno", "model_no", "disp_p", "disp_e", "disp_n", "disp_j"],
    header=0
)
astronet_results = astronet_results.rename(columns={'Sector': 'sector', 'Astro ID': 'astro_id'})
astronet_data = pd.read_csv(
    tce_catalog,
)[["astro_id", "tic_id", "planetno", "tmag", "period", "epoch", "depth", "duration", "centroid_distance_arcsec", "sector"]]
astronet_data["astro_id"] = astronet_data["astro_id"].astype(int)
astronet_data = astronet_data.drop_duplicates(subset=['astro_id'])#(subset=['tic_id', 'planetno'])

if not uses_sectors:
    astronet_data = astronet_data.drop(columns=['decision', 'as', 'ch', 'disp_b', 'disp_e', 'disp_j', 'disp_n', 'disp_p', 'disp_t', 'disp_u', 'dm', 'et', 'mk', 'md', 'total_votes', 'selected_total_votes'])

astronet_data = astronet_results.merge(astronet_data, on=['tic_id', 'astro_id', 'planetno'], how='right').dropna(subset='model_no')
#astronet_data = astronet_data.merge(astronet_results, on=["astro_id", "tic_id", "planetno"])
astronet_data = Table.from_pandas(astronet_data)

print(len(astronet_results))
print(len(astronet_data))

if uses_sectors:
    toi_data = (
        pd.read_csv(toi_info, comment="#")[[
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
    )

    toi_notes = pd.read_csv(toi_notes)
    toi_notes = toi_notes[['toi_id', 'is_qlp_s86_tev', 'disposition', 'is_first_detection', 'is_updated', 'notes']]
    toi_data = toi_data.merge(toi_notes, on='toi_id', how='left')

    toi_data = Table.from_pandas(toi_data)
    print(toi_data)

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

    print(astronet_data)
    matched_astronet_data = vstack(
        [
            get_ephemeris_matches(row, toi_data[toi_data["tic_id"] == row["tic_id"]])
            for row in astronet_data
            if row["tic_id"] in toi_data["tic_id"] and row["centroid_distance_arcsec"] < 21
        ]
    )

    # Here, `matched_astronet_data` contains the best match for each signal, but we want to throw out
    # matches that don't meet the criterion \sigma_P, \sigma_T > 3.
    matched_astronet_data = matched_astronet_data[
        (matched_astronet_data["sigma_p"] > 3) & (matched_astronet_data["sigma_t"] > 3)
    ]


    toi_data = matched_astronet_data.to_pandas()
    tce_data = astronet_data.to_pandas()

    tce_data['has_toi'] = tce_data['tic_id'].isin(toi_data['tic_id_astronet'])

    toi_subset = toi_data[['tic_id_astronet', 'planetno_astronet', 'toi_id', 'toi_disposition',
                        'detection_pipeline', 'match_strength', 'is_qlp_s86_tev', 'disposition', 'is_first_detection', 'is_updated', 'notes']].copy()
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

    # now add operator_passed
    operator_signals = Table.read(vetter_labels, names=["tic_id", "planetno"], format="ascii").to_pandas()
    operator_pairs = set(operator_signals.apply(tuple, axis=1))

    tce_data['operator_passed'] = tce_data.apply(
        lambda row: (row['tic_id'], row['planetno']) in operator_pairs,
        axis=1
    )

    # simulate vetter_passed
    # def compute_likeliness(row):
    #     base = row['disp_p']
    #     if row['operator_passed']:
    #         base += 0.2  # boost confidence if operator passed
    #     return np.clip(base + np.random.normal(0, 0.1), 0, 1)  # add noise, keep between 0–1

    # likeliness = tce_data.apply(compute_likeliness, axis=1)
    tce_data['vetter_passed'] = False
    print(len(tce_data[tce_data['has_toi']]))
else:
    tce_data = astronet_data.to_pandas()

print(f'Saved to {output_file}')
tce_data.to_csv(output_file, index=False)

"""
tce_data = tce_data.drop(tce_data.filter(regex="^fc_").columns, axis=1)
tce_data = tce_data.drop(columns=['sector', 'model_no', 'disp_p', 'disp_e', 'disp_n', 'disp_j'])
tce_data['true_label'] = tce_data['first_letter']
tce_data.to_csv('/pdo/astronet-data/data/labels/vetting_09_04_2025_test.csv')
print(tce_data["true_label"].value_counts())


df1 = pd.read_csv("/pdo/astronet-data/data/labels/sector_85_to_87_analysis_with_embeddings.csv")
df2 = pd.read_csv("/pdo/astronet-data/data/labels/entire-vetting-09-06-2025-with-embeddings.csv")

# Add domain column
df1["domain"] = "sector"
df2["domain"] = "dataset"

# Concatenate, aligning columns by name
combined = pd.concat([df1, df2], ignore_index=True)

# Save
combined.to_csv("/pdo/astronet-data/data/labels/all_data_embeddings.csv", index=False)
"""

# tce_data['true_label'] = tce_data['first_letter']
# saved_model_results = tce_data[['astro_id', 'tic_id', 'model_no', 'disp_p', 'disp_e', 'disp_n',  'disp_j', 'true_label']]
# saved_model_results.to_csv('/pdo/astronet-data/exodash/cached_model_results/vetting_09_04_2025_test.csv')

# toi_data['tic_id_astronet'] = toi_data['tic_id_astronet'].astype(tce_data['tic_id'].dtype)
# joined_df = toi_data.merge(
#     tce_data,
#     left_on='tic_id_astronet',
#     right_on='tic_id',
#     how='right',
#     suffixes=('_toi', '_tce')  # optional, helps distinguish overlapping column names
# )
# print(joined_df.columns)

# operator_signals = Table.read("data/pcs.ls", names=["tic_id", "planetno"], format="ascii")
# operator_signals[:5]





# # modify the true_labels column (p --> true, j --> false) and drop unnamed:0 
# if 'Unnamed: 0' in base_properties_df.columns:
#     base_properties_df = base_properties_df.drop(columns=['Unnamed: 0'])
# base_properties_df['operator_passed'] = base_properties_df['true_label'].map({'p': True, 'j': False})
# base_properties_df = base_properties_df.drop(columns=['true_label'])




# toi_info_df['TIC'] = toi_info_df['TIC'].astype(base_properties_df['tic_id'].dtype)
# merged_df = base_properties_df.merge(
#     toi_info_df,
#     left_on='tic_id',
#     right_on='TIC',
#     how='left'  # use 'inner' if you want to drop rows with no match
# )

# print(merged_df.columns)

# add in disp_<> ranking

# simulate vetter labels