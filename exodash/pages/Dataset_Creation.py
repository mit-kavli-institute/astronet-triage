


from astronet.util import config_util
from astronet import models
from astronet.astro_cnn_model import input_ds

from exodash.utils.filter import advanced_filter_sidebar
from exodash.utils.production_sector import ProductionSector, get_production_sector_selector
import streamlit as st
import pandas as pd
import os
import re
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="ExoDash - Dataset Creator", layout="wide")
st.title("Dataset Creation via Existing TFRecords")

if "initialized" not in st.session_state:
    st.session_state.initialized = False

orig_dataset = "/pdo/astronet-data/data/tfrecords/oct2025_original"# st.text_input("Original Dataset Dir (ex: /pdo/astronet-data/data/tfrecords/oct2025_original): ")
get_production_sector_selector()
current_sectors = tuple(sorted(st.session_state.selected_sectors))
if "last_selected_sectors" not in st.session_state:
    st.session_state.last_selected_sectors = None

def find_tfrecord_subdirs(base_dir):
    """Automatically find subfolders containing train/val/test TFRecords."""
    if not os.path.isdir(base_dir):
        raise ValueError(f"Directory not found: {base_dir}")

    split_dirs = {"train": [], "val": [], "test": []}

    for entry in os.listdir(base_dir):
        path = os.path.join(base_dir, entry)
        if not os.path.isdir(path):
            continue

        name_lower = entry.lower()

        # Check name pattern
        if re.search(r"(train|trn)", name_lower):
            split_dirs["train"].append('orig_train:' + path + '/*')
        elif re.search(r"(val|validation|eval)", name_lower):
            split_dirs["val"].append('orig_val:' + path + '/*')
        elif re.search(r"(test|tst)", name_lower):
            split_dirs["test"].append('orig_test:' + path + '/*')

    return split_dirs

subdirs = find_tfrecord_subdirs(orig_dataset)
eval_datasets = []

eval_files = []
properties = pd.read_csv('/pdo/astronet-data/data/labels/tces-vetting-v01-tois-triageJs-nocentroid-april2025-all.csv')

for subdir in subdirs:
    eval_files.extend(subdirs[subdir])
# add sector eval files
for sector in st.session_state.selected_sectors:
    production_sector = ProductionSector(sector)
    eval_files.extend(production_sector.eval_files)
    properties = pd.concat([properties, production_sector.properties_df], ignore_index=True)

#st.write(properties.describe())
#st.write(eval_files)

for file_pattern in eval_files:
    if ":" in file_pattern:
        name, pattern = file_pattern.split(":", 1)
    elif len(eval_files) == 1:
        name, pattern = "eval", file_pattern
    else:
        raise ValueError("Multiple datasets must be named as name:file_pattern")
    eval_datasets.append((name, pattern))

MODEL_CONFIG_PATH = "/pdo/astronet-data/models/vetting/baseline/AstroCNNModelVetting_cshallue_20250429_181612"

config = config_util.load_config(MODEL_CONFIG_PATH)
all_dfs = []

for name, file_pattern in eval_datasets:
    dataset = input_ds.build_eval_dataset(
        file_pattern=file_pattern,
        input_config=config.inputs,
        batch_size=config.hparams.batch_size,
        include_identifiers=True,
        include_labels=False,
    )

    for batch in dataset:
        x, identifiers = batch  # identifiers is a list of TIC IDs

        feature_dict = {}
        for k, v in x.items():
            arr = v.numpy()
            arr = np.squeeze(arr)  # remove extra dimensions like (512,1) → (512,)
            
            # Only include scalar or 1D arrays (per-row features)
            # Shape[0] should equal batch size; others are multi-dimensional
            if arr.ndim == 1:
                feature_dict[k] = arr
            elif arr.ndim == 0:  # scalar broadcast
                feature_dict[k] = np.repeat(arr, len(identifiers))
            else:
                # skip multi-dimensional features like light curves or image data
                continue

        # Build DataFrame only from valid (1D) arrays
        if feature_dict:
            df = pd.DataFrame(feature_dict)
            df.insert(0, "TIC ID", identifiers)
            df["source_name"] = name.lower()
            df["source_path"] = file_pattern
            all_dfs.append(df)

final_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

final_df = final_df.drop(columns=['n_folds', 'secondary_phase', 'local_scale', 'local_scale_0.3', 'local_scale_5.0', 'local_scale_present', 'local_scale_present_0.3', 'local_scale_present_5.0', 'secondary_phase_0.3', 'secondary_phase_5.0', 'secondary_scale', 'secondary_scale_0.3', 'secondary_scale_5.0', 'secondary_scale_present', 'secondary_scale_present_0.3', 'secondary_scale_present_5.0'])

# Convert properties to TFRecord names
# properties = properties.rename(columns={'depth': 'Depth', 'tic_id': 'TIC ID', 'astro_id': 'Astro ID', 'period': 'Period', 'duration': 'Duration', 'tmag': 'Tmag', 'Depth': 'Transit_Depth', 's_mass': 'star_mass', 's_mass_present': 'star_mass_present', 's_rad': 'star_rad', 's_rad_est': 'star_rad_est'})
# properties['star_mass_present'] = properties['star_mass'].notnull().astype(int)
# properties['star_rad_present'] = properties['star_mass'].notnull().astype(int)
# properties['star_rad_est_present'] = properties['star_mass'].notnull().astype(int)

# final_df = final_df[final_df.index.isin(properties.index)]

# # Update values column by column
# for col in final_df.columns:
#     if col in ['source_name', 'source_path']:
#         continue
#     if col in properties.columns:
#         final_df[col] = properties[col]
#     else:
#         # Drop the row if df2 doesn't have this column? 
#         # Actually, dropping by column makes more sense:
#         final_df = final_df.drop(columns=[col])


st.write("✅ Combined DataFrame shape:", final_df.shape)
st.dataframe(final_df.head(50))

def dataset_creator_filtered(df: pd.DataFrame):
    """
    Streamlit page to create train/val/test splits from a big df,
    using advanced sidebar filtering for custom allocation.
    """
    st.title("Filtered Dataset Creator")

    # Initialize session_state
    sector_changed = current_sectors != st.session_state.last_selected_sectors

    should_reset = (
        not st.session_state.initialized
        or sector_changed
    )
    if should_reset:
        st.session_state.unallocated_df = df.copy()
        st.session_state.train_df = pd.DataFrame(columns=df.columns)
        st.session_state.val_df = pd.DataFrame(columns=df.columns)
        st.session_state.test_df = pd.DataFrame(columns=df.columns)

        st.session_state.last_selected_sectors = current_sectors
        st.session_state.initialized = True

    st.subheader("Allocation Status")
    st.write(f"Unallocated rows: {len(st.session_state.unallocated_df)}")
    st.write(f"Train rows: {len(st.session_state.train_df)}")
    st.write(f"Validation rows: {len(st.session_state.val_df)}")
    st.write(f"Test rows: {len(st.session_state.test_df)}")

    # --- Filter sidebar ---
    st.sidebar.header("Advanced Filtering for Allocation")
    filtered_subset = advanced_filter_sidebar(st.session_state.unallocated_df)
    st.write(f"Filtered subset size: {len(filtered_subset)}")
    st.dataframe(filtered_subset.head())

    # --- Allocate filtered rows ---
    allocation_split = st.radio("Assign filtered rows to:", ["train", "val", "test"], horizontal=True)
    if st.button("Re-Allocate Original Dataset"):
        orig_train = df[df['source_name'] == 'orig_train']
        st.session_state.train_df = pd.concat([st.session_state.train_df, orig_train], ignore_index=True)
        orig_val = df[df['source_name'] == 'orig_val']
        st.session_state.val_df = pd.concat([st.session_state.val_df, orig_val], ignore_index=True)
        orig_test = df[df['source_name'] == 'orig_test']
        st.session_state.test_df = pd.concat([st.session_state.test_df, orig_test], ignore_index=True)
    if st.button("Allocate Filtered Subset"):
        if len(filtered_subset) == 0:
            st.warning("No rows in the filtered subset to allocate.")
        else:
            # Add to chosen split
            if allocation_split == "train":
                st.session_state.train_df = pd.concat([st.session_state.train_df, filtered_subset], ignore_index=True)
            elif allocation_split == "val":
                st.session_state.val_df = pd.concat([st.session_state.val_df, filtered_subset], ignore_index=True)
            elif allocation_split == "test":
                st.session_state.test_df = pd.concat([st.session_state.test_df, filtered_subset], ignore_index=True)

    # --- Randomly allocate remaining ---
    st.subheader("Randomly allocate remaining unallocated rows")
    frac_train = st.number_input("Fraction to train", value=0.7, min_value=0.0, max_value=1.0, step=0.05)
    frac_val = st.number_input("Fraction to val", value=0.15, min_value=0.0, max_value=1.0, step=0.05)
    frac_test = st.number_input("Fraction to test", value=0.15, min_value=0.0, max_value=1.0, step=0.05)

    if st.button("Allocate Remaining Randomly"):
        remaining = st.session_state.unallocated_df
        n_total = len(remaining)
        if n_total == 0:
            st.info("No unallocated rows left.")
        else:
            shuffled = remaining.sample(frac=1, random_state=42).reset_index(drop=True)
            n_train = int(frac_train * n_total)
            n_val = int(frac_val * n_total)
            n_test = n_total - n_train - n_val  # remainder

            st.session_state.train_df = pd.concat([st.session_state.train_df, shuffled.iloc[:n_train]], ignore_index=True)
            st.session_state.val_df = pd.concat([st.session_state.val_df, shuffled.iloc[n_train:n_train+n_val]], ignore_index=True)
            st.session_state.test_df = pd.concat([st.session_state.test_df, shuffled.iloc[n_train+n_val:]], ignore_index=True)
            st.session_state.unallocated_df = pd.DataFrame(columns=df.columns)
            st.success(f"Randomly allocated remaining {n_total} rows.")

    # --- Previews ---
    st.subheader("Preview splits")
    st.write("Train")
    st.dataframe(st.session_state.train_df.head())
    st.write("Validation")
    st.dataframe(st.session_state.val_df.head())
    st.write("Test")
    st.dataframe(st.session_state.test_df.head())

    allocated_df = pd.concat(
        [
            st.session_state.train_df,
            st.session_state.val_df,
            st.session_state.test_df,
        ],
        ignore_index=True,
    )
    st.session_state.unallocated_df = (
        st.session_state.unallocated_df[~st.session_state.unallocated_df['TIC ID'].isin(allocated_df['TIC ID'].unique())]
        .reset_index(drop=True)
    )
    st.success(f"Allocated {len(filtered_subset)} rows to {allocation_split}.")
dataset_creator_filtered(final_df)



def write_filtered_tfrecords_from_memory(config, eval_datasets, output_dir):
    """
    Rewrites filtered examples into new TFRecords for train/val/test splits.
    Works on in-memory datasets built from input_ds.build_eval_dataset.
    """
    st.session_state.train_df =  st.session_state.train_df
    st.session_state.val_df =  st.session_state.val_df
    st.session_state.test_df =  st.session_state.test_df
    def sample_ids(ids, n=500):
        ids = np.array(list(ids))
        if len(ids) > n:
            np.random.seed(42)
            return np.random.choice(ids, n, replace=False)
        return ids

    # Get raw unique IDs
    raw_train_ids = st.session_state.train_df['TIC ID'].unique()
    raw_val_ids   = st.session_state.val_df['TIC ID'].unique()
    raw_test_ids  = st.session_state.test_df['TIC ID'].unique()

    # Randomly sample up to 500 for each split
    train_ids = raw_train_ids#sample_ids(raw_train_ids, 100)
    val_ids   = raw_val_ids#sample_ids(raw_val_ids,   100)
    test_ids  = raw_test_ids#sample_ids(raw_test_ids,  100)
    st.write(train_ids)
    os.makedirs(output_dir, exist_ok=True)

    st.session_state.train_df.to_csv(output_dir + 'train.csv', index=False)

    # Define TFRecord writers
    writers = {
        "train": tf.io.TFRecordWriter(os.path.join(output_dir, "train.tfrecord")),
        "val": tf.io.TFRecordWriter(os.path.join(output_dir, "val.tfrecord")),
        "test": tf.io.TFRecordWriter(os.path.join(output_dir, "test.tfrecord")),
    }

    id_sets = {
        "train": set(train_ids),
        "val": set(val_ids),
        "test": set(test_ids),
    }

    def serialize_example(feature_dict):
        """Convert dict of numpy arrays into tf.train.Example"""
        features = {}
        for key, value in feature_dict.items():
            v = np.array(value)
            if v.dtype.kind in ('U', 'S', 'O'):  # string-like
                features[key] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[v.astype('S').tobytes()]))
            elif np.issubdtype(v.dtype, np.floating):
                features[key] = tf.train.Feature(float_list=tf.train.FloatList(value=v.flatten()))
            elif np.issubdtype(v.dtype, np.integer):
                features[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=v.flatten()))
        return tf.train.Example(features=tf.train.Features(feature=features))

    # Iterate through all source datasets
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Count total examples so we know how to scale the progress bar
    total_examples = sum(len(s) for s in id_sets.values())
    counter = 0

    for name, file_pattern in eval_datasets:
        dataset = input_ds.build_eval_dataset(
            file_pattern=file_pattern,
            input_config=config.inputs,
            batch_size=config.hparams.batch_size,
            include_identifiers=True,
            include_labels=False,
        )

        for batch in dataset:
            x, identifiers = batch
            for i, tic_id_tensor in enumerate(identifiers):
                tic_id = int(tic_id_tensor.numpy())

                # Determine split
                split = None
                for s, idset in id_sets.items():
                    if tic_id in idset:
                        split = s
                        break
                if split is None:
                    continue

                # Build feature dict
                feature_dict = {k: v[i].numpy() for k, v in x.items()}
                feature_dict["tic_id"] = np.array(tic_id)
                feature_dict["astro_id"] = np.array(tic_id)

                example = serialize_example(feature_dict)

                # Write (the slow part)
                writers[split].write(example.SerializeToString())

                # ---- Streamlit progress update ----
                counter += 1
                #progress_bar.progress(counter / total_examples)
                status_text.write(f"Processed {counter:,}/{total_examples:,} examples")

    # Close writers
    for w in writers.values():
        w.close()

    status_text.write(f"TFRecords written to: {output_dir}!")

if st.button("Write results"):
    write_filtered_tfrecords_from_memory(config, eval_datasets, '/pdo/users/dimond/tfrecord_exodash/')