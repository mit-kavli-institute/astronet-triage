import os
from typing import Optional
import pandas as pd
import streamlit as st

EXODASH_DATASET_DIR = "/pdo/astronet-data/exodash/dataset_configs/"
ROOT_DIR = "/pdo/astronet-data/"
CACHE_MODEL_DIR = "/pdo/astronet-data/exodash/cached_model_results"

@st.cache_data
def load_uploaded_df(uploaded_file):
    return pd.read_csv(uploaded_file)

def list_subdirs_and_files(path):
    """List subdirectories and files at given path."""
    items = os.listdir(path)
    dirs = [d for d in items if os.path.isdir(os.path.join(path, d))]
    files = [f for f in items if os.path.isfile(os.path.join(path, f))]
    return dirs, files

def dataset_selector() -> str:
    """
    For loading primary dataset and properties (TIC properties, labels, etc.)

    Allows the user to select a dataset from a list of supported datasets.
    """
    _, files = list_subdirs_and_files(EXODASH_DATASET_DIR)
    st.subheader("Select Dataset")
    selected_dataset = st.selectbox("Dataset", ["<Select a config>"] + files)
    if selected_dataset != "<Select a config>":
        file_path = os.path.join(EXODASH_DATASET_DIR, selected_dataset)
        st.success(f"Selected dataset: {file_path}\n\nPlease wait, loading...")
        return file_path
    
    return None


def annotation_file_selector():
    """
    For use in the labelling page.
    
    Allows the user to select which annotation file to use.
    """

def local_navigation_handler(key: str = "") -> Optional[pd.DataFrame]:
    """key: unique prefix to namespace all session_state keys for this instance."""
    _dir_key = f"{key}_current_dir"

    st.subheader("Browse Files")
    if st.button("⬆️ Go up one level", key=f"{key}_go_up"):
        st.session_state[_dir_key] = os.path.dirname(st.session_state[_dir_key])
        st.rerun()

    dirs, files = list_subdirs_and_files(st.session_state[_dir_key])
    selected_dir = st.selectbox(
        f"Folders (in: {st.session_state[_dir_key]})",
        ["<Select a folder>"] + sorted(dirs),
        key=f"{key}_dir_select",
    )

    if selected_dir != "<Select a folder>":
        st.session_state[_dir_key] = os.path.join(st.session_state[_dir_key], selected_dir)
        st.rerun()

    selected_file = st.selectbox(
        "Files",
        ["<Select a file>"] + files,
        key=f"{key}_file_select",
    )
    if selected_file != "<Select a file>":
        file_path = os.path.join(st.session_state[_dir_key], selected_file)
        st.success(f"Selected file: {file_path}")
        return load_uploaded_df(file_path)
    return None


def upload_handler(key: str = "") -> Optional[pd.DataFrame]:
    """key: unique prefix to namespace all session_state keys for this instance."""
    _upload_key = f"{key}_uploaded_file"

    st.subheader("Upload CSV")
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        key=f"{key}_file_uploader",
    )
    if uploaded_file is not None:
        st.session_state[_upload_key] = uploaded_file
        st.success("Uploaded file successfully")

    if _upload_key in st.session_state:
        return load_uploaded_df(st.session_state[_upload_key])
    return None


def cached_model_handler(key: str = "") -> Optional[pd.DataFrame]:
    """key: unique prefix to namespace all session_state keys for this instance."""
    _, files = list_subdirs_and_files(CACHE_MODEL_DIR)
    st.subheader("Cached Models")
    selected_model = st.selectbox(
        "Cached Models",
        ["<Select a model result>"] + files,
        key=f"{key}_cached_select",
    )
    if selected_model != "<Select a model result>":
        file_path = os.path.join(CACHE_MODEL_DIR, selected_model)
        st.success(f"Selected model: {file_path}\n\nPlease wait, loading...")
        return load_uploaded_df(file_path)
    return None


def direct_path_handler(key: str = "") -> Optional[pd.DataFrame]:
    """
    Load a CSV file or combine multiple result files from a directory.
    Persists the DataFrame in st.session_state so it's remembered across reruns.

    key: unique prefix to namespace all session_state keys for this instance.
    """
    _path_key = f"{key}_direct_path"
    _df_key = f"{key}_direct_df"

    st.subheader("Load from Direct Path")

    if _path_key not in st.session_state:
        st.session_state[_path_key] = ""
    if _df_key not in st.session_state:
        st.session_state[_df_key] = None

    path_input = st.text_input(
        "Enter path to CSV file or results directory",
        value=st.session_state[_path_key],
        placeholder="/path/to/file.csv or /path/to/results/",
        key=f"{key}_path_input",
    )

    st.session_state[_path_key] = path_input

    if st.button("Load from path", key=f"{key}_load_btn") or st.session_state[_df_key] is None:
        path = os.path.expanduser(os.path.expandvars(path_input))

        if not path:
            st.warning("Please enter a path.")
            st.session_state[_df_key] = None
        elif not os.path.exists(path):
            st.error(f"Path not found: {path}")
            st.session_state[_df_key] = None
        elif os.path.isfile(path):
            if not path.lower().endswith(".csv"):
                st.error("Only CSV files are supported.")
                st.session_state[_df_key] = None
            else:
                try:
                    st.success(f"Loading single file: {path}")
                    st.session_state[_df_key] = load_uploaded_df(path)
                except Exception as e:
                    st.error(f"Failed to load file: {e}")
                    st.session_state[_df_key] = None
        elif os.path.isdir(path):
            st.info(f"Scanning directory: {path}")
            combined_dfs = []
            model_no = 1
            for subdir_name in sorted(os.listdir(path)):
                subdir_path = os.path.join(path, subdir_name)
                if not os.path.isdir(subdir_path):
                    continue
                target_file = os.path.join(subdir_path, "evaluation", "test_exodash_results.csv")
                if os.path.exists(target_file):
                    try:
                        df = pd.read_csv(target_file)
                        df["model_no"] = model_no
                        combined_dfs.append(df)
                        model_no += 1
                    except Exception as e:
                        st.warning(f"Failed to load {target_file}: {e}")
                else:
                    st.warning(f"No test_exodash_results.csv in {subdir_path}")
            if combined_dfs:
                st.session_state[_df_key] = pd.concat(combined_dfs, ignore_index=True)
                st.success(f"Loaded {len(combined_dfs)} model result files.")
            else:
                st.error("No valid result files found in the provided directory.")
                st.session_state[_df_key] = None
        else:
            st.error(f"Invalid path: {path}")
            st.session_state[_df_key] = None

    return st.session_state[_df_key]


def model_result_selector(
    allow_local_navigation: bool = True,
    allow_direct_path: bool = True,
    allow_upload: bool = True,
    allow_cached_models: bool = True,
    local_root_dir: str = ROOT_DIR,
    key: str = "",  # <-- NEW: unique prefix so multiple instances don't collide
) -> Optional[pd.DataFrame]:
    """
    Display selectable methods for loading model results.
    Each method (local, upload, cached) is placed in its own column if enabled.

    Pass a unique `key` when using more than one instance on the same page,
    e.g. key="model_a" and key="model_b" on the comparison page.
    """
    _dir_key = f"{key}_current_dir"
    # Only reset current_dir on first render for this key
    if _dir_key not in st.session_state:
        st.session_state[_dir_key] = local_root_dir

    loaders = []

    if allow_cached_models:
        loaders.append(cached_model_handler)
    if allow_upload:
        loaders.append(upload_handler)
    if allow_local_navigation:
        loaders.append(local_navigation_handler)
    if allow_direct_path:
        loaders.append(direct_path_handler)

    cols = st.columns(len(loaders))
    for col, loader in zip(cols, loaders):
        with col:
            result = loader(key=key)
            if result is not None:
                return result
    return None