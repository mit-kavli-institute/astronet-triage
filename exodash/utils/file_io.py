
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

def local_navigation_handler() -> Optional[pd.DataFrame]:
    st.subheader("Browse Files")
    if st.button("⬆️ Go up one level"):
        st.session_state.current_dir = os.path.dirname(st.session_state.current_dir)
        st.rerun()

    dirs, files = list_subdirs_and_files(st.session_state.current_dir)
    selected_dir = st.selectbox(f"Folders (in: {st.session_state.current_dir})", ["<Select a folder>"] + sorted(dirs))

    if selected_dir != "<Select a folder>":
        st.session_state.current_dir = os.path.join(st.session_state.current_dir, selected_dir)
        st.rerun()

    selected_file = st.selectbox("Files", ["<Select a file>"] + files)
    if selected_file != "<Select a file>":
        file_path = os.path.join(st.session_state.current_dir, selected_file)
        st.success(f"Selected file: {file_path}")
        return load_uploaded_df(file_path)
    return None


def upload_handler() -> Optional[pd.DataFrame]:
    st.subheader("Upload CSV")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.success("Uploaded file successfully")

    if 'uploaded_file' in st.session_state:
        return load_uploaded_df(st.session_state.uploaded_file)
    return None


def cached_model_handler() -> Optional[pd.DataFrame]:
    _, files = list_subdirs_and_files(CACHE_MODEL_DIR)
    st.subheader("Cached Models")
    selected_model = st.selectbox("Cached Models", ["<Select a model result>"] + files)
    if selected_model != "<Select a model result>":
        file_path = os.path.join(CACHE_MODEL_DIR, selected_model)
        st.success(f"Selected model: {file_path}\n\nPlease wait, loading...")
        return load_uploaded_df(file_path)
    return None

def model_result_selector(
    allow_local_navigation: bool = True,
    allow_upload: bool = True,
    allow_cached_models: bool = True,
    local_root_dir: str = ROOT_DIR
) -> pd.DataFrame:
    """
    Display selectable methods for loading model results.
    Each method (local, upload, cached) is placed in its own column if enabled.
    """
    st.session_state.pop("current_dir", None)
    st.session_state.current_dir = local_root_dir

    loaders = []

    if allow_cached_models:
        loaders.append(cached_model_handler)
    if allow_upload:
        loaders.append(upload_handler)
    if allow_local_navigation:
        loaders.append(local_navigation_handler)

    cols = st.columns(len(loaders))
    for col, loader in zip(cols, loaders):
        with col:
            result = loader()
            if result is not None:
                return result
    return None