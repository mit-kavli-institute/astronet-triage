import streamlit as st
from config_parser import DatasetConfig
from data_management.data_manager import DataManager
from data_management.light_curve_server import LightCurveServer
from exodash.utils.file_io import dataset_selector

# --- Page Config ---
st.set_page_config(page_title="ExoDash", layout="wide")

@st.cache_data
def load_data(_data_manager: DataManager):
    """Load and deduplicate the main dataset."""
    df = _data_manager.get_data_frame()
    df = df.drop_duplicates(subset=["tic_id", "astro_id"])
    return df

@st.cache_data
def load_properties(_data_manager: DataManager):
    """Load additional dataset properties."""
    return _data_manager.properties_df

# --- Data Loading ---
config_path = dataset_selector()
initialized = False
if config_path is not None:
    config = DatasetConfig.from_yaml(config_path)

    if "df" not in st.session_state:
        data_manager = DataManager(config=config)
        st.session_state.df = load_data(data_manager)
        st.session_state.properties = load_properties(data_manager)
        st.session_state.data_manager = data_manager

    if "light_curve_server" not in st.session_state:
        st.session_state.light_curve_server = LightCurveServer()

if config_path is None:
    st.warning("Please select a dataset to continue.")
    st.stop()

df = st.session_state.df

st.title("🔭 ExoDash")
st.write(
    "Navigate through the sidebar to explore model failures, TIC images, and dataset distributions."
)

# Dataset summary
st.header("Dataset Statistics")
st.write(f"**Total TICs:** {df.shape[0]}")
st.write(f"**Total Features:** {df.shape[1]}")

st.subheader("Summary Statistics")
st.write(df.describe())

if "label" in df.columns:
    st.subheader("Label Distribution")
    label_counts = df["label"].value_counts()
    st.bar_chart(label_counts)

st.subheader("Missing Data Overview")
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

if not missing_values.empty:
    st.write(missing_values)
else:
    st.success("No missing values in the dataset!")