from exodash.utils.filter import advanced_filter_sidebar
import streamlit as st
import pandas as pd
import plotly.express as px
from config_parser import DatasetConfig
from data_management.data_manager import DataManager
from data_management.light_curve_server import LightCurveServer
from exodash.utils.file_io import dataset_selector

# --- Page Config ---
st.set_page_config(page_title="ExoDash", layout="wide")

st.title("🔭 ExoDash")
st.write(
    "Navigate through the sidebar to explore model failures, TIC images, and dataset distributions."
)

@st.cache_data
def load_data(config_path: str):
    """Load and deduplicate the main dataset."""
    config = DatasetConfig.from_yaml(config_path)
    manager = DataManager(config=config)
    df = manager.get_data_frame()
    df = df.drop_duplicates(subset=["tic_id", "astro_id"])
    return df, manager

# --- Data Loading ---
config_path = dataset_selector()

if config_path not in st.session_state:
    st.session_state.config_path = None

if config_path is not None:
    config = DatasetConfig.from_yaml(config_path)
    if config_path != st.session_state.config_path:
        for key in ["df", "data_manager"]:
            st.session_state.pop(key, None)

    if "df" not in st.session_state or config_path != st.session_state.config_path:
        #data_manager = DataManager(config=config)
        st.session_state.df, st.session_state.data_manager = load_data(config_path)
        st.session_state.config_path = config_path
        #st.session_state.data_manager = data_manager

    if "light_curve_server" not in st.session_state:
        st.session_state.light_curve_server = LightCurveServer()

if config_path is None:
    st.warning("Please select a dataset to continue.")
    st.stop()

df = st.session_state.df

# Dataset summary
st.header("Dataset Statistics")
st.write(f"**Total # Astro IDs:** {df.shape[0]}")
st.write(f"**Total Features:** {df.shape[1]}")

st.subheader("Summary Statistics")
st.write(df.describe())

df = advanced_filter_sidebar(df)
categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

st.subheader("Feature Distribution")
features = df.columns[1:]
hist_feature = st.selectbox("Select Feature", features)
nbins = st.slider("Number of Bins", 5, 100, 50, step=5)


color_options = ["None"] + categorical_features
default_col = "true_label"
color_col = st.selectbox("Color By", color_options, index=color_options.index(default_col) if default_col in color_options else 0)

if pd.api.types.is_numeric_dtype(df[hist_feature]):
    filtered_df = df.dropna(subset=[hist_feature])
    fig_hist = px.histogram(
        filtered_df,
        x=hist_feature,
        color=color_col if color_col is not "None" else None,
        nbins=nbins,
        title=f"Distribution of {hist_feature}"
    )
else:
    filtered_df = df.dropna(subset=[hist_feature])
    fig_hist = px.histogram(
        filtered_df,
        x=hist_feature,
        color=color_col,
        title=f"Categorical Distribution of {hist_feature}"
    )

st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("Scatter Plot")

x_options = filtered_df.columns[1:]
default_x = "tic_id"
default_y = "per"

x_col = st.selectbox("X-axis", filtered_df.columns[1:], index=x_options.get_loc(default_x) if default_x in x_options else 0)
y_col = st.selectbox("Y-axis", filtered_df.columns[1:], index=x_options.get_loc(default_y) if default_y in x_options else 0)

# drop na for x col and y col
filtered_df = filtered_df.dropna(subset=[x_col, y_col])

fig_scatter = px.scatter(
    filtered_df,
    x=x_col,
    y=y_col,
    color=color_col if color_col != "None" else None,
    title="Feature Correlation"
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("Missing Data Overview")
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

if not missing_values.empty:
    st.write(missing_values)
else:
    st.success("No missing values in the dataset!")