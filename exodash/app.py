import streamlit as st
from data_management.data_manager import data_manager

# --- Page Config ---
st.set_page_config(page_title="ExoDash", layout="wide")

# --- Data Loading ---
@st.cache_data
def load_data():
    """Load and deduplicate the main dataset."""
    df = data_manager.get_data_frame()
    df = df.drop_duplicates(subset=["tic_id", "astro_id"])
    return df

@st.cache_data
def load_properties():
    """Load additional dataset properties."""
    return data_manager.properties_df

# --- Initialize Session State ---
if "df" not in st.session_state:
    st.session_state.df = load_data()
    st.session_state.properties = load_properties()

df = st.session_state.df  # Access shared dataset

# --- Title & Intro ---
st.title("🔭 ExoDash")
st.write(
    "Navigate through the sidebar to explore model failures, TIC images, and dataset distributions."
)

# --- Dataset Summary ---
st.header("📊 Dataset Statistics")
st.write(f"**Total TICs:** {df.shape[0]}")
st.write(f"**Total Features:** {df.shape[1]}")

# --- Dataset Split Overview ---
if "split" in df.columns:
    st.subheader("📂 Dataset Split Distribution")
    split_counts = df["split"].value_counts()
    st.bar_chart(split_counts)
    st.write(split_counts)

# --- Summary Statistics ---
st.subheader("📌 Summary Statistics")
st.write(df.describe())

# --- Label Distribution ---
if "label" in df.columns:
    st.subheader("🔍 Label Distribution")
    label_counts = df["label"].value_counts()
    st.bar_chart(label_counts)

# --- Missing Values ---
st.subheader("🚨 Missing Data Overview")
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

if not missing_values.empty:
    st.write(missing_values)
else:
    st.success("✅ No missing values in the dataset!")

# --- Sidebar Navigation ---
st.sidebar.header("📌 Navigation")
st.sidebar.page_link("pages/dataset_exploration.py", label="📈 Dataset Exploration")