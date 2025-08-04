from exodash.utils.filter import advanced_filter_sidebar
import streamlit as st
import pandas as pd
import plotly.express as px
from config_parser import DatasetConfig
from data_management.data_manager import DataManager
from data_management.light_curve_server import ALL_PAGE_TYPES, LightCurveServer
from exodash.utils.file_io import dataset_selector
from matplotlib_venn import venn3
import matplotlib.pyplot as plt
import numpy as np



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
    df = pd.read_csv('/pdo/astronet-data/data/labels/sector_85_to_87_analysis.csv')#manager.get_data_frame()
    #df = df.drop_duplicates(subset=["tic_id", "astro_id", "planetno"])
    #print(len(df))
    return df

@st.cache_data
def load_properties(config_path: str):
    """Load additional dataset properties."""
    config = DatasetConfig.from_yaml(config_path)
    manager = DataManager(config=config)
    return manager.properties_df

# --- Data Loading ---
config_path = dataset_selector()

if config_path not in st.session_state:
    st.session_state.config_path = None

if config_path is not None:
    config = DatasetConfig.from_yaml(config_path)
    if config_path != st.session_state.config_path:
        for key in ["df", "properties", "data_manager"]:
            st.session_state.pop(key, None)

    if "df" not in st.session_state or config_path != st.session_state.config_path:
        data_manager = DataManager(config=config)
        st.session_state.df = load_data(config_path)
        st.session_state.config_path = config_path
        st.session_state.properties = load_properties(config_path)
        st.session_state.data_manager = data_manager

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
categorical_features = df.select_dtypes(include=["object", "bool", "string[python]"]).columns.tolist()

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


astronet_threshold = st.sidebar.slider(
    label='Astronet Threshold (disp_p)',
    min_value=0.0,
    max_value=1.0,
    value=0.75,  # default
    step=0.01
)
astronet_mask = df['disp_p'] > astronet_threshold
operator_mask = df['operator_passed'] == True
vetter_mask = df['vetter_passed'] == True
toi_mask = df['has_toi'] == True

toi_df = df[df['has_toi'] == True]
thresholds = np.linspace(0.0, 1.0, 100)
recalls = []
for t in thresholds:
    tp = (toi_df['disp_p'] > t).sum()  # TOIs that pass the threshold
    fn = (toi_df['disp_p'] <= t).sum() # TOIs that fail
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    recalls.append(recall)
selected_recall = (toi_df['disp_p'] > astronet_threshold).sum() / len(toi_df)


fig, ax = plt.subplots()

ax.plot(thresholds, recalls, label='Recall vs disp_p', color='blue')
ax.axvline(astronet_threshold, color='red', linestyle='--', label=f'Selected threshold: {astronet_threshold}')
ax.annotate(f"Recall: {selected_recall:.2f}",
            xy=(astronet_threshold, selected_recall),
            xytext=(astronet_threshold + 0.02, selected_recall - 0.1),
            arrowprops=dict(arrowstyle="->", color='black'),
            fontsize=10, backgroundcolor='white')
ax.set_xlabel('Astronet disp_p Threshold')
ax.set_ylabel('Recall on TOI list')
ax.set_title('TOI Recall vs Astronet disp_p')
ax.grid(True)
ax.legend()
st.pyplot(fig)

# Build sets of indices
astronet = set(df[astronet_mask].index)
operator = set(df[operator_mask].index)
vetter = set(df[vetter_mask].index)
toi = set(df[toi_mask].index)
all_indices = set(df.index)


# Plot Venn
# fig, ax = plt.subplots()
# venn3([astronet, operator, vetter], (f'Astronet [{len(astronet)}]', f'QLP Operator [{len(operator)}]', f'Vetter [{len(vetter)}]'))
# st.pyplot(fig)

fig, ax = plt.subplots()
venn3([astronet, operator, toi], (f'Astronet [{len(astronet)}]', f'QLP Operator [{len(operator)}]', f'TOI List [{len(toi)}]'))
st.pyplot(fig)

venn_regions = {
    '[TOI] Astronet ∩ ~Operator': (astronet & (all_indices - operator) & toi),
    '[TOI] Operator ∩ ~Astronet': (operator & (all_indices - astronet) & toi),
    '[TOI] ~Astronet ∩ ~Operator': ((all_indices - astronet) & (all_indices - operator) & toi),
}

selected_region = st.selectbox("Select a Venn region to inspect", list(venn_regions.keys()))
indices = venn_regions[selected_region]
subset_df = df.loc[list(indices)]
selected_types = st.sidebar.multiselect("Select Report Page Types", sorted(ALL_PAGE_TYPES), default=['Summary', 'Depth-aperture Correlation', 'Difference Images'])
server = st.session_state.light_curve_server

from exodash.utils.reports import generate_report_for_tic_id, infer_planet_number
st.write(f'Showing {len(subset_df)} reports')
i = 0
for idx, row in subset_df.iterrows():
    i += 1
    if i >= 2:
        break
    tic_id = row['tic_id']
    astro_id = row['astro_id']
    st.write(f'Astro ID: {astro_id} ({i} / {len(subset_df)})')
    st.write(f"Astronet scores: disp_p: {row['disp_p']} disp_e: {row['disp_e']} disp_j: {row['disp_j']}")
    st.write(f"Disposition: {row['disposition']}, first detection: {row['is_first_detection']}")
    st.write(f"Notes: {row['notes']}")


    planet_number = infer_planet_number(tic_id=tic_id, astro_id=astro_id)
    pages = server.get_report_pages(tic_id, planet_number=planet_number)
    generate_report_for_tic_id(tic_id=tic_id, planet_number=planet_number, pages=pages, selected_types=selected_types)

df = subset_df

st.subheader("Missing Data Overview")
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

if not missing_values.empty:
    st.write(missing_values)
else:
    st.success("No missing values in the dataset!")