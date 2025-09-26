from clustering import ClusterParams, Clustering
from exodash.utils.filter import advanced_filter_sidebar
from exodash.utils.tic_visualization import TICVisualizer
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
from streamlit_plotly_events import plotly_events

#FILE = "/pdo/astronet-data/data/labels/all_data_embeddings.csv"
#
#FILE = "/pdo/astronet-data/data/labels/sectors_85_to_87_with_test_set_embeddings.csv"
# FILE = "/pdo/astronet-data/data/labels/vetting-09-04-2025-test-with-embeddings.csv"
#"/pdo/astronet-data/data/labels/tces-vetting-v01-tois-triageJs-nocentroid-april2025-all.csv"
#FILE = '/pdo/astronet-data/data/labels/sector_85_to_87_analysis_with_embeddings.csv'
#FILE = '/pdo/astronet-data/data/labels/sector_85_to_87_analysis_with_fixed_tfrecords.csv'
#FILE = '/pdo/astronet-data/data/labels/tces-vetting-v01-tois-triageJs-nocentroid-april2025-all.csv'


FILE = '/pdo/astronet-data/data/labels/sector_85_to_87_analysis.csv'
FILE = '/pdo/astronet-data/data/labels/sector-87-reprocessed-with-embeddings.csv'

viz_file_list = [
    '/pdo/astronet-data/data/labels/sector_85_to_87_analysis.csv',
    '/pdo/astronet-data/data/labels/sector_85_to_87_analysis_with_embeddings.csv',
    '/pdo/astronet-data/data/labels/sector-87-reprocessed-with-embeddings.csv'
]

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
    manager = DataManager(config=None)
    df = pd.read_csv(FILE)#sector_86_multi_data_source.csv')#manager.get_data_frame()
    if FILE in viz_file_list:
        df = df.drop_duplicates(subset=["tic_id", "astro_id", "planetno"])
    df = df.drop_duplicates(subset=["tic_id", "astro_id", "planetno"])
    #print(len(df))
    return df

@st.cache_data
def load_properties(config_path: str):
    """Load additional dataset properties."""
    config = DatasetConfig.from_yaml(config_path)
    manager = DataManager(config=None)
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
        data_manager = DataManager(config=None)
        st.session_state.df = load_data(config_path)
        st.session_state.config_path = config_path
        #st.session_state.properties = load_properties(config_path)
        st.session_state.data_manager = data_manager

    if "light_curve_server" not in st.session_state:
        st.session_state.light_curve_server = LightCurveServer()

if config_path is None:
    st.warning("Please select a dataset to continue.")
    st.stop()

df = st.session_state.df
orig_df = df.copy()
server = st.session_state.light_curve_server

# Dataset summary
st.header("Dataset Statistics")
st.write(f"**Total # Astro IDs:** {df.shape[0]}")
st.write(f"**Total Features:** {df.shape[1]}")

st.subheader("Summary Statistics")
st.write(df.describe())

orig_df['domain'] = 'sector'
df['domain'] = 'sector'
df = advanced_filter_sidebar(df)
categorical_features = df.select_dtypes(include=["object", "bool", "string[python]"]).columns.tolist()
selected_types = st.sidebar.multiselect("Select Report Page Types", sorted(ALL_PAGE_TYPES), default=['Summary', 'Depth-aperture Correlation', 'Difference Images'])


if FILE in viz_file_list:
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
        'All': (astronet | operator | toi),
        'TOI': toi,
        'Operator': operator,
        'Astronet': astronet,
        '[TOI] Astronet ∩ ~Operator': (astronet & (all_indices - operator) & toi),
        '[TOI] Operator ∩ ~Astronet': (operator & (all_indices - astronet) & toi),
        '[TOI] ~Astronet ∩ ~Operator': ((all_indices - astronet) & (all_indices - operator) & toi),
    }

    selected_region = st.selectbox("Select a Venn region to inspect", list(venn_regions.keys()))
    indices = venn_regions[selected_region]
    subset_df = df.loc[list(indices)]

    from exodash.utils.reports import generate_report_for_tic_id, infer_planet_number
    subset_df = subset_df.loc[:, ~subset_df.columns.str.startswith("fc_")]
    st.write(f'Showing {len(subset_df)} reports')
    i = 0
    for idx, row in subset_df.iterrows():
        i += 1
        if i >= 0:
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
    st.write(df)

    
if FILE == '/pdo/astronet-data/data/labels/sectors_85_to_87_with_test_set_embeddings.csv':
    data_dir = '/pdo/astronet-data/data/fits/all/'
    eval_files = ["test:/pdo/astronet-data/data/tfrecords/vetting-aug-2025/*", "test:/pdo/astronet-data/data/tfrecords/sector-85/*", "test:/pdo/astronet-data/data/tfrecords/sector-86/*", "test:/pdo/astronet-data/data/tfrecords/sector-87/*"]
    config_path = "/pdo/users/pablomer/mnt/tess/models/vetting/20250502/cshallue/AstroCNNModelVetting_cshallue_20250502_000812"
    properties_csv = FILE
elif FILE == '/pdo/astronet-data/data/labels/sector_85_to_87_analysis.csv' or FILE == '/pdo/astronet-data/data/labels/sector_85_to_87_analysis_with_embeddings.csv':
    data_dir = '/pdo/astronet-data/data/fits/all/'
    eval_files = ["test:/pdo/astronet-data/data/tfrecords/sector-85/*", "test:/pdo/astronet-data/data/tfrecords/sector-86/*", "test:/pdo/astronet-data/data/tfrecords/sector-87/*"]
    config_path = "/pdo/users/pablomer/mnt/tess/models/vetting/20250502/cshallue/AstroCNNModelVetting_cshallue_20250502_000812"
    properties_csv = FILE
else:
    data_dir = '/pdo/astronet-data/data/fits/sector-87-reprocessed/'
    #data_dir = '/pdo/users/dimond/mnt/tess/fits_files/'
    eval_files = [
        # "test:/pdo/astronet-data/data/tfrecords/vetting-aug-2025-test/*",
        # "test:/pdo/astronet-data/data/tfrecords/vetting-aug-2025-train/*",
        # "test:/pdo/astronet-data/data/tfrecords/vetting-aug-2025-val/*",
        # "test:/pdo/astronet-data/data/tfrecords/sector-85/*",
        # "test:/pdo/astronet-data/data/tfrecords/sector-86/*",
        "test:/pdo/astronet-data/data/tfrecords/sector-87-reprocessed-3/*"
    ]
    config_path = "/pdo/users/pablomer/mnt/tess/models/vetting/20250502/cshallue/AstroCNNModelVetting_cshallue_20250502_000812"
    properties_csv = FILE


params = ClusterParams(
    pca_components=32, whiten=True, cosine_normalize=True,
    min_cluster_size=10, min_samples=10, cluster_selection_epsilon=0.01,
    umap_n_neighbors=15, umap_min_dist=0.05,
    postassign_prob_floor=0.35,
)

clu = Clustering(
    df=orig_df,
    data_dir=data_dir,
    eval_files=eval_files,
    config_path=config_path,
    view_key="global_view",
    params=params,
)

use_embeddings = st.checkbox("Use embeddings (vs global view)?", value=True)
data_source = 'embeddings' if use_embeddings else 'tfrecords'
clu.load_views(data_source=data_source)#ids_to_filter=set(df["astro_id"]))   # TFRecords -> id_to_view -> X
clu.fit_pca()          # PCA/HDBSCAN/UMAP + soft memberships


# For ExoDash: get a dataframe to plot & filter
visualizer = TICVisualizer(server=server, df=df)


"""
"""
for i, astro_id in enumerate(df["astro_id"].head(5)):
    # nearest_neighbors = clu.get_nearest_neighbors(astro_id=astro_id, df=df, n=10, include_self=True, filter_ids=set(df["astro_id"]))
    # neighbors_df = nearest_neighbors.merge(
    #     df, on="query_id", how="left"
    # )
    # st.write(nearest_neighbors)
    # 1/0
    row = df.loc[df['astro_id'] == astro_id].iloc[0]
    tic_id = df.loc[df["astro_id"] == astro_id, "tic_id"].values[0]
    planet_number = int(str(astro_id)[-2:])
    st.write(f'Astro ID: {astro_id} ({i} / {len(df)})')
    st.write(f"Astronet scores: disp_p: {row['disp_p']} disp_e: {row['disp_e']} disp_j: {row['disp_j']}")
    #st.write(f"Disposition: {row['disposition']}, first detection: {row['is_first_detection']}")
    #st.write(f"Notes: {row['notes']}")

    if astro_id < 10000:
        planet_number = 0

    visualizer.visualize_tic_ids(tic_ids=[tic_id], planet_numbers=[planet_number], selected_types=selected_types)
    st.write(astro_id)
    fig = clu.show_nearest_neighbors(astro_id=astro_id, df=df, n=8, layout="grid", cols=4, include_self=True, filter_ids=set(df["astro_id"]))
    st.pyplot(fig, use_container_width=True)
"""
"""

color_by = 'domain'

num_highlight = len(set(df['astro_id']))
num_orig = len(set(orig_df['astro_id']))

highlight_ids = None
if num_highlight <= 0.8 * num_orig:
    highlight_ids = set(df['astro_id'])

show_2d_map = st.checkbox("Show 2D clustering projection?", value=True)

with st.form("cluster_controls"):
    # (build widgets bound to your params object)
    # e.g. self.params.min_cluster_size = st.slider(...)
    recompute = st.form_submit_button("Compute / Recompute")

if recompute or "clu_ready" not in st.session_state:
    st.session_state["clu_ready"] = True
    clu.fit_clusters()

if show_2d_map:
    clu.fit_clusters()
    # controls
    col1, col2, col3 = st.columns(3)
    with col1:
        color_by = st.selectbox(
            "Color by",
            options=['domain'],#[None] + [c for c in orig_df.columns if c != "astro_id"],
            index=0
        )
    with col2:
        select_mode = st.radio("Selection tool", options=["box", "lasso"], horizontal=True, index=0)
    with col3:
        annotate = st.checkbox("Annotate highlights (non-interactive overlay)", value=False)

    # build interactive fig
    fig, to_ids = clu.plot_interactive(
        df=orig_df,
        color_by=color_by,
        select_mode=select_mode,
        # optionally pass your existing highlight_ids for a bold overlay:
        highlight_ids=highlight_ids,
    )

    # render + capture selection
    # Note: set select_event=True to enable box/lasso capture
    selected_points = plotly_events(
        fig,
        select_event=True,
        override_height=640,
        override_width="100%",   # or int
        key="umap_interactive"
    )

    selected_ids = to_ids(selected_points)

    st.caption(f"Selected: {len(selected_ids)} points")
    if selected_ids:
        # show a peek of the selected rows
        st.dataframe(
            orig_df.loc[orig_df["astro_id"].isin(selected_ids)].head(100),
            use_container_width=True
        )

        # handy export
        csv = pd.DataFrame({"astro_id": selected_ids}).to_csv(index=False)
        st.download_button(
            "Download selected astro_id list",
            data=csv,
            file_name="selected_astro_ids.csv",
            mime="text/csv",
        )

        # stash in session_state for downstream tools
        st.session_state["umap_selected_ids"] = selected_ids
# if show_2d_map:
#     clu.fit_clusters()
#     st.pyplot(clu.plot(use_post_labels=False, df=orig_df, color_by=color_by, highlight_ids=highlight_ids))


num_to_visualize = st.slider("# of Astro IDs to Visualize", 0, 25, 1)
num_viz = 0
for astro_id in selected_ids:
    if num_viz >= num_to_visualize:
        break
    # nearest_neighbors = clu.get_nearest_neighbors(astro_id=astro_id, df=df, n=10, include_self=True, filter_ids=set(df["astro_id"]))
    # neighbors_df = nearest_neighbors.merge(
    #     df, on="query_id", how="left"
    # )
    # st.write(nearest_neighbors)
    # 1/0
    try:
        row = orig_df.loc[orig_df['astro_id'] == astro_id].iloc[0]
        tic_id = orig_df.loc[orig_df["astro_id"] == astro_id, "tic_id"].values[0]
        planet_number = int(str(astro_id)[-2:])
        st.write(f'Astro ID: {astro_id}')
        st.write(f"Astronet scores: disp_p: {row['disp_p']} disp_e: {row['disp_e']} disp_j: {row['disp_j']}")
    except Exception:
        continue
    #st.write(f"Disposition: {row['disposition']}, first detection: {row['is_first_detection']}")
    #st.write(f"Notes: {row['notes']}")

    if astro_id < 30000:
        planet_number = 1

    visualizer.visualize_tic_ids(tic_ids=[tic_id], planet_numbers=[planet_number], selected_types=selected_types)
    st.write(astro_id)
    fig = clu.show_nearest_neighbors(astro_id=astro_id, df=df, n=8, layout="grid", cols=4, include_self=True, filter_ids=set(df["astro_id"]))
    st.pyplot(fig, use_container_width=True)
    num_viz += 1

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

# CANDIDATE_FEATURES = [
#     'planetno', 'tmag', 'period', 'epoch', 'depth', 'duration'
# ]
# use_cols = [c for c in CANDIDATE_FEATURES if c in subset_df.columns]

# st.subheader("Clustering")
# st.write(subset_df)
# st.caption(f"Using features: {', '.join(use_cols) if use_cols else '(none found)'}")

# if len(use_cols) < 2:
#     st.warning("Not enough numeric features found to cluster. Add at least 2 (e.g., per, depth, snr).")
# else:
#     import numpy as np
#     import pandas as pd
#     from sklearn.preprocessing import StandardScaler

#     # Clean and scale
#     X = subset_df[use_cols].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
#     keep_idx = X.index
#     scaler = StandardScaler()
#     Xs = scaler.fit_transform(X)

#     # --- Embedding (UMAP) ---
#     import umap.umap_ as umap
#     n_neighbors = st.slider("UMAP n_neighbors", 5, 50, 15, step=1)
#     min_dist = st.slider("UMAP min_dist", 0.0, 0.99, 0.1, step=0.01)
#     emb = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42).fit_transform(Xs)

#     # --- Clustering (HDBSCAN) ---
#     try:
#         import hdbscan
#         min_cluster_size = st.slider("HDBSCAN min_cluster_size", 5, 100, 20, step=5)
#         min_samples = st.slider("HDBSCAN min_samples", 1, 50, 10, step=1)
#         labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(emb)
#     except Exception as e:
#         st.info("HDBSCAN unavailable; falling back to KMeans.")
#         from sklearn.cluster import KMeans
#         k = st.slider("KMeans: number of clusters (k)", 2, 10, 4, step=1)
#         labels = KMeans(n_clusters=k, random_state=42).fit_predict(emb)

#     # Attach results back to subset_df
#     subset_df.loc[keep_idx, 'umap_x'] = emb[:, 0]
#     subset_df.loc[keep_idx, 'umap_y'] = emb[:, 1]
#     subset_df.loc[keep_idx, 'cluster'] = labels

#     # --- Plot ---
#     import plotly.express as px
#     fig_umap = px.scatter(
#         subset_df.loc[keep_idx],
#         x='umap_x',
#         y='umap_y',
#         color='cluster',
#         hover_data=['tic_id', 'astro_id', 'period', 'depth', 'tmag'],
#         title="UMAP of selected Venn region (clusters)"
#     )
#     st.plotly_chart(fig_umap, use_container_width=True)

#     # --- Cluster summaries ---
#     st.subheader("Cluster summaries")
#     valid = subset_df.loc[keep_idx].copy()
#     # numeric summary: median and IQR
#     def iqr(s): return s.quantile(0.75) - s.quantile(0.25)
#     summary = valid.groupby('cluster')[use_cols].agg(['median', iqr, 'count'])
#     # tidy column names
#     summary.columns = [f"{a}_{b}" for a, b in summary.columns]
#     st.dataframe(summary)

#     # Top distinguishing features per cluster (Cohen's d vs rest)
#     import numpy as np
#     rows = []
#     for cl in sorted(valid['cluster'].unique()):
#         A = valid[valid['cluster'] == cl][use_cols].astype(float)
#         B = valid[valid['cluster'] != cl][use_cols].astype(float)
#         muA, muB = A.mean(), B.mean()
#         varA, varB = A.var(), B.var()
#         pooled = np.sqrt(0.5 * (varA + varB)).replace(0, np.nan)
#         d = ((muA - muB) / pooled).abs().sort_values(ascending=False)
#         top = d.head(5).index.tolist()
#         rows.append({"cluster": cl, "top_features": ", ".join(top)})
#     st.table(pd.DataFrame(rows))

st.subheader("Missing Data Overview")
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

if not missing_values.empty:
    st.write(missing_values)
else:
    st.success("No missing values in the dataset!")