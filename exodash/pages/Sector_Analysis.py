from data_management.light_curve_server import ALL_PAGE_TYPES, LightCurveServer
from exodash.utils.filter import advanced_filter_sidebar
from exodash.utils.production_sector import ProductionSector, get_production_sector_df
from exodash.utils.reports import generate_report_for_tic_id, infer_planet_number
from clustering import ClusterParams, Clustering
from exodash.utils.tic_visualization import TICVisualizer
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_venn import venn3
import plotly.express as px
from streamlit_plotly_events import plotly_events



st.set_page_config(page_title="ExoDash - Sector Analysis", layout="wide")
st.title("Production Sector Analysis")

if "selected_sectors" not in st.session_state:
    st.session_state.selected_sectors = set()
if "light_curve_server" not in st.session_state:
        st.session_state.light_curve_server = LightCurveServer()

MODEL_CONFIG_PATH = "/pdo/users/pablomer/mnt/tess/models/vetting/20250502/cshallue/AstroCNNModelVetting_cshallue_20250502_000812"
sectors = list(range(85, 95))
available = {85: True, 86: True, 87: True, 88: False, 89: False, 90: False, 91: False, 92: False, 93: False, 94: True}

custom_model = st.text_input("Custom model dir:")

cols = st.columns(len(sectors))

for i, astro_id in enumerate(sectors):
    with cols[i]:
        disabled = not available[astro_id]
        selected = astro_id in st.session_state.selected_sectors

        if st.button(
            f"{astro_id}",
            key=f"btn_{astro_id}",
            disabled=disabled,
            type="primary" if selected else "secondary",
            use_container_width=True,
        ):
            # Toggle selection
            if selected:
                st.session_state.selected_sectors.remove(astro_id)
            else:
                st.session_state.selected_sectors.add(astro_id)


if len(st.session_state.selected_sectors) == 0:
    st.stop()

sector_to_astronet_scores_override = {
    85: '/pdo/astronet-data/models/vetting/experimental/dimond/sectors_85_to_87_with_embeddings/test_predictions.csv',
    86: '/pdo/astronet-data/models/vetting/experimental/dimond/sectors_85_to_87_with_embeddings/test_predictions.csv',
    87: '/pdo/astronet-data/models/vetting/experimental/dimond/sectors_85_to_87_with_embeddings/test_predictions.csv',
}

df = get_production_sector_df(st.session_state.selected_sectors, custom_model, sector_to_astronet_scores_override=sector_to_astronet_scores_override)

eval_files = []
for sector in st.session_state.selected_sectors:
    production_sector = ProductionSector(sector)
    eval_files.append(production_sector.eval_files)

orig_df = df.copy()
server = st.session_state.light_curve_server

# Dataset summary
st.header("Dataset Statistics")
st.write(f"**Total # Astro IDs:** {df.shape[0]}")
st.write(f"**Total Features:** {df.shape[1]}")

st.subheader("Summary Statistics")
st.write(df.describe())
st.divider() 
orig_df['domain'] = 'sector'
df['domain'] = 'sector'
df = advanced_filter_sidebar(df)
categorical_features = df.select_dtypes(include=["object", "bool", "string[python]"]).columns.tolist()
selected_types = st.sidebar.multiselect("Select Report Page Types", sorted(ALL_PAGE_TYPES), default=['Summary', 'Depth-aperture Correlation', 'Difference Images'])


astronet_threshold = st.slider(
    label='Astronet Threshold (disp_p)',
    min_value=0.0,
    max_value=1.0,
    value=0.75,  # default
    step=0.01
)

left_col, right_col = st.columns(2)

astronet_mask = df['disp_p'] > astronet_threshold
operator_mask = df['operator_passed'] == True
#vetter_mask = df['vetter_passed'] == True
toi_mask = df['has_toi'] == True

toi_df = df[toi_mask]
toi_disp_mask = toi_df['toi_disposition'].isin(['PC', 'CP', 'KP'])
toi_disp_df = toi_df[toi_disp_mask]

thresholds = np.linspace(0.0, 1.0, 100)

# Recall for all TOIs
recalls = [(toi_df['disp_p'] > t).sum() / len(toi_df) for t in thresholds]

# Recall for selected dispositions
recalls_disp = [(toi_disp_df['disp_p'] > t).sum() / len(toi_disp_df) if len(toi_disp_df) > 0 else 0 for t in thresholds]

# Selected recall at the chosen threshold
selected_recall = (toi_df['disp_p'] > astronet_threshold).sum() / len(toi_df)
selected_recall_disp = (toi_disp_df['disp_p'] > astronet_threshold).sum() / len(toi_disp_df) if len(toi_disp_df) > 0 else 0

with left_col:
    fig, ax = plt.subplots()

    # Full TOI line
    ax.plot(thresholds, recalls, label='Recall vs disp_p (all TOIs)', color='blue')
    # Subset disposition line
    ax.plot(thresholds, recalls_disp, label='Recall vs disp_p (PC/CP/KP)', color='green')

    # Selected threshold line
    ax.axvline(astronet_threshold, color='red', linestyle='--', label=f'Selected threshold: {astronet_threshold}')

    # Annotations
    ax.annotate(f"Recall: {selected_recall:.2f}",
                xy=(astronet_threshold, selected_recall),
                xytext=(astronet_threshold + 0.02, selected_recall - 0.1),
                arrowprops=dict(arrowstyle="->", color='black'),
                fontsize=10, backgroundcolor='white')

    # ax.annotate(f"Recall (PC/CP/KP): {selected_recall_disp:.2f}",
    #             xy=(astronet_threshold, selected_recall_disp),
    #             #xytext=(astronet_threshold + 0.02, selected_recall_disp - 0.15),
    #             #arrowprops=dict(arrowstyle="->", color='green'),
    #             fontsize=10, backgroundcolor='white')

    ax.set_xlabel('Astronet disp_p Threshold')
    ax.set_ylabel('Recall on TOI list')
    ax.set_title('TOI Recall vs Astronet disp_p')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# Build sets of indices
astronet = set(df[astronet_mask].index)
operator = set(df[operator_mask].index)
#vetter = set(df[vetter_mask].index)
toi = set(df[toi_mask].index)
all_indices = set(df.index)


# Plot Venn
# fig, ax = plt.subplots()
# venn3([astronet, operator, vetter], (f'Astronet [{len(astronet)}]', f'QLP Operator [{len(operator)}]', f'Vetter [{len(vetter)}]'))
# st.pyplot(fig)

with right_col:
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

subset_df = subset_df.loc[:, ~subset_df.columns.str.startswith("fc_")]

num_to_visualize = st.slider("# of Astro IDs to Visualize", 0, 25, 1)
st.write(f'Showing {len(subset_df)} reports')
i = 0
for idx, row in subset_df.iterrows():
    if i >= num_to_visualize:
        break
    i += 1
    tic_id = row['tic_id']
    astro_id = row['astro_id']
    st.subheader(f'Astro ID: {astro_id} ({i} / {len(subset_df)}), TOI Disposition: {row["toi_disposition"]}')
    st.write(f"Astronet scores: disp_p: {row['disp_p']} disp_e: {row['disp_e']} disp_j: {row['disp_j']}")


    planet_number = infer_planet_number(tic_id=tic_id, astro_id=astro_id)
    pages = server.get_report_pages(tic_id, planet_number=planet_number)
    generate_report_for_tic_id(tic_id=tic_id, planet_number=planet_number, pages=pages, selected_types=selected_types)

df = subset_df

st.divider() 
st.header('Clustering Analysis')
params = ClusterParams(
    pca_components=16, whiten=True, cosine_normalize=True,
    min_cluster_size=5, min_samples=5, cluster_selection_epsilon=0.01,
    umap_n_neighbors=15, umap_min_dist=0.05,
    postassign_prob_floor=0.35,
)

clu = Clustering(
    df=orig_df,
    eval_files=eval_files,
    config_path=MODEL_CONFIG_PATH,
    view_key="global_view",
    params=params,
)

use_embeddings = st.checkbox("Use embeddings (vs global view)?", value=False)
data_source = 'embeddings' if use_embeddings else 'tfrecords'
clu.load_views(data_source=data_source)#ids_to_filter=set(df["astro_id"]))   # TFRecords -> id_to_view -> X
clu.fit_pca()          # PCA/HDBSCAN/UMAP + soft memberships


# For ExoDash: get a dataframe to plot & filter
visualizer = TICVisualizer(server=server, df=df)

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


num_to_visualize_2 = st.slider("# to Visualize", 0, 25, 1)
num_viz = 0
for astro_id in selected_ids:
    if num_viz >= num_to_visualize_2:
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