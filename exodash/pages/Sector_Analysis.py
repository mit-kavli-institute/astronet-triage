import math
from typing import Dict, List
from data_management.light_curve_server import ALL_PAGE_TYPES, LightCurveServer
from data_management.live_report_generator import LiveReportGenerator
from exodash.utils.annotation import AnnotationHandler
from exodash.utils.filter import advanced_filter_sidebar
from exodash.utils.production_sector import ProductionSector, get_production_sector_df
from exodash.utils.production_sectors import get_production_sector_selector
from exodash.utils.reports import infer_planet_number
from exodash.utils.reports_tfrecords import TFRecordReports
from clustering import ClusterParams, Clustering
from exodash.utils.tic_visualization import TICVisualizer
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_venn import venn3
import plotly.express as px
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go
import os
from joblib import Memory


st.set_page_config(page_title="ExoDash - Sector Analysis", layout="wide")
st.title("Production Sector Analysis")

if "selected_sectors" not in st.session_state:
    st.session_state.selected_sectors = set()
if "light_curve_server" not in st.session_state:
        st.session_state.light_curve_server = LightCurveServer()


memory = Memory(".cache/joblib", verbose=0)

MODEL_CONFIG_PATH = "/pdo/users/pablomer/mnt/tess/models/vetting/20250502/cshallue/AstroCNNModelVetting_cshallue_20250502_000812"

custom_model = st.text_input("Custom model dir (ex: /pdo/astronet-data/models/vetting/baseline/): ")
tfrecord_postfix = st.text_input("Custom tfrecords postfix for /pdo/astronet-data/data/tfrecords/ (ex: cadencebin)")
get_production_sector_selector(tfrecord_postfix)
if len(st.session_state.selected_sectors) == 0:
    st.stop()
sector_to_astronet_scores_override = {}

@memory.cache
def get_cached_production_sector_df(
    sectors: List[int],
    custom_model: str | None,
    sector_to_astronet_scores_override: Dict[int, str],
    tfrecord_postfix=tfrecord_postfix,
) -> pd.DataFrame:
    return get_production_sector_df(
        sectors,
        custom_model,
        sector_to_astronet_scores_override=sector_to_astronet_scores_override,
        tfrecord_postfix=tfrecord_postfix,
    )

df = get_cached_production_sector_df(st.session_state.selected_sectors, custom_model, sector_to_astronet_scores_override=sector_to_astronet_scores_override, tfrecord_postfix=tfrecord_postfix)
eps = 1e-12
prob_cols = ["disp_p", "disp_e", "disp_n", "disp_j"]  # adjust if disp_n name differs
if all(c in df.columns for c in prob_cols):
    probs = df[prob_cols].clip(eps, 1.0).astype(float)
    probs = probs.div(probs.sum(axis=1), axis=0)  # normalize just in case
    df["pred_entropy"] = -(probs * np.log(probs)).sum(axis=1)
    df["pred_margin"] = df["disp_p"] - probs.drop(columns=["disp_p"]).max(axis=1)

eval_files = []
for sector in st.session_state.selected_sectors:
    production_sector = ProductionSector(sector, tfrecord_postfix=tfrecord_postfix)
    eval_files.extend(production_sector.eval_files)
tfrecord_reports = TFRecordReports(eval_files=eval_files, model_config_path=MODEL_CONFIG_PATH)

orig_df = df.copy()
server = st.session_state.light_curve_server
live_report_generator = LiveReportGenerator()

# For ExoDash: get a dataframe to plot & filter
visualizer = TICVisualizer(server=server, df=df)

# Dataset summary
st.header("Dataset Statistics")
st.write(f"**Total # Astro IDs:** {df.shape[0]}")
st.write(f"**Total Features:** {df.shape[1]}")

st.subheader("Summary Statistics")
st.write(df.describe())

orig_df['domain'] = 'sector'
df['domain'] = 'sector'
df = advanced_filter_sidebar(df)


st.subheader("Feature Distribution")
features = df.columns[1:]
hist_feature = st.selectbox("Select Feature", features)
nbins = st.slider("Number of Bins", 5, 100, 50, step=5)

categorical_features = df.select_dtypes(include=["object", "bool", "string[python]"]).columns.tolist()
color_options = ["None"] + categorical_features
default_col = "true_label"
color_col = st.selectbox("Color By", color_options, index=color_options.index(default_col) if default_col in color_options else 0)

if pd.api.types.is_numeric_dtype(df[hist_feature]):
    filtered_df = df.dropna(subset=[hist_feature])
    fig_hist = px.histogram(
        filtered_df,
        x=hist_feature,
        color=color_col if color_col != "None" else None,
        nbins=nbins,
        barmode="stack", 
        title=f"Distribution of {hist_feature}"
    )
else:
    filtered_df = df.dropna(subset=[hist_feature])
    fig_hist = px.histogram(
        filtered_df,
        x=hist_feature,
        barmode="stack", 
        color=color_col if color_col != "None" else None,
        title=f"Categorical Distribution of {hist_feature}",
    )
st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("Scatter Plot")
log_x = st.checkbox("Log scale X-axis", value=False)

x_options = filtered_df.columns[1:]
default_x = "tic_id"
default_y = "per"

x_col = st.selectbox("X-axis", filtered_df.columns[1:], index=x_options.get_loc(default_x) if default_x in x_options else 0)
y_col = st.selectbox("Y-axis", filtered_df.columns[1:], index=x_options.get_loc(default_y) if default_y in x_options else 0)

# drop na for x col and y col
filtered_df = filtered_df.dropna(subset=[x_col, y_col])

is_period_radius_plot = (
    x_col.lower() == "period" and y_col.lower() in ["planet_radius", "planet_radius_rearth", "rp", "rp_rearth"]
)

if is_period_radius_plot:
    # Keep only positive periods for the power-law boundary
    plot_df = filtered_df[filtered_df[x_col] > 0].copy()

    # Blue shaded cutoff function: f(P) = 30 * P^(-1/3)
    plot_df["rp_cutoff"] = 30.0 * np.power(plot_df[x_col], -1.0 / 3.0)

    # Define passing / non-passing
    # non-passing if above horizontal 30 OR above blue curve
    plot_df["passes_radius_filter"] = (
        (plot_df[y_col] <= 30.0) &
        (plot_df[y_col] <= plot_df["rp_cutoff"])
    )

    passing_df = plot_df[plot_df["passes_radius_filter"]].copy()
    failing_df = plot_df[~plot_df["passes_radius_filter"]].copy()

    # Build x-grid for smooth overlays
    x_min = plot_df[x_col].min()
    x_max = plot_df[x_col].max()
    x_grid = np.logspace(np.log10(x_min), np.log10(x_max), 300)

    y_cut_curve = 30.0 * np.power(x_grid, -1.0 / 3.0)
    y_top = max(plot_df[y_col].max(), 35)

    fig_scatter = go.Figure()

    # Passing points
    fig_scatter.add_trace(
        go.Scatter(
            x=passing_df[x_col],
            y=passing_df[y_col],
            mode="markers",
            name="Passing",
            marker=dict(size=7),
            customdata=passing_df[["astro_id"]],
            hovertemplate=(
                f"{x_col}: %{{x}}<br>"
                f"{y_col}: %{{y}}<br>"
                "astro_id: %{customdata[0]}<extra></extra>"
            ),
        )
    )

    # Non-passing points in red
    fig_scatter.add_trace(
        go.Scatter(
            x=failing_df[x_col],
            y=failing_df[y_col],
            mode="markers",
            name="Non-passing",
            marker=dict(size=8, color="red", symbol="x"),
            customdata=failing_df[["astro_id"]],
            hovertemplate=(
                f"{x_col}: %{{x}}<br>"
                f"{y_col}: %{{y}}<br>"
                "astro_id: %{customdata[0]}<extra></extra>"
            ),
        )
    )

    # Horizontal line at Rp = 30
    fig_scatter.add_trace(
        go.Scatter(
            x=x_grid,
            y=np.full_like(x_grid, 30.0),
            mode="lines",
            name="Rp = 30",
            line=dict(color="black", dash="dash"),
        )
    )

    # Blue cutoff curve
    fig_scatter.add_trace(
        go.Scatter(
            x=x_grid,
            y=y_cut_curve,
            mode="lines",
            name="Cutoff: 30·P^(-1/3)",
            line=dict(color="blue"),
        )
    )

    # Blue shaded region above the curve
    fig_scatter.add_trace(
        go.Scatter(
            x=np.concatenate([x_grid, x_grid[::-1]]),
            y=np.concatenate([np.full_like(x_grid, y_top), y_cut_curve[::-1]]),
            fill="toself",
            fillcolor="rgba(0, 0, 255, 0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="Excluded region",
            showlegend=True,
        )
    )

    fig_scatter.update_layout(
        title="Planet Radius vs Period",
        xaxis_title=x_col,
        yaxis_title=y_col,
    )

    if log_x:
        fig_scatter.update_xaxes(type="log")

else:
    # Default generic scatter behavior
    fig_scatter = px.scatter(
        filtered_df,
        x=x_col,
        y=y_col,
        color=color_col if color_col != "None" else None,
        title="Feature Correlation",
        hover_data=["astro_id"],
    )

    if log_x:
        fig_scatter.update_xaxes(type="log")

st.plotly_chart(fig_scatter, use_container_width=True)

st.divider() 
selected_types = st.sidebar.multiselect("Select Report Page Types", sorted(ALL_PAGE_TYPES), default=['Summary', 'Depth-aperture Correlation', 'Difference Images', 'TFRecord Global View'])


#astronet_threshold = float(st.text_input("Astronet Threshold (disp_p)", 0.75))
astronet_threshold = st.slider(
    label='Astronet Threshold (disp_p)',
    min_value=0.0,
    max_value=1.0,
    value=0.75,  # default
    step=0.001
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
operator_recall = operator_mask[toi_mask].sum() / len(toi_df)

# Recall for selected dispositions
recalls_disp = [(toi_disp_df['disp_p'] > t).sum() / len(toi_disp_df) if len(toi_disp_df) > 0 else 0 for t in thresholds]

# Selected recall at the chosen threshold
selected_recall = (toi_df['disp_p'] > astronet_threshold).sum() / len(toi_df)
selected_recall_disp = (toi_disp_df['disp_p'] > astronet_threshold).sum() / len(toi_disp_df) if len(toi_disp_df) > 0 else 0

sector_recalls_vs_threshold = {}

for sector, g in toi_df.groupby('sector'):
    if len(g) == 0:
        continue
    sector_recalls_vs_threshold[sector] = [
        (g['disp_p'] > t).sum() / len(g)
        for t in thresholds
    ]

with left_col:
    fig, ax = plt.subplots()

    for sector, recalls_sector in sector_recalls_vs_threshold.items():
        ax.plot(
            thresholds,
            recalls_sector,
            linestyle='--',
            linewidth=1,
            alpha=0.35,
        )

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
    
    ax.axhline(
        operator_recall,
        color='green',
        linestyle=':',
        label=f'Operator recall (constant): {operator_recall:.2f}'
    )


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
    'Operator ∩ ~Astronet ∩ ~TOI': ((all_indices - astronet) &  operator & (all_indices - toi)),
    'Operator ∩ ~TOI': (astronet & (all_indices - operator) & (all_indices - toi)),
    'Astronet': astronet,
    'Astronet ∩ ~Operator ∩ ~TOI': (astronet & (all_indices - operator) & (all_indices - toi)),
    'TOIs Astronet Missed': (((all_indices - astronet) & (all_indices - operator) & toi) | (operator & (all_indices - astronet) & toi)),
    '[TOI] Astronet ∩ ~Operator': (astronet & (all_indices - operator) & toi),
    '[TOI] Operator ∩ ~Astronet': (operator & (all_indices - astronet) & toi),
    '[TOI] ~Astronet ∩ ~Operator': ((all_indices - astronet) & (all_indices - operator) & toi),
}

selected_region = st.selectbox("Select a Venn region to inspect", list(venn_regions.keys()))
indices = venn_regions[selected_region]
subset_df = df.loc[list(indices)]

subset_df = subset_df.loc[:, ~subset_df.columns.str.startswith("fc_")]

sortable_cols = subset_df.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

sort_col = st.selectbox(
    "Sort by column",
    options=["(none)"] + sortable_cols,
    index=0
)

sort_ascending = st.radio(
    "Sort order",
    options=["Descending", "Ascending"],
    horizontal=True,
    index=0
)
if sort_col != "(none)":
    subset_df = subset_df.sort_values(
        by=sort_col,
        ascending=(sort_ascending == "Ascending"),
        na_position="last"
    )

num_to_visualize = st.slider("# of Astro IDs to Visualize", 0, 25, 1)
st.write(f'Showing {len(subset_df)} reports')
if sort_col != "(none)":
    st.caption(
        f"Showing top {num_to_visualize} rows sorted by "
        f"{sort_col} ({sort_ascending.lower()})"
    )
i = 0
annotate = True
# --- config ---
PAGE_SIZE = num_to_visualize  # N items per page

# --- init state ---
if "page" not in st.session_state:
    st.session_state.page = 0

total = len(subset_df)
total_pages = max(1, math.ceil(total / PAGE_SIZE))
st.session_state.page = max(0, min(st.session_state.page, total_pages - 1))

start = st.session_state.page * PAGE_SIZE
end = min(start + PAGE_SIZE, total)

# --- pager UI ---
c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
with c1:
    prev_disabled = st.session_state.page == 0
    if st.button("⬅️ Previous", disabled=prev_disabled, use_container_width=True):
        st.session_state.page -= 1
        st.rerun()
with c2:
    next_disabled = st.session_state.page >= total_pages - 1
    if st.button("Next ➡️", disabled=next_disabled, use_container_width=True):
        st.session_state.page += 1
        st.rerun()
with c3:
    st.caption(f"Showing {start + 1}-{end} of {total}  •  Page {st.session_state.page + 1}/{total_pages}")
with c4:
    # optional: jump to page
    new_page = st.number_input("Page", 1, total_pages, st.session_state.page + 1, label_visibility="collapsed")
    if new_page - 1 != st.session_state.page:
        st.session_state.page = new_page - 1
        st.rerun()

# --- slice df for current page ---
page_df = subset_df.iloc[start:end]

@st.fragment
def render_annotation(astro_id, row, model_version, data_version):
    AnnotationHandler(
        astro_id=astro_id,
        row=row,
        model_version=model_version,
        data_version=data_version,
    )

# --- render page items ---
for j, (_, row) in enumerate(page_df.iterrows(), start=start + 1):
    tic_id = row["tic_id"]
    astro_id = row["astro_id"]

    st.subheader(f'Astro ID: {astro_id} ({j} / {total}), TOI Disposition: {row["toi_disposition"]}')
    st.write(f"Astronet scores: disp_p: {row['disp_p']} disp_e: {row['disp_e']} disp_j: {row['disp_j']}")
    st.dataframe(row.to_frame())  # cleaner than row.T

    planet_number = infer_planet_number(tic_id=tic_id, astro_id=astro_id)
    visualizer.visualize_tic_ids(
        tic_ids=[tic_id],
        planet_numbers=[planet_number],
        selected_types=selected_types,
        tfrecord_reports=tfrecord_reports,
    )

    # for page in [0, 1, 2, 3, 5, 6, 7]:
    #     try:
    #         print(f'Trying to get page {page}...')
    #         img_path = live_report_generator.generate_summary(
    #             tic_id=row["tic_id"],
    #             planetno=row["planetno"],
    #             ccd=row["ccd"],
    #             cam=row["cam"],
    #             sector=row["sector"],
    #             page_num=page,
    #         )
    #         st.image(img_path)
    #     except Exception:
    #         st.warning(f"Failed to locate page {page}")

    # render_annotation(
    #     astro_id=astro_id,
    #     row=row,
    #     model_version=os.path.basename(custom_model or MODEL_CONFIG_PATH),
    #     data_version="s" + "s".join(str(s) for s in sorted(st.session_state.selected_sectors)),
    # )

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

color_by = 'sector'

num_highlight = len(set(df['astro_id']))
num_orig = len(set(orig_df['astro_id']))

highlight_ids = None
if num_highlight <= 0.8 * num_orig:
    highlight_ids = set(df['astro_id'])

show_2d_map = st.checkbox("Show 2D clustering projection?", value=True)

highlighted_only = st.checkbox("Filter to only highlighted points (NOT WORKING)?")
if highlighted_only:
    selected_df = df
else:
    selected_df = orig_df

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
            options=['domain', 'sector'],#[None] + [c for c in orig_df.columns if c != "astro_id"],
            index=0
        )
    with col2:
        select_mode = st.radio("Selection tool", options=["box", "lasso"], horizontal=True, index=0)
    with col3:
        annotate = st.checkbox("Annotate highlights (non-interactive overlay)", value=False)

    # build interactive fig
    fig, to_ids = clu.plot_interactive(
        df=selected_df,
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
        st.dataframe(
            selected_df.loc[selected_df["astro_id"].isin(selected_ids)].head(100),
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
    try:
        row = selected_df.loc[selected_df['astro_id'] == astro_id].iloc[0]
        tic_id = selected_df.loc[selected_df["astro_id"] == astro_id, "tic_id"].values[0]
        planet_number = int(str(astro_id)[-2:])
        st.write(f'Astro ID: {astro_id}')
        st.write(f"Astronet scores: disp_p: {row['disp_p']} disp_e: {row['disp_e']} disp_j: {row['disp_j']}")
    except Exception:
        continue

    if astro_id < 30000:
        planet_number = 1

    visualizer.visualize_tic_ids(tic_ids=[tic_id], planet_numbers=[planet_number], selected_types=selected_types, tfrecord_reports=tfrecord_reports)
    st.write(astro_id)
    # TODO fix nearest neighbors
    fig = clu.show_nearest_neighbors(astro_id=astro_id, df=selected_df, n=8, layout="grid", cols=4, include_self=True, filter_ids=set(selected_df["astro_id"]))
    st.pyplot(fig, use_container_width=True)
    num_viz += 1