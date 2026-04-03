"""
ExoDash — Model Comparison Page
================================
Directly compare two model inference CSVs side-by-side.
Mirrors the structure of the Model Performance page but renders
every diagnostic in paired columns so differences are immediately visible.
"""

from exodash.utils.file_io import model_result_selector
from exodash.utils.filter import advanced_filter_sidebar
from exodash.utils.model_visualization import (
    analyze_features,
    plot_pr_curve,
    plot_prediction_score_distribution,
    show_all_model_performance,
)
from exodash.eval_utils import REQUIRED_MODEL_COLUMNS, EvalUtils
from data_management.type_mapping import HUMAN_LABEL_MAP, PREDICTION_MAPPING, PREDICTION_LABELS

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
)

# ---------------------------------------------------------------------------
# Guard: require landing page state
# ---------------------------------------------------------------------------
if "df" not in st.session_state or "light_curve_server" not in st.session_state:
    st.error("Dataset not found. Please use the landing page first.")
    st.stop()

properties_df = df = st.session_state.df

st.set_page_config(page_title="ExoDash — Model Comparison", layout="wide")
st.title("⚖️ Model Comparison")
st.write(
    "Upload two sets of model inference results to directly compare their "
    "ensemble performance, error patterns, and confidence distributions."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
LABEL_COLS = list(HUMAN_LABEL_MAP.values())
# Confusion matrix display order
CLASS_NAMES = ["Planet", "Eclipsing Binary", "Noise", "Junk"]


def _pr_auc(ens: pd.DataFrame) -> float:
    """Compute macro-averaged PR AUC across all classes."""
    aucs = []
    for class_label, prob_col in HUMAN_LABEL_MAP.items():
        if prob_col not in ens.columns:
            continue
        y_true = (ens["true_label"] == class_label).astype(int)
        if y_true.sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_true, ens[prob_col])
        aucs.append(auc(rec, prec))
    return float(sum(aucs) / len(aucs)) if aucs else 0.0


def _compute_metrics(ens: pd.DataFrame) -> dict:
    """Return a flat dict of scalar metrics for an ensemble result frame."""
    y_true = ens["true_label"]
    y_pred = ens["predicted_label"]
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (w)": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall (w)": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 (w)": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "PR AUC (macro)": _pr_auc(ens),
    }


def _metrics_comparison_table(m1: dict, m2: dict, name1: str, name2: str) -> None:
    """Render a styled metric comparison table with delta column."""
    rows = []
    for metric, v1 in m1.items():
        v2 = m2[metric]
        delta = v2 - v1
        rows.append(
            {
                "Metric": metric,
                name1: f"{v1:.4f}",
                name2: f"{v2:.4f}",
                "Δ (B − A)": f"{delta:+.4f}",
            }
        )
    st.dataframe(pd.DataFrame(rows).set_index("Metric"), use_container_width=True)


def _confusion_matrix_fig(ens: pd.DataFrame, title: str) -> go.Figure:
    cm = confusion_matrix(ens["true_label"], ens["predicted_label"], labels=CLASS_NAMES)
    # Normalise rows so colours reflect per-class recall
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    fig = go.Figure(
        go.Heatmap(
            z=cm_norm,
            x=CLASS_NAMES,
            y=CLASS_NAMES,
            colorscale="Blues",
            showscale=False,
            text=cm,
            texttemplate="%{text}",
            hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="True",
        height=400,
        margin=dict(t=50, b=50, l=80, r=20),
    )
    return fig


def _delta_confusion_fig(ens1: pd.DataFrame, ens2: pd.DataFrame, name1: str, name2: str) -> go.Figure:
    """Heatmap showing count(B) − count(A) for each confusion cell."""
    cm1 = confusion_matrix(ens1["true_label"], ens1["predicted_label"], labels=CLASS_NAMES)
    cm2 = confusion_matrix(ens2["true_label"], ens2["predicted_label"], labels=CLASS_NAMES)
    delta = cm2.astype(int) - cm1.astype(int)

    fig = go.Figure(
        go.Heatmap(
            z=delta,
            x=CLASS_NAMES,
            y=CLASS_NAMES,
            colorscale="RdBu",
            zmid=0,
            text=delta,
            texttemplate="%{text:+d}",
            hovertemplate="True: %{y}<br>Pred: %{x}<br>Δ count: %{text:+d}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Confusion delta ({name2} − {name1})",
        xaxis_title="Predicted",
        yaxis_title="True",
        height=400,
        margin=dict(t=50, b=50, l=80, r=20),
    )
    return fig


def _fpr_fnr_comparison(ens1, ens2, numeric_cols, name1, name2, rate: str = "FPR") -> None:
    """Side-by-side FPR or FNR bar charts for a user-selected property and class."""
    assert rate in ("FPR", "FNR")

    prop = st.selectbox(f"Property for {rate}:", numeric_cols, key=f"{rate}_prop_cmp")
    n_bins = st.slider(f"Bins ({rate}):", 3, 10, 5, key=f"{rate}_bins_cmp")
    label = st.selectbox(f"Class for {rate}:", CLASS_NAMES, key=f"{rate}_label_cmp")

    def compute_rate(group, lbl, r):
        if r == "FPR":
            fp = ((group["predicted_label"] == lbl) & (group["true_label"] != lbl)).sum()
            tn = ((group["predicted_label"] != lbl) & (group["true_label"] != lbl)).sum()
            return fp / (fp + tn) if (fp + tn) > 0 else 0
        else:
            fn = ((group["predicted_label"] != lbl) & (group["true_label"] == lbl)).sum()
            tp = ((group["predicted_label"] == lbl) & (group["true_label"] == lbl)).sum()
            return fn / (fn + tp) if (fn + tp) > 0 else 0

    y_col = "false_positive_rate" if rate == "FPR" else "false_negative_rate"

    col1, col2 = st.columns(2)
    for col_ui, ens, name in [(col1, ens1, name1), (col2, ens2, name2)]:
        tmp = ens.copy()
        if prop not in tmp.columns:
            col_ui.warning(f"`{prop}` not available in {name}.")
            continue
        tmp["bin"] = pd.cut(tmp[prop], bins=n_bins).astype(str)
        rate_by_bin = (
            tmp.groupby("bin", observed=True)
            .apply(lambda g: compute_rate(g, label, rate))
            .reset_index()
        )
        rate_by_bin.columns = ["bin", y_col]
        fig = px.bar(
            rate_by_bin, x="bin", y=y_col,
            title=f"{rate} — '{label}' across {prop} [{name}]",
        )
        col_ui.plotly_chart(fig, use_container_width=True)


def _score_distribution_overlay(ens1, ens2, name1, name2) -> None:
    """Overlay prediction score histograms for a chosen class column."""
    score_col = st.selectbox("Score column to compare:", LABEL_COLS, key="score_dist_cmp")
    combined = pd.concat(
        [
            ens1[[score_col]].assign(model=name1),
            ens2[[score_col]].assign(model=name2),
        ]
    )
    fig = px.histogram(
        combined, x=score_col, color="model",
        barmode="overlay", opacity=0.6, nbins=50,
        title=f"Score distribution: {score_col}",
    )
    st.plotly_chart(fig, use_container_width=True)


def _correlation_comparison(ens1, ens2, numeric_cols, name1, name2) -> None:
    """Bar chart of misclassification correlations for both models, side by side."""
    def _corrs(ens, cols):
        is_wrong = (ens["true_label"] != ens["predicted_label"]).astype(int)
        return pd.Series(
            {c: ens[c].corr(is_wrong) for c in cols if c in ens.columns},
        ).sort_values(key=abs, ascending=True)

    all_cols = [
        c for c in numeric_cols
        if c not in {
            "planetno", "duration_hours", "Teff", "period_days", "period_y",
            "duration_x", "Unnamed:0", "bls_points_pre_transit",
            "bls_points_post_transit", "star_pm_ra", "star_pm_dec",
            "duration_y", "period_x", "snr_bls",
        }
    ]

    corr1 = _corrs(ens1, all_cols)
    corr2 = _corrs(ens2, all_cols)

    # Align on same index
    all_idx = corr1.index.union(corr2.index)
    corr1 = corr1.reindex(all_idx, fill_value=0.0)
    corr2 = corr2.reindex(all_idx, fill_value=0.0)

    compare_df = pd.DataFrame(
        {name1: corr1.values, name2: corr2.values},
        index=all_idx,
    ).reset_index().rename(columns={"index": "Property"})

    fig = px.bar(
        compare_df.melt(id_vars="Property", var_name="Model", value_name="Correlation"),
        x="Correlation", y="Property", color="Model",
        barmode="group", orientation="h",
        title="Property correlations with misclassification",
        color_discrete_sequence=["#4C78A8", "#F58518"],
        height=max(400, len(all_idx) * 28),
    )
    fig.update_layout(margin=dict(l=200))
    st.plotly_chart(fig, use_container_width=True)


def _disagreement_table(ens1, ens2, name1, name2) -> None:
    """Show rows where both models got the true label but predicted differently."""
    merged = pd.merge(
        ens1[["astro_id", "true_label", "predicted_label"] + LABEL_COLS].rename(
            columns={"predicted_label": f"pred_{name1}", **{c: f"{c}_{name1}" for c in LABEL_COLS}}
        ),
        ens2[["astro_id", "true_label", "predicted_label"] + LABEL_COLS].rename(
            columns={"predicted_label": f"pred_{name2}", **{c: f"{c}_{name2}" for c in LABEL_COLS}}
        ),
        on=["astro_id", "true_label"],
        how="inner",
    )
    disagreements = merged[merged[f"pred_{name1}"] != merged[f"pred_{name2}"]].copy()
    st.write(f"**{len(disagreements)} cases** where {name1} and {name2} predict differently.")
    if not disagreements.empty:
        st.dataframe(disagreements, use_container_width=True)


# ---------------------------------------------------------------------------
# Load two model CSVs
# ---------------------------------------------------------------------------
st.header("1 · Load Model Results")

col_load1, col_load2 = st.columns(2)

with col_load1:
    st.subheader("Model A")
    model_name_1 = st.text_input("Name for Model A", value="Model A", key="name_a")
    results_1 = model_result_selector(
        allow_cached_models=True,
        allow_local_navigation=False,
        allow_upload=True,
        allow_direct_path=True,
        key="model_a",           # pass unique key if selector supports it
    )

with col_load2:
    st.subheader("Model B")
    model_name_2 = st.text_input("Name for Model B", value="Model B", key="name_b")
    results_2 = model_result_selector(
        allow_cached_models=True,
        allow_local_navigation=False,
        allow_upload=True,
        allow_direct_path=True,
        key="model_b",
    )

if results_1 is None or results_2 is None:
    st.warning("Please load both Model A and Model B results to continue.")
    st.stop()

for name, res in [(model_name_1, results_1), (model_name_2, results_2)]:
    if not REQUIRED_MODEL_COLUMNS.issubset(res.columns):
        st.error(f"❌ {name} is missing required columns. Check your CSV.")
        st.stop()

# ---------------------------------------------------------------------------
# EvalUtils + ensemble results
# ---------------------------------------------------------------------------
eval_a = EvalUtils(results_1)
eval_b = EvalUtils(results_2)

eclipsing_binary_as_junk = st.sidebar.checkbox("Show eclipsing binaries as junk?", value=False)

use_thresholds = st.sidebar.checkbox("🔧 Use custom thresholds?", value=False)
thresholds = None
if use_thresholds:
    st.sidebar.subheader("Thresholds (applied to both models)")
    thresholds = {
        cls: st.sidebar.slider(
            f"Threshold — {PREDICTION_MAPPING[cls]}",
            0.0, 1.0, 0.5, 0.01,
            key=f"thresh_{cls}",
        )
        for cls in PREDICTION_LABELS
    }

def _get_ensemble(eval_utils, thresholds, eclipsing_binary_as_junk):
    ens = eval_utils.get_ensemble_results(
        thresholds, include_labels=False, include_properties=True, dropna=True
    )
    new_cols = ["astro_id"] + [c for c in properties_df.columns if c not in ens.columns]
    ens = pd.merge(ens, properties_df[new_cols], on="astro_id", how="left")
    if eclipsing_binary_as_junk:
        ens.loc[ens["predicted_label"] == "Eclipsing Binary", "predicted_label"] = "Junk"
    return ens

ens_a_orig = _get_ensemble(eval_a, thresholds, eclipsing_binary_as_junk)
ens_b_orig = _get_ensemble(eval_b, thresholds, eclipsing_binary_as_junk)

# Optional filter (shared sidebar filter — applies to both)
st.sidebar.markdown("---")
st.sidebar.subheader("Filters (applied to both)")
ens_a = advanced_filter_sidebar(ens_a_orig, key="model_a")
ens_b = advanced_filter_sidebar(ens_b_orig, key="model_b")

# Determine shared numeric property columns
base_cols = set(eval_a.get_ensemble_results(thresholds, include_labels=False, include_properties=True, dropna=False).columns)
numeric_property_cols = [
    c for c in properties_df.columns
    if c not in base_cols and c != "astro_id"
    and pd.api.types.is_numeric_dtype(ens_a.get(c, pd.Series(dtype=float)))
]

# ---------------------------------------------------------------------------
# 2 · Per-model performance tables
# ---------------------------------------------------------------------------
st.header("2 · Individual Model Performance")
perf_col1, perf_col2 = st.columns(2)
with perf_col1:
    st.subheader(model_name_1)
    perf_a = eval_a.compute_performance()
    show_all_model_performance(perf_a, key="model_a")
with perf_col2:
    st.subheader(model_name_2)
    perf_b = eval_b.compute_performance()
    show_all_model_performance(perf_b, key="model_b")

# ---------------------------------------------------------------------------
# 3 · Ensemble metric comparison
# ---------------------------------------------------------------------------
st.header("3 · Ensemble Metric Comparison")
m_a = _compute_metrics(ens_a)
m_b = _compute_metrics(ens_b)

# Big delta callouts
metric_cols = st.columns(len(m_a))
for mc, (metric, v_a) in zip(metric_cols, m_a.items()):
    v_b = m_b[metric]
    delta = v_b - v_a
    mc.metric(
        label=metric,
        value=f"{v_b:.4f}",
        delta=f"{delta:+.4f}  vs {model_name_1}",
    )

st.markdown("#### Full comparison table")
_metrics_comparison_table(m_a, m_b, model_name_1, model_name_2)

# ---------------------------------------------------------------------------
# 4 · Confusion matrices
# ---------------------------------------------------------------------------
st.header("4 · Confusion Matrices")
cm_col1, cm_col2, cm_col3 = st.columns(3)
with cm_col1:
    st.plotly_chart(_confusion_matrix_fig(ens_a, model_name_1), use_container_width=True)
with cm_col2:
    st.plotly_chart(_confusion_matrix_fig(ens_b, model_name_2), use_container_width=True)
with cm_col3:
    st.plotly_chart(_delta_confusion_fig(ens_a, ens_b, model_name_1, model_name_2), use_container_width=True)

# ---------------------------------------------------------------------------
# 5 · PR curves — overlay
# ---------------------------------------------------------------------------
st.header("5 · PR Curves")
pr_col1, pr_col2 = st.columns(2)
with pr_col1:
    st.subheader(model_name_1)
    plot_pr_curve(ens_a_orig, ens_a, key="model_a")
with pr_col2:
    st.subheader(model_name_2)
    plot_pr_curve(ens_b_orig, ens_b, key="model_b")

# ---------------------------------------------------------------------------
# 6 · Score distributions
# ---------------------------------------------------------------------------
st.header("6 · Prediction Score Distributions")
tabs_dist = st.tabs([f"{model_name_1}", f"{model_name_2}", "Overlay"])
with tabs_dist[0]:
    plot_prediction_score_distribution(ens_a, key="model_a")
with tabs_dist[1]:
    plot_prediction_score_distribution(ens_b, key="model_b")
with tabs_dist[2]:
    _score_distribution_overlay(ens_a, ens_b, model_name_1, model_name_2)

# ---------------------------------------------------------------------------
# 7 · Feature analysis
# ---------------------------------------------------------------------------
st.header("7 · Feature Analysis")
fa_col1, fa_col2 = st.columns(2)
with fa_col1:
    st.subheader(model_name_1)
    analyze_features(ens_a, key="model_a")
with fa_col2:
    st.subheader(model_name_2)
    analyze_features(ens_b, key="model_b")

# ---------------------------------------------------------------------------
# 8 · FPR / FNR by property bin
# ---------------------------------------------------------------------------
if numeric_property_cols:
    st.header("8 · Error Rates by Property Bin")
    st.markdown("#### False Positive Rate")
    _fpr_fnr_comparison(ens_a, ens_b, numeric_property_cols, model_name_1, model_name_2, "FPR")
    st.markdown("#### False Negative Rate")
    _fpr_fnr_comparison(ens_a, ens_b, numeric_property_cols, model_name_1, model_name_2, "FNR")

    # ---------------------------------------------------------------------------
    # 9 · Property correlation with misclassification
    # ---------------------------------------------------------------------------
    st.header("9 · Property Correlation with Misclassification")
    _correlation_comparison(ens_a, ens_b, numeric_property_cols, model_name_1, model_name_2)

# ---------------------------------------------------------------------------
# 10 · Disagreement analysis
# ---------------------------------------------------------------------------
st.header("10 · Disagreement Analysis")
st.write(
    "Cases where the two models predict **different** labels for the same target, "
    "regardless of which is correct."
)
_disagreement_table(ens_a, ens_b, model_name_1, model_name_2)

# ---------------------------------------------------------------------------
# 11 · Confusion matrix query (joint)
# ---------------------------------------------------------------------------
st.header("11 · Confusion Matrix Query")
q_col1, q_col2 = st.columns(2)
with q_col1:
    selected_true = st.selectbox("True label:", CLASS_NAMES, key="q_true")
with q_col2:
    selected_pred = st.selectbox("Predicted label:", CLASS_NAMES, key="q_pred")

q_a = ens_a[(ens_a["true_label"] == selected_true) & (ens_a["predicted_label"] == selected_pred)]
q_b = ens_b[(ens_b["true_label"] == selected_true) & (ens_b["predicted_label"] == selected_pred)]

tab_a, tab_b = st.tabs([model_name_1, model_name_2])
with tab_a:
    st.write(f"{len(q_a)} records in **{model_name_1}**")
    st.dataframe(q_a, use_container_width=True)
with tab_b:
    st.write(f"{len(q_b)} records in **{model_name_2}**")
    st.dataframe(q_b, use_container_width=True)

# ---------------------------------------------------------------------------
# 12 · Scatter plot
# ---------------------------------------------------------------------------
st.header("12 · Scatter Plot")
sc_col1, sc_col2 = st.columns(2)
with sc_col1:
    l1 = st.selectbox("X axis:", LABEL_COLS, index=0, key="sc_x")
with sc_col2:
    l2 = st.selectbox("Y axis:", LABEL_COLS, index=1, key="sc_y")

labels_to_include = st.multiselect(
    "True labels to include:", CLASS_NAMES, default=CLASS_NAMES, key="sc_filter"
)

if l1 != l2:
    sc_tab1, sc_tab2 = st.tabs([model_name_1, model_name_2])
    for tab, ens, name in [(sc_tab1, ens_a, model_name_1), (sc_tab2, ens_b, model_name_2)]:
        with tab:
            filt = ens[ens["true_label"].isin(labels_to_include)]
            fig = px.scatter(
                filt, x=l1, y=l2, color="true_label",
                opacity=0.6, hover_data=["astro_id"],
                title=f"{l1} vs. {l2} — {name}",
            )
            st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Select two different score columns for the scatter plot.")