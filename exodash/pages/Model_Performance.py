from exodash.utils.file_io import model_result_selector
from exodash.utils.filter import advanced_filter_sidebar
from exodash.utils.mast import fetch_tic_rows_by_id
from exodash.utils.model_visualization import analyze_features, plot_pr_curve, plot_prediction_score_distribution, show_all_model_performance
from exodash.utils.reports import generate_report_for_tic_id, infer_planet_number
import streamlit as st
import pandas as pd
from data_management.light_curve_server import ALL_PAGE_TYPES
from data_management.type_mapping import HUMAN_LABEL_MAP, PREDICTION_MAPPING, PREDICTION_LABELS
from exodash.eval_utils import REQUIRED_MODEL_COLUMNS, EvalUtils
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.express as px

if "df" not in st.session_state or "light_curve_server" not in st.session_state:
    st.error("Dataset not found. Please use the landing page first.")
    st.stop()

properties_df = df = st.session_state.df
server = st.session_state.light_curve_server

# Streamlit UI
st.set_page_config(page_title="ExoDash - Model Performance", layout="wide")
st.title("Model Performance Overview")
st.write("Compare individual model performance against ensemble predictions. Upload model inference results to analyze predictions and errors.")

def _find_interesting_astro_ids(ensemble_results, N=5):
    """
    Selects the most interesting Astro IDs based on:
    1. High-confidence misclassification.
    2. Low-confidence correct predictions.
    3. High variance in predictions.

    Returns a list of dictionaries containing:
    - astro_id
    - true_label
    - predicted_label
    - disp scores
    - reason for selection
    """

    # Step 1: Find misclassified samples
    misclassified = ensemble_results[ensemble_results["true_label"] != ensemble_results["predicted_label"]].copy()
    probability_columns = list(HUMAN_LABEL_MAP.values())
    misclassified["max_wrong_confidence"] = misclassified[probability_columns].max(axis=1)
    misclassified = misclassified.sort_values(by="max_wrong_confidence", ascending=False)
    misclassified["selection_reason"] = "High-confidence misclassification"

    # Step 2: Find low-confidence correct predictions
    correct_predictions = ensemble_results[ensemble_results["true_label"] == ensemble_results["predicted_label"]].copy()
    correct_predictions["max_confidence"] = correct_predictions[probability_columns].max(axis=1)
    correct_predictions = correct_predictions.sort_values(by="max_confidence", ascending=True)
    correct_predictions["selection_reason"] = "Low-confidence correct prediction"

    # Step 3: Find high-variance cases (same Astro ID has different model scores)
    model_variance = ensemble_results.groupby("astro_id")[probability_columns].std().sum(axis=1)
    high_variance_cases = model_variance.sort_values(ascending=False).head(N).index.tolist()
    high_variance_df = ensemble_results[ensemble_results["astro_id"].isin(high_variance_cases)].copy()
    high_variance_df["selection_reason"] = "High variance in predictions"

    # Merge and limit results to N
    selected_cases = pd.concat([misclassified.head(N), correct_predictions.head(N), high_variance_df.head(N)])
    selected_cases = selected_cases.drop_duplicates(subset=["astro_id"]).head(N)

    return selected_cases.to_dict(orient="records")  # Convert to list of dictionaries

# --- Model result loading --- 
individual_model_results = model_result_selector(allow_cached_models=True, allow_local_navigation=False, allow_upload=True, allow_direct_path=True)
if individual_model_results is None:
    st.warning("Please upload model results to continue.")
    st.stop()

st.subheader("Processing Uploaded Model Predictions")
eval_utils = EvalUtils(individual_model_results)
eclipsing_binary_as_junk = st.checkbox("Show eclipsing binaries as junk?", value=False)


if not REQUIRED_MODEL_COLUMNS.issubset(individual_model_results.columns):
    st.error("! ERROR ! Please ensure the model results has all columns:")
    st.stop()
performance_df = eval_utils.compute_performance()
show_all_model_performance(performance_df)

# Sidebar sliders for setting thresholds dynamically
use_thresholds = st.sidebar.checkbox("🔧 Use Custom Thresholds", value=False)

thresholds = None
if use_thresholds:
    st.sidebar.subheader("Adjust Classification Thresholds")
    thresholds = {}
    for class_col in PREDICTION_LABELS:
        thresholds[class_col] = st.sidebar.slider(
            f"Threshold for {PREDICTION_MAPPING[class_col]}",
            min_value=0.0,
            max_value=1.0,
            value=0.5,  # Default threshold
            step=0.01
        )

# Compute ensemble results with the selected thresholds
ensemble_results = eval_utils.get_ensemble_results(thresholds, include_labels=False, include_properties=True, dropna=True)

# Only keep columns from properties_df that don't already exist in ensemble_results (plus the join key)
new_cols = ["astro_id"] + [c for c in properties_df.columns if c not in ensemble_results.columns]
ensemble_results = pd.merge(
    ensemble_results,
    properties_df[new_cols],
    on="astro_id",
    how="left",
)

if eclipsing_binary_as_junk:
    ensemble_results.loc[ensemble_results["predicted_label"] == "Eclipsing Binary", "predicted_label"] = "Junk"

ensemble_results_orig = ensemble_results.copy()
ensemble_results = advanced_filter_sidebar(ensemble_results)

analyze_features(ensemble_results)
# _feature_confidence(ensemble_results)
# _feature_performance_heatmap(ensemble_results)

plot_pr_curve(ensemble_results_orig, ensemble_results)
plot_prediction_score_distribution(ensemble_results)

# Compute ensemble performance metrics
ensemble_acc = accuracy_score(ensemble_results["true_label"], ensemble_results["predicted_label"])
ensemble_prec = precision_score(ensemble_results["true_label"], ensemble_results["predicted_label"], average="weighted", zero_division=0)
ensemble_rec = recall_score(ensemble_results["true_label"], ensemble_results["predicted_label"], average="weighted", zero_division=0)
ensemble_f1 = f1_score(ensemble_results["true_label"], ensemble_results["predicted_label"], average="weighted", zero_division=0)

# Display results
if thresholds:
    st.subheader("Ensemble Model Performance (Custom thresholds)")
else:
    st.subheader("Ensemble Model Performance (Max of average)")
st.write(f"**Accuracy:** {ensemble_acc:.4f}")
st.write(f"**Precision:** {ensemble_prec:.4f}")
st.write(f"**Recall:** {ensemble_rec:.4f}")
st.write(f"**F1-score:** {ensemble_f1:.4f}")


# Confusion Matrix with Query Feature
st.subheader("Confusion Matrix Query")
conf_matrix = confusion_matrix(ensemble_results["true_label"], ensemble_results["predicted_label"], labels=list(PREDICTION_MAPPING.values()))
conf_matrix_df = pd.DataFrame(conf_matrix, index=list(PREDICTION_MAPPING.values()), columns=list(PREDICTION_MAPPING.values()))
st.write("Confusion Matrix:")
st.dataframe(conf_matrix_df)

# --- Feature Correlation with Error Types ---
st.subheader("Feature Correlation with Classification Errors")

# Build error type column
ensemble_results["error_type"] = "Correct"
ensemble_results.loc[
    (ensemble_results["true_label"] != ensemble_results["predicted_label"]),
    "error_type"
] = ensemble_results["true_label"] + " → " + ensemble_results["predicted_label"]

# Identify new columns from properties_df
DROPNA = False
base_cols = set(eval_utils.get_ensemble_results(thresholds, include_labels=False, include_properties=True, dropna=DROPNA).columns)
new_property_cols = [c for c in properties_df.columns if c not in base_cols and c != "astro_id"]
numeric_property_cols = [c for c in new_property_cols if pd.api.types.is_numeric_dtype(ensemble_results[c])]

if not numeric_property_cols:
    st.warning("No new numeric property columns found to correlate.")
else:
    # 1. False Positive Rate per property bin
    st.markdown("#### False Positive Rate by Property Bins")
    fpr_prop = st.selectbox("Select property to bin by (FPR):", numeric_property_cols)
    n_bins = st.slider("Number of bins:", 3, 10, 5)
    
    temp = ensemble_results.copy()
    temp["prop_bin"] = pd.cut(temp[fpr_prop], bins=n_bins)
    fpr_label = st.selectbox("Select class to compute FPR for:", list(PREDICTION_MAPPING.values()))
    
    def compute_fpr(group, label):
        fp = ((group["predicted_label"] == label) & (group["true_label"] != label)).sum()
        tn = ((group["predicted_label"] != label) & (group["true_label"] != label)).sum()
        return fp / (fp + tn) if (fp + tn) > 0 else 0

    fpr_by_bin = temp.groupby("prop_bin", observed=True).apply(lambda g: compute_fpr(g, fpr_label)).reset_index()
    fpr_by_bin.columns = ["bin", "false_positive_rate"]
    fpr_by_bin["bin"] = fpr_by_bin["bin"].astype(str)
    fig_fpr = px.bar(fpr_by_bin, x="bin", y="false_positive_rate",
                     title=f"False Positive Rate for '{fpr_label}' across {fpr_prop} bins")
    st.plotly_chart(fig_fpr)

    # 1b. False Negative Rate per property bin
    st.markdown("#### False Negative Rate by Property Bins")
    fnr_prop = st.selectbox("Select property to bin by (FNR):", numeric_property_cols, key="fnr_prop")
    n_bins_fnr = st.slider("Number of bins (FNR):", 3, 10, 5, key="fnr_bins")

    temp_fnr = ensemble_results.copy()
    temp_fnr["prop_bin"] = pd.cut(temp_fnr[fnr_prop], bins=n_bins_fnr)
    fnr_label = st.selectbox("Select class to compute FNR for:", list(PREDICTION_MAPPING.values()), key="fnr_label")

    def compute_fnr(group, label):
        fn = ((group["predicted_label"] != label) & (group["true_label"] == label)).sum()
        tp = ((group["predicted_label"] == label) & (group["true_label"] == label)).sum()
        return fn / (fn + tp) if (fn + tp) > 0 else 0

    fnr_by_bin = temp_fnr.groupby("prop_bin", observed=True).apply(lambda g: compute_fnr(g, fnr_label)).reset_index()
    fnr_by_bin.columns = ["bin", "false_negative_rate"]
    fnr_by_bin["bin"] = fnr_by_bin["bin"].astype(str)
    fig_fnr = px.bar(fnr_by_bin, x="bin", y="false_negative_rate",
                     title=f"False Negative Rate for '{fnr_label}' across {fnr_prop} bins")
    st.plotly_chart(fig_fnr)

    # 1c. False Positive breakdown by true label per bin
    st.markdown("#### False Positive Composition by True Label")
    fp_only = temp[(temp["predicted_label"] == fpr_label) & (temp["true_label"] != fpr_label)].copy()
    fp_breakdown = fp_only.groupby(["prop_bin", "true_label"], observed=True).size().reset_index(name="count")
    fp_breakdown["prop_bin"] = fp_breakdown["prop_bin"].astype(str)
    fig_fp_breakdown = px.bar(
        fp_breakdown, x="prop_bin", y="count", color="true_label",
        barmode="stack",
        title=f"What are the false positives for '{fpr_label}' in each {fpr_prop} bin?",
        labels={"prop_bin": fpr_prop, "count": "Number of False Positives", "true_label": "True Label"},
    )
    st.plotly_chart(fig_fp_breakdown)

    # 2. Property distributions: correct vs misclassified
    st.markdown("#### Property Distribution: Correct vs. Misclassified")
    dist_prop = st.selectbox("Select property to compare:", numeric_property_cols, key="dist_prop")
    fig_dist = px.histogram(
        ensemble_results, x=dist_prop, color="error_type",
        barmode="overlay", opacity=0.6, nbins=40,
        title=f"Distribution of {dist_prop} by Prediction Outcome"
    )
    st.plotly_chart(fig_dist)

    # 3. Correlation heatmap: numeric properties vs error indicator
    st.markdown("#### Correlation of Properties with Misclassification")
    st.write(ensemble_results[['star_t_eff', 'true_label', 'predicted_label']].notna().sum())

    numeric_property_cols_for_graph = [col for col in numeric_property_cols if col not in ['planetno', 'duration_hours', 'Teff', 'period_days', 'period_y', 'duration_x', 'Unnamed:0', 'bls_points_pre_transit', 'bls_points_post_transit', 'star_pm_ra', 'star_pm_dec', 'duration_y', 'period_x', 'snr_bls']]
    is_misclassified = (ensemble_results["true_label"] != ensemble_results["predicted_label"]).astype(int)
    correlations = pd.Series(
        {col: ensemble_results[col].corr(is_misclassified) 
         for col in numeric_property_cols_for_graph},
    ).sort_values(key=abs, ascending=True)
    fig_corr = px.bar(
        x=correlations.values, y=correlations.index,
        labels={"x": "Correlation with Misclassification", "y": "Property"},
        title="Property Correlations with Misclassification Rate",
        color=correlations.values, color_continuous_scale="RdBu_r",
        orientation="h"
    )
    fig_corr.update_layout(
        height=max(400, len(correlations) * 30),
        margin=dict(l=200)
    )
    st.plotly_chart(fig_corr)
    # 3b. Distribution of high-correlation properties by classification outcome
    st.markdown("#### Distribution of High-Correlation Properties")
    corr_threshold = st.slider("Minimum absolute correlation to show:", 0.0, 1.0, 0.1, 0.01)
    high_corr_props = correlations[correlations.abs() >= corr_threshold].index.tolist()

    if not high_corr_props:
        st.info("No properties meet the correlation threshold.")
    else:
        ensemble_results["Classification"] = ensemble_results.apply(
            lambda r: "Correct" if r["true_label"] == r["predicted_label"] else "Misclassified", axis=1
        )
        for prop in high_corr_props:
            col_data = ensemble_results[prop].dropna()
            p1, p99 = float(col_data.quantile(0.01)), float(col_data.quantile(0.99))
            x_min, x_max = st.slider(
                f"X-axis range for {prop}:",
                min_value=float(col_data.min()),
                max_value=float(col_data.max()),
                value=(p1, p99),
                key=f"range_{prop}"
            )
            filtered = ensemble_results[(ensemble_results[prop] >= x_min) & (ensemble_results[prop] <= x_max)]
            fig_dist = px.histogram(
                filtered, x=prop, color="Classification",
                barmode="overlay", opacity=0.7, nbins=40,
                color_discrete_map={"Correct": "steelblue", "Misclassified": "tomato"},
                title=f"{prop} (r={correlations[prop]:.3f})",
            )
            st.plotly_chart(fig_dist)

    # 4. Per-class breakdown by property
    st.markdown("#### Property vs. Prediction Confidence by True Label")
    conf_prop = st.selectbox("Select property:", numeric_property_cols, key="conf_prop")
    conf_label_col = st.selectbox("Select confidence score column:", list(HUMAN_LABEL_MAP.values()), key="conf_label")
    fig_conf = px.scatter(
        ensemble_results, x=conf_prop, y=conf_label_col,
        color="true_label", symbol="error_type",
        opacity=0.6, title=f"{conf_prop} vs. {conf_label_col} confidence",
        hover_data=["astro_id", "predicted_label"]
    )
    st.plotly_chart(fig_conf)

    

# Scatter plot filters
st.subheader("Scatter Plot with Filters")
labels_to_include = st.multiselect("Select Labels to Include:", list(PREDICTION_MAPPING.values()), default=list(PREDICTION_MAPPING.values()))
filtered_df = ensemble_results[ensemble_results["true_label"].isin(labels_to_include)]

probability_columns = list(HUMAN_LABEL_MAP.values())
l1 = st.selectbox("Select First Disp Column:", probability_columns, index=0)
l2 = st.selectbox("Select Second Disp Column:", probability_columns, index=1)

if l1 != l2:
    fig_scatter = px.scatter(
        filtered_df, x=l1, y=l2, color="true_label",
        title=f"{l1} vs. {l2} Scatter Plot",
        labels={l1: f"{l1} Probability", l2: f"{l2} Probability"},
        opacity=0.6,
        hover_data=["astro_id"]  # Show additional data on hover
    )
    st.plotly_chart(fig_scatter)
else:
    st.warning("Please select two different probability columns for the scatter plot.")

# Select category from matrix
st.subheader("Query TIC IDs from Confusion Matrix")
selected_true_label = st.selectbox("Select True Label:", list(PREDICTION_MAPPING.values()))
selected_pred_label = st.selectbox("Select Predicted Label:", list(PREDICTION_MAPPING.values()))

query_results = ensemble_results[(ensemble_results["true_label"] == selected_true_label) & (ensemble_results["predicted_label"] == selected_pred_label)]
st.write(f"Total matching records: {query_results.shape[0]}")
st.dataframe(query_results)


# Display Misclassified Samples
misclassified = ensemble_results[ensemble_results["true_label"] != ensemble_results["predicted_label"]]
st.write(f"Total Misclassified Samples: {misclassified.shape[0]}")
st.dataframe(misclassified)

N_TO_ANALYZE = st.sidebar.slider(
    f"Number of interesting cases to analyze",
    1, 10
)
num_analyzed = 0
interesting_cases = _find_interesting_astro_ids(ensemble_results, 1000)
cur_case = 0
st.subheader(f"Top {N_TO_ANALYZE} Most Interesting Astro IDs")
st.write("These Astro IDs were selected based on key failure modes: misclassification, uncertainty, or high variance.")
selected_types = st.sidebar.multiselect("Select Report Page Types", sorted(ALL_PAGE_TYPES), default=['Summary', 'Depth-aperture Correlation', 'Difference Images'])

while num_analyzed < N_TO_ANALYZE:
    cur_case += 1
    case = interesting_cases[cur_case]
    astro_id = case["astro_id"]
    true_label = case["true_label"]
    predicted_label = case["predicted_label"]
    disp_scores = {label: case[label] for label in HUMAN_LABEL_MAP.values()}  # Extract disp scores
    selection_reason = case["selection_reason"]
    try:
        astro_props = properties_df[properties_df["astro_id"] == astro_id].to_dict(orient="records")[0]  # Extract single record
    except Exception as e:
        continue


    tic_id = astro_props['tic_id']
    sector = astro_props['sector']
    planet_number = infer_planet_number(tic_id=tic_id, astro_id=astro_id)
    pages = server.get_report_pages(tic_id, planet_number=planet_number, sector=int(sector))
    if not pages:
        st.write(f"No report for astro ID {astro_id}, skipping...")
    else:
        # Display metadata
        st.subheader(f"Report for Astro ID: {astro_id} [{num_analyzed+1}/{N_TO_ANALYZE}]")
        st.write(f"**True Label:** {true_label}, **Predicted Label:** {predicted_label}")
        st.write(f"**disp Scores:** {disp_scores}")
        st.write(f"**Reason for Selection:** {selection_reason}")
        generate_report_for_tic_id(tic_id=tic_id, planet_number=planet_number, pages=pages, selected_types=selected_types, sector=sector)
        num_analyzed += 1