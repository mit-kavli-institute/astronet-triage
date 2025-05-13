import ast
import os
from typing import List
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from data_management.light_curve_server import PAGE_NUMBER_TO_TYPE, LightCurveServer
from data_management.type_mapping import HUMAN_LABEL_MAP, PREDICTION_MAPPING, PREDICTION_LABELS
from exodash.eval_utils import EvalUtils
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, auc


properties_df = df = st.session_state.df  # Access shared dataset

# Streamlit UI
st.set_page_config(page_title="ExoDash - Model Performance", layout="wide")
st.title("Model Performance Overview")
st.write("Compare individual model performance against ensemble predictions. Upload model inference results to analyze predictions and errors.")

# Root directory (change this to your desired base path on the server)
ROOT_DIR = "/pdo/users/"

# Initialize session state to track current directory
if "current_dir" not in st.session_state:
    st.session_state.current_dir = ROOT_DIR

def list_subdirs_and_files(path):
    """List subdirectories and files at given path."""
    items = os.listdir(path)
    dirs = [d for d in items if os.path.isdir(os.path.join(path, d))]
    files = [f for f in items if os.path.isfile(os.path.join(path, f))]
    return dirs, files



def _advanced_filter_sidebar(df):
    filtered_df = df.copy()  # Start with full dataset

    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    selected_num_filters = st.sidebar.multiselect("Select Numeric Features to Filter", numeric_features)
    for feature in selected_num_filters:
        min_val, max_val = st.sidebar.slider(
            f"Range for {feature}",
            float(df[feature].min()), float(df[feature].max()),
            (float(df[feature].min()), float(df[feature].max()))
        )
        filtered_df = filtered_df[(filtered_df[feature] >= min_val) & (filtered_df[feature] <= max_val)]

    # Allow users to filter categorical features
    selected_cat_filters = st.sidebar.multiselect("Select Categorical Features to Filter", categorical_features)

    for feature in selected_cat_filters:
        unique_values = df[feature].dropna().unique()
        selected_values = st.sidebar.multiselect(f"Filter by {feature}", unique_values, default=unique_values)
        filtered_df = filtered_df[filtered_df[feature].isin(selected_values)]
    
    return filtered_df

def _show_all_model_performance(performance_df):
    metric_to_plot = st.selectbox("Select Metric to Compare", ["accuracy", "precision", "recall", "f1_score"])
    fig = px.bar(performance_df, x="model_no", y=metric_to_plot, text=metric_to_plot, title=f"{metric_to_plot.capitalize()} Across Models")
    st.plotly_chart(fig)

def _plot_pr_curve(ensemble_results_orig, ensemble_results_filtered):
    st.subheader("Precision-Recall Curve (All Classes)")

    # Check if filtering is active (i.e., the filtered df is different from the original)
    filtering_active = not ensemble_results_filtered.equals(ensemble_results_orig)

    # Sidebar option to hide the original PR curve
    hide_orig = st.sidebar.checkbox("Hide Original PR Curve", value=False) if filtering_active else False

    # Get class sample counts
    class_counts_orig = ensemble_results_orig["true_label"].value_counts().to_dict()
    class_counts_filtered = ensemble_results_filtered["true_label"].value_counts().to_dict() if filtering_active else {}

    fig, ax = plt.subplots(figsize=(16, 9))  # Set figure size

    for class_label, prob_column in HUMAN_LABEL_MAP.items():
        count_orig = class_counts_orig.get(class_label, 0)
        count_filtered = class_counts_filtered.get(class_label, 0) if filtering_active else 0

        if not hide_orig:
            # Original dataset
            y_true_orig = (ensemble_results_orig["true_label"] == class_label).astype(int)
            y_scores_orig = ensemble_results_orig[prob_column]  

            precision_orig, recall_orig, _ = precision_recall_curve(y_true_orig, y_scores_orig)
            pr_auc_orig = auc(recall_orig, precision_orig)

            # Plot PR curve for original dataset with sample count
            ax.plot(
                recall_orig, precision_orig, marker='.',
                label=f'{class_label} ({count_orig} samples) - AUC: {pr_auc_orig:.4f}'
            )

        # Only plot the filtered dataset if filtering is active
        if filtering_active:
            y_true_filtered = (ensemble_results_filtered["true_label"] == class_label).astype(int)
            y_scores_filtered = ensemble_results_filtered[prob_column]

            precision_filtered, recall_filtered, _ = precision_recall_curve(y_true_filtered, y_scores_filtered)
            pr_auc_filtered = auc(recall_filtered, precision_filtered)

            ax.plot(
                recall_filtered, precision_filtered, linestyle='dashed', marker='.', 
                label=f'{class_label} ({count_filtered} samples, Filtered) - AUC: {pr_auc_filtered:.4f}'
            )

    # Set labels and title
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve for All Classes')
    ax.legend()

    st.pyplot(fig, use_container_width=False)

def _plot_prediction_score_distribution(ensemble_results):
    st.subheader("Prediction Score Distribution Analysis")

    # 1️⃣ Select the score distribution to visualize
    selected_disp_label = st.selectbox("Select Score to Analyze:", list(HUMAN_LABEL_MAP.values()), index=0)

    # 2️⃣ Select the differentiator class (to split the data)
    selected_differentiator_class = st.selectbox("Select Differentiator Class (True Label):", 
                                                 list(PREDICTION_MAPPING.values()), index=1)

    # Split data based on the differentiator class
    df_with_class = ensemble_results[ensemble_results["true_label"] == selected_differentiator_class]
    df_without_class = ensemble_results[ensemble_results["true_label"] != selected_differentiator_class]

    # Rename columns for consistency
    df_with_class = df_with_class[["astro_id", selected_disp_label, "true_label"]].copy()
    df_without_class = df_without_class[["astro_id", selected_disp_label, "true_label"]].copy()

    df_with_class["Category"] = f"True Label = {selected_differentiator_class}"
    df_without_class["Category"] = f"True Label ≠ {selected_differentiator_class}"

    combined_df = pd.concat([df_with_class, df_without_class], axis=0)

    # Violin Plot for Score Distributions
    fig_violin = px.violin(
        combined_df, x="Category", y=selected_disp_label, box=True, points="all",
        title=f"Distribution of {selected_disp_label} Scores (Grouped by True Label)",
        labels={selected_disp_label: "Prediction Score", "Category": "True Label Group"},
        color="Category",
        hover_data=["astro_id"]
    )

    # Display the charts side by side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"{selected_disp_label} Scores where True Label = {selected_differentiator_class}")
        fig_with = px.histogram(df_with_class, x=selected_disp_label, nbins=20, opacity=0.7,
                                title=f"True Label = {selected_differentiator_class}",
                                labels={selected_disp_label: "Prediction Score"})
        st.plotly_chart(fig_with, use_container_width=True)

    with col2:
        st.subheader(f"{selected_disp_label} Scores where True Label ≠ {selected_differentiator_class}")
        fig_without = px.histogram(df_without_class, x=selected_disp_label, nbins=20, opacity=0.7,
                                   title=f"True Label ≠ {selected_differentiator_class}",
                                   labels={selected_disp_label: "Prediction Score"})
        st.plotly_chart(fig_without, use_container_width=True)

    # Display violin plot below for overall distribution
    st.plotly_chart(fig_violin, use_container_width=True)

def _analyze_features(ensemble_results):
    # Identify numeric and categorical features
    numeric_features = ensemble_results.select_dtypes(include=['int64', 'float64']).columns.tolist()
    all_features = numeric_features

    st.subheader("Feature-wise Performance Analysis")

    # Use Streamlit columns to align dropdowns side by side
    col1, col2 = st.columns(2)
    with col1:
        selected_feature = st.selectbox("Select Feature", all_features)
    with col2:
        selected_metric = st.selectbox(
            "Select Metric",
            ["Accuracy", "F1-score", "Precision", "Recall"]
        )

    # Define a function to compute the selected metric
    def compute_metric(y_true, y_pred, metric_type):
        if metric_type == "Accuracy":
            return accuracy_score(y_true, y_pred)
        elif metric_type == "F1-score":
            return f1_score(y_true, y_pred, average="weighted", zero_division=0)
        elif metric_type == "Precision":
            return precision_score(y_true, y_pred, average="weighted", zero_division=0)
        elif metric_type == "Recall":
            return recall_score(y_true, y_pred, average="weighted", zero_division=0)
        else:
            return 0.0

    # Bin if numeric; otherwise, use categories directly
    if selected_feature in numeric_features:
        bin_count = st.slider("Number of Bins", min_value=2, max_value=20, value=10)
        ensemble_results["feature_bin"] = pd.cut(ensemble_results[selected_feature], bins=bin_count)
    else:
        ensemble_results["feature_bin"] = ensemble_results[selected_feature]

    # Group and compute metrics
    grouped = ensemble_results.groupby("feature_bin").agg({
        "true_label": lambda x: list(x),
        "predicted_label": lambda x: list(x),
        selected_feature: "count"
    }).rename(columns={selected_feature: "count"})

    performance_by_bin = []
    for idx, row in grouped.iterrows():
        metric_value = compute_metric(row["true_label"], row["predicted_label"], selected_metric)
        performance_by_bin.append({
            "bin": str(idx),
            "metric_value": metric_value,
            "count": row["count"]
        })

    perf_df = pd.DataFrame(performance_by_bin)

    # Plot with metric and sample count
    fig = px.bar(perf_df, x="bin", y="metric_value", title=f"{selected_metric} vs. {selected_feature}")
    fig.add_scatter(x=perf_df["bin"], y=perf_df["count"], mode="lines+markers", yaxis="y2", name="Sample Count")
    fig.update_layout(
        yaxis=dict(title=selected_metric),
        yaxis2=dict(title="Sample Count", overlaying="y", side="right", showgrid=False),
        legend=dict(x=0.75, y=1.1)
    )
    st.plotly_chart(fig)


def _feature_confidence(ensemble_results):
    st.subheader("Prediction Confidence by Feature Bin")
    
    numeric_features = ensemble_results.select_dtypes(include=['int64', 'float64']).columns.tolist()


    selected_feature = st.selectbox("Select Feature for Binning (Boxplot)", numeric_features)
    conf_column = st.selectbox("Select Confidence Score Column", HUMAN_LABEL_MAP.values(), index=0)
    bin_count = st.slider("Number of Bins (Boxplot)", min_value=2, max_value=20, value=10)

    if selected_feature in ensemble_results.columns and conf_column in ensemble_results.columns:
        ensemble_results["feature_bin"] = pd.cut(ensemble_results[selected_feature], bins=bin_count)

        fig_box = px.box(
            ensemble_results,
            x="feature_bin",
            y=conf_column,
            color="true_label",
            title=f"{conf_column} vs. {selected_feature} Bins",
            labels={conf_column: "Prediction Confidence"},
        )
        st.plotly_chart(fig_box)

def _feature_performance_heatmap(ensemble_results):
    st.subheader("Heatmap of Accuracy vs. Two Features")
    
    numeric_features = ensemble_results.select_dtypes(include=['int64', 'float64']).columns.tolist()

    feature_x = st.selectbox("Select Feature for X-Axis", numeric_features, index=0)
    feature_y = st.selectbox("Select Feature for Y-Axis", numeric_features, index=1)
    bin_count_x = st.slider("Bins for X Feature", min_value=2, max_value=20, value=6, key="bins_x")
    bin_count_y = st.slider("Bins for Y Feature", min_value=2, max_value=20, value=6, key="bins_y")

    # Bin both features
    ensemble_results["x_bin"] = pd.cut(ensemble_results[feature_x], bins=bin_count_x)
    ensemble_results["y_bin"] = pd.cut(ensemble_results[feature_y], bins=bin_count_y)

    # Group and compute accuracy
    heatmap_data = []
    grouped = ensemble_results.groupby(["x_bin", "y_bin"])
    for (x_bin, y_bin), group in grouped:
        if len(group) > 0:
            acc = accuracy_score(group["true_label"], group["predicted_label"])
            heatmap_data.append({
                feature_x: str(x_bin),
                feature_y: str(y_bin),
                "accuracy": acc,
                "count": len(group)
            })

    heatmap_df = pd.DataFrame(heatmap_data)

    fig_heat = px.density_heatmap(
        heatmap_df,
        x=feature_x,
        y=feature_y,
        z="accuracy",
        text_auto=True,
        color_continuous_scale="Viridis",
        title=f"Accuracy Heatmap: {feature_x} vs. {feature_y}"
    )
    st.plotly_chart(fig_heat)

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

def _has_report_pages(server, astro_id) -> bool:
    tic_id = properties_df.loc[properties_df["astro_id"] == astro_id, "tic_id"]
    if tic_id.empty:
        st.warning(f"No TIC ID found for Astro ID: {astro_id}")
        return False
    else:
        tic_id = tic_id.iloc[0]
    pages = server.get_report_pages(tic_id)
    return pages


def generate_report_for_astro_id(server, astro_id: int, pages, selected_types: List[str]):
    # Get report paths from dataframe
    tic_id = properties_df.loc[properties_df["astro_id"] == astro_id, "tic_id"]
    if tic_id.empty:
        st.warning(f"No TIC ID found for Astro ID: {astro_id}")
        return
    tic_id = tic_id.values[0]
    type_to_page = {PAGE_NUMBER_TO_TYPE.get(p): p for p in pages if PAGE_NUMBER_TO_TYPE.get(p) in selected_types}

    cols = st.columns(3)
    for i, (ptype, page_num) in enumerate(type_to_page.items()):
        with cols[i % 3]:
            image = server.get_page_image(tic_id, page_num)
            if isinstance(image, Image.Image):
                st.image(image, caption=f"{ptype} (Page {page_num})", use_container_width=True)
            else:
                st.warning(f"Image for {ptype} not available.")
    return True



col1, col2 = st.columns(2)
individual_model_results = None

with col1:
    # Navigation: go up one directory
    if st.button("⬆️ Go up one level"):
        parent_dir = os.path.dirname(st.session_state.current_dir)
        # Prevent going above ROOT_DIR
        if os.path.commonpath([ROOT_DIR, parent_dir]) == ROOT_DIR:
            st.session_state.current_dir = parent_dir
            st.rerun()
    # List subdirectories and files
    dirs, files = list_subdirs_and_files(st.session_state.current_dir)

    # Folder navigation
    selected_dir = st.selectbox(f"Folders (selected: {st.session_state.current_dir})", ["<Select a folder>"] + sorted(dirs))
    if selected_dir != "<Select a folder>":
        new_path = os.path.join(st.session_state.current_dir, selected_dir)
        st.session_state.current_dir = new_path
        st.rerun()

    # File selection
    selected_file = st.selectbox("Files", ["<Select a file>"] + files)
    if selected_file != "<Select a file>":
        file_path = os.path.join(st.session_state.current_dir, selected_file)
        st.success(f"Selected file: {file_path}")
        # You can now load the file
        individual_model_results = pd.read_csv(file_path)
with col2:
    st.header("Upload File")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        individual_model_results = pd.read_csv(uploaded_file)
        st.success("Uploaded file successfully")

if individual_model_results is not None:
    st.subheader("Processing Uploaded Model Predictions")
    eval_utils = EvalUtils(individual_model_results)
    eclipsing_binary_as_junk = st.checkbox("Show eclipsing binaries as junk?", value=True)


    if {"astro_id", "model_no", "disp_p", "disp_e", "disp_n", "disp_j"}.issubset(individual_model_results.columns):
        performance_df = eval_utils.compute_performance()
        _show_all_model_performance(performance_df)
        
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
        if eclipsing_binary_as_junk:
            ensemble_results.loc[ensemble_results["predicted_label"] == "Eclipsing Binary", "predicted_label"] = "Junk"

        ensemble_results_orig = ensemble_results.copy()
        ensemble_results = _advanced_filter_sidebar(ensemble_results)

        _analyze_features(ensemble_results)
        #_feature_confidence(ensemble_results)
        # _feature_performance_heatmap(ensemble_results)

        _plot_pr_curve(ensemble_results_orig, ensemble_results)
        _plot_prediction_score_distribution(ensemble_results)

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
        all_page_types = ["Summary", "BLS Spectrum", "Depth-aperture Correlation", "Difference Images", "Full Detrended LC", "Full Raw LC + Folded Detrended LC", "MCMC Fit", "Matches to Known Signals"]
        selected_types = st.sidebar.multiselect("Select Report Page Types", sorted(all_page_types), default=sorted(all_page_types))
        server = LightCurveServer()

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

            pages = _has_report_pages(server, astro_id)
            if not pages:
                st.write(f"No report for astro ID {astro_id}, skipping...")
            else:
                # Display metadata
                st.subheader(f"Report for Astro ID: {astro_id}")
                st.write(f"**True Label:** {true_label}, **Predicted Label:** {predicted_label}")
                st.write(f"**disp Scores:** {disp_scores}")
                st.write(f"**Reason for Selection:** {selection_reason}")
                generate_report_for_astro_id(server, astro_id=astro_id, pages=pages, selected_types=selected_types)
                num_analyzed += 1
else:
    st.info("Please upload model inference results to proceed.")