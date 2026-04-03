import pandas as pd
from data_management.type_mapping import HUMAN_LABEL_MAP, PREDICTION_MAPPING
import streamlit as st
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc


def show_all_model_performance(performance_df: pd.DataFrame, key: str = "") -> None:
    def _k(name: str) -> str:
        return f"{key}_{name}" if key else name

    metric_to_plot = st.selectbox(
        "Select Metric to Compare",
        ["accuracy", "precision", "recall", "f1_score"],
        key=_k("metric_select"),
    )
    fig = px.bar(
        performance_df, x="model_no", y=metric_to_plot,
        text=metric_to_plot,
        title=f"{metric_to_plot.capitalize()} Across Models",
    )
    st.plotly_chart(fig)


def plot_pr_curve(
    ensemble_results_orig: pd.DataFrame,
    ensemble_results_filtered: pd.DataFrame,
    key: str = "",
) -> None:
    def _k(name: str) -> str:
        return f"{key}_{name}" if key else name

    st.subheader("Precision-Recall Curve (All Classes)")

    filtering_active = not ensemble_results_filtered.equals(ensemble_results_orig)

    hide_orig = (
        st.sidebar.checkbox("Hide Original PR Curve", value=False, key=_k("hide_orig_pr"))
        if filtering_active
        else False
    )

    class_counts_orig = ensemble_results_orig["true_label"].value_counts().to_dict()
    class_counts_filtered = (
        ensemble_results_filtered["true_label"].value_counts().to_dict()
        if filtering_active
        else {}
    )

    fig, ax = plt.subplots(figsize=(16, 9))

    for class_label, prob_column in HUMAN_LABEL_MAP.items():
        count_orig = class_counts_orig.get(class_label, 0)
        count_filtered = class_counts_filtered.get(class_label, 0) if filtering_active else 0

        if not hide_orig:
            y_true_orig = (ensemble_results_orig["true_label"] == class_label).astype(int)
            y_scores_orig = ensemble_results_orig[prob_column]

            precision_orig, recall_orig, _ = precision_recall_curve(y_true_orig, y_scores_orig)
            pr_auc_orig = auc(recall_orig, precision_orig)

            ax.plot(
                recall_orig, precision_orig, marker=".",
                label=f"{class_label} ({count_orig} samples) - AUC: {pr_auc_orig:.4f}",
            )

        if filtering_active:
            y_true_filtered = (ensemble_results_filtered["true_label"] == class_label).astype(int)
            y_scores_filtered = ensemble_results_filtered[prob_column]

            precision_filtered, recall_filtered, _ = precision_recall_curve(
                y_true_filtered, y_scores_filtered
            )
            pr_auc_filtered = auc(recall_filtered, precision_filtered)

            ax.plot(
                recall_filtered, precision_filtered, linestyle="dashed", marker=".",
                label=f"{class_label} ({count_filtered} samples, Filtered) - AUC: {pr_auc_filtered:.4f}",
            )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve for All Classes")
    ax.legend()

    st.pyplot(fig, use_container_width=False)


def plot_prediction_score_distribution(
    ensemble_results: pd.DataFrame,
    key: str = "",
) -> None:
    def _k(name: str) -> str:
        return f"{key}_{name}" if key else name

    st.subheader("Prediction Score Distribution Analysis")

    selected_disp_label = st.selectbox(
        "Select Score to Analyze:",
        list(HUMAN_LABEL_MAP.values()),
        index=0,
        key=_k("score_dist_label"),
    )

    selected_differentiator_class = st.selectbox(
        "Select Differentiator Class (True Label):",
        list(PREDICTION_MAPPING.values()),
        index=1,
        key=_k("score_dist_class"),
    )

    df_with_class = ensemble_results[
        ensemble_results["true_label"] == selected_differentiator_class
    ].copy()
    df_without_class = ensemble_results[
        ensemble_results["true_label"] != selected_differentiator_class
    ].copy()

    df_with_class = df_with_class[["astro_id", selected_disp_label, "true_label"]]
    df_without_class = df_without_class[["astro_id", selected_disp_label, "true_label"]]

    df_with_class = df_with_class.copy()
    df_without_class = df_without_class.copy()
    df_with_class["Category"] = f"True Label = {selected_differentiator_class}"
    df_without_class["Category"] = f"True Label ≠ {selected_differentiator_class}"

    combined_df = pd.concat([df_with_class, df_without_class], axis=0)

    fig_violin = px.violin(
        combined_df, x="Category", y=selected_disp_label, box=True, points="all",
        title=f"Distribution of {selected_disp_label} Scores (Grouped by True Label)",
        labels={selected_disp_label: "Prediction Score", "Category": "True Label Group"},
        color="Category",
        hover_data=["astro_id"],
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"{selected_disp_label} Scores where True Label = {selected_differentiator_class}")
        fig_with = px.histogram(
            df_with_class, x=selected_disp_label, nbins=20, opacity=0.7,
            title=f"True Label = {selected_differentiator_class}",
            labels={selected_disp_label: "Prediction Score"},
        )
        st.plotly_chart(fig_with, use_container_width=True)

    with col2:
        st.subheader(f"{selected_disp_label} Scores where True Label ≠ {selected_differentiator_class}")
        fig_without = px.histogram(
            df_without_class, x=selected_disp_label, nbins=20, opacity=0.7,
            title=f"True Label ≠ {selected_differentiator_class}",
            labels={selected_disp_label: "Prediction Score"},
        )
        st.plotly_chart(fig_without, use_container_width=True)

    st.plotly_chart(fig_violin, use_container_width=True)


def analyze_features(ensemble_results: pd.DataFrame, key: str = "") -> None:
    def _k(name: str) -> str:
        return f"{key}_{name}" if key else name

    numeric_features = ensemble_results.select_dtypes(include=["int64", "float64"]).columns.tolist()
    all_features = numeric_features

    st.subheader("Feature-wise Performance Analysis")

    col1, col2 = st.columns(2)
    with col1:
        selected_feature = st.selectbox(
            "Select Feature",
            all_features,
            key=_k("analyze_feature_select"),
        )
    with col2:
        selected_metric = st.selectbox(
            "Select Metric",
            ["Accuracy", "F1-score", "Precision", "Recall"],
            key=_k("analyze_metric_select"),
        )

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

    if selected_feature in numeric_features:
        bin_count = st.slider(
            "Number of Bins",
            min_value=2,
            max_value=20,
            value=10,
            key=_k("analyze_bins_slider"),
        )
        ensemble_results = ensemble_results.copy()
        ensemble_results["feature_bin"] = pd.cut(ensemble_results[selected_feature], bins=bin_count)
    else:
        ensemble_results = ensemble_results.copy()
        ensemble_results["feature_bin"] = ensemble_results[selected_feature]

    grouped = ensemble_results.groupby("feature_bin").agg({
        "true_label": lambda x: list(x),
        "predicted_label": lambda x: list(x),
        selected_feature: "count",
    }).rename(columns={selected_feature: "count"})

    performance_by_bin = []
    for idx, row in grouped.iterrows():
        metric_value = compute_metric(row["true_label"], row["predicted_label"], selected_metric)
        performance_by_bin.append({
            "bin": str(idx),
            "metric_value": metric_value,
            "count": row["count"],
        })

    perf_df = pd.DataFrame(performance_by_bin)

    fig = px.bar(perf_df, x="bin", y="metric_value", title=f"{selected_metric} vs. {selected_feature}")
    fig.add_scatter(
        x=perf_df["bin"], y=perf_df["count"],
        mode="lines+markers", yaxis="y2", name="Sample Count",
    )
    fig.update_layout(
        yaxis=dict(title=selected_metric),
        yaxis2=dict(title="Sample Count", overlaying="y", side="right", showgrid=False),
        legend=dict(x=0.75, y=1.1),
    )
    st.plotly_chart(fig)


def feature_confidence(ensemble_results: pd.DataFrame, key: str = "") -> None:
    def _k(name: str) -> str:
        return f"{key}_{name}" if key else name

    st.subheader("Prediction Confidence by Feature Bin")

    numeric_features = ensemble_results.select_dtypes(include=["int64", "float64"]).columns.tolist()

    selected_feature = st.selectbox(
        "Select Feature for Binning (Boxplot)",
        numeric_features,
        key=_k("fc_feature_select"),
    )
    conf_column = st.selectbox(
        "Select Confidence Score Column",
        HUMAN_LABEL_MAP.values(),
        index=0,
        key=_k("fc_conf_select"),
    )
    bin_count = st.slider(
        "Number of Bins (Boxplot)",
        min_value=2,
        max_value=20,
        value=10,
        key=_k("fc_bins_slider"),
    )

    if selected_feature in ensemble_results.columns and conf_column in ensemble_results.columns:
        ensemble_results = ensemble_results.copy()
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


def feature_performance_heatmap(ensemble_results: pd.DataFrame, key: str = "") -> None:
    def _k(name: str) -> str:
        return f"{key}_{name}" if key else name

    st.subheader("Heatmap of Accuracy vs. Two Features")

    numeric_features = ensemble_results.select_dtypes(include=["int64", "float64"]).columns.tolist()

    feature_x = st.selectbox(
        "Select Feature for X-Axis",
        numeric_features,
        index=0,
        key=_k("heatmap_x_select"),
    )
    feature_y = st.selectbox(
        "Select Feature for Y-Axis",
        numeric_features,
        index=1,
        key=_k("heatmap_y_select"),
    )
    bin_count_x = st.slider(
        "Bins for X Feature",
        min_value=2, max_value=20, value=6,
        key=_k("bins_x"),
    )
    bin_count_y = st.slider(
        "Bins for Y Feature",
        min_value=2, max_value=20, value=6,
        key=_k("bins_y"),
    )

    ensemble_results = ensemble_results.copy()
    ensemble_results["x_bin"] = pd.cut(ensemble_results[feature_x], bins=bin_count_x)
    ensemble_results["y_bin"] = pd.cut(ensemble_results[feature_y], bins=bin_count_y)

    heatmap_data = []
    grouped = ensemble_results.groupby(["x_bin", "y_bin"])
    for (x_bin, y_bin), group in grouped:
        if len(group) > 0:
            acc = accuracy_score(group["true_label"], group["predicted_label"])
            heatmap_data.append({
                feature_x: str(x_bin),
                feature_y: str(y_bin),
                "accuracy": acc,
                "count": len(group),
            })

    heatmap_df = pd.DataFrame(heatmap_data)

    fig_heat = px.density_heatmap(
        heatmap_df,
        x=feature_x,
        y=feature_y,
        z="accuracy",
        text_auto=True,
        color_continuous_scale="Viridis",
        title=f"Accuracy Heatmap: {feature_x} vs. {feature_y}",
    )
    st.plotly_chart(fig_heat)
