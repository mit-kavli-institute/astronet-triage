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

def _has_report_pages(server, astro_id) -> bool:
    tic_id = properties_df.loc[properties_df["astro_id"] == astro_id, "tic_id"]
    if tic_id.empty:
        st.warning(f"No TIC ID found for Astro ID: {astro_id}")
        return False
    else:
        tic_id = tic_id.iloc[0]
    pages = server.get_report_pages(tic_id)
    return pages

st.title("Model Comparison")
properties_df = df = st.session_state.df  # Access shared dataset

col1, col2 = st.columns(2)

# Upload both CSVs
with col1:
    st.subheader("Model A Results")
    model_a_file = st.file_uploader("Upload CSV for Model A", type=["csv"], key="model_a")
with col2:
    st.subheader("Model B Results")
    model_b_file = st.file_uploader("Upload CSV for Model B", type=["csv"], key="model_b")

if model_a_file and model_b_file:
    eclipsing_binary_as_junk = st.checkbox("Show eclipsing binaries as junk?", value=True)
    df_a = pd.read_csv(model_a_file)
    df_b = pd.read_csv(model_b_file)

    # Validate
    common_ids = set(df_a["astro_id"]) & set(df_b["astro_id"])
    if not common_ids:
        st.error("! No overlapping astro_ids found between the models. !")

    # Compute performance for both
    st.header("Overall Performance Comparison")
    eval_a = EvalUtils(df_a)
    eval_b = EvalUtils(df_b)

    ensemble_results_a = eval_a.get_ensemble_results(None, include_labels=False, include_properties=True, dropna=True)
    ensemble_results_b = eval_b.get_ensemble_results(None, include_labels=False, include_properties=True, dropna=True)
    if eclipsing_binary_as_junk:
        ensemble_results_a.loc[ensemble_results_a["predicted_label"] == "Eclipsing Binary", "predicted_label"] = "Junk"
        ensemble_results_b.loc[ensemble_results_b["predicted_label"] == "Eclipsing Binary", "predicted_label"] = "Junk"



    perf_a = eval_a.compute_performance()
    perf_b = eval_b.compute_performance()

    perf_a["model"] = "Model A"
    perf_b["model"] = "Model B"
    perf_combined = pd.concat([perf_a, perf_b], axis=0)

    # Show comparative performance
    st.dataframe(perf_combined)
    st.dataframe(perf_combined.sort_values(by=["model_no", "model"]))

    # Plot side-by-side F1 scores (or others)
    fig = px.bar(
        perf_combined,
        x="model_no",
        y="f1_score",
        color="model",
        barmode="group",
        title="Per-Class F1 Score Comparison",
        text_auto=".2f"
    )
    st.plotly_chart(fig)

    # Merge and compare predictions
    pred_a = ensemble_results_a[["astro_id", "predicted_label", "true_label"]].rename(columns={"predicted_label": "model_a_pred"})
    pred_b = ensemble_results_b[["astro_id", "predicted_label"]].rename(columns={"predicted_label": "model_b_pred"})

    # Merge on astro_id
    merged_preds = pd.merge(pred_a, pred_b, on="astro_id", how="inner")
    merged_preds["match"] = merged_preds["model_a_pred"] == merged_preds["model_b_pred"]

    # Summary stats
    total = len(merged_preds)
    mismatches = len(merged_preds[~merged_preds["match"]])
    st.write(f"Total Matched Predictions: {total - mismatches}")
    st.write(f"Total Differing Predictions: {mismatches}")

    st.sidebar.subheader("Prediction Comparison Filters")

    diff_filter = st.sidebar.radio(
        "Filter Type",
        ["Show All", "Only Matches", "Only Mismatches"],
        index=2
    )

    unique_labels = sorted(set(merged_preds["model_a_pred"].unique()) | set(merged_preds["model_b_pred"].unique()))
    label_a_filter = st.sidebar.multiselect("Model A Predicted Label(s)", unique_labels, default=unique_labels)
    label_b_filter = st.sidebar.multiselect("Model B Predicted Label(s)", unique_labels, default=unique_labels)
    true_label_filter = st.sidebar.multiselect("True Label(s)", unique_labels, default=unique_labels)

    filtered_preds = merged_preds[
        (merged_preds["model_a_pred"].isin(label_a_filter)) &
        (merged_preds["model_b_pred"].isin(label_b_filter)) &
        (merged_preds["true_label"].isin(true_label_filter))
    ]

    if diff_filter == "Only Matches":
        filtered_preds = filtered_preds[filtered_preds["match"]]
    elif diff_filter == "Only Mismatches":
        filtered_preds = filtered_preds[~filtered_preds["match"]]

    # Display
    st.subheader("Filtered Prediction Comparison")
    st.write(f"Showing {len(filtered_preds)} of {total} predictions based on filters.")
    st.dataframe(filtered_preds.reset_index(drop=True))

    N_TO_ANALYZE = st.sidebar.slider(
        f"Number of interesting cases to analyze",
        1, 10
    )
    num_analyzed = 0
    interesting_cases = merged_preds.to_dict(orient="records")
    cur_case = 0
    st.subheader(f"Reports")
    all_page_types = ["Summary", "BLS Spectrum", "Depth-aperture Correlation", "Difference Images", "Full Detrended LC", "Full Raw LC + Folded Detrended LC", "MCMC Fit", "Matches to Known Signals"]
    selected_types = st.sidebar.multiselect("Select Report Page Types", sorted(all_page_types), default=sorted(all_page_types))
    server = LightCurveServer()

    while num_analyzed < N_TO_ANALYZE:
        cur_case += 1
        case = interesting_cases[cur_case]
        astro_id = case["astro_id"]
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
            generate_report_for_astro_id(server, astro_id=astro_id, pages=pages, selected_types=selected_types)
            num_analyzed += 1

else:
    st.info("Please upload two CSV files to compare models.")