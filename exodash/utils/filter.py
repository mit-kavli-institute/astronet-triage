
import streamlit as st
import pandas as pd
import numpy as np

slider_overrides = {
    "snr": {
        "ui_min": 1.0,
        "ui_max": 200.0,
        "step": 0.1,
        "allow_overflow": True
    },
    "period": {
        "ui_min": 0.1,
        "ui_max": 100.0,
        "step": 0.01,
        "allow_overflow": True
    },
    "bls_points_in_transit": {
        "ui_min": 0,
        "ui_max": 2000,
        "step": 0.01,
        "allow_overflow": True
    }
}

def advanced_filter_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a streamlit side panel for advanced filtering for a specified df.

    Supports both numeric and categorical labels.
    """
    filtered_df = df.copy()

    numeric_features = df.select_dtypes(include=["int", "float"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object", "bool", "string[python]"]).columns.tolist()

    # Manually specify which numeric fields should use direct input instead of slider
    direct_input_fields = {"astro_id", "tic_id"}
    selected_num_filters = st.sidebar.multiselect("Select Numeric Features to Filter", numeric_features)

    for feature in selected_num_filters:
        col = df[feature].dropna()

        # Handle all-null numeric columns gracefully
        if col.empty:
            st.sidebar.warning(f"Column '{feature}' has no non-NaN values — skipping filter.")
            continue

        # Direct input fields (exact match)
        if feature in direct_input_fields:
            # Choose an appropriate default based on dtype
            if pd.api.types.is_integer_dtype(df[feature]):
                default_val = int(col.min())
                value = st.sidebar.number_input(
                    f"Enter value for {feature}",
                    value=default_val,
                    step=1,
                    format="%d",
                    key=f"{feature}_direct"
                )
                value = int(value)
            else:
                default_val = float(col.min())
                value = st.sidebar.number_input(
                    f"Enter value for {feature}",
                    value=default_val,
                    key=f"{feature}_direct"
                )
                value = float(value)

            filtered_df = filtered_df[filtered_df[feature] == value]
            continue

        # Override logic (capped slider + optional include_over)
        if feature in slider_overrides:
            config = slider_overrides[feature]
            ui_min = float(config["ui_min"])
            ui_max = float(config["ui_max"])
            step = float(config.get("step", 0.1))
            allow_overflow = bool(config.get("allow_overflow", False))

            # Optional: clamp ui_min/ui_max against actual data range to avoid weirdness
            data_min, data_max = float(col.min()), float(col.max())
            ui_min = max(ui_min, data_min)
            ui_max = min(ui_max, data_max) if data_max >= ui_min else ui_min

            selected_range = st.sidebar.slider(
                f"Range for {feature}",
                min_value=ui_min,
                max_value=ui_max,
                value=(ui_min, ui_max),
                step=step,
                key=f"{feature}_slider"
            )

            mask = (
                (filtered_df[feature] >= selected_range[0]) &
                (filtered_df[feature] <= selected_range[1])
            )

            if allow_overflow:
                include_over = st.sidebar.checkbox(
                    f"Include {feature} > {ui_max}",
                    value=False,
                    key=f"{feature}_include_over"
                )

                # Only extend mask if user maxed slider at cap
                if include_over and selected_range[1] == ui_max:
                    mask |= (filtered_df[feature] > ui_max)

            filtered_df = filtered_df[mask]
            continue

        # Default behavior (data-driven slider)
        min_val, max_val = float(col.min()), float(col.max())

        # If effectively constant, avoid a broken slider
        if np.isclose(min_val, max_val):
            st.sidebar.info(f"'{feature}' is constant ({min_val:g}) — no range filter applied.")
            continue

        selected_range = st.sidebar.slider(
            f"Range for {feature}",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val),
            key=f"{feature}_slider_default"
        )

        filtered_df = filtered_df[
            (filtered_df[feature] >= selected_range[0]) &
            (filtered_df[feature] <= selected_range[1])
        ]

    # Allow users to filter categorical features
    selected_cat_filters = st.sidebar.multiselect("Select Categorical Features to Filter", categorical_features)

    for feature in selected_cat_filters:
        unique_values = df[feature].dropna().unique().tolist()
        unique_values_with_nan = unique_values + ["NaN"] if df[feature].isna().any() else unique_values

        selected_values = st.sidebar.multiselect(
            f"Filter by {feature}",
            unique_values_with_nan,
            default=unique_values,
            key=f"{feature}_cat"
        )

        mask = df[feature].isin([v for v in selected_values if v != "NaN"])
        if "NaN" in selected_values:
            mask |= df[feature].isna()

        filtered_df = filtered_df[mask]

    # Custom filter implementations
    
    filter_long_period = st.sidebar.checkbox("Long Period", key="filter_long_period")
    if filter_long_period:
        filtered_df = filtered_df[filtered_df["period"] >= 20]

    small_planet = st.sidebar.checkbox("Small Planet", key="filter_small_planet")
    if small_planet:
        filtered_df = filtered_df[filtered_df["planet_radius"] <= 5]

    filter_ebs_by_period = st.sidebar.checkbox("Filter EBs by period", key="filter_ebs_by_period")
    if filter_ebs_by_period:
        tolerance = 0.05

        def keep_highest_planet_within_tolerance(group):
            if len(group) == 1:
                return group
            group = group.sort_values(by="planetno", ascending=False)
            top_period = group.iloc[0]["period"]
            mask = np.abs(group["period"] - top_period) <= tolerance
            return group[mask]

        filtered_df = (
            filtered_df.groupby("tic_id", group_keys=False)
            .apply(keep_highest_planet_within_tolerance)
            .reset_index(drop=True)
        )

    filter_tois_by_qlp_only = st.sidebar.checkbox("Filter TOIs by QLP detection only?", key="filter_tois_by_qlp_only")
    if filter_tois_by_qlp_only:
        if "detection_pipeline" in filtered_df.columns:
            dp = filtered_df["detection_pipeline"]
            filtered_df = filtered_df[
                dp.isna() |
                (dp.astype(str).str.upper() == "QLP") |
                (dp.astype(str).str.upper() == "SPOC/QLP")
            ].reset_index(drop=True)
        else:
            st.sidebar.warning("Column 'detection_pipeline' not found — QLP filter not applied.")

    return filtered_df
