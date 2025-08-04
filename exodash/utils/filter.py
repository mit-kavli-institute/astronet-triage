
import streamlit as st
import pandas as pd

def advanced_filter_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a streamlit side panel for advanced filtering for a specified df.

    Supperts both numeric and categorical labels.
    """
    filtered_df = df.copy()

    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=["object", "bool", "string[python]"]).columns.tolist()

    # Manually specify which numeric fields should use text/number input instead of slider
    direct_input_fields = {"astro_id", "tic_id"}

    selected_num_filters = st.sidebar.multiselect("Select Numeric Features to Filter", numeric_features)

    for feature in selected_num_filters:
        if feature in direct_input_fields:
            value = st.sidebar.number_input(f"Enter value for {feature}", value=int(df[feature].min()))
            filtered_df = filtered_df[filtered_df[feature] == value]
        else:
            min_val, max_val = float(df[feature].min()), float(df[feature].max())
            selected_range = st.sidebar.slider(
                f"Range for {feature}",
                min_val, max_val, (min_val, max_val)
            )
            filtered_df = filtered_df[(filtered_df[feature] >= selected_range[0]) & (filtered_df[feature] <= selected_range[1])]

    # Allow users to filter categorical features
    selected_cat_filters = st.sidebar.multiselect("Select Categorical Features to Filter", categorical_features)

    for feature in selected_cat_filters:
        unique_values = df[feature].dropna().unique().tolist()
        unique_values_with_nan = unique_values + ["NaN"] if df[feature].isna().any() else unique_values
        selected_values = st.sidebar.multiselect(f"Filter by {feature}", unique_values_with_nan, default=unique_values)
        mask = df[feature].isin([v for v in selected_values if v != "NaN"])
        if "NaN" in selected_values:
            mask |= df[feature].isna()

        filtered_df = filtered_df[mask]
    return filtered_df