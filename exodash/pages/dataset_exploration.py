import streamlit as st
import plotly.express as px
import pandas as pd

# ---------------------------------------
# Check that required data is available
# ---------------------------------------
if "df" not in st.session_state or "properties" not in st.session_state:
    st.error("Dataset not found. Please start from the main ExoDash app.")
    st.stop()

df = st.session_state.df.copy()
properties_df = st.session_state.properties

st.set_page_config(page_title="Dataset Explorer", layout="wide")
st.title("Dataset Explorer")

# ---------------------------------------
# Sidebar: Filtering Controls
# ---------------------------------------
st.sidebar.header("Advanced Filtering")

# Fill NaNs in label_simplified for plotting
df["label_simplified"] = df["label_simplified"].fillna("Unknown")

# Identify feature types
numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

# Initialize filtered_df
filtered_df = df.copy()

# --- Numeric Filters ---
selected_num_filters = st.sidebar.multiselect(
    "Numeric Features to Filter", numeric_features
)

for feature in selected_num_filters:
    min_val, max_val = float(df[feature].min()), float(df[feature].max())
    selected_range = st.sidebar.slider(
        f"Range for {feature}", min_val, max_val, (min_val, max_val)
    )
    filtered_df = filtered_df[
        (filtered_df[feature] >= selected_range[0]) &
        (filtered_df[feature] <= selected_range[1])
    ]

# --- Categorical Filters ---
selected_cat_filters = st.sidebar.multiselect(
    "Categorical Features to Filter", categorical_features
)

for feature in selected_cat_filters:
    unique_values = df[feature].dropna().unique()
    selected_values = st.sidebar.multiselect(
        f"Filter by {feature}", unique_values, default=unique_values
    )
    filtered_df = filtered_df[filtered_df[feature].isin(selected_values)]

# ---------------------------------------
# Scatter Plot
# ---------------------------------------
st.subheader("Scatter Plot")

x_col = st.selectbox("X-axis", df.columns[1:], index=0)
y_col = st.selectbox("Y-axis", df.columns[1:], index=1)
color_col = st.selectbox("Color By", ["None"] + categorical_features, index=0)

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

# ---------------------------------------
# Histogram Plot
# ---------------------------------------
st.subheader("Feature Distribution")

hist_feature = st.selectbox("Select Feature", df.columns[1:])
nbins = st.slider("Number of Bins", 5, 100, 50, step=5)

if pd.api.types.is_numeric_dtype(df[hist_feature]):
    fig_hist = px.histogram(
        filtered_df,
        x=hist_feature,
        color="label_simplified",
        nbins=nbins,
        title=f"Distribution of {hist_feature}"
    )
else:
    fig_hist = px.histogram(
        filtered_df,
        x=hist_feature,
        color="label_simplified",
        title=f"Categorical Distribution of {hist_feature}"
    )

st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------------------
# Filtered TIC List (Clickable)
# ---------------------------------------
st.subheader("Filtered TIC List (Top 50 Results)")

required_cols = ["tic_id", "astro_id", "label_simplified"]

if all(col in filtered_df.columns for col in required_cols):
    display_df = filtered_df[required_cols].head(50).copy()

    # Create clickable TIC links
    def make_tic_link(tic_id):
        return f"[{tic_id}](./tic_exploration?tic_id={tic_id})"

    display_df["tic_id"] = display_df["tic_id"].apply(make_tic_link)

    # Show as markdown table
    st.markdown(display_df.to_markdown(index=False), unsafe_allow_html=True)
else:
    st.warning("Missing required columns: tic_id, astro_id, or label_simplified.")