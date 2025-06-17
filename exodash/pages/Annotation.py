import os
from exodash.utils.file_io import model_result_selector
from exodash.utils.filter import advanced_filter_sidebar
from exodash.utils.reports import generate_report_for_tic_id, infer_planet_number
import streamlit as st
import pandas as pd
import getpass
from data_management.light_curve_server import ALL_PAGE_TYPES
from data_management.type_mapping import HUMAN_LABEL_MAP
from exodash.eval_utils import REQUIRED_MODEL_COLUMNS, EvalUtils

label_1 = None
label_2 = None
notes = None
annotator_name = None
single_transit = None

if "df" not in st.session_state or "light_curve_server" not in st.session_state:
    st.error("Dataset not found. Please use the landing page first.")
    st.stop()

server = st.session_state.light_curve_server
properties_df = df = st.session_state.df
if "skipped_astro_ids" not in st.session_state:
    st.session_state.skipped_astro_ids = []


# Streamlit UI
st.set_page_config(page_title="ExoDash - Annotation", layout="wide")
st.title("Annotation")
st.write("Create manual labels for Astro IDs to create new test/eval sets.")

default_uid = getpass.getuser()
annotator_name = st.text_input("Annotator Name", value=default_uid)
if annotator_name is None:
    st.warning("Please enter your name to continue.")
    st.stop()
ANNOTATION_PATH = f"/pdo/astronet-data/data/labels/labels_{annotator_name}.csv"
st.write(f'Will save annotation results to {ANNOTATION_PATH}!')

model_results = model_result_selector(allow_cached_models=True, allow_local_navigation=False, allow_upload=False)

if model_results is None:
    st.warning("Please select model results to continue.")
    st.stop()

eval_utils = EvalUtils(model_results)
eclipsing_binary_as_junk = st.sidebar.checkbox("Show eclipsing binaries as junk?", value=True)

if not REQUIRED_MODEL_COLUMNS.issubset(model_results.columns):
    st.error("! ERROR ! Please ensure the model results has all columns:")
    st.stop()

thresholds = None
ensemble_results = eval_utils.get_ensemble_results(thresholds, include_labels=False, include_properties=True, dropna=True)
if eclipsing_binary_as_junk:
    ensemble_results.loc[ensemble_results["predicted_label"] == "Eclipsing Binary", "predicted_label"] = "Junk"

ensemble_results = advanced_filter_sidebar(ensemble_results)
if 'filter_len' not in st.session_state or st.session_state.filter_len != len(ensemble_results):
    st.session_state.filter_len = len(ensemble_results)
    st.session_state.pop("astro_row", None)
    st.session_state.pop("pages", None)
selected_types = st.sidebar.multiselect("Select Report Page Types", sorted(ALL_PAGE_TYPES), default=['Summary', 'Depth-aperture Correlation', 'Difference Images'])

skipped_astro_ids = []
if os.path.exists(ANNOTATION_PATH):
    existing_annotations = pd.read_csv(ANNOTATION_PATH)
else:
    existing_annotations = pd.DataFrame()
st.write(f'Existing Annotations Table at {ANNOTATION_PATH}')
st.dataframe(existing_annotations)
st.write("Skipped Astro IDs (reset the session if you want to go back to them)")
st.write(st.session_state.skipped_astro_ids)
    

if 'astro_row' not in st.session_state:
    sampled_ids = ensemble_results['astro_id'].sample(frac=1).tolist()
    for idx, astro_id in enumerate(sampled_ids):
        match = ensemble_results[ensemble_results["astro_id"] == astro_id]
        if match.empty:
            continue
        row = match.iloc[0]
        tic_id = row['tic_id']
        planet_number = infer_planet_number(tic_id=tic_id, astro_id=astro_id)
        pages = server.get_report_pages(tic_id, planet_number=planet_number)

        if 'astro_id' in existing_annotations.columns:
            existing_annotations_df = existing_annotations[existing_annotations['astro_id'] == astro_id]
        else:
            existing_annotations_df = pd.DataFrame()
        if pages and len(existing_annotations_df) == 0 and astro_id not in st.session_state.skipped_astro_ids:
            st.session_state.astro_row = row
            st.session_state.idx = idx+1
            st.session_state.pages = pages
            break

if 'astro_row' in st.session_state:
    row = st.session_state.astro_row
    astro_id = row['astro_id']
    true_label = row['true_label']
    predicted_label = row['predicted_label']
    disp_scores = {label: row[label] for label in HUMAN_LABEL_MAP.values() if label in row}
    pages = st.session_state.pages

    st.subheader(f"Annotate Astro ID: {astro_id} [{st.session_state.idx}/{len(ensemble_results)}]")
    st.write(f"**True Label:** {true_label}, **Predicted Label:** {predicted_label}")
    st.write(f"**disp Scores:** {disp_scores}")

    tic_id = row['tic_id']
    planet_number = infer_planet_number(tic_id=tic_id, astro_id=astro_id)
    generate_report_for_tic_id(tic_id=tic_id, planet_number=planet_number, pages=pages, selected_types=selected_types)

    label_1 = st.radio("First label (Planet/Eclipsing/Unknown/Junk)", ['p', 'e', 'j'], key="label_1")
    #label_2 = st.radio("Second label (on Target/Background/Unknown)", ['t', 'b', 'u'], key="label_2")
    notes = st.text_input("Notes (optional)")
    single_transit = st.checkbox("Single Transit (s)", key="single_transit")

    if label_1 and annotator_name:
        if st.button("Save annotation and move on"):
            annotation = {
                "annotator": annotator_name,
                "astro_id": astro_id,
                "label_1": label_1,
                #"label_2": label_2,
                "single_transit": single_transit,
                "notes": notes,
            }
            file_exists = os.path.exists(ANNOTATION_PATH)
            pd.DataFrame([annotation]).to_csv(ANNOTATION_PATH, mode='a', header=not file_exists, index=False)

            del st.session_state.astro_row
            del st.session_state.idx
            del st.session_state.pages
            st.rerun()
        if st.button("Skip this one"):
            st.session_state.skipped_astro_ids.append(astro_id)
            del st.session_state.astro_row
            del st.session_state.idx
            del st.session_state.pages
            st.rerun()
    else:
        st.info("Set the label and enter your name to continue.")