from data_management.light_curve_server import ALL_PAGE_TYPES, PAGE_NUMBER_TO_TYPE
from PIL import Image
from exodash.utils.filter import advanced_filter_sidebar
from exodash.utils.tic_visualization import TICVisualizer
import streamlit as st

if "df" not in st.session_state or "light_curve_server" not in st.session_state:
    st.error("Dataset not found. Please use the landing page first.")
    st.stop()

server = st.session_state.light_curve_server

df = st.session_state.df
df = advanced_filter_sidebar(df)
st.title("TIC Explorer")
st.write(df.head())

st.sidebar.header("Select TIC IDs")
tic_ids = st.sidebar.text_area("Enter TIC IDs (comma-separated)", "").strip()
tic_ids = [int(tic.strip()) for tic in tic_ids.split(",") if tic.strip().isdigit()]
planet_numbers = [1 for x in tic_ids]
astro_ids = st.sidebar.text_area("Enter Astro IDs [tic_id + planet_no] (comma-separated)", "").strip()
astro_ids = [int(astro_id.strip()) for astro_id in astro_ids.split(",") if astro_id.strip().isdigit()]

for astro_id in astro_ids:
    tic_ids.append(int(str(astro_id)[:-2]))
    planet_numbers.append(int(str(astro_id)[-2:]))

visualizer = TICVisualizer(server=server, df=df)

st.subheader(f"Report Pages for Selected TICs ({len(tic_ids)} TICs)")
selected_types = st.sidebar.multiselect("Select Report Page Types", sorted(ALL_PAGE_TYPES), default=sorted(ALL_PAGE_TYPES))

if tic_ids:
    visualizer.visualize_tic_ids(tic_ids=tic_ids, planet_numbers=planet_numbers, selected_types=selected_types)

else:
    st.sidebar.warning("Enter valid TIC IDs to view images.")