from data_management.light_curve_server import PAGE_NUMBER_TO_TYPE
from PIL import Image
from exodash.utils.filter import advanced_filter_sidebar
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

if tic_ids:
    all_page_types = set()
    tic_pages = {}

    for idx, tic_id in enumerate(tic_ids):
        planet_number = planet_numbers[idx]
        pages = server.get_report_pages(tic_id=tic_id, planet_number=planet_number)
        page_types = [PAGE_NUMBER_TO_TYPE.get(p) for p in pages if p in PAGE_NUMBER_TO_TYPE]
        page_types = [ptype for ptype in page_types if ptype is not None]
        all_page_types.update(page_types)
        tic_pages[tic_id] = pages

    selected_types = st.sidebar.multiselect("Select Report Page Types", sorted(all_page_types), default=sorted(all_page_types))

    st.subheader(f"Report Pages for Selected TICs ({len(tic_ids)} TICs)")

    for idx, tic_id in enumerate(tic_ids):
        planet_number = planet_numbers[idx]
        pages = tic_pages.get(tic_id, [])
        type_to_page = {PAGE_NUMBER_TO_TYPE.get(p): p for p in pages if PAGE_NUMBER_TO_TYPE.get(p) in selected_types}

        tic_info = df.loc[df["tic_id"] == tic_id].dropna().head(1)
        label = tic_info["true_label"].values[0] if not tic_info.empty else "Unknown"

        st.markdown(f"### TIC {tic_id} (Label: `{label}`)")

        cols = st.columns(3)
        for i, (ptype, page_num) in enumerate(type_to_page.items()):
            with cols[i % 3]:
                image = server.get_page_image(tic_id, page_num)
                if isinstance(image, Image.Image):
                    st.image(image, caption=f"{ptype} (Page {page_num})", use_container_width=True)
                else:
                    st.warning(f"Image for {ptype} not available.")

else:
    st.sidebar.warning("Enter valid TIC IDs to view images.")