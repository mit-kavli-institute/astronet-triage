from data_management.light_curve_server import PAGE_NUMBER_TO_TYPE, LightCurveServer
from PIL import Image
import streamlit as st

if "df" not in st.session_state:
    st.error("Dataset not found. Please run `app.py` first.")
    st.stop()

df = st.session_state.df
st.title("TIC Explorer")

st.sidebar.header("🔎 Select TIC IDs")
tic_ids = st.sidebar.text_area("Enter TIC IDs (comma-separated)", "").strip()
tic_ids = [int(tic.strip()) for tic in tic_ids.split(",") if tic.strip().isdigit()]
astro_ids = st.sidebar.text_area("Enter Astro IDs [tic_id + planet_no] (comma-separated)", "").strip()

# Instantiate your LightCurveServer
server = LightCurveServer()

if tic_ids:
    all_page_types = set()
    tic_pages = {}

    for tic_id in tic_ids:
        pages = server.get_report_pages(tic_id)
        page_types = [PAGE_NUMBER_TO_TYPE.get(p) for p in pages if p in PAGE_NUMBER_TO_TYPE]
        page_types = [ptype for ptype in page_types if ptype is not None]
        all_page_types.update(page_types)
        tic_pages[tic_id] = pages

    selected_types = st.sidebar.multiselect("Select Report Page Types", sorted(all_page_types), default=sorted(all_page_types))

    st.subheader(f"Report Pages for Selected TICs ({len(tic_ids)} TICs)")

    for tic_id in tic_ids:
        pages = tic_pages.get(tic_id, [])
        type_to_page = {PAGE_NUMBER_TO_TYPE.get(p): p for p in pages if PAGE_NUMBER_TO_TYPE.get(p) in selected_types}

        tic_info = df.loc[df["tic_id"] == tic_id, ["label"]].dropna().head(1)
        label = tic_info["label"].values[0] if not tic_info.empty else "Unknown"

        st.markdown(f"### TIC {tic_id} (Label: `{label}`)")

        cols = st.columns(3)
        for i, (ptype, page_num) in enumerate(type_to_page.items()):
            with cols[i % 3]:
                image = server.get_page_image(tic_id, page_num)
                if isinstance(image, Image.Image):
                    st.image(image, caption=f"{ptype} (Page {page_num})", use_container_width=True)
                else:
                    st.warning(f"Image for {ptype} not available.")

    if st.button("🔄 Refresh Page"):
        st.experimental_rerun()

else:
    st.sidebar.warning("Enter valid TIC IDs to view images.")