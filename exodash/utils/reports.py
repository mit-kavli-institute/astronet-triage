from io import BytesIO
from typing import Dict, List, Tuple
import streamlit as st
from PIL import Image
from data_management.light_curve_server import PAGE_NUMBER_TO_TYPE, LightCurveServer

if "light_curve_server" not in st.session_state:
    st.session_state.light_curve_server = LightCurveServer()

server = st.session_state.light_curve_server

def infer_planet_number(astro_id: int, tic_id: int):
    return int(str(astro_id)[-2:]) if str(tic_id) in str(astro_id) else 1

@st.cache_data
def _get_page_images(tic_id: int, planet_number: int, page_map: Dict[str, int]) -> Dict[str, Tuple[int, bytes]]:
    result = {}
    for ptype, page_num in page_map.items():
        image = server.get_page_image(tic_id=tic_id, planet_number=planet_number, page_number=page_num)
        if isinstance(image, Image.Image):
            buf = BytesIO()
            image.save(buf, format='PNG')
            result[ptype] = (page_num, buf.getvalue())

    return result

def generate_report_for_tic_id(tic_id: int, planet_number: int, pages: List[int], selected_types: List[str]):
    type_to_page = {
        PAGE_NUMBER_TO_TYPE.get(p): p
        for p in pages if PAGE_NUMBER_TO_TYPE.get(p) in selected_types
    }

    image_data = _get_page_images(tic_id=tic_id, planet_number=planet_number, page_map=type_to_page)

    if not image_data:
        st.warning(f"No report images found for TIC ID {tic_id}, planet # {planet_number}.")
        return

    cols = st.columns(1)
    for i, (ptype, (page_num, image_bytes)) in enumerate(image_data.items()):
        with cols[i % 1]:
            st.image(image_bytes, caption=f"{ptype} (Page {page_num})", use_container_width=True)
