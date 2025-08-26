
import streamlit as st
from typing import List
from data_management.light_curve_server import PAGE_NUMBER_TO_TYPE, LightCurveServer
from PIL import Image
import pandas as pd


class TICVisualizer:
    def __init__(self, server: LightCurveServer, df: pd.DataFrame) -> None:
        self.server = server
        self.df = df

    def visualize_tic_ids(self, tic_ids: List[int], planet_numbers: List[int], selected_types: List[str]):
        tic_pages = {}

        for idx, tic_id in enumerate(tic_ids):
            planet_number = planet_numbers[idx]
            pages = self.server.get_report_pages(tic_id=tic_id, planet_number=planet_number)
            tic_pages[tic_id] = pages

        for idx, tic_id in enumerate(tic_ids):
            planet_number = planet_numbers[idx]
            pages = tic_pages.get(tic_id, [])
            type_to_page = {PAGE_NUMBER_TO_TYPE.get(p): p for p in pages if PAGE_NUMBER_TO_TYPE.get(p) in selected_types}

            tic_info = self.df.loc[self.df["tic_id"] == tic_id].dropna().head(1)
            label = tic_info["true_label"].values[0] if not tic_info.empty and "true_label" in tic_info else "Unknown"

            st.markdown(f"### TIC {tic_id} (Label: `{label}`)")

            cols = st.columns(3)
            for i, (ptype, page_num) in enumerate(type_to_page.items()):
                with cols[i % 3]:
                    image = self.server.get_page_image(tic_id, page_num)
                    if isinstance(image, Image.Image):
                        st.image(image, caption=f"{ptype} (Page {page_num})", use_container_width=True)
                    else:
                        st.warning(f"Image for {ptype} not available.")