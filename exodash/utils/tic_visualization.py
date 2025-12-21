from io import BytesIO
from typing import Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image

from exodash.utils.reports_tfrecords import TFRecordReports
from data_management.light_curve_server import (
    PAGE_NUMBER_TO_TYPE,
    PAGE_TYPE_TO_TFRECORD_KEY,
    LightCurveServer,
)

# --- Session singleton server ---
if "light_curve_server" not in st.session_state:
    st.session_state.light_curve_server = LightCurveServer()
server: LightCurveServer = st.session_state.light_curve_server


def infer_planet_number(astro_id: int, tic_id: int) -> int:
    return int(str(astro_id)[-2:]) if str(tic_id) in str(astro_id) else 1


def infer_astro_id(tic_id: int, planet_no: int) -> int:
    return int(f"{tic_id}{planet_no:02d}")


def _normalize_to_png_bytes(image) -> Optional[bytes]:
    """Accept PIL Image or bytes. Return PNG bytes if possible."""
    if image is None:
        return None
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    if isinstance(image, Image.Image):
        buf = BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
    return None


@st.cache_data(show_spinner=False)
def _get_page_images_cached(
    tic_id: int,
    planet_number: int,
    page_map: Dict[str, int],
    use_tfrecord: Dict[str, bool],
    # cache key: rely on object identity is bad; pass a lightweight token instead
    tfrecord_cache_token: Optional[str] = None,
) -> Dict[str, Tuple[int, bytes]]:
    """
    NOTE: We *do not* fetch TFRecordReports inside cache because it's not serializable/stable.
    We cache only server-rendered images here. TFRecord images will be added outside cache.
    """
    out: Dict[str, Tuple[int, bytes]] = {}
    for ptype, page_num in page_map.items():
        if use_tfrecord.get(ptype, False):
            continue  # skip; handled outside cache
        img = server.get_page_image(tic_id=tic_id, planet_number=planet_number, page_number=page_num)
        b = _normalize_to_png_bytes(img)
        if b:
            out[ptype] = (page_num, b)
    return out


def _get_page_images(
    tic_id: int,
    planet_number: int,
    page_map: Dict[str, int],
    tfrecord_reports: Optional[TFRecordReports] = None,
) -> Dict[str, Tuple[int, bytes]]:
    """
    Unified fetcher:
    - non-TFRecord pages via LightCurveServer (cached)
    - TFRecord pages via TFRecordReports (not cached unless you add your own token strategy)
    """
    use_tfrecord = {ptype: ("TFRecord" in ptype) for ptype in page_map.keys()}

    # cache only the LightCurveServer part
    cached = _get_page_images_cached(
        tic_id=tic_id,
        planet_number=planet_number,
        page_map=page_map,
        use_tfrecord=use_tfrecord,
        tfrecord_cache_token=None,  # optionally pass a stable string if you want
    )

    # add TFRecord pages (if provided)
    if tfrecord_reports:
        for ptype, page_num in page_map.items():
            if not use_tfrecord.get(ptype, False):
                continue
            page_name = PAGE_TYPE_TO_TFRECORD_KEY.get(ptype)
            if not page_name:
                continue
            img = tfrecord_reports.get_page(
                astro_id=infer_astro_id(tic_id=tic_id, planet_no=planet_number),
                page_name=page_name,
            )
            b = _normalize_to_png_bytes(img)
            if b:
                cached[ptype] = (page_num, b)

    return cached


def _selected_type_to_page_map(
    pages: List[int],
    selected_types: List[str],
    tfrecord_reports: Optional[TFRecordReports] = None,
) -> Dict[str, int]:
    # optionally include TFRecord “pages” even if they aren’t in server.get_report_pages()
    if tfrecord_reports:
        pages = list(pages) + [
            p for p, label in PAGE_NUMBER_TO_TYPE.items()
            if "TFRecord" in label
        ]

    return {
        PAGE_NUMBER_TO_TYPE.get(p): p
        for p in pages
        if PAGE_NUMBER_TO_TYPE.get(p) in selected_types
    }


def _render_image_grid(
    image_data: Dict[str, Tuple[int, bytes]],
    n_cols: int = 3,
):
    if not image_data:
        return
    n_cols = max(1, int(n_cols))
    cols = st.columns(n_cols)
    for i, (ptype, (page_num, image_bytes)) in enumerate(image_data.items()):
        with cols[i % n_cols]:
            st.image(image_bytes, caption=f"{ptype} (Page {page_num})", use_container_width=True)


class TICVisualizer:
    def __init__(self, server: LightCurveServer, df) -> None:
        self.server = server
        self.df = df

    def visualize_tic_ids(
        self,
        tic_ids: List[int],
        planet_numbers: List[int],
        selected_types: List[str],
        tfrecord_reports: Optional[TFRecordReports] = None,
        n_cols: int = 3,
    ):
        # prefetch pages (optional; keeps your previous behavior)
        tic_pages: Dict[int, List[int]] = {}
        for tic_id, planet_number in zip(tic_ids, planet_numbers):
            tic_pages[tic_id] = self.server.get_report_pages(
                tic_id=tic_id,
                planet_number=planet_number,
            )

        for tic_id, planet_number in zip(tic_ids, planet_numbers):
            pages = tic_pages.get(tic_id, [])
            page_map = _selected_type_to_page_map(
                pages=pages,
                selected_types=selected_types,
                tfrecord_reports=tfrecord_reports,
            )

            tic_info = self.df.loc[self.df["tic_id"] == tic_id].dropna().head(1)
            label = (
                tic_info["true_label"].values[0]
                if (not tic_info.empty and "true_label" in tic_info)
                else "Unknown"
            )

            st.markdown(f"### TIC {tic_id} (Label: `{label}`)")

            image_data = _get_page_images(
                tic_id=tic_id,
                planet_number=planet_number,
                page_map=page_map,
                tfrecord_reports=tfrecord_reports,
            )

            if not image_data:
                st.warning(f"No report images found for TIC {tic_id}, planet # {planet_number}.")
                continue

            _render_image_grid(image_data, n_cols=n_cols)