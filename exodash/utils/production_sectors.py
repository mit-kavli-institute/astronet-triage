from collections import defaultdict
import streamlit as st
from typing import Optional
import os

if "selected_sectors" not in st.session_state:
    st.session_state.selected_sectors = set()
def get_production_sector_selector(tfrecord_postfix: Optional[str] = None):
    sectors = list(range(73, 95))
    available = {}
    errors = defaultdict(list)
    for sector in sectors:
        if sector == 85:
            available[sector] = True
            continue
        tfrecord_dir = f'/pdo/astronet-data/data/tfrecords/sector-{sector}-{tfrecord_postfix}' if tfrecord_postfix else f'/pdo/astronet-data/data/tfrecords/sector-{sector}'
        tfrecords_exist = os.path.isdir(tfrecord_dir) and os.listdir(tfrecord_dir)
        properties_exist = os.path.isfile(f'/pdo/astronet-data/data/properties/tces-sector{sector}.csv')
        astronet_scores_exist = os.path.isfile(f'/pdo/qlp-data/sector-{sector}/ffi/run/astronet_vetting_scores_cam1.alltriage.csv')
        qlp_centroid_filter_exists = os.path.isfile(f'/pdo/qlp-data/sector-{sector}/ffi/run/centroid_cam1.ls')
        qlp_delivery_exists = os.path.isfile(f'/pdo/qlp-data/tev/qlp-delivery/sector-{sector}/batch3/cand.ls')
        if properties_exist and tfrecords_exist and qlp_delivery_exists:
            available[sector] = True
        else:
            available[sector] = False
        if not tfrecords_exist:
            errors[sector].append(f'Missing TFRecords: {tfrecord_dir}')
        if not properties_exist:
            errors[sector].append('Missing properties CSV in /pdo/astronet-data/data/properties/')
        if not astronet_scores_exist:
            errors[sector].append(f'Missing Astronet scores in /pdo/qlp-data/sector-{sector}/ffi/run/')
        if not qlp_delivery_exists:
            errors[sector].append(f'Missing QLP delivery cand.ls in /pdo/qlp-data/tev/qlp-delivery/sector-{sector}/batch3/')
        if not qlp_centroid_filter_exists:
            errors[sector].append(f'Missing QLP centroid filter .ls in /pdo/qlp-data/sector-{sector}/ffi/run/')
    
    cols = st.columns(len(sectors))

    for i, sector in enumerate(sectors):
        with cols[i]:
            disabled = not available[sector]
            selected = sector in st.session_state.selected_sectors

            if st.button(
                f"{sector}",
                key=f"btn_{sector}",
                disabled=disabled,
                type="primary" if selected else "secondary",
                use_container_width=True,
            ):
                if selected:
                    st.session_state.selected_sectors.remove(sector)
                else:
                    st.session_state.selected_sectors.add(sector)
                st.rerun()

    valid_sectors = [s for s in sectors if available.get(s, False)]
    c_all, c_none = st.columns([1, 1])

    with c_all:
        if st.button("ALL", use_container_width=True):
            st.session_state.selected_sectors = set(valid_sectors)
            st.rerun()

    with c_none:
        if st.button("NONE", use_container_width=True):
            st.session_state.selected_sectors = set()
            st.rerun()
    for sector in errors:
        st.warning(f'Sector {sector} failed with errors: {errors[sector]}')
