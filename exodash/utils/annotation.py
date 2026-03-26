import streamlit as st
import getpass
import os
import pandas as pd
from datetime import datetime


class AnnotationHandler:

    PRIMARY_LABELS = ["planet", "eb", "junk", "unclear"]
    SECONDARY_LABELS = ["obvious", "subtle", "systematic"]

    def __init__(
        self,
        astro_id: int,
        row: pd.Series,
        model_version: str,
        data_version: str,
        base_dir: str = "/pdo/astronet-data/labels",
    ):
        self.astro_id = astro_id
        self.row = row
        self.filepath = self._get_filepath(model_version, data_version, base_dir)
        self._render()

    def _get_filepath(self, model_version: str, data_version: str, base_dir: str) -> str:
        user = getpass.getuser()
        filename = f"{user}_model-{model_version}_data-{data_version}.csv"
        return os.path.join(base_dir, filename)

    def _load(self) -> pd.DataFrame:
        if os.path.exists(self.filepath):
            return pd.read_csv(self.filepath)
        return pd.DataFrame(columns=["astro_id", "primary", "secondary", "notes", "labeled_at"])

    def _save(self, annotation: dict):
        df = self._load()
        df = df[df["astro_id"] != annotation["astro_id"]]
        df = pd.concat([df, pd.DataFrame([annotation])], ignore_index=True)
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        df.to_csv(self.filepath, index=False)

    def _render(self):
        existing = self._load()
        prior = existing[existing["astro_id"] == self.astro_id]
        already_labeled = not prior.empty
        prev = prior.iloc[0] if already_labeled else None

        # Astronet scores for context
        score_cols = [c for c in ["disp_p", "disp_e", "disp_j", "disp_n"] if c in self.row.index]
        if score_cols:
            cols = st.columns(len(score_cols))
            for col, c in zip(cols, score_cols):
                col.metric(c, f"{self.row[c]:.3f}")

        if already_labeled:
            st.success(f"Previously labeled: **{prev['primary']}** / **{prev['secondary']}**")

        c1, c2 = st.columns(2)
        with c1:
            primary = st.radio(
                "What is it?",
                self.PRIMARY_LABELS,
                index=self.PRIMARY_LABELS.index(prev["primary"]) if already_labeled else 3,
                key=f"primary_{self.astro_id}",
                horizontal=True,
            )
        with c2:
            secondary = st.radio(
                "Why did the model get it wrong/right?",
                self.SECONDARY_LABELS,
                index=self.SECONDARY_LABELS.index(prev["secondary"]) if already_labeled else 1,
                key=f"secondary_{self.astro_id}",
                horizontal=True,
            )

        notes = st.text_input(
            "Notes",
            value=prev["notes"] if already_labeled and pd.notna(prev["notes"]) else "",
            key=f"notes_{self.astro_id}",
            placeholder="optional",
        )

        b1, b2 = st.columns(2)
        with b1:
            if st.button("💾 Save & Next", key=f"save_{self.astro_id}", use_container_width=True, type="primary"):
                self._save({
                    "astro_id": self.astro_id,
                    "primary": primary,
                    "secondary": secondary,
                    "notes": notes,
                    "labeled_at": datetime.now().isoformat(),
                })
                st.session_state.page += 1
                st.rerun()
        with b2:
            if st.button("⏭️ Skip", key=f"skip_{self.astro_id}", use_container_width=True):
                if "skipped_astro_ids" not in st.session_state:
                    st.session_state.skipped_astro_ids = []
                st.session_state.skipped_astro_ids.append(self.astro_id)
                st.session_state.page += 1
                st.rerun()