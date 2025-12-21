

from collections import defaultdict
from io import BytesIO
from typing import List
from astronet.util import config_util
from data_management.light_curve_server import PAGE_TYPE_TO_TFRECORD_KEY
import numpy as np
import streamlit as st
import pandas as pd
from astronet.astro_cnn_model import input_ds
import matplotlib.pyplot as plt
from PIL import Image

AVAILABLE_PAGES = list(PAGE_TYPE_TO_TFRECORD_KEY.values())

class TFRecordReports:
    def __init__(self, eval_files: List[str], model_config_path: str) -> None:
        self.eval_files = eval_files
        self.model_config_path = model_config_path
        self.pages_to_mapping = defaultdict(dict) # {'global_view': {0001: global_view_data}}

        eval_datasets = []
        for file_pattern in eval_files:
            if ":" in file_pattern:
                name, pattern = file_pattern.split(":", 1)
            elif len(eval_files) == 1:
                name, pattern = "eval", file_pattern
            else:
                raise ValueError("Multiple datasets must be named as name:file_pattern")
            eval_datasets.append((name, pattern))


        all_x = []
        all_ids = []
                    
        config = config_util.load_config(self.model_config_path)

        for name, file_pattern in eval_datasets:
            dataset = input_ds.build_eval_dataset(
                file_pattern=file_pattern,
                input_config=config.inputs,
                batch_size=config.hparams.batch_size,
                include_identifiers=True,
                include_labels=False,
            )

            for features, identifiers in dataset:
                ids = identifiers.numpy()
                for view_key in AVAILABLE_PAGES:
                    views = features[view_key].numpy()  # shape: (B, ...)

                    for astro_id, v in zip(ids, views):
                        astro_id = int(astro_id)
                        self.pages_to_mapping[view_key][astro_id] = v
            

    def get_page(self, astro_id: int, page_name: str):
        image_data = self.pages_to_mapping[page_name].get(astro_id)

        if image_data is None:
            return None

        fig, ax = plt.subplots()
        ax.plot(image_data)
        ax.set_title(page_name)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)   # IMPORTANT: prevent memory leaks

        buf.seek(0)
        return Image.open(buf)