import os
from typing import List
import numpy as np
import tensorflow as tf
import pandas as pd
#import streamlit as st

from astronet import models
from astronet.util import config_util

from astronet.astro_cnn_model import input_ds

MODEL: str = "AstroCNNModelVetting"

class LiveEvaluation():
    def __init__(self, model_dir: str) -> None:
      self.model_dir = model_dir
      self.ensemble_models = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]

    def evaluate(self, eval_files: List[str]) -> pd.DataFrame:
        #st.write(f'Evaluating model on {eval_files}')
        result_dfs = []

        total_models = len(self.ensemble_models)
        #progress_bar = st.progress(0)
        #status_text = st.empty()

        for model_no, model_path in enumerate(self.ensemble_models):
            #status_text.write(f"Evaluating model {model_no + 1}/{total_models}: `{os.path.basename(model_path)}`")
            #progress_bar.progress((model_no + 1) / total_models)

            config = config_util.load_config(model_path)
            
            model_class = models.get_model_class(MODEL)
            model = model_class(config, return_embeddings=True)
            model = models.load_model(MODEL, model_path)
            if True:
                setattr(model, "return_embeddings", True)

            # Build named datasets
            eval_datasets = []
            for file_pattern in eval_files:
                if ":" in file_pattern:
                    name, pattern = file_pattern.split(":", 1)
                elif len(eval_files) == 1:
                    name, pattern = "eval", file_pattern
                else:
                    raise ValueError("Multiple datasets must be named as name:file_pattern")
                eval_datasets.append((name, pattern))

            for name, file_pattern in eval_datasets:
                dataset = input_ds.build_eval_dataset(
                    file_pattern=file_pattern,
                    input_config=config.inputs,
                    batch_size=config.hparams.batch_size,
                    include_identifiers=True,
                    include_labels=False,
                )

                all_logits = []
                all_embeddings = []
                all_ids = []
                for batch in dataset:
                    x, identifiers = batch
                    out = model(x, training=False)
                    logits, emb = out
                    all_logits.append(logits.numpy())
                    all_embeddings.append(emb.numpy())
                    all_ids.extend([int(x_id) for x_id in identifiers.numpy()])

                if not all_logits:
                    continue

                # Concatenate predictions/embeddings
                preds = np.concatenate(all_logits, axis=0)
                if all_embeddings:
                    embeddings = np.concatenate(all_embeddings, axis=0)
                    # L2-normalize for cosine/Euclidean distance work
                    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                else:
                    # Shouldn't happen with return_embeddings=True, but guard anyway
                    raise RuntimeError("Embeddings were not returned by the model. "
                                    "Ensure return_embeddings=True in the model and loader.")
                astro_ids = np.array(all_ids)

                # Metadata columns
                tic_ids = [int(str(x)[:-2]) for x in astro_ids]
                planet_nos = [int(str(x)[-2:]) for x in astro_ids]

                meta_df = pd.DataFrame({
                    "dataset": [name] * len(astro_ids),
                    "astro_id": astro_ids,
                    "tic_id": tic_ids,
                    "planetno": planet_nos,
                    "model_no": [model_no] * len(astro_ids),
                })

                pred_df = pd.DataFrame(
                    preds, columns=["disp_p", "disp_e", "disp_n", "disp_j"]
                )

                emb_cols = [f"fc_{i}" for i in range(embeddings.shape[1])]
                emb_df = pd.DataFrame(embeddings, columns=emb_cols)

                df = pd.concat([meta_df, pred_df, emb_df], axis=1)
                result_dfs.append(df)

        #progress_bar.progress(1.0)
        #status_text.success("✅ All models evaluated.")
        return pd.concat(result_dfs, ignore_index=True)