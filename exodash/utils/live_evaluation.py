import os
from typing import List
import numpy as np
import tensorflow as tf
import pandas as pd

from astronet import models
from astronet.util import config_util

from astronet.astro_cnn_model import input_ds

MODEL: str = "AstroCNNModelVetting"

class LiveEvaluation():
    def __init__(self, model_dir: str) -> None:
      self.model_dir = model_dir

    def evaluate(self, eval_files: List[str]) -> pd.DataFrame:
        config = config_util.load_config(self.model_dir)
        # model_class = models.get_model_class(MODEL)
        # model = model_class(config)
        model = models.load_model(MODEL, self.model_dir)

        # Set up evaluation datasets
        eval_datasets = []
        for file_pattern in eval_files:
            if ":" in file_pattern:
                name, pattern = file_pattern.split(":", 1)
            elif len(eval_files) == 1:
                name, pattern = "eval", file_pattern
            else:
                raise ValueError("Multiple datasets must be named as name:file_pattern")
            eval_datasets.append((name, pattern))

        output_dir = os.path.join(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        result_dfs = []
        for name, file_pattern in eval_datasets:
            dataset = input_ds.build_eval_dataset(
                file_pattern=file_pattern,
                input_config=config.inputs,
                batch_size=config.hparams.batch_size,
                include_identifiers=True,
                include_labels=False,
            )

            astro_ids = []

            # If you have a sector per file_pattern, parse it; otherwise keep your placeholder
            sector = None#int(file_pattern.split("-")[-1].split("/")[0])#  if applicable

            for batch in dataset:
                x, identifiers = batch
                astro_ids.extend([x_id for x_id in identifiers.numpy()])

                # Forward pass (model returns (logits, embeddings) when return_embeddings=True)
                out = model(x, training=False)

                # Concatenate predictions/embeddings
                y_pred = np.concatenate(out, axis=0)

            # Derive metadata from astro_ids (after full pass so lengths match)
            tic_ids = [int(str(x)[:-2]) for x in astro_ids]
            planet_nos = [int(str(x)[-2:]) for x in astro_ids]
            model_nos = [0] * len(astro_ids)
            sectors = [sector] * len(astro_ids)

            # Build DataFrame
            pred_df = pd.DataFrame(y_pred, columns=["disp_p", "disp_e", "disp_n", "disp_j"])

            meta_df = pd.DataFrame({
                "Sector": sectors,
                "Astro ID": astro_ids,
                "tic_id": tic_ids,
                "planetno": planet_nos,
                "model_no": model_nos,
            })

            df = pd.concat([meta_df, pred_df], axis=1)
            result_dfs.append(df)

        return pd.concat(result_dfs, ignore_index=True)