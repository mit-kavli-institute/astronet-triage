"""Script for evaluating a trained AstroNet model."""

import os
import numpy as np
import tensorflow as tf
from absl import app, flags, logging
import pandas as pd

from astronet import evaluation, models
from astronet.util import config_util

from astronet.astro_cnn_model import input_ds

flags.DEFINE_string("model", None, "Name of the model class.", required=True)

flags.DEFINE_string(
    "model_dir", None,
    "Directory of the trained model to evaluate (must contain config.yaml).",
    required=True)

flags.DEFINE_string(
    "output_dir", None,
    "Directory of where to store results.",
    required=True)

flags.DEFINE_multi_string(
    "eval_files", None,
    "File patterns matching the TFRecord files in the evaluation dataset(s). "
    "Each dataset can be named with the format name:file_patterns. If a single "
    "pattern is passed, it defaults to the name 'eval'.",
    required=True)

flags.DEFINE_float("threshold", 0.215,
                   "Threshold for binary classification evaluation.")

FLAGS = flags.FLAGS


def main(_):
  return_embeddings = True
  config = config_util.load_config(FLAGS.model_dir)
  model_class = models.get_model_class(FLAGS.model)
  model = model_class(config, return_embeddings=return_embeddings)
  model = models.load_model(FLAGS.model, FLAGS.model_dir)
  if return_embeddings:
    setattr(model, "return_embeddings", True)

  output_dir = FLAGS.output_dir

  # Set up evaluation datasets
  eval_datasets = []
  for file_pattern in FLAGS.eval_files:
    if ":" in file_pattern:
      name, pattern = file_pattern.split(":", 1)
    elif len(FLAGS.eval_files) == 1:
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
    logits_list = []
    emb_list = []

    # If you have a sector per file_pattern, parse it; otherwise keep your placeholder
    sector = int(file_pattern.split("-")[-1].split("/")[0])#  if applicable

    for batch in dataset:
      x, identifiers = batch
      astro_ids.extend([x_id for x_id in identifiers.numpy()])

      # Forward pass (model returns (logits, embeddings) when return_embeddings=True)
      out = model(x, training=False)
      if isinstance(out, (list, tuple)) and len(out) == 2:
        logits, emb = out
      else:
        # Fallback: if your model is still single-output for some reason
        logits, emb = out, None

      logits_list.append(logits.numpy())
      if emb is not None:
        emb_list.append(emb.numpy())

    # Concatenate predictions/embeddings
    y_pred = np.concatenate(logits_list, axis=0)
    if emb_list:
      embeddings = np.concatenate(emb_list, axis=0)
      # L2-normalize for cosine/Euclidean distance work
      embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    else:
      # Shouldn't happen with return_embeddings=True, but guard anyway
      raise RuntimeError("Embeddings were not returned by the model. "
                         "Ensure return_embeddings=True in the model and loader.")

    # Derive metadata from astro_ids (after full pass so lengths match)
    tic_ids = [int(str(x)[:-2]) for x in astro_ids]
    planet_nos = [int(str(x)[-2:]) for x in astro_ids]
    model_nos = [0] * len(astro_ids)
    sectors = [sector] * len(astro_ids)

    # Build DataFrame
    pred_df = pd.DataFrame(y_pred, columns=["disp_p", "disp_e", "disp_n", "disp_j"])
    emb_cols = [f"fc_{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)

    meta_df = pd.DataFrame({
        "Sector": sectors,
        "Astro ID": astro_ids,
        "tic_id": tic_ids,
        "planetno": planet_nos,
        "model_no": model_nos,
    })

    df = pd.concat([meta_df, pred_df, emb_df], axis=1)
    result_dfs.append(df)

  combined_df = pd.concat(result_dfs, ignore_index=True)
  csv_path = os.path.join(output_dir, f"{name}_predictions.csv")
  print(f'Saved results to {csv_path}')
  combined_df.to_csv(csv_path, index=False)


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  app.run(main)