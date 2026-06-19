#!/usr/bin/env python3
import json
import os
import numpy as np
import pandas as pd
from absl import app, flags, logging

# This script combines model results from an ensemble of models into a single CSV file.

FLAGS = flags.FLAGS
flags.DEFINE_string("base_path", None, "Base path to the model folders.")
flags.DEFINE_bool("output_rp", False, "If True, export an extra CSV with computed r_p.")
flags.DEFINE_string("rp_metadata_csv", None, "Optional metadata CSV with astro_id/depth/s_rad columns.")
flags.DEFINE_string("rp_output_filename", "all_preds_with_rp.csv", "Output CSV name when output_rp is enabled.")
flags.mark_flag_as_required("base_path")

def _normalize_cols(df):
    renamed = {}
    for col in df.columns:
        new_col = col.strip().lower().replace(" ", "_")
        renamed[col] = new_col
    return df.rename(columns=renamed)

def _load_metadata(path):
    meta = pd.read_csv(path)
    meta = _normalize_cols(meta)
    if "astro_id" not in meta.columns:
        raise ValueError(f"rp metadata file {path} is missing astro_id/Astro ID column")
    if "depth" not in meta.columns:
        raise ValueError(f"rp metadata file {path} is missing depth/Depth column")
    if "s_rad" not in meta.columns and "srad" in meta.columns:
        meta = meta.rename(columns={"srad": "s_rad"})
    if "s_rad" not in meta.columns:
        raise ValueError(f"rp metadata file {path} is missing s_rad/SRad column")
    keep = meta[["astro_id", "depth", "s_rad"]].copy()
    keep["astro_id"] = pd.to_numeric(keep["astro_id"], errors="coerce")
    keep["depth"] = pd.to_numeric(keep["depth"], errors="coerce")
    keep["s_rad"] = pd.to_numeric(keep["s_rad"], errors="coerce")
    keep = keep.dropna(subset=["astro_id"]).drop_duplicates(subset=["astro_id"])
    keep["astro_id"] = keep["astro_id"].astype(int)
    return keep

def _load_config_output_rp(model_folder):
    config_path = os.path.join(model_folder, "config.json")
    if not os.path.exists(config_path):
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return bool(cfg.get("output_rp", False))
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning(f"Could not read output_rp from {config_path}: {exc}")
        return False

def main(argv):
    base_path = FLAGS.base_path
    label_names = ["p", "e", "n", "j"]

    # Discover model folders
    subfolders = sorted([
        os.path.join(base_path, f) for f in os.listdir(base_path)
        if (
            os.path.isdir(os.path.join(base_path, f)) and
            os.path.exists(os.path.join(base_path, f, "evaluation", "test_pred.npy"))
        )
    ])
    if not subfolders:
        raise ValueError(f"No model subfolders with evaluation/test_pred.npy found under {base_path}")

    all_preds = {}
    true_labels_dict = {}
    astro_ids = None
    exodash_df = None

    for i, folder in enumerate(subfolders, start=1):
        eval_path = os.path.join(folder, "evaluation")
        pred_path = os.path.join(eval_path, "test_pred.npy")

        if os.path.exists(pred_path):
            preds = np.load(pred_path)
            for j, label in enumerate(label_names):
                col_name = f"disp_{label.replace(' ', '_')}_{i}"
                all_preds[col_name] = preds[:, j]

        if i == 1:
            astro_ids = np.load(os.path.join(eval_path, "test_astro_ids.npy"))
            true_labels = np.load(os.path.join(eval_path, "test_label.npy"))

            # Try loading ExoDash CSV
            exodash_path = os.path.join(eval_path, "test_exodash_results.csv")
            if os.path.exists(exodash_path):
                exodash_df = pd.read_csv(exodash_path)
                exodash_df = exodash_df.drop(columns=["model_no", "disp_p", "disp_e", "disp_n", "disp_j"], errors='ignore')
            else:
                logging.warning(f"ExoDash results not found at {exodash_path}. Proceeding without merge.")

            for j, label in enumerate(label_names):
                true_labels_dict[f"true_{label.replace(' ', '_')}"] = true_labels[:, j]

    # Assemble DataFrame
    df = pd.DataFrame(all_preds)
    df.insert(0, "astro_id", astro_ids)

    for col, values in true_labels_dict.items():
        df.insert(1, col, values)

    # Merge exodash if available
    if exodash_df is not None:
        df = df.merge(exodash_df, on="astro_id", how="left")

    print(df.head())
    print(f"Final shape: {df.shape}")

    output_path = os.path.join(base_path, "all_preds.csv")
    df.to_csv(output_path, index=False)
    print(f"DataFrame saved to {output_path}")

    output_rp_enabled = FLAGS.output_rp or _load_config_output_rp(subfolders[0])
    if not output_rp_enabled:
        return

    rp_df = df.copy()
    if FLAGS.rp_metadata_csv:
        metadata = _load_metadata(FLAGS.rp_metadata_csv)
        rp_df = rp_df.merge(metadata, on="astro_id", how="left")
    rp_df = _normalize_cols(rp_df)
    if "srad" in rp_df.columns and "s_rad" not in rp_df.columns:
        rp_df = rp_df.rename(columns={"srad": "s_rad"})
    if "depth" not in rp_df.columns or "s_rad" not in rp_df.columns:
        raise ValueError(
            "Could not compute r_p because depth/s_rad columns are missing. "
            "Pass --rp_metadata_csv with columns astro_id, Depth, and SRad."
        )

    rp_df["depth"] = pd.to_numeric(rp_df["depth"], errors="coerce")
    rp_df["s_rad"] = pd.to_numeric(rp_df["s_rad"], errors="coerce")
    rp_df["r_p"] = rp_df["s_rad"] * np.sqrt(rp_df["depth"] / (10 ** 6)) * 109.076

    rp_output_path = os.path.join(base_path, FLAGS.rp_output_filename)
    rp_df.to_csv(rp_output_path, index=False)
    print(f"DataFrame with r_p saved to {rp_output_path}")

if __name__ == "__main__":
    app.run(main)
