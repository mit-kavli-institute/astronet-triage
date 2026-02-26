#!/usr/bin/env python3
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from absl import app, logging

REPO_ROOT = "/pdo/users/pablomer/Astronet-Triage"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from downstream_tasks.load_model import load_model_from_checkpoint
from astronet.astro_cnn_model import input_ds

MODEL_DIR = (
    "/pdo/astronet-data/models/vetting/experimental/pablomer/"
    "dec2025_cad_scat_v5_duration24/20260226/pablomer-2k-nopretrained-rp/"
    "AstroCNNModelVetting_pablomer_20260226_141255"
)
SECTORS = [80, 81, 82, 83, 84]
TFRECORD_GLOB_TEMPLATE = "/pdo/astronet-data/data/tfrecords/sector-{sector}-scatter/*"

OUTPUT_DIR = "/pdo/users/pablomer/Astronet-Triage/rp_filter"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "live_sector_80_84_predictions_with_rp.csv")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "live_sector_80_84_rp_distribution_threshold_0p9.png")

PLANET_THRESHOLD = 0.9
HARD_RP_LIMIT = 30.0
MAX_RP_FOR_PLOT = 100.0

DEPTH_CANDIDATES = ["Transit_Depth", "depth", "Depth"]
STAR_RAD_CANDIDATES = ["star_rad", "s_rad", "SRad", "star_rad_est"]


def _find_feature_name(features_dict, candidates):
    for key in candidates:
        if key in features_dict:
            return key
    return None


def _flatten_numpy(x):
    arr = np.asarray(x)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    return arr.reshape(arr.shape[0], -1)[:, 0]


def _extract_raw_metadata(file_patterns):
    """Read raw (unscaled) astro_id/depth/star_rad directly from TFRecords."""
    filenames = tf.io.gfile.glob(file_patterns)
    if not filenames:
        raise ValueError(f"No files matched file_patterns={file_patterns}")

    raw_rows = []
    for raw_record in tf.data.TFRecordDataset(filenames):
        ex = tf.train.Example()
        ex.ParseFromString(raw_record.numpy())
        feats = ex.features.feature

        astro_vals = feats["astro_id"].int64_list.value if "astro_id" in feats else []
        depth_vals = feats["Transit_Depth"].float_list.value if "Transit_Depth" in feats else []
        srad_vals = feats["star_rad"].float_list.value if "star_rad" in feats else []
        srad_est_vals = feats["star_rad_est"].float_list.value if "star_rad_est" in feats else []

        astro_id = int(astro_vals[0]) if astro_vals else np.nan
        depth = float(depth_vals[0]) if depth_vals else np.nan
        if srad_vals:
            s_rad = float(srad_vals[0])
        elif srad_est_vals:
            s_rad = float(srad_est_vals[0])
        else:
            s_rad = np.nan

        # Match CSV convention: encode missing star radius as NaN.
        if s_rad == 0:
            s_rad = np.nan

        raw_rows.append({"astro_id": astro_id, "depth_raw": depth, "s_rad_raw": s_rad})

    return pd.DataFrame(raw_rows)


def main(argv):
    del argv
    logging.set_verbosity(logging.INFO)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model, config = load_model_from_checkpoint(MODEL_DIR, compile_model=False)
    logging.info("Loaded model from %s", MODEL_DIR)

    file_patterns = []
    used_sectors = []
    missing_sectors = []
    for s in SECTORS:
        pattern = TFRECORD_GLOB_TEMPLATE.format(sector=s)
        files = tf.io.gfile.glob(pattern)
        if files:
            file_patterns.append(pattern)
            used_sectors.append(s)
        else:
            missing_sectors.append(s)

    if not file_patterns:
        raise ValueError("No TFRecord files found for requested sectors.")

    if missing_sectors:
        logging.warning("Missing sectors (no tfrecords found): %s", missing_sectors)
    logging.info("Using sectors: %s", used_sectors)

    ds = input_ds.build_eval_dataset(
        file_pattern=file_patterns,
        input_config=config.inputs,
        batch_size=config.hparams.batch_size,
        include_identifiers=True,
        include_labels=False,
    )

    all_astro_ids = []
    all_depth = []
    all_srad = []
    all_preds = []

    depth_name = None
    srad_name = None

    for batch_i, batch in enumerate(ds):
        features, astro_ids = batch

        if depth_name is None:
            depth_name = _find_feature_name(features, DEPTH_CANDIDATES)
            srad_name = _find_feature_name(features, STAR_RAD_CANDIDATES)
            if depth_name is None or srad_name is None:
                raise ValueError(
                    f"Could not find depth/star radius features. "
                    f"depth candidates={DEPTH_CANDIDATES}, star_rad candidates={STAR_RAD_CANDIDATES}. "
                    f"available keys={list(features.keys())[:30]}"
                )
            logging.info("Using depth feature: %s", depth_name)
            logging.info("Using star radius feature: %s", srad_name)

        pred_batch = model(features, training=False).numpy()

        all_preds.append(pred_batch)
        all_astro_ids.append(_flatten_numpy(astro_ids.numpy()))
        all_depth.append(_flatten_numpy(features[depth_name].numpy()))
        all_srad.append(_flatten_numpy(features[srad_name].numpy()))

        if (batch_i + 1) % 10 == 0:
            logging.info("Processed %d batches", batch_i + 1)

    preds = np.concatenate(all_preds, axis=0)
    astro_ids = np.concatenate(all_astro_ids, axis=0).astype(np.int64)

    # IMPORTANT: depth/star radius for r_p must be raw physical values, not scaled features.
    raw_meta = _extract_raw_metadata(file_patterns)
    if len(raw_meta) != len(astro_ids):
        logging.warning(
            "Raw metadata length (%d) != predictions length (%d). Falling back to astro_id merge.",
            len(raw_meta), len(astro_ids)
        )
        pred_index = pd.DataFrame({"astro_id": astro_ids})
        raw_meta = raw_meta.drop_duplicates(subset=["astro_id"], keep="first")
        aligned = pred_index.merge(raw_meta, on="astro_id", how="left")
        depth = aligned["depth_raw"].to_numpy(dtype=np.float64)
        s_rad = aligned["s_rad_raw"].to_numpy(dtype=np.float64)
    else:
        raw_ids = raw_meta["astro_id"].to_numpy(dtype=np.int64)
        if np.array_equal(raw_ids, astro_ids):
            depth = raw_meta["depth_raw"].to_numpy(dtype=np.float64)
            s_rad = raw_meta["s_rad_raw"].to_numpy(dtype=np.float64)
        else:
            logging.warning(
                "Raw metadata order does not match prediction order. Falling back to astro_id merge."
            )
            pred_index = pd.DataFrame({"astro_id": astro_ids})
            raw_meta = raw_meta.drop_duplicates(subset=["astro_id"], keep="first")
            aligned = pred_index.merge(raw_meta, on="astro_id", how="left")
            depth = aligned["depth_raw"].to_numpy(dtype=np.float64)
            s_rad = aligned["s_rad_raw"].to_numpy(dtype=np.float64)

    df = pd.DataFrame({
        "astro_id": astro_ids,
        "depth": depth,
        "s_rad": s_rad,
        "disp_p": preds[:, 0],
        "disp_e": preds[:, 1],
        "disp_n": preds[:, 2],
        "disp_j": preds[:, 3],
    })

    df["r_p"] = df["s_rad"] * np.sqrt(df["depth"] / (10 ** 6)) * 109.076
    df.to_csv(OUTPUT_CSV, index=False)
    logging.info("Saved predictions+r_p CSV: %s", OUTPUT_CSV)

    df_plot = df.dropna(subset=["disp_p", "r_p"]).copy()
    predicted_planet_all = df_plot[df_plot["disp_p"] >= PLANET_THRESHOLD]["r_p"].values
    predicted_non_planet_all = df_plot[df_plot["disp_p"] < PLANET_THRESHOLD]["r_p"].values

    n_predicted_planet_all = int(len(predicted_planet_all))
    n_predicted_non_planet_all = int(len(predicted_non_planet_all))
    n_predicted_planet_above_hard_limit = int(np.sum(predicted_planet_all > HARD_RP_LIMIT))
    frac_above_limit = (
        100.0 * n_predicted_planet_above_hard_limit / n_predicted_planet_all
        if n_predicted_planet_all > 0 else 0.0
    )

    predicted_planet = predicted_planet_all[predicted_planet_all <= MAX_RP_FOR_PLOT]
    predicted_non_planet = predicted_non_planet_all[predicted_non_planet_all <= MAX_RP_FOR_PLOT]

    bins_max = min(np.nanpercentile(df_plot["r_p"], 99), MAX_RP_FOR_PLOT)
    bins = np.linspace(0, bins_max, 50)

    plt.figure(figsize=(10, 6))
    plt.hist(
        predicted_planet,
        bins=bins,
        alpha=0.6,
        label=f"Predicted planet (>= {PLANET_THRESHOLD}) [N={n_predicted_planet_all}]",
        color="#2ca02c",
    )
    plt.hist(
        predicted_non_planet,
        bins=bins,
        alpha=0.6,
        label=f"Predicted non-planet (< {PLANET_THRESHOLD}) [N={n_predicted_non_planet_all}]",
        color="#d62728",
    )
    plt.axvline(HARD_RP_LIMIT, color="black", linestyle="--", linewidth=1.5, label="hard rp limit")
    ymax = plt.ylim()[1]
    plt.text(HARD_RP_LIMIT + 1.0, ymax * 0.95, "hard rp limit", rotation=90, va="top", ha="left")

    plt.xlabel("r_p")
    plt.ylabel("Count")
    plt.title(f"Live sectors {used_sectors}: Distribution of r_p by predicted class")
    plt.text(
        0.98,
        0.98,
        (
            f"Pred. planets > {HARD_RP_LIMIT:g}: "
            f"{n_predicted_planet_above_hard_limit} ({frac_above_limit:.1f}%)"
        ),
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=220)

    print(f"Saved r_p distribution plot: {OUTPUT_PNG}")
    print(f"N(predicted planet, all)={n_predicted_planet_all}")
    print(f"N(predicted non-planet, all)={n_predicted_non_planet_all}")
    print(f"N(predicted planet, plotted <= {MAX_RP_FOR_PLOT:g})={len(predicted_planet)}")
    print(f"N(predicted non-planet, plotted <= {MAX_RP_FOR_PLOT:g})={len(predicted_non_planet)}")
    print(
        f"N(predicted planet with r_p > {HARD_RP_LIMIT:g}) before r_p>{MAX_RP_FOR_PLOT:g} drop="
        f"{n_predicted_planet_above_hard_limit} ({frac_above_limit:.1f}%)"
    )


if __name__ == "__main__":
    app.run(main)
