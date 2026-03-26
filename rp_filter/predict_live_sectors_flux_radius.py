#!/usr/bin/env python3
import argparse
import configparser
import io
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from absl import logging

REPO_ROOT = "/pdo/users/pablomer/Astronet-Triage"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from downstream_tasks.load_model import load_model_from_checkpoint
from astronet.astro_cnn_model import input_ds

# Physical constants (same logic as radius_plot.py)
STEFAN_BOLTZMANN = 5.670374419e-8
G = 6.67430e-11
R_SUN = 6.96e8
M_SUN = 1.989e30
EARTH_FLUX = 1361.0
R_EARTH = 6.371e6

# MODEL_DIR = (
#     "/pdo/astronet-data/models/vetting/experimental/pablomer/"
#     "dec2025_cad_scat_v5_duration24/20260226/pablomer-2k-nopretrained-rp/"
#     "AstroCNNModelVetting_pablomer_20260226_141255"
# )

MODEL_DIR = (
    "/pdo/astronet-data/models/vetting/experimental/pablomer/march2026/20260305/pablomer_final-final-3k-ensemble/AstroCNNModelVetting_pablomer_final_20260305_155254/"
)

TFRECORD_GLOB_TEMPLATE = "/pdo/astronet-data/data/tfrecords/sector-{sector}-scatter/*"
PROPERTIES_DIR = "/pdo/astronet-data/data/properties"
OUTPUT_DIR = "/pdo/users/pablomer/Astronet-Triage/rp_filter"
TIC_CONFIG = str(Path.home() / ".config" / "tic" / "db.conf")


def parse_sector_range(spec: str) -> list[int]:
    m = re.fullmatch(r"(\d+)-(\d+)", spec.strip())
    if not m:
        raise ValueError(f"Invalid sector range: {spec}. Expected format like 80-93")
    a, b = int(m.group(1)), int(m.group(2))
    if b < a:
        raise ValueError(f"Invalid sector range: {spec}")
    return list(range(a, b + 1))


def _flatten_numpy(x):
    arr = np.asarray(x)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    return arr.reshape(arr.shape[0], -1)[:, 0]


def find_tce_csv_for_sector(sector: int, properties_dir: str) -> str:
    candidates = [
        f"tces-sector{sector}-with-filenames.csv",
        f"tces-sector{sector}.csv",
        f"tces-sector{sector}-qlp.csv",
    ]
    for name in candidates:
        path = os.path.join(properties_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No TCE CSV found for sector {sector} under {properties_dir}")


def normalize_tce_df(df: pd.DataFrame, sector: int) -> pd.DataFrame:
    tce = df.copy()
    if "Astro ID" not in tce.columns:
        raise ValueError(f"TCE CSV for sector {sector} is missing 'Astro ID' column")

    tce["astro_id"] = pd.to_numeric(tce["Astro ID"], errors="coerce")
    tce = tce.dropna(subset=["astro_id"]).copy()
    tce["astro_id"] = tce["astro_id"].astype(np.int64)
    tce["sector"] = sector
    tce["tic_id_base"] = (tce["astro_id"] // 100).astype(np.int64)
    return tce


def load_tic_conn(conf_path: str, section: str = "tic_82") -> dict:
    cfg = configparser.ConfigParser()
    if not cfg.read(conf_path):
        raise FileNotFoundError(f"Could not read TIC config: {conf_path}")
    if section not in cfg:
        raise KeyError(f"Section {section} not found in {conf_path}")
    sec = cfg[section]
    return {
        "username": sec["username"],
        "password": sec["password"],
        "database": sec["database"],
        "host": sec["host"],
        "port": str(sec.get("port", "5433")),
    }


def query_tic_all_columns(base_tic_ids: list[int], conn: dict) -> pd.DataFrame:
    uniq = sorted(set(int(x) for x in base_tic_ids if pd.notna(x)))
    if not uniq:
        return pd.DataFrame()

    env = os.environ.copy()
    env["PGPASSWORD"] = conn["password"]
    chunk_size = 5000
    frames = []

    for i in range(0, len(uniq), chunk_size):
        chunk = uniq[i : i + chunk_size]
        ids_sql = ",".join(str(x) for x in chunk)
        query = (
            f"COPY (SELECT * FROM ticentries WHERE id IN ({ids_sql}) ORDER BY id) "
            "TO STDOUT WITH CSV HEADER"
        )
        cmd = [
            "/usr/pgsql-14/bin/psql",
            "-h",
            conn["host"],
            "-U",
            conn["username"],
            "-p",
            conn["port"],
            "-d",
            conn["database"],
            "-c",
            query,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"TIC query failed on chunk {(i // chunk_size) + 1}: {proc.stderr.strip()}"
            )
        if proc.stdout.strip():
            frames.append(pd.read_csv(io.StringIO(proc.stdout)))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"], keep="first")


def calculate_planetary_flux_array(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    R_star = df["SRad"].to_numpy(dtype=np.float64) * R_SUN
    T_eff = df["teff"].to_numpy(dtype=np.float64)
    P = df["Per"].to_numpy(dtype=np.float64) * 24 * 3600
    M_star = df["SMass"].to_numpy(dtype=np.float64) * M_SUN

    flux_W = (
        R_star**2
        * STEFAN_BOLTZMANN
        * T_eff**4
        * (4 * np.pi**2 / (P**2 * G * M_star)) ** (2 / 3)
    )
    flux_earth = flux_W / EARTH_FLUX
    return flux_W, flux_earth


def calculate_planet_radius_earth(df: pd.DataFrame) -> np.ndarray:
    R_star = df["SRad"].to_numpy(dtype=np.float64) * R_SUN
    depth_ppm = df["Depth"].to_numpy(dtype=np.float64)
    depth_frac = depth_ppm / 1e6
    rp_m = R_star * np.sqrt(depth_frac)
    return rp_m / R_EARTH


def rp_limit(flux_w: float) -> float:
    if flux_w > 1e5:
        return 13 + (np.log10(flux_w) - 5) * 5
    return 13

def rp_limit_v2(flux_w: float) -> float:
    """Calculate the upper bound on the radius of a planet given the flux."""
    if flux_w > 1e5:
        return min(18 + (np.log10(flux_w) - 5) * 10, 27.5)
    return 18

def run_predictions(model_dir: str, sectors: list[int]) -> tuple[pd.DataFrame, list[int], list[int]]:
    model, config = load_model_from_checkpoint(model_dir, compile_model=False)
    logging.info("Loaded model: %s", model_dir)

    patterns = []
    used, missing = [], []
    for s in sectors:
        pattern = TFRECORD_GLOB_TEMPLATE.format(sector=s)
        files = tf.io.gfile.glob(pattern)
        if files:
            patterns.append(pattern)
            used.append(s)
        else:
            missing.append(s)

    if not patterns:
        raise ValueError("No tfrecords found for requested sectors")

    ds = input_ds.build_eval_dataset(
        file_pattern=patterns,
        input_config=config.inputs,
        batch_size=config.hparams.batch_size,
        include_identifiers=True,
        include_labels=False,
    )

    all_preds = []
    all_ids = []
    for i, batch in enumerate(ds):
        features, astro_ids = batch
        pred_batch = model(features, training=False).numpy()
        all_preds.append(pred_batch)
        all_ids.append(_flatten_numpy(astro_ids.numpy()))
        if (i + 1) % 20 == 0:
            logging.info("Processed %d batches", i + 1)

    preds = np.concatenate(all_preds, axis=0)
    astro_ids = np.concatenate(all_ids, axis=0).astype(np.int64)

    df = pd.DataFrame(
        {
            "astro_id": astro_ids,
            "tic_id_base": astro_ids // 100,
            "disp_p": preds[:, 0],
            "disp_e": preds[:, 1],
            "disp_n": preds[:, 2],
            "disp_j": preds[:, 3],
        }
    )
    return df, used, missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict on live sectors and run radius-vs-flux rp-limit analysis."
    )
    parser.add_argument("--sectors", default="80-93", help="Sector range, e.g. 80-93")
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--properties-dir", default=PROPERTIES_DIR)
    parser.add_argument("--tic-config", default=TIC_CONFIG)
    parser.add_argument(
        "--output-prefix",
        default="live_sector_80_93_flux_radius",
        help="Prefix for output CSV/PNG files",
    )
    parser.add_argument(
        "--reuse-predictions-csv",
        default="",
        help="Optional existing predictions CSV to skip model inference",
    )
    parser.add_argument(
        "--reuse-merged-csv",
        default="",
        help="Optional existing merged CSV to skip model inference, TCE merge, and TIC DB query",
    )
    args = parser.parse_args()

    logging.set_verbosity(logging.INFO)
    os.makedirs(args.output_dir, exist_ok=True)

    sectors = parse_sector_range(args.sectors)
    pred_csv = None
    merged_csv = None

    if args.reuse_merged_csv:
        merged = pd.read_csv(args.reuse_merged_csv)
        if "disp_p" not in merged.columns:
            raise ValueError("reuse merged CSV must include disp_p column")
        if "sector" in merged.columns:
            sector_vals = (
                pd.to_numeric(merged["sector"], errors="coerce").dropna().astype(np.int64).unique().tolist()
            )
            used_sectors = sorted(int(x) for x in sector_vals)
            if not used_sectors:
                used_sectors = sectors
        else:
            used_sectors = sectors
        missing_sectors = []
    else:
        if args.reuse_predictions_csv:
            pred_df = pd.read_csv(args.reuse_predictions_csv)
            if "astro_id" not in pred_df.columns or "disp_p" not in pred_df.columns:
                raise ValueError("reuse CSV must include astro_id and disp_p columns")
            pred_df["astro_id"] = pd.to_numeric(pred_df["astro_id"], errors="coerce").astype("Int64")
            pred_df = pred_df.dropna(subset=["astro_id"]).copy()
            pred_df["astro_id"] = pred_df["astro_id"].astype(np.int64)
            if "tic_id_base" not in pred_df.columns:
                pred_df["tic_id_base"] = pred_df["astro_id"] // 100
            used_sectors = sectors
            missing_sectors = []
        else:
            pred_df, used_sectors, missing_sectors = run_predictions(args.model_dir, sectors)

        pred_csv = os.path.join(args.output_dir, f"{args.output_prefix}_predictions.csv")
        pred_df.to_csv(pred_csv, index=False)

        tce_frames = []
        for s in used_sectors:
            path = find_tce_csv_for_sector(s, args.properties_dir)
            tce_df = pd.read_csv(path)
            tce_df = normalize_tce_df(tce_df, s)
            tce_df["tce_source_csv"] = path
            tce_frames.append(tce_df)

        if not tce_frames:
            raise ValueError("No TCE tables loaded")

        tce_all = pd.concat(tce_frames, ignore_index=True)

        merged = pred_df.merge(tce_all, on=["astro_id", "tic_id_base"], how="left")

        conn = load_tic_conn(args.tic_config, section="tic_82")
        tic_df = query_tic_all_columns(merged["tic_id_base"].dropna().astype(int).tolist(), conn)
        merged = merged.merge(
            tic_df, left_on="tic_id_base", right_on="id", how="left", suffixes=("", "_tic")
        )

        merged_csv = os.path.join(args.output_dir, f"{args.output_prefix}_merged.csv")
        merged.to_csv(merged_csv, index=False)

    # Radius/flux analysis (same core logic as planet_radius/radius_plot.py)
    planet_df = merged[pd.to_numeric(merged["disp_p"], errors="coerce") >= args.threshold].copy()

    for col in ["SRad", "teff", "Per", "SMass", "Depth"]:
        if col not in planet_df.columns:
            planet_df[col] = np.nan
        planet_df[col] = pd.to_numeric(planet_df[col], errors="coerce")

    # Keep rows missing SRad/SMass for fallback radius-only plotting.
    missing_srad_or_smass = planet_df["SRad"].isna() | planet_df["SMass"].isna()
    fallback_candidates = planet_df[missing_srad_or_smass].copy()

    analysis_df = planet_df.dropna(subset=["SRad", "teff", "Per", "SMass", "Depth"]).copy()
    if analysis_df.empty:
        raise ValueError("No predicted planets with complete SRad/teff/Per/SMass/Depth for flux-radius analysis")

    flux_w, flux_earth = calculate_planetary_flux_array(analysis_df)
    rp_earth = calculate_planet_radius_earth(analysis_df)

    analysis_df["flux_calculated_W"] = flux_w
    analysis_df["flux_calculated_earthflux"] = flux_earth
    analysis_df["planet_radius_earth"] = rp_earth
    analysis_df["rp_limit_earth"] = np.array([rp_limit_v2(f) for f in flux_w])
    analysis_df["excluded_by_rp_limit"] = analysis_df["planet_radius_earth"] > analysis_df["rp_limit_earth"]

    # Fallback radius estimate for rows missing SRad or SMass:
    # r_p = s_rad * sqrt(depth / 1e6) * 109.076 ; default s_rad=1 when unavailable.
    if "depth" in fallback_candidates.columns:
        depth_fallback = pd.to_numeric(fallback_candidates["depth"], errors="coerce")
    else:
        depth_fallback = pd.to_numeric(fallback_candidates["Depth"], errors="coerce")
    if "s_rad" in fallback_candidates.columns:
        srad_fallback = pd.to_numeric(fallback_candidates["s_rad"], errors="coerce")
    else:
        srad_fallback = pd.Series(np.nan, index=fallback_candidates.index, dtype=np.float64)
    srad_fallback = srad_fallback.fillna(1.0)
    fallback_candidates["planet_radius_earth"] = (
        srad_fallback * np.sqrt(depth_fallback / (10**6)) * 109.076
    )
    fallback_plot_df = fallback_candidates.dropna(subset=["planet_radius_earth"]).copy()
    fallback_plot_df["excluded_by_rp_limit"] = (
        fallback_plot_df["planet_radius_earth"].to_numpy(dtype=np.float64) >= 27.5
    )

    n_total = len(analysis_df)
    n_excluded_regular = int(analysis_df["excluded_by_rp_limit"].sum())
    n_fallback = len(fallback_plot_df)
    n_excluded_fallback = int(fallback_plot_df["excluded_by_rp_limit"].sum())
    n_total_including_fallback = n_total + n_fallback
    n_excluded_total = n_excluded_regular + n_excluded_fallback
    pct_excluded_total = (
        100.0 * n_excluded_total / n_total_including_fallback
        if n_total_including_fallback
        else 0.0
    )

    analysis_csv = os.path.join(args.output_dir, f"{args.output_prefix}_analysis.csv")
    analysis_df.to_csv(analysis_csv, index=False)

    flux_linspace = np.logspace(np.log10(flux_w.min()), np.log10(flux_w.max()), 200)
    rp_lim = np.array([rp_limit_v2(f) for f in flux_linspace])

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(
        flux_w,
        rp_earth,
        alpha=0.6,
        s=40,
        color="seagreen",
        edgecolors="darkgreen",
        linewidth=0.5,
        label=f"Predicted planets (disp_p >= {args.threshold}) [N={n_total}]",
        zorder=3,
    )
    # Plot fallback radius-only points at a fixed x-position on the left side.
    if not fallback_plot_df.empty:
        # fallback_x = np.full(len(fallback_plot_df), 1e6)
        center_x = 1e6
        jitter = np.random.normal(loc=0, scale=50000, size=len(fallback_plot_df))
        fallback_x_jittered = center_x + jitter


        ax.scatter(
            fallback_x_jittered,
            fallback_plot_df["planet_radius_earth"].to_numpy(dtype=np.float64),
            alpha=0.8,
            s=36,
            color="royalblue",
            edgecolors="navy",
            linewidth=0.5,
            label=f"Fallback radius (missing SRad/SMass) [N={len(fallback_plot_df)}]",
            zorder=5,
        )
    ax.plot(
        flux_linspace,
        rp_lim,
        color="crimson",
        linewidth=3,
        label="rp limit",
        zorder=4,
    )
    ax.fill_between(
        flux_linspace,
        rp_lim,
        40,
        alpha=0.15,
        color="red",
        label=f"excluded area ({pct_excluded_total:.1f}% planets)",
        zorder=2,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Insolation Flux (W/m²)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Planet Radius (Earth Radii)", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Live sectors {used_sectors}: Radius vs Flux (threshold={args.threshold})",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.7, zorder=1)
    ax.legend(fontsize=11, loc="upper left", framealpha=0.95, edgecolor="black")
    # ax.set_ylim(0, 30)
    # ax.set_xlim(flux_w.min() * 0.8, flux_w.max() * 1.2)
    ax.set_ylim(0, 40)
    ax.set_xlim(flux_w.min() * 0.8, flux_w.max() * 1.2)
    ax.tick_params(labelsize=11)

    plot_png = os.path.join(args.output_dir, f"{args.output_prefix}.png")
    plt.tight_layout()
    plt.savefig(plot_png, dpi=300, bbox_inches="tight")

    if pred_csv is not None:
        print(f"Saved predictions CSV: {pred_csv}")
    else:
        print(f"Reused merged CSV: {args.reuse_merged_csv}")
    if merged_csv is not None:
        print(f"Saved merged CSV: {merged_csv}")
    print(f"Saved analysis CSV: {analysis_csv}")
    print(f"Saved plot: {plot_png}")
    print(f"Used sectors: {used_sectors}")
    print(f"Missing sectors (no scatter tfrecords): {missing_sectors}")
    print(f"Predicted planets at threshold {args.threshold}: {len(planet_df)}")
    print(f"With full analysis columns: {n_total}")
    print(f"Fallback plotted (missing SRad/SMass): {n_fallback}")
    print(f"Fallback excluded by rp>=27.5: {n_excluded_fallback}")
    print(f"Excluded by rp limit (regular): {n_excluded_regular}")
    print(f"Excluded by rp limit (including fallback): {n_excluded_total} ({pct_excluded_total:.1f}%)")


if __name__ == "__main__":
    main()
