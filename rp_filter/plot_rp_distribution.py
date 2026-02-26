#!/usr/bin/env python3
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from absl import app

INPUT_CSV = (
    "/pdo/astronet-data/models/vetting/experimental/pablomer/"
    "dec2025_cad_scat_v5_duration24/20260226/"
    "pablomer-2k-nopretrained-rp/all_preds_with_rp.csv"
)
THRESHOLD = 0.9
OUTPUT_PNG = (
    "/pdo/users/pablomer/Astronet-Triage/rp_filter/"
    "rp_distribution_20260226_pablomer_threshold_0p9.png"
)
OUTPUT_PNG_TRUE_LABEL_STACKED = (
    "/pdo/users/pablomer/Astronet-Triage/rp_filter/"
    "rp_distribution_20260226_pablomer_threshold_0p9_true_label_stacked.png"
)
HARD_RP_LIMIT = 30.0
MAX_RP_FOR_PLOT = 100.0


def _find_planet_score_column(df):
    if "disp_p_mean" in df.columns:
        return "disp_p_mean"
    disp_p_cols = sorted([c for c in df.columns if c.startswith("disp_p_")])
    if disp_p_cols:
        df["disp_p_mean"] = df[disp_p_cols].mean(axis=1)
        return "disp_p_mean"
    if "disp_p" in df.columns:
        return "disp_p"
    raise ValueError("Could not find planet score column (disp_p, disp_p_mean, or disp_p_*)")


def main(argv):
    df = pd.read_csv(INPUT_CSV)

    if "r_p" not in df.columns:
        raise ValueError("input_csv is missing r_p column")

    score_col = _find_planet_score_column(df)
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df["r_p"] = pd.to_numeric(df["r_p"], errors="coerce")
    df = df.dropna(subset=[score_col, "r_p"])
    df = df[df["r_p"] <= 100]

    predicted_planet_all = df[df[score_col] >= THRESHOLD]["r_p"].values
    predicted_non_planet_all = df[df[score_col] < THRESHOLD]["r_p"].values

    # Count before applying any upper clipping used for plotting.
    n_predicted_planet_above_hard_limit = int(np.sum(predicted_planet_all > HARD_RP_LIMIT))

    predicted_planet = predicted_planet_all[predicted_planet_all <= MAX_RP_FOR_PLOT]
    predicted_non_planet = predicted_non_planet_all[predicted_non_planet_all <= MAX_RP_FOR_PLOT]

    plt.figure(figsize=(10, 6))
    bins_max = min(np.nanpercentile(df["r_p"], 99), MAX_RP_FOR_PLOT)
    bins = np.linspace(0, bins_max, 50)
    plt.hist(predicted_planet, bins=bins, alpha=0.6, label=f"Predicted planet (>= {THRESHOLD})", color="#2ca02c")
    plt.hist(predicted_non_planet, bins=bins, alpha=0.6, label=f"Predicted non-planet (< {THRESHOLD})", color="#d62728")
    plt.axvline(HARD_RP_LIMIT, color="black", linestyle="--", linewidth=1.5, label="hard rp limit")
    ymax = plt.ylim()[1]
    plt.text(HARD_RP_LIMIT + 1.0, ymax * 0.95, "hard rp limit", rotation=90, va="top", ha="left")
    plt.xlabel("r_p")
    plt.ylabel("Count")
    plt.title(f"Distribution of r_p by predicted class with threshold P={THRESHOLD}")
    plt.text(
        0.98,
        0.8,
        f"Pred. planets wirth rp > {HARD_RP_LIMIT:g}: {n_predicted_planet_above_hard_limit}",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=220)
    print(f"Saved r_p distribution plot: {OUTPUT_PNG}")
    print(f"N(predicted planet)={len(predicted_planet)}")
    print(f"N(predicted non-planet)={len(predicted_non_planet)}")
    print(
        f"N(predicted planet with r_p > {HARD_RP_LIMIT:g}) before r_p>{MAX_RP_FOR_PLOT:g} drop="
        f"{n_predicted_planet_above_hard_limit}"
    )

    # Second plot: stacked histograms by true label for each predicted group.
    if "true_label" in df.columns:
        df_plot = df[df["r_p"] <= MAX_RP_FOR_PLOT].copy()
        df_planet = df_plot[df_plot[score_col] >= THRESHOLD]
        df_non_planet = df_plot[df_plot[score_col] < THRESHOLD]
        true_label_order = ["p", "e", "j"]
        true_label_colors = {
            "p": "#1f77b4",
            "e": "#ff7f0e",
            "j": "#d62728",
        }

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for ax, subdf, title in [
            (axes[0], df_planet, f"Predicted planet (>= {THRESHOLD})"),
            (axes[1], df_non_planet, f"Predicted non-planet (< {THRESHOLD})"),
        ]:
            series_list = [
                pd.to_numeric(subdf[subdf["true_label"] == t]["r_p"], errors="coerce").dropna().values
                for t in true_label_order
            ]
            labels = [f"true_{t}" for t in true_label_order]
            colors = [true_label_colors[t] for t in true_label_order]
            ax.hist(series_list, bins=bins, stacked=True, label=labels, color=colors, alpha=0.85)
            ax.axvline(HARD_RP_LIMIT, color="black", linestyle="--", linewidth=1.2)
            ax.set_title(title)
            ax.set_xlabel("r_p")
            ax.grid(alpha=0.2, linestyle=":")

        axes[0].set_ylabel("Count")
        handles, labels = axes[1].get_legend_handles_labels()
        fig.suptitle("r_p Distribution by Predicted Group and True Label", y=0.99)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=3,
            frameon=True,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.86])
        os.makedirs(os.path.dirname(OUTPUT_PNG_TRUE_LABEL_STACKED), exist_ok=True)
        fig.savefig(OUTPUT_PNG_TRUE_LABEL_STACKED, dpi=220, bbox_inches="tight")
        print(f"Saved stacked true-label r_p plot: {OUTPUT_PNG_TRUE_LABEL_STACKED}")
    else:
        print("Skipping stacked true-label plot: true_label column not found.")


if __name__ == "__main__":
    app.run(main)
