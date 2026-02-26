#!/usr/bin/env python3
"""Plot 1-vs-1 precision-recall curves for ablation vs baseline models."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


def parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--ablation_csv", required=True)
  parser.add_argument("--baseline_csv", required=True)
  parser.add_argument("--target_class", default="p", choices=["p", "e", "n", "j"])
  parser.add_argument("--output_png", required=True)
  parser.add_argument("--output_csv", default=None)
  parser.add_argument("--title", default=None)
  parser.add_argument("--x_min", type=float, default=0.0)
  parser.add_argument("--x_max", type=float, default=1.0)
  parser.add_argument("--y_min", type=float, default=0.0)
  parser.add_argument("--y_max", type=float, default=1.02)
  parser.add_argument("--show_prevalence", action="store_true")
  return parser.parse_args()


def load_labels_and_scores(csv_path, target_class):
  df = pd.read_csv(csv_path)
  score_col = f"disp_{target_class}"
  if score_col not in df.columns:
    raise ValueError(f"Column '{score_col}' not found in {csv_path}")
  if "true_label" not in df.columns:
    raise ValueError(f"Column 'true_label' not found in {csv_path}")

  y_true = (df["true_label"].astype(str).str.lower() == target_class.lower()).astype(int).to_numpy()
  y_score = df[score_col].to_numpy(dtype=float)
  return y_true, y_score


def main():
  args = parse_args()
  os.makedirs(os.path.dirname(args.output_png), exist_ok=True)
  if args.output_csv:
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

  y_true_ab, score_ab = load_labels_and_scores(args.ablation_csv, args.target_class)
  y_true_bl, score_bl = load_labels_and_scores(args.baseline_csv, args.target_class)

  if len(y_true_ab) != len(y_true_bl):
    raise ValueError(
        "Ablation and baseline rows differ; expected aligned evaluation sets. "
        f"Got {len(y_true_ab)} vs {len(y_true_bl)}.")
  if not np.array_equal(y_true_ab, y_true_bl):
    mismatch = int(np.sum(y_true_ab != y_true_bl))
    raise ValueError(f"Label mismatch between ablation/baseline rows: {mismatch} rows differ")

  y_true = y_true_ab
  prevalence = float(np.mean(y_true))

  prec_ab, rec_ab, _ = precision_recall_curve(y_true, score_ab)
  prec_bl, rec_bl, _ = precision_recall_curve(y_true, score_bl)
  ap_ab = float(average_precision_score(y_true, score_ab))
  ap_bl = float(average_precision_score(y_true, score_bl))

  fig, ax = plt.subplots(figsize=(10.5, 8))
  ax.plot(rec_bl, prec_bl, color="#264653", lw=3.0, label=f"Baseline (AP={ap_bl:.4f})")
  ax.plot(rec_ab, prec_ab, color="#e76f51", lw=3.0, linestyle="--", label=f"Ablation G+L (AP={ap_ab:.4f})")
  if args.show_prevalence:
    ax.axhline(prevalence, color="#8d99ae", lw=1.8, linestyle=":", label=f"Prevalence={prevalence:.4f}")

  title = args.title
  if not title:
    title = f"PR Curve Comparison for class '{args.target_class.upper()}'"
  ax.set_title(title, fontsize=17)
  ax.set_xlabel("Recall", fontsize=13)
  ax.set_ylabel("Precision", fontsize=13)
  ax.set_xlim(args.x_min, args.x_max)
  ax.set_ylim(args.y_min, args.y_max)
  ax.grid(alpha=0.3, linewidth=0.8, linestyle="-")
  ax.minorticks_on()
  ax.grid(which="minor", alpha=0.18, linewidth=0.5, linestyle=":")
  ax.legend(loc="lower left", frameon=False, fontsize=11)
  plt.tight_layout()
  fig.savefig(args.output_png, dpi=220)
  plt.close(fig)

  if args.output_csv:
    out = pd.DataFrame({
        "curve": (["baseline"] * len(prec_bl)) + (["ablation_global_local"] * len(prec_ab)),
        "recall": np.concatenate([rec_bl, rec_ab]),
        "precision": np.concatenate([prec_bl, prec_ab]),
    })
    out.to_csv(args.output_csv, index=False)

  print(f"[OK] Wrote PR figure: {args.output_png}")
  if args.output_csv:
    print(f"[OK] Wrote PR points: {args.output_csv}")
  print(f"[INFO] AP baseline={ap_bl:.6f} | AP ablation={ap_ab:.6f} | prevalence={prevalence:.6f}")


if __name__ == "__main__":
  main()
