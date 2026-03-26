# light_curve_metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd


def deficit_from_flux(flux: np.ndarray) -> np.ndarray:
    """
    Deficit for canonicalized flux: baseline ~0, transit negative.
    deficit = max(0, -flux)
    """
    v = np.asarray(flux, dtype=np.float32)
    return np.maximum(0.0, -v)


def find_transit_window(
    flux: np.ndarray,
    frac: float = 0.08,
    cap_hw: int = 300,
) -> tuple[int, int, int, int]:
    """
    Finds a window around the deepest deficit point using thresholding:
      g >= peak * frac

    Returns:
      (center_idx, hw_left, hw_right, hw_min)
    """
    g = deficit_from_flux(flux)
    n = g.size
    c = int(np.argmax(g))
    peak = float(g[c])

    if not np.isfinite(peak) or peak <= 0:
        return c, 0, 0, 0

    thr = peak * float(frac)

    L = c
    steps = 0
    while L > 0 and steps < cap_hw and g[L] >= thr:
        L -= 1
        steps += 1

    R = c
    steps = 0
    while R < n - 1 and steps < cap_hw and g[R] >= thr:
        R += 1
        steps += 1

    hwL = c - L
    hwR = R - c
    hw = int(min(hwL, hwR))
    return c, int(hwL), int(hwR), hw


def _half_slices_deficit(flux: np.ndarray, c: int, hw: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (left, right) deficit arrays:
      left:  center outward on left side (reversed)
      right: center outward on right side
    """
    g = deficit_from_flux(flux)
    left = g[c - hw:c][::-1]
    right = g[c:c + hw]
    return left.astype(np.float32), right.astype(np.float32)


def _t_recover(x: np.ndarray, thr: float, hw: int) -> int:
    idx = np.where(x <= thr)[0]
    return int(idx[0]) if len(idx) else int(hw)


def extract_tail_metrics(
    flux: np.ndarray,
    window_frac: float = 0.08,
    cap_hw: int = 300,
    halfdepth_frac: float = 0.5,
    recovery_level: float = 0.2,
    eps: float = 1e-4,
    min_hw: int = 15,
) -> Dict[str, Any]:
    """
    Computes dust-tail-inspired metrics from a canonicalized 1D flux vector.

    Notes:
      - Baseline assumed ~0; transit dips negative.
      - Uses deficit g=max(0,-flux).
      - Window is defined by g >= peak * window_frac.

    Returns dict with:
      basic: center_idx, peak_deficit, hw_left/right/min, valid
      asymmetry: wr_wl_halfdepth, slope_ratio_ingress_egress,
                 recovery_time_ratio_20pct, asym_area
      tail model: tau_egress_exp, n_points_tail_fit
      tail specificity: egress_monotonic_frac, egress_smoothness, ingress_impulse
    """
    v = np.asarray(flux, dtype=np.float32)
    g = deficit_from_flux(v)
    n = g.size

    c = int(np.argmax(g))
    peak = float(g[c])

    metrics: Dict[str, Any] = {
        "center_idx": c,
        "peak_deficit": peak,
        "valid": False,
    }

    if not np.isfinite(peak) or peak <= 0:
        return metrics

    c, hwL, hwR, hw = find_transit_window(v, frac=window_frac, cap_hw=cap_hw)
    metrics.update({"hw_left": hwL, "hw_right": hwR, "hw_min": hw})

    if hw < min_hw:
        return metrics

    left, right = _half_slices_deficit(v, c, hw)

    # --- half-depth width ratio (right wider => tail-like) ---
    thr_hd = peak * float(halfdepth_frac)
    WL = np.where(left >= thr_hd)[0]
    WR = np.where(right >= thr_hd)[0]
    wl = (int(WL[-1]) + 1) if len(WL) else np.nan
    wr = (int(WR[-1]) + 1) if len(WR) else np.nan
    metrics["wr_wl_halfdepth"] = float(wr / wl) if (isinstance(wl, (int, np.integer)) and wl > 0) else np.nan

    # --- ingress/egress slope ratio (sharp ingress) ---
    k = max(6, hw // 30)
    k = min(k, hw)
    # steepness: deficit near center minus deficit a bit outward
    s_in = float((left[0] - left[k - 1]) / max(k - 1, 1))
    s_out = float((right[0] - right[k - 1]) / max(k - 1, 1))
    metrics["slope_ratio_ingress_egress"] = float(s_in / (s_out + 1e-6))

    # --- recovery ratio at 20% of peak ---
    thr_rec = peak * float(recovery_level)
    tL = _t_recover(left, thr_rec, hw)
    tR = _t_recover(right, thr_rec, hw)
    metrics["recovery_time_ratio_20pct"] = float(tR / max(tL, 1))

    # --- asymmetry area (left-right deficit difference) ---
    metrics["asym_area"] = float(np.trapz(left - right))

    # --- exponential tail fit on egress: right(t) ~ A exp(-t/tau) ---
    m = right > float(eps)
    metrics["n_points_tail_fit"] = int(np.sum(m))
    if np.sum(m) >= 10:
        t = np.arange(hw, dtype=np.float32)[m]
        y = np.log(right[m])
        slope = float(np.polyfit(t, y, 1)[0])  # y = a + slope*t
        metrics["tau_egress_exp"] = float(-1.0 / slope) if slope < 0 else np.nan
    else:
        metrics["tau_egress_exp"] = np.nan

    # -----------------------------
    # Extra specificity metrics (helps reject wiggly imposters)
    # -----------------------------

    # Egress monotonicity: fraction of steps that decrease (should be high)
    dr = np.diff(right)
    metrics["egress_monotonic_frac"] = float(np.mean(dr <= 0.0)) if dr.size else np.nan

    # Egress smoothness: normalized 2nd-derivative energy (should be low)
    d2 = np.diff(right, 2)
    denom = float(np.sum(right * right) + 1e-6)
    metrics["egress_smoothness"] = float(np.sum(d2 * d2) / denom) if d2.size else np.nan

    # Ingress impulsiveness: fraction of deficit "mass" in the first 20% (should be high)
    kk = max(3, int(0.2 * hw))
    denom2 = float(np.sum(left) + 1e-6)
    metrics["ingress_impulse"] = float(np.sum(left[:kk]) / denom2)

    metrics["valid"] = True
    return metrics


def export_all_metrics(
    id_to_view: Dict[int, np.ndarray],
    window_frac: float = 0.08,
    cap_hw: int = 300,
    min_hw: int = 15,
) -> pd.DataFrame:
    """
    Computes metrics for all astro IDs in id_to_view and returns a DataFrame.
    """
    rows = []
    for astro_id, flux in id_to_view.items():
        m = extract_tail_metrics(
            flux,
            window_frac=window_frac,
            cap_hw=cap_hw,
            min_hw=min_hw,
        )
        m["astro_id"] = astro_id
        rows.append(m)
    return pd.DataFrame(rows)
