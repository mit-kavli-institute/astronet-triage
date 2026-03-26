# similarity_search.py
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .light_curve_metrics import deficit_from_flux, find_transit_window


def _resample_1d(x: np.ndarray, m: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return np.zeros(m, dtype=np.float32)
    if x.size == 1:
        return np.full(m, x[0], dtype=np.float32)
    xp = np.linspace(0.0, 1.0, x.size, dtype=np.float32)
    xq = np.linspace(0.0, 1.0, m, dtype=np.float32)
    return np.interp(xq, xp, x).astype(np.float32)


def tail_signature(
    flux: np.ndarray,
    sig_frac: float = 0.03,
    sig_cap_hw: int = 300,
    sig_m: int = 160,
    min_hw: int = 12,
    fallback_hw: int = 60,
) -> Optional[np.ndarray]:
    """
    Fixed-length signature capturing left-vs-right deficit structure:
      signature = resample(left_deficit, m) - resample(right_deficit, m)

    Windowing:
      - primary: threshold window using peak*sig_frac
      - fallback: fixed half-width around deepest deficit if window too narrow

    Returns None if no usable transit.
    """
    v = np.asarray(flux, dtype=np.float32)
    g = deficit_from_flux(v)
    n = g.size
    c = int(np.argmax(g))
    peak = float(g[c])
    if not np.isfinite(peak) or peak <= 0:
        return None

    c, hwL, hwR, hw = find_transit_window(v, frac=sig_frac, cap_hw=sig_cap_hw)

    if hw < min_hw:
        hw = int(min(fallback_hw, c, n - c - 1))

    if hw < min_hw:
        return None

    left = g[c - hw:c][::-1]
    right = g[c:c + hw]

    left_r = _resample_1d(left, sig_m)
    right_r = _resample_1d(right, sig_m)
    return (left_r - right_r).astype(np.float32)


def make_weights(m: int) -> np.ndarray:
    """
    Weighted emphasis near center (ingress) + mid region (tail).
    """
    x = np.linspace(0, 1, m, dtype=np.float32)
    w = np.ones_like(x)
    w += 4.0 * np.exp(-0.5 * ((x - 0.08) / 0.06) ** 2)
    w += 2.0 * np.exp(-0.5 * ((x - 0.45) / 0.18) ** 2)
    return w


def weighted_l2(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.sum(w * d * d)))


def _plot_view(view: np.ndarray, astro_id: int) -> None:
    phase = np.linspace(0, 1, len(view), endpoint=False)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(phase, view, marker=".", linestyle="-")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Normalized flux")
    ax.set_title(f"Astro ID {astro_id}")
    ax.grid(True)
    fig.tight_layout()
    plt.show()


def similarity_search(
    id_to_view: Dict[int, np.ndarray],
    df: pd.DataFrame,
    target_astro_id: int,
    topn: int = 30,
    # signature parameters (windowing)
    sig_frac: float = 0.03,
    sig_cap_hw: int = 300,
    sig_m: int = 160,
    sig_min_hw: int = 12,
    sig_fallback_hw: int = 60,
    # constraints
    min_slope_ratio: float = 1.2,
    min_recovery_ratio: float = 1.2,
    min_tau: float = 20.0,
    min_wrwl: float = 1.1,
    # optional speedup
    prefilter_k: Optional[int] = 200,
    # optional extra quality gates (recommended)
    min_egress_monotonic: Optional[float] = None,    # e.g. 0.75
    max_egress_smoothness: Optional[float] = None,   # e.g. 0.08
    min_ingress_impulse: Optional[float] = None,     # e.g. 0.35
) -> pd.DataFrame:
    """
    Returns a DataFrame of top-N most similar candidates to target_astro_id.

    Expects `df` to contain columns from light_curve_metrics.export_all_metrics()
    AND any additional filters you joined in (e.g. snr, disp_j).
    Must include: astro_id, valid, slope_ratio_ingress_egress,
                  recovery_time_ratio_20pct, tau_egress_exp, wr_wl_halfdepth
    """
    if "astro_id" not in df.columns:
        raise ValueError("df must have an 'astro_id' column")

    if target_astro_id not in id_to_view:
        raise KeyError(f"target_astro_id {target_astro_id} not in id_to_view")

    # build target signature
    q_sig = tail_signature(
        id_to_view[target_astro_id],
        sig_frac=sig_frac,
        sig_cap_hw=sig_cap_hw,
        sig_m=sig_m,
        min_hw=sig_min_hw,
        fallback_hw=sig_fallback_hw,
    )
    if q_sig is None:
        raise ValueError(
            "Target signature is None. Try lowering sig_frac (e.g. 0.02), "
            "or increase sig_fallback_hw."
        )

    w = make_weights(sig_m)

    # metric-filter pool
    pool = df[df["valid"] == True].copy()
    pool = pool[pool["astro_id"] != target_astro_id]

    pool = pool[pool["slope_ratio_ingress_egress"] >= float(min_slope_ratio)]
    pool = pool[pool["recovery_time_ratio_20pct"] >= float(min_recovery_ratio)]
    pool = pool[np.isfinite(pool["tau_egress_exp"]) & (pool["tau_egress_exp"] >= float(min_tau))]
    pool = pool[np.isfinite(pool["wr_wl_halfdepth"]) & (pool["wr_wl_halfdepth"] >= float(min_wrwl))]

    # optional extra gates (helps reject wiggly imposters)
    if min_egress_monotonic is not None and "egress_monotonic_frac" in pool.columns:
        pool = pool[pool["egress_monotonic_frac"] >= float(min_egress_monotonic)]
    if max_egress_smoothness is not None and "egress_smoothness" in pool.columns:
        pool = pool[pool["egress_smoothness"] <= float(max_egress_smoothness)]
    if min_ingress_impulse is not None and "ingress_impulse" in pool.columns:
        pool = pool[pool["ingress_impulse"] >= float(min_ingress_impulse)]

    # prefilter (speed): choose strongest tail-ish subset before full distance
    if prefilter_k is not None and len(pool) > int(prefilter_k):
        # cheap score: favor longer tau + sharper ingress + longer recovery
        score = (
            0.8 * pool["tau_egress_exp"].fillna(0) +
            1.0 * pool["slope_ratio_ingress_egress"].fillna(0) +
            1.0 * pool["recovery_time_ratio_20pct"].fillna(0) +
            0.2 * pool["wr_wl_halfdepth"].fillna(0)
        )
        pool = pool.loc[score.sort_values(ascending=False).head(int(prefilter_k)).index]

    rows = []
    for _, r in pool.iterrows():
        aid = int(r["astro_id"])
        view = id_to_view.get(aid, None)
        if view is None:
            continue

        sig = tail_signature(
            view,
            sig_frac=sig_frac,
            sig_cap_hw=sig_cap_hw,
            sig_m=sig_m,
            min_hw=sig_min_hw,
            fallback_hw=sig_fallback_hw,
        )
        if sig is None:
            continue

        d = weighted_l2(sig, q_sig, w)
        rr = r.to_dict()
        rr["distance"] = d
        rows.append(rr)

    res = pd.DataFrame(rows)
    if res.empty:
        return res

    res = res.sort_values("distance", ascending=True).head(topn)

    # Put key cols first
    front = [
        "astro_id", "distance",
        "tau_egress_exp",
        "slope_ratio_ingress_egress",
        "recovery_time_ratio_20pct",
        "wr_wl_halfdepth",
        "egress_monotonic_frac",
        "egress_smoothness",
        "ingress_impulse",
        "peak_deficit",
        "hw_left", "hw_right", "hw_min",
    ]
    cols = [c for c in front if c in res.columns] + [c for c in res.columns if c not in front]
    return res[cols]


def show_top_similar(
    id_to_view: Dict[int, np.ndarray],
    df: pd.DataFrame,
    target_astro_id: int,
    topn: int = 30,
    plot_n: int = 8,
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience wrapper that runs similarity_search() and plots results.
    """
    res = similarity_search(
        id_to_view=id_to_view,
        df=df,
        target_astro_id=target_astro_id,
        topn=topn,
        **kwargs,
    )

    if res.empty:
        print("No results. Relax constraints (min_*), lower sig_frac, or increase prefilter_k.")
        return res

    print(res[[
        "astro_id", "distance",
        "tau_egress_exp",
        "slope_ratio_ingress_egress",
        "recovery_time_ratio_20pct",
        "wr_wl_halfdepth",
    ]].head(min(len(res), 15)))

    _plot_view(id_to_view[target_astro_id], target_astro_id)
    for aid in res["astro_id"].head(plot_n).tolist():
        _plot_view(id_to_view[int(aid)], int(aid))

    return res
