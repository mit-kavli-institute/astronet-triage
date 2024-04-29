from typing import Optional

import numpy as np
import numpy.typing as npt

from ..preprocess import preprocess


def global_features(
    tic: int,
    time: np.ndarray,
    flux: np.ndarray,
    transit_mask: npt.NDArray[np.bool_],
    period: float,
):
    view, std, mask, _, _ = preprocess.global_view(tic, time, flux, period)
    transit_mask, _, _, _, _ = preprocess.tr_mask_view(tic, time, transit_mask, period)
    return {
        "global_view": view,
        "global_std": std,
        "global_mask": mask,
        "global_transit_mask": transit_mask,
    }


def local_features(
    tic: int,
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    duration: float,
):
    view, std, mask, scale, depth = preprocess.local_view(
        tic, time, flux, period, duration
    )
    return (
        {
            "local_view": view,
            "local_std": std,
            "local_mask": mask,
            "local_scale": np.array([scale]) if scale is not None else np.array([0.0]),
            "local_scale_present": np.array([scale is not None]).astype(float),
        },
        scale,
        depth,
    )


def aperture_features(
    aperture_name: str,
    tic: int,
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    duration: float,
    scale: Optional[float],
    depth: Optional[float],
):
    view, _, _, _, _ = preprocess.local_view(
        tic, time, flux, period, duration, scale=scale, depth=depth
    )
    return {f"local_aperture_{aperture_name}": view}


def odd_features(
    odd_mask: npt.NDArray[np.bool_],
    tic: int,
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    duration: float,
    scale: Optional[float],
    depth: Optional[float],
):
    view, std, mask, _, _ = preprocess.local_view(
        tic, time[odd_mask], flux[odd_mask], period, duration, scale=scale, depth=depth
    )
    return {
        "local_view_odd": view,
        "local_std_odd": std,
        "local_mask_odd": mask,
    }


def even_features(
    even_mask: npt.NDArray[np.bool_],
    tic: int,
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    duration: float,
    scale: Optional[float],
    depth: Optional[float],
):
    view, std, mask, _, _ = preprocess.local_view(
        tic,
        time[even_mask],
        flux[even_mask],
        period,
        duration,
        scale=scale,
        depth=depth,
    )
    return {
        "local_view_even": view,
        "local_std_even": std,
        "local_mask_even": mask,
    }


def secondary_features(
    tic: int,
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    duration: float,
    scale: Optional[float],
    depth: Optional[float],
):
    (_, _, _, secondary_scale, _), _ = preprocess.secondary_view(
        tic, time, flux, period, duration
    )
    (view, std, mask, scale, _), t0 = preprocess.secondary_view(
        tic, time, flux, period, duration, scale=scale, depth=depth
    )
    return (
        {
            "secondary_view": view,
            "secondary_std": std,
            "secondary_mask": mask,
            "secondary_phase": np.array([t0 / period]),
            "secondary_scale": np.array([secondary_scale])
            if secondary_scale is not None
            else np.array([0.0]),
            "secondary_scale_present": np.array([secondary_scale is not None]).astype(
                float
            ),
        },
        secondary_scale,
    )


def sample_segments_features(
    tic: int,
    time: np.ndarray,
    flux: np.ndarray,
    fold_num: np.ndarray,
    odd_mask: npt.NDArray[np.bool_],
    even_mask: npt.NDArray[np.bool_],
    period: float,
    duration: float,
):
    view = preprocess.sample_segments_view(tic, time, flux, fold_num, period, duration)
    odd_view = preprocess.sample_segments_view(
        tic,
        time[odd_mask],
        flux[odd_mask],
        fold_num[odd_mask],
        period,
        duration,
        num_bins=61,
        num_transits=4,
        local=True,
    )
    even_view = preprocess.sample_segments_view(
        tic,
        time[even_mask],
        flux[even_mask],
        fold_num[even_mask],
        period,
        duration,
        num_bins=61,
        num_transits=4,
        local=True,
    )
    local_view = np.concatenate([odd_view, even_view], axis=-1)
    return {
        "sample_segments_view": view,
        "sample_segments_local_view": local_view,
    }


def double_period_features(
    tic: int,
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
):
    view, std, mask, _, _ = preprocess.global_view(tic, time, flux, period * 2)
    return {
        "global_view_double_period": view,
        "global_view_double_period_std": std,
        "global_view_double_period_mask": mask,
    }


def half_period_features(
    tic: int,
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    duration: float,
):
    global_view, global_std, global_mask, _, _ = preprocess.global_view(
        tic, time, flux, period / 2
    )
    local_view, local_std, local_mask, _, _ = preprocess.local_view(
        tic, time, flux, period / 2, duration
    )
    return {
        "global_view_half_period": global_view,
        "global_view_half_period_std": global_std,
        "global_view_half_period_mask": global_mask,
        "local_view_half_period": local_view,
        "local_view_half_period_std": local_std,
        "local_view_half_period_mask": local_mask,
    }
