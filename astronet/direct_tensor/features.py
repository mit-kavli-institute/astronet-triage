"""
Functions for assembling various astronet input views.

See section 3 of https://doi.org/10.3847/1538-3881/acad85 for a full
description of the model input representation.
"""

from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from ..preprocess import preprocess

# Type for feature dictionaries whose feature are arrays and boolean masks
feature_dict = dict[str, Union[npt.NDArray[np.float_], npt.NDArray[np.bool_]]]


def global_features(
    tic: int,
    time: np.ndarray,
    flux: np.ndarray,
    transit_mask: npt.NDArray[np.bool_],
    period: float,
) -> feature_dict:
    """
    View of the full lightcurve.

    Note: time/flux should be folded *before* being passed to this method.

    Returns
    -------
    global_features: dict[str, array[float | bool]]
        global_view: The full light curve folded on the reported period with
            201 bins.
        global_std: The standard deviations for each bin.
        global_mask: A mask indicating whether the bin was empty.
        global_transit_mask: A mask indicating whether the bin falls inside
            the detected transit.
    """
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
) -> tuple[feature_dict, Optional[float], Optional[float]]:
    """
    View of points within two transit durations of the transit center, for a
    full time span of four transit durations.

    Note: time/flux should be folded *before* being passed to this method.

    Returns
    -------
    local_features: dict[str, array[float | bool]]
        local_view: Points within two transit durations of the transit center
            folded on the reported period with 61 bins.
        local_std: The standard deviations for each bin.
        local_mask: A mask indicating whether the bin was empty.
        local_scale: The scale factor used in normalization.
        local_scale_present: Whether the scale factor could be reported.
    local_scale: float | None
        Scale factor used in normalization.
    local_depth: float | None
        The transit depth pre-normalization (not used as a feature).
    """
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
) -> feature_dict:
    """
    Local view for flux calculated with a small/medium/large aperture.

    Used in the vetting model.

    Note: time/flux should be folded *before* being passed to this method.

    Returns
    -------
    aperture_features: dict[str, array[float]]
        local_aperture_{aperture_name}: Points within two transit durations of
            the transit center folded on the reported period with 61 bins.
    """
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
) -> feature_dict:
    """
    Local view using only points near odd-numbered transits (1st, 3rd, ...).

    Note: time/flux should be folded *before* being passed to this method.

    Returns
    -------
    odd_features: dict[str, array[float | bool]]
        local_view_odd: Points within two transit durations of an odd transit
            center folded on the reported period with 61 bins.
        local_std_odd: The standard deviations for each bin.
        local_mask_odd: A mask indicating whether the bin was empty.
    """
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
) -> feature_dict:
    """
    Local view using only points near even-numbered transits (2nd, 4th, ...).

    Note: time/flux should be folded *before* being passed to this method.

    Returns
    -------
    even_features: dict[str, array[float | bool]]
        local_view_even: Points within two transit durations of an even transit
            center folded on the reported period with 61 bins.
        local_std_even: The standard deviations for each bin.
        local_mask_even: A mask indicating whether the bin was empty.
    """
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
) -> tuple[feature_dict, Optional[float]]:
    """
    View centered around the most significant secondary transit.

    Note: time/flux should be folded *before* being passed to this method.

    Returns
    -------
    secondary_features: dict[str, array[float | bool]]
        secondary_view: Points within two transit durations of the most
            significant secondary transit, folded on the reported period with
            61 bins.
            secondary_std: The standard deviations for each bin.
            secondary_mask: A mask indicating whether the bin was empty.
            secondary_phase: The phase of the secondary transit's center.
            secondary_scale: The normalization scale factor.
        secondary_scale: float | None
            Scale factor used in normalization.
    """
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
) -> feature_dict:
    """
    Global or local view of points close to one of the transits with the most
    points.

    Note: time/flux should be folded *before* being passed to this method.

    Returns
    -------
    sample_segments_features: dict[str, array[float]]
        sample_segments_view: Global views of up to seven of the folds that
            contain the most points. Each fold is independently binned with 201
            bins.
        sample_segments_local_view: Local views of up to four of the folds that
            contain the most points. Each fold is independently binned with 61
            bins.
    """
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
) -> feature_dict:
    """
    Global view folded at double the reported period.

    Note: time/flux should be folded *before* being passed to this method.

    Note: the light curve should be folded with the transit center offset
    by 1/4 period: t0 = epoch - period / 2.
    This is to center the view between the two transits contained in the view,
    rather than having one in the middle and the other split at the boundaries.

    Returns
    -------
    double_period_features: dict[str, array[float | bool]]
        global_view_double_period: The full light curve folded on double the
            reported period with 201 bins.
        global_view_double_period_std: The standard deviation for each bin.
        global_view_double_period_mask: A mask indicating whether each bin was
            empty.
    """
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
) -> feature_dict:
    """
    Global and local views folded at half the reported period.

    Note: time/flux should be folded *before* being passed to this method.

    Returns
    -------
    half_period_features: dict[str, array[float | bool]]
        global_view_half_period: The full light curve folded on half the
            reported period with 201 bins.
        global_view_half_period_std: The standard deviation for each global
            bin.
        global_view_half_period_mask: A mask indicating whether each global bin
            was empty.
        local_view_half_period_std: Points within two transit durations of the
            transit center folded on half the reported period with 61 bins.
        local_view_half_period_std: The standard deviation for each local bin.
        local_view_half_period_mask: A mask indicating whether each local bin
            was empty.
    """
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
