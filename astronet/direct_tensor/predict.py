"""Make astronet predictions without creating tf.training.Example objects and files."""

import json
from itertools import starmap
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Literal, Optional, Protocol

import numpy as np
import pandas as pd
import tensorflow as tf

from astronet.direct_tensor.features import (
    aperture_features,
    double_period_features,
    even_features,
    global_features,
    half_period_features,
    local_features,
    odd_features,
    sample_segments_features,
    secondary_features,
)
from astronet.preprocess import preprocess


class LCGetter(Protocol):
    """Take astro_id and optionally aperture name and return (time, flux)."""

    def __call__(
        self, astro_id: int, aperture: Optional[Literal["s", "m", "l"]] = None
    ) -> tuple[np.ndarray, np.ndarray]: ...


def standard_view_features(
    tic: int,
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
    breakspace: Optional[float],
    aperture_fluxes: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """
    Process lightcurve and create standard view inputs that depend on time/flux values.

    Params
    ------
    tic: int
        TIC ID of target. Note: pretty sure this is not used by any of the methods where it's used.
    time: array[float]
        Array of time values in lightcurve.
    flux: array[float]
        Array of flux values in lightcurve.
    period: float
        Period of transit signal; used for lightcurve folding.
    epoch: float
        Reference transit time for signal; used for lightcurve folding.
    duration: float
        Duration of transits.
    breakspace: float | None
        Breakspace to use in spline detrending. None indicates that various breakspaces should be
        tried and the best selected.
    aperture_fluxes: dict[str, tuple[np.ndarray, np.ndarray]]
        For triage model: empty dict.
        For vetting model: dict of {aperture_name: (lightcurve_time, lightcurve_flux)} for all
        apertures to be considered ('s', 'm', 'l').


    Returns
    -------
    features: dict[str, float | array[float]]
        Dict of {feature_name: feature_value} for features that are or depend on views of the
        lightcurve.
    fold_num: array[float]
        Array containing the fold number of each point in the lightcurve. Used to calculate the
        total number of folds in the lightcurve for the "n_folds" feature.
    """
    tag = "" if breakspace is None else f"_{breakspace}"
    all_features = {}

    det_time, det_flux, transit_mask = preprocess.detrend_and_filter(
        tic, time, flux, period, epoch, duration, breakspace
    )
    folded_time, folded_flux, fold_num, normal_transit_mask = (
        preprocess.phase_fold_and_sort_light_curve(
            det_time, det_flux, transit_mask, period, epoch
        )
    )
    odd_mask = fold_num % 2 == 1
    even_mask = fold_num % 2 == 1

    all_features.update(
        global_features(tic, folded_time, folded_flux, normal_transit_mask, period)
    )

    loc_features, local_scale, local_depth = local_features(
        tic, folded_time, folded_flux, period, duration
    )
    all_features.update(loc_features)

    for aperture, (ap_time, ap_flux) in aperture_fluxes.items():
        ap_det_time, ap_det_flux, ap_transit_mask = preprocess.detrend_and_filter(
            tic, ap_time, ap_flux
        )
        ap_folded_time, ap_folded_flux, _, _ = (
            preprocess.phase_fold_and_sort_light_curve(
                ap_det_time, ap_det_flux, ap_transit_mask, period, epoch
            )
        )
        all_features.update(
            aperture_features(
                aperture,
                tic,
                ap_folded_time,
                ap_folded_flux,
                period,
                duration,
                local_scale,
                local_depth,
            )
        )

    all_features.update(
        odd_features(
            odd_mask,
            tic,
            folded_time,
            folded_flux,
            period,
            duration,
            local_scale,
            local_depth,
        )
    )
    all_features.update(
        even_features(
            even_mask,
            tic,
            folded_time,
            folded_flux,
            period,
            duration,
            local_scale,
            local_depth,
        )
    )

    sec_features, secondary_scale = secondary_features(
        tic, folded_time, folded_flux, period, duration, local_scale, local_depth
    )
    all_features.update(sec_features)

    all_features.update(
        sample_segments_features(
            tic,
            folded_time,
            folded_flux,
            fold_num,
            odd_mask,
            even_mask,
            period,
            duration,
        )
    )

    double_fold_time, double_fold_flux, _, _ = (
        preprocess.phase_fold_and_sort_light_curve(
            det_time, det_flux, transit_mask, period * 2, epoch - period / 2
        )
    )
    all_features.update(
        double_period_features(tic, double_fold_time, double_fold_flux, period)
    )

    half_fold_time, half_fold_flux, _, _ = preprocess.phase_fold_and_sort_light_curve(
        det_time, det_flux, transit_mask, period / 2, epoch
    )
    all_features.update(
        half_period_features(tic, half_fold_time, half_fold_flux, period, duration)
    )

    return {k + tag: v for k, v in all_features.items()}, fold_num


def prediction_features(
    tce: pd.Series,
    time: np.ndarray,
    flux: np.ndarray,
    aperture_fluxes: dict[str, tuple[np.ndarray, np.ndarray]],
    breakspaces: list[Optional[float]] = [0.3, 5.0, None],
):
    """
    Assemble all input features for a lightcurve.

    Params
    ------
    tce: pd.Series
        Row of dataframe containing relevant TCE parameters. Should include "Astro ID", "Per",
        "Epoc", "Dur", "Depth", "Tmag", "SMass", "SRad", "SRadEst".
    time: array[float]
        Array of time values in lightcurve.
    flux: array[float]
        Array of flux values in lightcurve.
    aperture_fluxes: dict[str, tuple[np.ndarray, np.ndarray]]
        For triage model: empty dict.
        For vetting model: dict of {aperture_name: (lightcurve_time, lightcurve_flux)} for all
        apertures to be considered ('s', 'm', 'l').
    breakspaces: list[Optional[float]]
        Breakspaces to use in spline detrending. None indicates that various breakspaces should be
        tried and the best selected. Default = [0.3, 5.0, None].

    Returns
    -------
    features: dict[str, float | array[float]]
        Dict of {feature_name: feature_value} for all inputs for the lightcurve.
    """
    all_features = {}

    for breakspace in breakspaces:
        breakspace_features, fold_num = standard_view_features(
            tce["Astro ID"],
            time,
            flux,
            tce["Per"],
            tce["Epoc"],
            tce["Dur"],
            breakspace,
            aperture_fluxes,
        )
        all_features.update(breakspace_features)

    scalar_features = {
        "astro_id": tce["Astro ID"],
        "Period": tce["Per"],
        "Duration": tce["Dur"],
        "Transit_Depth": tce["Depth"],
        "Tmag": tce["Tmag"],
        "star_mass": tce["SMass"] if not np.isnan(tce["SMass"]) else 0.0,
        "star_mass_present": float(np.isnan(tce["SMass"])),
        "star_rad": tce["SRad"] if not np.isnan(tce["SRad"]) else 0.0,
        "star_rad_present": float(np.isnan(tce["SRad"])),
        "star_rad_est": tce["SRadEst"] if not np.isnan(tce["SRadEst"]) else 0.0,
        "star_rad_est_present": float(np.isnan(tce["SRadEst"])),
        "n_folds": len(set(fold_num)),
        "n_points": len(fold_num),
    }
    for feature, value in scalar_features.items():
        assert not np.isnan(
            value
        ), f"Bad nan feature for Astro ID {tce['Astro ID']}: {feature}"

    all_features.update(
        {feature: np.array([value]) for feature, value in scalar_features.items()}
    )

    return all_features


def prepare_input(
    feature_cfg: dict,
    tce: pd.Series,
    get_lc: LCGetter,
    mode: Literal["triage", "vetting"],
) -> dict:
    """Assemble input features for TCE and normalize values where necessary."""
    time, flux = get_lc(tce["Astro ID"])
    aperture_fluxes = {}
    if mode == "vetting":
        aperture_fluxes = {
            aperture: get_lc(tce["Astro ID"], aperture) for aperture in ["s", "m", "l"]
        }
    tce_features = prediction_features(tce, time, flux, aperture_fluxes)
    tce_features = {
        name: value for name, value in tce_features.items() if name in feature_cfg
    }

    features = {}
    assert set(tce_features.keys()) == set(feature_cfg.keys())
    for name, value in tce_features.items():
        cfg = feature_cfg[name]
        if not cfg["is_time_series"]:
            if cfg.get("scale", None) == "log":
                value = tf.cast(value, tf.float64)
                value = tf.clip_by_value(value, cfg["min_val"], cfg["max_val"])
                value = value - cfg["min_val"] + 1
                value = tf.math.log(value) / tf.math.log(
                    tf.constant(cfg["max_val"], tf.float64)
                )
                value = tf.cast(value, tf.float32)
            elif cfg.get("scale", None) == "norm":
                value = (value - cfg["mean"]) / cfg["std"]
        features[name.lower()] = value

    return {k: [v] for k, v in features.items()}


def build_dataset(
    feature_cfg: dict,
    tces: pd.DataFrame,
    get_lc: LCGetter,
    mode: Literal["triage", "vetting"],
    nprocs: int = 1,
) -> tf.data.Dataset:
    """Create Dataset object containing input tensors for all tces."""
    tasks = [(feature_cfg, tce.to_dict(), get_lc, mode) for _, tce in tces.iterrows()]
    if nprocs == 1:
        all_tces_features = starmap(prepare_input, tasks)
    else:
        with Pool(nprocs) as pool:
            all_tces_features = pool.starmap(prepare_input, tasks)
    feature_table = pd.DataFrame(all_tces_features)
    dataset = feature_table.to_dict(orient="list")
    return tf.data.Dataset.from_tensor_slices(dataset)


def find_checkpoints(base_dir: Path, nruns: Optional[int] = None) -> list[Path]:
    """Find model directories, assuming base directory structure of model training checkpoints."""
    if nruns is None:
        nruns = len(list(base_dir.iterdir()))
    return [next((base_dir / str(i)).iterdir()) for i in range(1, nruns + 1)]


def batch_predict(
    checkpoints_dir: Path,
    tces: pd.DataFrame,
    get_lc: LCGetter,
    mode: Literal["triage", "vetting"],
    nruns: Optional[int] = None,
    nprocs: int = 1,
):
    """
    Run predictions from multiple model checkpoints for all TCEs.

    Assembles dataset in parallel, then runs model predictions in serial.

    Returns
    -------
    predictions: pd.DataFrame
        Indexed by Astro ID and model number. Column names are output labels and values are model
        predictions.
    """
    model_dirs = find_checkpoints(checkpoints_dir, nruns)
    first_model_dir = model_dirs[0]
    with (first_model_dir / "config.json").open("r") as config_file:
        config = json.load(config_file)
    input_features_cfg = config["inputs"]["features"]
    output_labels = config["inputs"]["label_columns"]
    # Ensure all configured inputs/outputs are the same
    for model_dir in model_dirs:
        with (model_dir / "config.json").open("r") as cfg_file:
            model_cfg = json.load(cfg_file)
        if model_cfg["inputs"]["features"] != input_features_cfg:
            raise ValueError(
                f"Configured inputs in {model_dir} do not match first checkpoint."
                f"\nFirst checkpoint:\n{input_features_cfg}"
                f"\n{model_dir}:\n{model_cfg['inputs']['features']}"
            )
        if model_cfg["inputs"]["label_columns"] != output_labels:
            raise ValueError(
                f"Configured output labels in {model_dir} do not match first checkpoint."
                f"\nFirst checkpoint:\n{output_labels}"
                f"\n{model_dir}:\n{model_cfg['inputs']['label_columns']}"
            )

    dataset = build_dataset(input_features_cfg, tces, get_lc, mode, nprocs)
    predictions = [
        tf.keras.models.load_model(model_dir).predict(dataset)
        for model_dir in model_dirs
    ]
    prediction_dfs = [
        pd.DataFrame(
            pred,
            index=pd.MultiIndex.from_product(
                [tces["Astro ID"], [i]], names=["Astro ID", "model_no"]
            ),
            columns=output_labels,
        )
        for i, pred in enumerate(predictions)
    ]
    return pd.concat(prediction_dfs)
