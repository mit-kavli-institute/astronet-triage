# Scalar Ablation (Time-Series-Only) Summary

## Naming
- Requested name: **scalar ablation**
- Suggested technical name: **time-series-only ablation**
  - Reason: it removes scalar auxiliary inputs and keeps all time-series streams.

## Goal
Keep all non-scalar (time-series) model inputs and remove scalar parameters from the model input pipeline.

## What was changed

### 1) New model class
- `ablation_studies/models/astro_cnn_model_scalar_ablation.py`
- Adds `AstroCNNModelScalarAblation` (subclass of `AstroCNNModel`) with guardrails:
  - `hparams.aux_inputs` must be empty.
  - all declared `inputs.features` must be time-series (`is_time_series=True`).

### 2) New scalar-ablation config module
- `ablation_studies/models/configurations_vetting_scalar_ablation.py`
- Provides `pablomer()` config for vetting:
  - uses triage `pablomer` time-series features only (filters out scalar features),
  - keeps vetting label setup: `["disp_p", "disp_e", "disp_n", "disp_j"]`,
  - keeps all original time-series blocks (`time_series_hidden`) from triage config,
  - keeps vetting aperture channels (`local_aperture_s/m/l`),
  - sets `aux_inputs: []`,
  - defaults to `init_from_pretrained_model: false`.

### 3) Model registry
- `astronet/models.py`
- Added:
  - model name: `AstroCNNModelVettingScalarAblation`
  - class: `AstroCNNModelScalarAblation`
  - config module: `configurations_vetting_scalar_ablation`

### 4) New training script
- `ablation_studies/scripts/ensemble_train_vetting_2025_scalar_ablation.sh`
- Analogous to prior vetting scripts.
- Uses:
  - `--model=AstroCNNModelVettingScalarAblation`
  - `--config_name=pablomer`
  - `CONFIG_OVERRIDES="train_steps=2000,init_from_pretrained_model=false,hparams.pre_logits_hidden_layer_size=512"`
- Runs 2 models and then combines predictions with `combine_model_results.py`.

## Training run completed

Script run:
- `bash /pdo/users/pablomer/Astronet-Triage/ablation_studies/scripts/ensemble_train_vetting_2025_scalar_ablation.sh`

Output base dir:
- `/pdo/astronet-data/models/vetting/experimental/pablomer/dec2025_cad_scat_v5_duration24/20260226/pablomer-scalar-ablation-2k-nopretrained-z_dim512/`

Produced runs:
- `AstroCNNModelVettingScalarAblation_pablomer_20260226_134905`
- `AstroCNNModelVettingScalarAblation_pablomer_20260226_135114`

Combined predictions:
- `.../pablomer-scalar-ablation-2k-nopretrained-z_dim512/all_preds.csv`
- final shape reported by script: `(1548, 14)`

## PR curve generated (1 vs 1)

Comparison:
- Scalar ablation model:
  - `.../AstroCNNModelVettingScalarAblation_pablomer_20260226_134905/evaluation/test_exodash_results.csv`
- Baseline model:
  - `/pdo/astronet-data/models/vetting/experimental/pablomer/dec2025_cad_scat_v5_duration24/20251217/pablomer-2k-nopretrained/AstroCNNModelVetting_pablomer_20251217_133625/evaluation/test_exodash_results.csv`

Command used:
- `/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python /pdo/users/pablomer/Astronet-Triage/ablation_studies/scripts/pr_curve_ablation_vs_baseline.py --ablation_csv <scalar_ablation_test_exodash_results.csv> --baseline_csv <baseline_test_exodash_results.csv> --target_class p --output_png /pdo/users/pablomer/Astronet-Triage/ablation_studies/plots/pr_curve_scalar_ablation_vs_baseline_testP.png --output_csv /pdo/users/pablomer/Astronet-Triage/ablation_studies/plots/pr_curve_scalar_ablation_vs_baseline_testP_points.csv --title "Baseline vs Scalar Ablation PR (Test, class P)" --x_min 0.6 --x_max 1.0 --y_min 0.6 --y_max 1.0`

Artifacts:
- `ablation_studies/plots/pr_curve_scalar_ablation_vs_baseline_testP.png`
- `ablation_studies/plots/pr_curve_scalar_ablation_vs_baseline_testP_points.csv`

AP summary from the script output:
- baseline AP = `0.975030`
- scalar ablation AP = `0.957694`
- prevalence = `0.375969` (not plotted in this figure)
