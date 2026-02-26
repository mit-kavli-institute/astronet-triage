# Mixed Ablation (Global + Local + Scalars) Summary

## Objective
Create an ablation that keeps:
- `global_view` (time series),
- `local_view` (time series),
- scalar auxiliary parameters (`aux_inputs` from triage `pablomer` config),

and removes all other time-series views/channels.

## Files Added/Updated

### New model class
- `ablation_studies/models/astro_cnn_model_mixed_ablation.py`
- Class: `AstroCNNModelMixedAblation`
- Guardrails:
  - only `global_view` + `local_view` are allowed as time-series blocks,
  - feature set must equal `{global_view, local_view} U aux_inputs`,
  - aux inputs must map to non-time-series feature specs.

### New config module
- `ablation_studies/models/configurations_vetting_mixed_ablation.py`
- Config function: `pablomer()`
- Uses vetting labels: `disp_p, disp_e, disp_n, disp_j`
- Time-series features: only `global_view`, `local_view`
- Scalar features: copied from triage `pablomer` feature specs for names listed in triage `hparams.aux_inputs`
- `hparams.aux_inputs`: kept (scalar parameters included)
- `hparams.time_series_hidden`: only blocks for `global_view` and `local_view`
- `init_from_pretrained_model`: `False`

### Model registry update
- `astronet/models.py`
- Added model name:
  - `AstroCNNModelVettingMixedAblation`
  - mapped to:
    - class: `AstroCNNModelMixedAblation`
    - config module: `configurations_vetting_mixed_ablation`

### New training script
- `ablation_studies/scripts/ensemble_train_vetting_2025_mixed_ablation.sh`
- Runs 2 models with:
  - `--model=AstroCNNModelVettingMixedAblation`
  - `--config_name=pablomer`
  - `CONFIG_OVERRIDES="train_steps=2000,init_from_pretrained_model=false,hparams.pre_logits_hidden_layer_size=512"`
- Generates ensemble CSV via `combine_model_results.py`.

### PR script minor enhancement
- `ablation_studies/scripts/pr_curve_ablation_vs_baseline.py`
- Added `--ablation_label` argument for plot legend naming.

## Training Run (Completed)

Executed:
- `bash /pdo/users/pablomer/Astronet-Triage/ablation_studies/scripts/ensemble_train_vetting_2025_mixed_ablation.sh`

Output base:
- `/pdo/astronet-data/models/vetting/experimental/pablomer/dec2025_cad_scat_v5_duration24/20260226/pablomer-mixed-ablation-2k-nopretrained-z_dim512/`

Produced runs:
- `AstroCNNModelVettingMixedAblation_pablomer_20260226_152748`
- `AstroCNNModelVettingMixedAblation_pablomer_20260226_152926`

Combined predictions:
- `.../pablomer-mixed-ablation-2k-nopretrained-z_dim512/all_preds.csv`
- script-reported shape: `(1548, 14)`

## PR Comparison Plot (1 vs 1)

Baseline:
- `/pdo/astronet-data/models/vetting/experimental/pablomer/dec2025_cad_scat_v5_duration24/20251217/pablomer-2k-nopretrained/AstroCNNModelVetting_pablomer_20251217_133625/evaluation/test_exodash_results.csv`

Mixed ablation run:
- `/pdo/astronet-data/models/vetting/experimental/pablomer/dec2025_cad_scat_v5_duration24/20260226/pablomer-mixed-ablation-2k-nopretrained-z_dim512/AstroCNNModelVettingMixedAblation_pablomer_20260226_152748/evaluation/test_exodash_results.csv`

Generated with:
- `PYTHONPATH=/pdo/users/pablomer/Astronet-Triage /pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python /pdo/users/pablomer/Astronet-Triage/ablation_studies/scripts/pr_curve_ablation_vs_baseline.py --ablation_csv <mixed_run_csv> --baseline_csv <baseline_csv> --target_class p --ablation_label "Mixed Ablation" --output_png /pdo/users/pablomer/Astronet-Triage/ablation_studies/plots/pr_curve_mixed_ablation_vs_baseline_testP.png --output_csv /pdo/users/pablomer/Astronet-Triage/ablation_studies/plots/pr_curve_mixed_ablation_vs_baseline_testP_points.csv --title "Baseline vs Mixed Ablation PR (Test, class P)" --x_min 0.6 --x_max 1.0 --y_min 0.6 --y_max 1.0`

Outputs:
- `ablation_studies/plots/pr_curve_mixed_ablation_vs_baseline_testP.png`
- `ablation_studies/plots/pr_curve_mixed_ablation_vs_baseline_testP_points.csv`

Run summary:
- baseline AP = `0.975030`
- mixed ablation AP = `0.994092`
- prevalence = `0.375969` (not drawn unless `--show_prevalence` is passed)
