# Global+Local Ablation Changes

## Goal
Create a vetting ablation that uses only `global_view` and `local_view` from TFRecords, and provide a train script analogous to `astronet/ensemble_train_vetting_2025.sh`.

## Files Added

### `ablation_studies/models/astro_cnn_model_global_local_ablation.py`
- Adds `AstroCNNModelGlobalLocalAblation`, a subclass of `AstroCNNModel`.
- Enforces ablation constraints at init time:
  - features must be exactly `global_view` and `local_view`
  - time-series blocks must be exactly `global_view` and `local_view`
  - `aux_inputs` must be empty
- Purpose: hard guardrail to ensure the model is not accidentally fed extra inputs.

### `ablation_studies/models/configurations_vetting_global_local_ablation.py`
- Adds `pablomer()` config for vetting ablation.
- Keeps vetting labels:
  - `label_columns = ["disp_p", "disp_e", "disp_n", "disp_j"]`
  - `label_scheme = "binary"`
- Restricts `inputs.features` to:
  - `global_view` shape `[201]`
  - `local_view` shape `[61]`
- Sets `aux_inputs` to `[]`.
- Defines CNN blocks only for `global_view` and `local_view`.

### `ablation_studies/scripts/ensemble_train_vetting_2025_global_local_ablation.sh`
- New training launcher, modeled after `ensemble_train_vetting_2025.sh`.
- Uses:
  - `--model=AstroCNNModelVettingGlobalLocalAblation`
  - `--config_name=pablomer`
  - `CONFIG_OVERRIDES="train_steps=2000,init_from_pretrained_model=false,hparams.pre_logits_hidden_layer_size=512"`
- Includes post-train metrics/prediction aggregation via:
  - `astronet/combine_model_results.py --base_path="$OUTPUT_DIR"`
- Adds `PYTHONPATH="$CODE_DIR"` on python invocations so `astronet` imports resolve.

## Files Modified

### `astronet/models.py`
- Registered new model entry:
  - `"AstroCNNModelVettingGlobalLocalAblation"`
- Wired to:
  - class: `AstroCNNModelGlobalLocalAblation`
  - config module: `configurations_vetting_global_local_ablation`

## Runtime Validation Done
- Executed:
  - `bash /pdo/users/pablomer/Astronet-Triage/ablation_studies/scripts/ensemble_train_vetting_2025_global_local_ablation.sh`
- First run failed due missing `PYTHONPATH`; script was patched.
- Relaunched successfully.
- Confirmed:
  - config override applied (`pre_logits_hidden_layer_size=512`)
  - GPUs detected
  - model built as `astro_cnn_model_global_local_ablation`
  - training actively progressed (step logs observed beyond 900/2000)

## PR Curve Comparison Added
- Added script:
  - `ablation_studies/scripts/pr_curve_ablation_vs_baseline.py`
- Purpose:
  - Plot a 1-vs-1 PR curve comparison from `test_exodash_results.csv` for one ablation model vs one baseline model.
- Run used:
  - `/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python /pdo/users/pablomer/Astronet-Triage/ablation_studies/scripts/pr_curve_ablation_vs_baseline.py --ablation_csv /pdo/astronet-data/models/vetting/experimental/pablomer/dec2025_cad_scat_v5_duration24/20260226/pablomer-global-local-ablation-2k-nopretrained-z_dim512/AstroCNNModelVettingGlobalLocalAblation_pablomer_20260226_115209/evaluation/test_exodash_results.csv --baseline_csv /pdo/astronet-data/models/vetting/experimental/pablomer/dec2025_cad_scat_v5_duration24/20251217/pablomer-2k-nopretrained/AstroCNNModelVetting_pablomer_20251217_133625/evaluation/test_exodash_results.csv --target_class p --output_png /pdo/users/pablomer/Astronet-Triage/ablation_studies/plots/pr_curve_ablation_vs_baseline_testP.png --output_csv /pdo/users/pablomer/Astronet-Triage/ablation_studies/plots/pr_curve_ablation_vs_baseline_testP_points.csv --title "Baseline vs Global+Local Ablation PR (Test, class P)"`
- Outputs created:
  - `ablation_studies/plots/pr_curve_ablation_vs_baseline_testP.png`
  - `ablation_studies/plots/pr_curve_ablation_vs_baseline_testP_points.csv`
- AP summary from this run:
  - baseline AP = `0.975030`
  - ablation AP = `0.903624`
  - prevalence = `0.375969`

## Notes
- Original files were preserved; new ablation behavior is isolated in new files plus model registry entry.
- The script currently trains 2 models (`for i in {1..2}`), then runs `combine_model_results.py`.
