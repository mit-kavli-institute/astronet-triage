# Global+Local Ablation: Detailed Description

## Objective
Build a vetting model variant that uses only two TFRecord light-curve views as input:
- `global_view`
- `local_view`

Everything else in the TFRecords is intentionally excluded from model inputs for this ablation.

---

## 1. What is available in the TFRecords

`astronet/preprocess/generate_input_records_2.py` writes many features per example, including:
- Core light-curve views: `global_view`, `local_view`, `secondary_view`
- Alternate-view variants: `_0.3`, `_5.0`, odd/even, half/double-period views
- Masks/std/scales: e.g. `global_std`, `local_std`, masks, scale flags
- Aperture views (vetting): `local_aperture_s/m/l`
- Scalar aux/star features: `Period`, `Duration`, `Transit_Depth`, `Tmag`, stellar params
- Labels: vetting labels such as `disp_p`, `disp_e`, `disp_n`, `disp_j` (plus others in source data)

For this ablation, only `global_view` and `local_view` are parsed as model features.

---

## 2. Exact model inputs used in the ablation

In `astronet/astro_cnn_model/configurations_vetting_global_local_ablation.py`:
- `inputs.features` contains exactly:
  - `global_view`: shape `[201]`, time-series
  - `local_view`: shape `[61]`, time-series
- `hparams.aux_inputs` is `[]` (no scalar/aux features).
- `hparams.time_series_hidden` contains exactly CNN blocks for:
  - `global_view`
  - `local_view`

So the input parser reads only these two keys from each TFExample and ignores all other stored features.

---

## 3. Labels/loss setup

The ablation keeps the same vetting label setup as your recent vetting configs:
- `label_columns = ["disp_p", "disp_e", "disp_n", "disp_j"]`
- `exclusive_labels = True`
- `label_scheme = "binary"`

Training/evaluation metrics are produced by existing training code and stored in `evaluation/metrics.json`:
- `roc_auc`, `average_precision`, `loss`
- `precision`, `recall`, `pr_auc`
- `precision_thresh`, `recall_thresh`, `f1_thresh`

---

## 4. Guardrails to keep the ablation strict

`astronet/astro_cnn_model/astro_cnn_model_global_local_ablation.py` adds runtime checks:
- Feature set must be exactly `{"global_view", "local_view"}`
- Time-series block set must be exactly `{"global_view", "local_view"}`
- `aux_inputs` must be empty

If any extra feature/block/aux input is added by mistake, model init fails with a clear error.

---

## 5. Model registry wiring

`astronet/models.py` registers a new model name:
- `AstroCNNModelVettingGlobalLocalAblation`

Mapped to:
- Class: `AstroCNNModelGlobalLocalAblation`
- Config module: `configurations_vetting_global_local_ablation`

This keeps the original model entries untouched and allows ablation runs via `--model=...`.

---

## 6. Training script and run behavior

`astronet/ensemble_train_vetting_2025_global_local_ablation.sh`:
- Mirrors the structure of `ensemble_train_vetting_2025.sh`
- Uses:
  - `--model=AstroCNNModelVettingGlobalLocalAblation`
  - `--config_name=pablomer`
  - `CONFIG_OVERRIDES="train_steps=2000,init_from_pretrained_model=false,hparams.pre_logits_hidden_layer_size=512"`
- Trains two runs (`for i in {1..2}`)
- Sets `PYTHONPATH=$CODE_DIR` so `astronet` imports resolve
- Runs `combine_model_results.py` at the end

Output base:
- `/pdo/astronet-data/models/vetting/experimental/pablomer/dec2025_cad_scat_v5_duration24/<DATE>/pablomer-global-local-ablation-2k-nopretrained-z_dim512/`

Per-run artifacts include:
- `evaluation/metrics.json`
- `evaluation/*_pred.npy`, `*_label.npy`, `*_astro_ids.npy`
- `evaluation/*_exodash_results.csv`

Combined output:
- `all_preds.csv` at the base output directory.

---

## 7. What this ablation isolates

This setup isolates the contribution of:
- global phase-folded shape (`global_view`)
- transit-local shape (`local_view`)

It removes contributions from:
- secondary/eclipse view information
- multi-aperture channels
- alternate cadence/odd-even/period-perturbed views
- scalar astrophysical/context features

So performance deltas vs full vetting models primarily reflect information lost by excluding those additional channels/features.

---

## 8. PR curve workflow (1-vs-1 baseline vs ablation)

A standalone script was added:
- `astronet/pr_curve_ablation_vs_baseline.py`

### Inputs
- `--ablation_csv`: path to one ablation model `evaluation/test_exodash_results.csv`
- `--baseline_csv`: path to one baseline model `evaluation/test_exodash_results.csv`
- `--target_class`: one-vs-rest class in `{p,e,n,j}`

Expected CSV columns:
- `true_label`
- `disp_p`, `disp_e`, `disp_n`, `disp_j`

### What it computes
- Builds one-vs-rest labels from `true_label == target_class`
- Computes PR curves via `precision_recall_curve`
- Computes AP via `average_precision_score`
- Plots:
  - baseline PR line
  - ablation PR line
  - horizontal prevalence reference line
- Optionally exports all curve points to CSV

### Command used for your comparison
- Baseline model:
  - `/pdo/astronet-data/models/vetting/experimental/pablomer/dec2025_cad_scat_v5_duration24/20251217/pablomer-2k-nopretrained/AstroCNNModelVetting_pablomer_20251217_133625/evaluation/test_exodash_results.csv`
- Ablation model:
  - `/pdo/astronet-data/models/vetting/experimental/pablomer/dec2025_cad_scat_v5_duration24/20260226/pablomer-global-local-ablation-2k-nopretrained-z_dim512/AstroCNNModelVettingGlobalLocalAblation_pablomer_20260226_115209/evaluation/test_exodash_results.csv`

Executed:
- `/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python /pdo/users/pablomer/Astronet-Triage/astronet/pr_curve_ablation_vs_baseline.py --ablation_csv <ablation_csv> --baseline_csv <baseline_csv> --target_class p --output_png /pdo/users/pablomer/Astronet-Triage/astronet/pr_curve_ablation_vs_baseline_testP.png --output_csv /pdo/users/pablomer/Astronet-Triage/astronet/pr_curve_ablation_vs_baseline_testP_points.csv --title "Baseline vs Global+Local Ablation PR (Test, class P)"`

### Outputs
- Figure:
  - `astronet/pr_curve_ablation_vs_baseline_testP.png`
- Curve points:
  - `astronet/pr_curve_ablation_vs_baseline_testP_points.csv`
- Summary from run:
  - baseline AP = `0.975030`
  - ablation AP = `0.903624`
  - prevalence = `0.375969`
