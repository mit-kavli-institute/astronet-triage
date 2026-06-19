# Changelog

## v3.1.0 — 2026-06-19 — ⚠️ BIG UPDATE by Pablo

**New astronet-vetting model + cadence-aware preprocessing.**

- New vetting model: **10-model ensemble**, config `pablomer_final`, trained Mar-2026.
  Uses cadence-aware / scatter-weighted preprocessing and `local_aperture_s/m/l` views.
- Deployed to `/pdo/astronet-data/models/vetting/production`; consumed by QLP
  `estools astronet --vetting` (step 7 of the QLP pipeline — replaces the human triage
  that normally happens at that stage).
- Architecture refactor in `astronet/astro_cnn_model/astro_cnn_model.py`
  (`backbone()` / `head()` / `get_embeddings()`); preprocessing changes in
  `astronet/preprocess/{preprocess,generate_input_records}.py` and
  `light_curve_util/median_filter2.py` (3.33-min cadence phase, scatter weighting).
- The QLP↔astronet public API is unchanged: `predict.batch_predict(models_dir, data_files)`
  and `preprocess.generate_input_records.create(..., mode=)`.

> ### ⏪ If something breaks, THIS is the reference point.
> Roll back by pointing `/sw/astronet` back to `astronet-3.0.1` (tag **`v3.0.1`**) and
> restoring the previous model `AstroCNNModelVetting_cshallue_20250429_181612` from
> `/pdo/astronet-data/models/vetting/archive/`.

This release was landed as a **code-only** curated merge from `pablomer-dev-training`:
notebooks, data files, and scratch/experiment scripts were intentionally excluded
(see `.gitignore`). `astronet/models.py` was kept from `main`, which drops the
experimental `ablation_studies` imports that are not part of production.

See `UPDATING.md` for how to put this version on your `PYTHONPATH`.

---

## v3.0.1 and earlier

Previous production vetting model: `AstroCNNModelVetting_cshallue_20250429_181612`.
See git history for details.
