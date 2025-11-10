# Soft Labels Generation Process

This directory contains scripts to generate ensemble predictions and update TFRecord files with soft labels (averaged predictions from multiple models).

## Overview

The process involves:
1. Generating predictions from all models in an ensemble
2. Averaging the predictions across models
3. Updating TFRecord files with the averaged predictions as soft labels

## Step-by-Step Instructions

### Option 1: Use the Automated Pipeline Script (Recommended)

The easiest way is to use the bash script that processes both train and validation datasets:

1. **Edit the paths in `run_softlabels_pipeline.sh`**:
   ```bash
   ENSEMBLE_DIR='/path/to/your/ensemble/models/directory'
   TRAIN_DATA_DIR='/path/to/your/train/tfrecords/*'
   VAL_DATA_DIR='/path/to/your/val/tfrecords/*'
   BASE_OUTPUT_DIR='/path/to/output/directory'
   ```

2. **Run the pipeline**:
   ```bash
   cd Astronet-Triage/distillation
   ./run_softlabels_pipeline.sh
   ```

This will automatically:
- Generate predictions for both train and validation datasets
- Create separate output directories for each (`{BASE_OUTPUT_DIR}/train` and `{BASE_OUTPUT_DIR}/val`)
- Update TFRecord files for both datasets

### Option 2: Run Scripts Manually

If you prefer to run the scripts individually:

#### 1. Generate Ensemble Predictions

Run the script with command-line arguments:

```bash
cd Astronet-Triage/distillation
python generate_ensemble_predictions.py \
    --ensemble_dir '/path/to/your/ensemble/models/directory' \
    --data_dir '/path/to/your/tfrecords/*' \
    --output_dir '/path/to/output/directory'
```

**What this does:**
- Finds all model directories in `ensemble_dir`
- Generates predictions for each model on the dataset specified in `data_dir`
- Creates three CSV files in `output_dir`:
  - `ensemble_predictions_all.csv`: All predictions from all models (one row per astro_id per model)
  - `ensemble_predictions_averaged.csv`: Averaged predictions with true labels
  - `ensemble_predictions_averaged_no_labels.csv`: Averaged predictions without true labels

**Expected output:**
- Progress messages showing which models are being processed
- Summary statistics (number of predictions, models, etc.)
- CSV files saved to `output_dir`

#### 2. Update TFRecord Files with Soft Labels

Run the script to replace original labels in TFRecord files with the averaged predictions:

```bash
python update_tfrecords_with_predictions.py \
    --input_tfrecord_dir '/path/to/input/tfrecords/directory' \
    --predictions_csv '/path/to/ensemble_predictions_averaged.csv' \
    --output_tfrecord_dir '/path/to/output/directory'  # Optional, defaults to input_dir + '_softlabels'
```

**What this does:**
- Loads `ensemble_predictions_averaged.csv`
- Reads all TFRecord shard files from the input directory
- For each example in each TFRecord:
  - Extracts `astro_id`
  - Looks up averaged predictions (`avg_disp_p`, `avg_disp_e`, `avg_disp_n`, `avg_disp_j`)
  - Replaces original `disp_p`, `disp_e`, `disp_n`, `disp_j` values with averaged predictions
- Writes updated TFRecords to output directory (defaults to input directory name + `_softlabels`)

**Expected output:**
- Progress bar showing which TFRecord files are being processed
- Summary statistics (total records, updated records, missing records)
- Updated TFRecord files in the output directory

## Output Files

### From `generate_ensemble_predictions.py`:

1. **`ensemble_predictions_all.csv`**
   - Columns: `astro_id`, `model_no`, `model_dir`, `disp_p`, `disp_e`, `disp_n`, `disp_j`
   - Contains all predictions from all models

2. **`ensemble_predictions_averaged.csv`**
   - Columns: `astro_id`, `avg_disp_p`, `avg_disp_e`, `avg_disp_n`, `avg_disp_j`, `disp_p`, `disp_e`, `disp_n`, `disp_j`
   - Contains averaged predictions (avg_*) and true labels (disp_*)

3. **`ensemble_predictions_averaged_no_labels.csv`**
   - Columns: `astro_id`, `avg_disp_p`, `avg_disp_e`, `avg_disp_n`, `avg_disp_j`
   - Contains only averaged predictions, no true labels

### From `update_tfrecords_with_predictions.py`:

- Updated TFRecord files in a new directory (original name + `_softlabels`)
- Each TFRecord file has the same structure as the original, but with `disp_p`, `disp_e`, `disp_n`, `disp_j` replaced by the averaged predictions

## Troubleshooting

### Issue: "No models found in ensemble_dir"
- Check that `ensemble_dir` points to a directory containing model subdirectories
- Each model directory should contain `config.json` and `train_flags.json`

### Issue: "No files found in input directory"
- Verify the input TFRecord directory path in `update_tfrecords_with_predictions.py`
- Check that TFRecord files exist (they typically have no file extension)

### Issue: "Records without predictions"
- Some `astro_id` values in TFRecords might not be in the predictions CSV
- These records will keep their original labels
- Check the summary output to see how many records were updated vs. kept original

## Example Workflow

```bash
# 1. Edit paths
nano ensemblelabels.py

# 2. Generate predictions
python generate_ensemble_predictions.py

# 3. Update TFRecords
python update_tfrecords_with_predictions.py

# 4. Verify output
ls /pdo/astronet-data/data/tfrecords/oct2025_30minbin_v2/tfrecords-vetting-v01-tois-triageJs-nocentroid-april2025-train_softlabels/
```
