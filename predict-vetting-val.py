#!/usr/bin/env python3
"""
Astronet Vetting Model Prediction and Evaluation Script

This script loads a trained Astronet model and evaluates its performance on test data,
generating precision-recall curves and other metrics.

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    classification_report,
)

# TensorFlow imports

import tensorflow as tf
from astronet import models, evaluation
from astronet.util import config_util
from astronet.astro_cnn_model import input_ds


test_tfrecord_pattern = '../mnt/tess/astronet/tfrecords-vetting-v01-tois-triageJs-nocentroid-april2025-test/*'
live_sector=False

class AstronetEvaluator:
    """Class to handle Astronet model evaluation and visualization."""

    def __init__(self, model_dir, test_tfrecord_pattern, live_sector=False):
        """
        Initialize the evaluator.

        Args:
            model_dir: Path to the trained model directory
            test_tfrecord_pattern: Pattern for test TFRecord files
            live_sector: Whether this is live sector data (affects label processing)
        """
        self.model_dir = model_dir
        self.test_tfrecord_pattern = test_tfrecord_pattern
        self.live_sector = live_sector

        # Load model and configs
        self._load_model_and_config()

        # Initialize data containers
        self.preds = None
        self.labels = None
        self.astro_ids = None
        self.df = None

    def _load_model_and_config(self):
        """Load the trained model and configuration files."""
        try:
            # Load configuration files
            train_flags = config_util.load_config(os.path.join(self.model_dir, 'train_flags.json'))
            self.config = config_util.load_config(os.path.join(self.model_dir, 'config.json'))

            # Load the model
            model_name = train_flags['model']
            with tf.device('/CPU:0'):
                self.model = models.load_model(model_name, self.model_dir)

            self.model.summary()

            # Compile the model
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[
                    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                    tf.keras.metrics.AUC(name="auc")
                ]
            )

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _extract_identifiers_from_batch(self, batch):
        """Extract identifiers from a batch and convert to proper format."""
        inputs, identifiers = batch
        astro_ids = identifiers.numpy()

        # Convert astro_ids to proper format
        astro_ids_list = []
        tic_ids = []
        planet_nos = []

        for astro_id in astro_ids:
            astro_id_str = str(astro_id)
            astro_ids_list.append(astro_id)
            tic_ids.append(int(astro_id_str[:-2]))
            planet_nos.append(int(astro_id_str[-2:]))

        return astro_ids_list, tic_ids, planet_nos

    def generate_predictions(self):
        """Generate predictions for the test dataset."""
        batch_size = self.config.hparams.batch_size

        # Build dataset for predictions
        ds_pred = input_ds.build_eval_dataset(
            file_pattern=self.test_tfrecord_pattern,
            input_config=self.config.inputs,
            batch_size=batch_size,
            include_identifiers=True,
            include_labels=False
        )

        # Generate predictions
        with tf.device('/CPU:0'):
            self.preds = self.model.predict(ds_pred)

        print(f"Predictions shape: {self.preds.shape}")

        # Process identifiers for live sector data
        if self.live_sector:
            self._process_live_sector_data(ds_pred)
        else:
            self._process_standard_data()

    def _process_live_sector_data(self, ds_pred):
        """Process data for live sector mode."""
        astro_ids = []
        tic_ids = []
        planet_nos = []

        # Extract identifiers from all batches
        for batch in ds_pred:
            batch_astro_ids, batch_tic_ids, batch_planet_nos = self._extract_identifiers_from_batch(batch)
            astro_ids.extend(batch_astro_ids)
            tic_ids.extend(batch_tic_ids)
            planet_nos.extend(batch_planet_nos)

        # Create DataFrame with all columns at once
        df_data = {
            "Astro ID": astro_ids,
            "tic_id": tic_ids,
            "planetno": planet_nos,
            "model_no": [0] * len(astro_ids)
        }

        # Add prediction columns
        for i, col in enumerate(["disp_p", "disp_e", "disp_n", "disp_j"]):
            df_data[col] = self.preds[:, i]

        self.df = pd.DataFrame(df_data)

        # Load true labels from CSV
        self._load_true_labels_from_csv()

    def _process_standard_data(self):
        """Process data for standard (non-live) mode."""
        batch_size = self.config.hparams.batch_size

        # Build dataset for labels
        ds_lbl = input_ds.build_eval_dataset(
            file_pattern=self.test_tfrecord_pattern,
            input_config=self.config.inputs,
            batch_size=batch_size,
            include_identifiers=True,
            include_labels=True
        )

        # Extract labels and IDs
        labels_list = []
        ids_list = []

        for features, labels_batch, weight_batch, id_batch in ds_lbl:
            labels_list.append(labels_batch.numpy())
            ids_list.append(id_batch.numpy())

        self.labels = np.concatenate(labels_list, axis=0)
        self.astro_ids = np.concatenate(ids_list, axis=0)

        print(f"Labels shape: {self.labels.shape}")
        print(f"Astro IDs shape: {self.astro_ids.shape}")

    def _load_true_labels_from_csv(self):
        """Load true labels from CSV file for live sector data."""
        true_labels_csv_path = '/pdo/users/dimond/eval_test/pablo_model/test_predictions_with_label.csv'

        try:
            true_labels_df = pd.read_csv(true_labels_csv_path)[['astro_id', 'true_label']]
            print(f"Using labels loaded from CSV file: {true_labels_csv_path}")

                        # Convert labels to one-hot encoding
            def convert_to_one_hot(label):
                if label == 'p': return [1, 0, 0, 0]
                if label == 'e': return [0, 1, 0, 0]
                if label == 'n': return [0, 0, 1, 0]
                if label == 'j': return [0, 0, 0, 1]
                return [0, 0, 0, 0]  # Default case

            # Apply the conversion function
            true_labels_df = true_labels_df.copy()
            true_labels_df['one_hot'] = true_labels_df['true_label'].apply(convert_to_one_hot)
            true_labels_df = true_labels_df.drop(columns='true_label')

            # Check if IDs match
            if self.df is not None:
                ids_match = np.array_equal(
                    true_labels_df['astro_id'].to_numpy(),
                    self.df['Astro ID'].to_numpy()
                )
            else:
                ids_match = False

            if ids_match:
                print("IDs match!")
            else:
                print("Warning: IDs do not match!")

            # Extract labels
            self.labels = np.array(true_labels_df['one_hot'].tolist())
            print(f'Labels shape: {self.labels.shape}')

        except Exception as e:
            print(f"Error loading true labels from CSV: {e}")
            raise

    def compute_statistics(self):
        """Compute basic statistics about predictions and labels."""
        if self.live_sector and self.df is not None:
            try:
                # Count true planets
                true_labels_df = pd.read_csv(
                    '/pdo/users/dimond/eval_test/pablo_model/test_predictions_with_label.csv'
                )
                true_labels_column = true_labels_df['true_label']
                count_p = (true_labels_column == 'p').sum()
                print(f"Number of true planets: {count_p}")

                # Count predicted planets
                pred_cols = ['disp_p', 'disp_e', 'disp_n', 'disp_j']
                count_max_p = (self.df['disp_p'] == self.df[pred_cols].max(axis=1)).sum()
                print(f"Number of predicted planets: {count_max_p}")
            except Exception as e:
                print(f"Error computing statistics: {e}")

    def prepare_evaluation_data(self):
        """Prepare data for evaluation by removing unnecessary columns."""
        test_labels = self.labels.copy()
        test_pred = self.preds.copy()

        # Drop the "Not Sure" column (index 2)
        test_labels = np.delete(test_labels, 2, axis=1)
        test_pred = np.delete(test_pred, 2, axis=1)

        if self.live_sector:
            # For live sector, also drop eclipsing binary column (index 1)
            test_labels = np.delete(test_labels, 1, axis=1)
            test_pred = np.delete(test_pred, 1, axis=1)
            label_names = ["Planet", "Junk"]
        else:
            label_names = ["Planet", "Eclipsing Binary", "Junk"]

        return test_labels, test_pred, label_names

    def create_precision_recall_plot(self, test_labels, test_pred, label_names):
        """Create an interactive precision-recall plot."""
        counts = test_labels.sum(axis=0).astype(int)

        # Compute PR curves for each class
        pr_data = []
        for i in range(len(label_names)):
            y_true = test_labels[:, i]
            y_score = test_pred[:, i]
            p, r, t = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            pr_data.append({
                "precision": p,
                "recall": r,
                "thresholds": t,
                "ap": ap
            })

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

        for i, name in enumerate(label_names):
            data = pr_data[i]
            lbl = f"{name} (N={counts[i]}, AUC={data['ap']:.3f})"
            ax.plot(data["recall"][:-1], data["precision"][:-1],
                    lw=2, label=lbl, marker='o', markersize=2)

        ax.set_xlabel("Recall", fontsize=14)
        ax.set_ylabel("Precision", fontsize=14)
        ax.set_title("Precision–Recall Curves", fontsize=18)

        # Set specific tick marks including 0.9 and 0.95
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])

        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.minorticks_on()
        ax.grid(True, which="minor", alpha=0.15)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left", fontsize=12, framealpha=0.8)
        plt.tight_layout()
        plt.show()

        return fig, pr_data

    def run_evaluation(self):
        """Run the complete evaluation pipeline."""
        print("Starting Astronet evaluation...")

        # Generate predictions
        self.generate_predictions()

        # Compute statistics
        self.compute_statistics()

        # Prepare evaluation data
        test_labels, test_pred, label_names = self.prepare_evaluation_data()

        # Create precision-recall plot
        fig, pr_data = self.create_precision_recall_plot(test_labels, test_pred, label_names)

        print("Evaluation complete!")
        return fig, pr_data


def main():
    """Main function to run the evaluation."""
    # Configuration
    model_dir = '../mnt/tess/models/vetting/20250429/cshallue/AstroCNNModelVetting_cshallue_20250429_181612'
    test_tfrecord_pattern = '../mnt/tess/astronet/tfrecords-vetting-v01-tois-triageJs-nocentroid-april2025-test/*'
    live_sector = False

    # Alternative configuration for live sector
    # test_tfrecord_pattern = '../../dimond/mnt/tess/astronet/tfrecords-vetting-sector86-all-test/*'
    # live_sector = True


    # Create evaluator and run evaluation
    evaluator = AstronetEvaluator(model_dir, test_tfrecord_pattern, live_sector)
    fig, pr_data = evaluator.run_evaluation()
    plt.savefig('pr_curve.png')
    plt.show()


if __name__ == "__main__":
    main()
