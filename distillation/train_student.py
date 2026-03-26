#!/usr/bin/env python3
"""
Train a student model using soft labels from ensemble predictions.

This script trains a student model using a combination of:
- KL divergence loss on soft labels (ensemble predictions)
- Standard loss on hard labels (original ground truth)
"""

import datetime
import os
import argparse
import numpy as np
import tensorflow as tf
from absl import logging
import pandas as pd

from astronet import models, evaluation
from astronet.util import config_util
from astronet.astro_cnn_model import input_ds
from astronet.training import ThresholdPrecision, ThresholdRecall


class SoftLabelExampleParser:
    """Parser that extracts both hard and soft labels from TFRecord examples."""

    def __init__(self, config, include_identifiers=False):
        self.config = config
        self.include_identifiers = include_identifiers
        self.hard_label_cols = config.label_columns
        self.soft_label_cols = [f"{col}_soft" for col in config.label_columns]

    def _extract_features(self, parsed_features):
        """Extracts and processes features from raw parsed features."""
        features = {}
        for name, cfg in self.config.features.items():
            value = parsed_features.pop(name)
            if not cfg.is_time_series:
                if cfg.get("scale") == "log":
                    value = tf.cast(value, tf.float64)
                    value = tf.maximum(value, cfg.min_val)
                    value = tf.minimum(value, cfg.max_val)
                    value = value - cfg.min_val + 1
                    value = tf.math.log(value) / tf.math.log(
                        tf.constant(cfg.max_val, tf.float64))
                    value = tf.cast(value, tf.float32)
                elif cfg.get("scale") == "norm":
                    value = (value - cfg["mean"]) / cfg["std"]
            features[name] = value
        return features

    def _extract_hard_labels(self, parsed_features):
        """Extracts hard labels (original ground truth)."""
        label_features = [
            parsed_features.pop(name) for name in self.hard_label_cols
        ]
        label_features = tf.cast(tf.stack(label_features), tf.float32)

        label_scheme = self.config.get("label_scheme", "binary")
        is_single_label = len(self.hard_label_cols) == 1
        if is_single_label and label_scheme != "binary":
            raise ValueError("Single label requires label_scheme=binary")
        if label_scheme == "binary":
            labels = tf.squeeze(tf.minimum(label_features, 1))
        elif label_scheme == "distribution":
            labels = label_features / tf.reduce_sum(label_features)
        elif label_scheme == "maximum":
            labels = tf.floor(label_features / tf.reduce_max(label_features))
            labels /= tf.reduce_sum(labels)
        else:
            labels = label_features / tf.reduce_sum(label_features)

        weight = 1.0
        if self.config.get("uncertainty_weight"):
            if len(self.hard_label_cols) == 1:
                raise ValueError("uncertainty_weight requires multiple labels")
            weight = tf.reduce_max(label_features) / tf.maximum(
                tf.reduce_sum(label_features), 1.0)

        downweight_factor = self.config.get("non_primary_downweight_factor", 2.0)
        primary_class = 0 if is_single_label else self.config.primary_class
        if downweight_factor and label_features[primary_class] < 1:
            weight /= downweight_factor

        return labels, weight

    def _extract_soft_labels(self, parsed_features):
        """Extracts soft labels (ensemble predictions)."""
        soft_label_features = [
            parsed_features.pop(name) for name in self.soft_label_cols
        ]
        soft_labels = tf.cast(tf.stack(soft_label_features), tf.float32)
        # Soft labels are already probabilities from ensemble models
        # No normalization needed - use as-is
        return soft_labels

    def __call__(self, serialized_example):
        """Parses a single tf.Example into features, hard labels, and soft labels."""
        data_fields = {
            feature_name: tf.io.FixedLenFeature(feature.shape, tf.float32)
            for feature_name, feature in self.config.features.items()
        }

        # Add hard labels (int64)
        for name in self.hard_label_cols:
            data_fields[name] = tf.io.FixedLenFeature([], tf.int64)

        # Add soft labels (float32)
        for name in self.soft_label_cols:
            data_fields[name] = tf.io.FixedLenFeature([], tf.float32)

        if self.include_identifiers:
            assert "astro_id" not in data_fields
            data_fields["astro_id"] = tf.io.FixedLenFeature([], tf.int64)

        parsed_features = tf.io.parse_single_example(
            serialized_example, features=data_fields)

        features = self._extract_features(parsed_features)
        hard_labels, weight = self._extract_hard_labels(parsed_features)
        soft_labels = self._extract_soft_labels(parsed_features)

        if self.include_identifiers:
            identifiers = parsed_features.pop("astro_id")
            return features, hard_labels, soft_labels, weight, identifiers
        else:
            return features, hard_labels, soft_labels, weight


class CombinedDistillationLoss(tf.keras.losses.Loss):
    """Combined loss function for knowledge distillation.

    Combines:
    - KL divergence loss between student predictions and soft labels (teacher)
    - Standard loss between student predictions and hard labels
    """

    def __init__(self,
                 soft_label_weight=1.0,
                 hard_label_weight=0.1,
                 temperature=1.0,
                 hard_loss_fn=None,
                 name='combined_distillation_loss'):
        super().__init__(name=name)
        self.soft_label_weight = soft_label_weight
        self.hard_label_weight = hard_label_weight
        self.temperature = temperature
        self.hard_loss_fn = hard_loss_fn or tf.keras.losses.BinaryCrossentropy()

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: Tuple of (hard_labels, soft_labels)
            y_pred: Student model predictions (logits)
        """
        hard_labels, soft_labels = y_true

        # KL divergence loss with temperature scaling
        # Apply temperature to logits
        scaled_logits = y_pred / self.temperature

        # Determine if we're using multi-class (softmax) or multi-label (sigmoid)
        # Check the hard loss function type
        is_categorical = isinstance(self.hard_loss_fn, tf.keras.losses.CategoricalCrossentropy)

        if is_categorical:
            # Multi-class: use softmax
            student_probs = tf.nn.softmax(scaled_logits)
            # KL divergence: KL(soft_labels || student_probs)
            # = sum(soft_labels * log(soft_labels / student_probs))
            # First term is constant, so we minimize: -sum(soft_labels * log(student_probs))
            epsilon = 1e-8
            kl_loss = -tf.reduce_sum(
                soft_labels * tf.math.log(student_probs + epsilon), axis=-1
            )
        else:
            # Multi-label: use sigmoid
            student_probs = tf.nn.sigmoid(scaled_logits)
            # For multi-label, compute KL divergence per label and sum
            # KL(soft_labels || student_probs) = sum over labels. For each label, we have:
            #   soft_labels * log(soft_labels / student_probs) +
            #   (1 - soft_labels) * log((1 - soft_labels) / (1 - student_probs))
            epsilon = 1e-8
            kl_per_label = (
                soft_labels * tf.math.log((soft_labels + epsilon) / (student_probs + epsilon)) +
                (1 - soft_labels) * tf.math.log((1 - soft_labels + epsilon) / (1 - student_probs + epsilon))
            )
            kl_loss = tf.reduce_sum(kl_per_label, axis=-1)

        kl_loss = tf.reduce_mean(kl_loss)

        # Hard label loss (can be BinaryCrossentropy or CategoricalCrossentropy)
        hard_loss = self.hard_loss_fn(hard_labels, y_pred)

        # Combined loss
        # Temperature^2 scaling is standard in distillation literature
        total_loss = (self.soft_label_weight * kl_loss * (self.temperature ** 2) +
                     self.hard_label_weight * hard_loss)

        return total_loss


def build_train_dataset_with_soft_labels(file_pattern,
                                         input_config,
                                         batch_size,
                                         shuffle_values_buffer=2500,
                                         exclude_astro_ids=None):
    """Builds a training dataset that includes both hard and soft labels."""
    filenames = tf.io.gfile.glob(file_pattern)
    if not filenames:
        raise ValueError(f"Found no files matching '{file_pattern}'")

    ds = tf.data.Dataset.from_tensor_slices(filenames)
    ds = ds.shuffle(len(filenames))
    ds = ds.flat_map(tf.data.TFRecordDataset)

    # Parse examples with soft labels
    parse_identifiers = exclude_astro_ids is not None
    example_parser = SoftLabelExampleParser(input_config, parse_identifiers)
    ds = ds.map(example_parser)

    # Filtering step
    if exclude_astro_ids is not None:
        exclude_astro_ids_tf = tf.constant(list(exclude_astro_ids), dtype=tf.int64)

        def filter_fn(*args):
            astro_id = args[-1]
            is_excluded = tf.reduce_any(tf.equal(astro_id, exclude_astro_ids_tf))
            return ~is_excluded

        ds = ds.filter(filter_fn)
        logging.info(f"Filtered out {len(exclude_astro_ids)} astro_ids")

        # Remove identifiers if not needed
        def strip_identifiers(features, hard_labels, soft_labels, weight, astro_id):
            return features, hard_labels, soft_labels, weight
        ds = ds.map(strip_identifiers)

    # Apply data augmentation if needed
    # (TimeSeriesRandomReverser would go here if needed)

    # Combine hard and soft labels into a tuple for the loss function
    def combine_labels(features, hard_labels, soft_labels, weight):
        # Return (features, (hard_labels, soft_labels)) for loss function
        return features, (hard_labels, soft_labels)

    ds = ds.map(combine_labels)

    # Shuffle, repeat, batch
    ds = ds.shuffle(shuffle_values_buffer)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(10)

    return ds


def compile_model_student(model, config, soft_label_weight, hard_label_weight, temperature):
    """Compiles a model for student training with combined distillation loss."""
    # Set up the learning rate schedule (same as training.py)
    if config.hparams.learning_rate_schedule == "constant":
        if config.hparams.learning_rate_warmup_frac:
            raise ValueError(
                "Learning rate warmup is not supported with constant schedule")
        learning_rate = config.hparams.learning_rate
    elif config.hparams.learning_rate_schedule == "cosine":
        logging.info(f"Using cosine learning rate schedule")
        train_steps = config.train_steps
        warmup_frac = config.hparams.learning_rate_warmup_frac
        warmup_steps = int(warmup_frac * train_steps)
        peak_learning_rate = float(config.hparams.learning_rate)
        initial_learning_rate = peak_learning_rate / 1000
        decay_hparams = dict(
            initial_learning_rate=initial_learning_rate,
            warmup_target=peak_learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=train_steps - warmup_steps,
            alpha=float(config.hparams.learning_rate_decay_alpha),
        )
        logging.info(
            f"Using cosine learning rate decay with parameters {decay_hparams}")
        learning_rate = tf.keras.optimizers.schedules.CosineDecay(**decay_hparams)
    else:
        raise ValueError(config.hparams.learning_rate_schedule)

    # Set up the optimizer
    opt_hparams = dict(
        learning_rate=learning_rate,
        weight_decay=config.hparams.get("weight_decay"))
    if config.hparams.optimizer == "sgd":
        opt_hparams.update(momentum=1.0 - config.hparams.one_minus_momentum,)
        optimizer = tf.keras.optimizers.SGD(**opt_hparams)
    elif config.hparams.optimizer == "adam":
        opt_hparams.update(
            beta_1=1.0 - config.hparams.one_minus_adam_beta_1,
            beta_2=1.0 - config.hparams.one_minus_adam_beta_2,
            epsilon=config.hparams.adam_epsilon,
        )
        optimizer = tf.keras.optimizers.Adam(**opt_hparams)
    else:
        raise ValueError(config.hparams.optimizer)
    logging.info(
        f"Using '{optimizer.name}' optimizer with parameters {opt_hparams}")

    logging.info(f"Using combined distillation loss:")
    logging.info(f"  - Soft label weight: {soft_label_weight}")
    logging.info(f"  - Hard label weight: {hard_label_weight}")
    logging.info(f"  - Temperature: {temperature}")

    # Compile model (loss is handled in custom train_step)
    model.compile(
        optimizer=optimizer,
        loss=None,  # Handled in custom train_step
        metrics=[
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
            ThresholdPrecision(threshold=0.3),
            ThresholdRecall(threshold=0.3)
        ]
    )


class StudentModel(tf.keras.Model):
    """Wrapper model that handles combined distillation loss with tuple labels."""

    def __init__(self, base_model, combined_loss):
        super().__init__()
        self.base_model = base_model
        self.combined_loss = combined_loss

    def call(self, inputs, training=None):
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        """Custom training step to handle tuple labels."""
        x, y = data
        hard_labels, soft_labels = y

        # Debug print on first step only
        if not hasattr(self, '_first_step_printed'):
            logging.info("\n" + "="*70)
            logging.info("First training step - verifying labels and predictions:")
            logging.info("="*70)
            logging.info(f"Hard labels shape: {hard_labels.shape}, dtype: {hard_labels.dtype}")
            logging.info(f"Soft labels shape: {soft_labels.shape}, dtype: {soft_labels.dtype}")
            # Avoid .numpy() inside graph mode (train_step can run under tf.function).
            logging.info(f"First example hard label tensor: {hard_labels[0]}")
            logging.info(f"First example soft label tensor: {soft_labels[0]}")
            self._first_step_printed = True

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

            # Debug print on first step
            if not hasattr(self, '_first_pred_printed'):
                logging.info(f"Predictions shape: {y_pred.shape}, dtype: {y_pred.dtype}")
                logging.info(f"First example prediction tensor (logits): {y_pred[0]}")
                logging.info("="*70 + "\n")
                self._first_pred_printed = True

            loss = self.combined_loss((hard_labels, soft_labels), y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.compiled_metrics.update_state(hard_labels, y_pred)

        return {m.name: m.result() for m in self.metrics}


def train_student(model, config, train_files, shuffle_buffer_size=2500,
                 exclude_astro_ids=None, soft_label_weight=1.0,
                 hard_label_weight=0.1, temperature=1.0, use_binary_loss_for_hard_labels=False):
    """Trains a student model with soft labels."""
    # Build dataset with soft labels
    ds = build_train_dataset_with_soft_labels(
        file_pattern=train_files,
        input_config=config.inputs,
        batch_size=config.hparams.batch_size,
        shuffle_values_buffer=shuffle_buffer_size,
        exclude_astro_ids=exclude_astro_ids
    )

    # Debug: Print sample batch to verify soft labels are loaded correctly
    logging.info("\n" + "="*70)
    logging.info("Verifying soft labels in dataset...")
    logging.info("="*70)
    sample_batch = next(iter(ds))
    features, labels_tuple = sample_batch
    hard_labels, soft_labels = labels_tuple

    logging.info(f"Batch features keys: {list(features.keys())[:5]}...")  # Show first 5 keys
    logging.info(f"Hard labels shape: {hard_labels.shape}")
    logging.info(f"Soft labels shape: {soft_labels.shape}")
    logging.info(f"\nFirst 3 examples - Hard labels:")
    for i in range(min(3, hard_labels.shape[0])):
        logging.info(f"  Example {i}: {hard_labels[i].numpy()}")
    logging.info(f"\nFirst 3 examples - Soft labels:")
    for i in range(min(3, soft_labels.shape[0])):
        logging.info(f"  Example {i}: {soft_labels[i].numpy()}")
    logging.info(f"\nHard labels sum (should be 1.0 for one-hot): {tf.reduce_sum(hard_labels[0]).numpy():.4f}")
    logging.info(f"Soft labels sum (should be ~1.0 for probabilities): {tf.reduce_sum(soft_labels[0]).numpy():.4f}")
    logging.info(f"Soft labels min: {tf.reduce_min(soft_labels).numpy():.6f}, max: {tf.reduce_max(soft_labels).numpy():.6f}")
    logging.info("="*70 + "\n")

    # Create combined loss
    # For hard labels, use BinaryCrossentropy if explicitly requested or if labels are not exclusive
    # BinaryCrossentropy treats each class independently, which can be better for distillation
    n_labels = len(config.inputs.label_columns)

    if use_binary_loss_for_hard_labels:
        hard_loss_fn = tf.keras.losses.BinaryCrossentropy()
        logging.info("Using BinaryCrossentropy for hard labels (overriding config)")
    elif n_labels > 1 and config.inputs.get("exclusive_labels", False):
        hard_loss_fn = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=config.hparams.get("label_smoothing", 0.0))
        logging.info("Using CategoricalCrossentropy for hard labels (from config)")
    else:
        hard_loss_fn = tf.keras.losses.BinaryCrossentropy()
        logging.info("Using BinaryCrossentropy for hard labels (from config)")

    combined_loss = CombinedDistillationLoss(
        soft_label_weight=soft_label_weight,
        hard_label_weight=hard_label_weight,
        temperature=temperature,
        hard_loss_fn=hard_loss_fn
    )

    # Wrap model to handle tuple labels
    student_model = StudentModel(model, combined_loss)

    # Compile with optimizer and metrics (loss is handled in train_step)
    compile_model_student(
        student_model, config, soft_label_weight, hard_label_weight, temperature
    )

    # Train
    history = student_model.fit(ds, steps_per_epoch=config["train_steps"])
    return history, student_model


def main():
    parser = argparse.ArgumentParser(
        description='Train a student model using soft labels from ensemble predictions'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Name of the model class'
    )
    parser.add_argument(
        '--config_file',
        type=str,
        default=None,
        help='File containing the model and training configuration'
    )
    parser.add_argument(
        '--config_name',
        type=str,
        default=None,
        help='Name of the model and training configuration'
    )
    parser.add_argument(
        '--config_overrides',
        type=str,
        default=None,
        help='Overrides to the base configuration (comma-separated key=value pairs)'
    )
    parser.add_argument(
        '--train_files',
        type=str,
        required=True,
        help='Comma-separated list of file patterns matching TFRecord files with soft labels'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Directory for model checkpoints and summaries'
    )
    parser.add_argument(
        '--soft_label_weight',
        type=float,
        default=1.0,
        help='Weight for soft label (KL divergence) loss (default: 1.0)'
    )
    parser.add_argument(
        '--hard_label_weight',
        type=float,
        default=0.1,
        help='Weight for hard label loss (default: 0.1)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Temperature for distillation (default: 1.0, no scaling)'
    )
    parser.add_argument(
        '--use_binary_loss_for_hard_labels',
        action='store_true',
        help='Use BinaryCrossentropy for hard labels even if config specifies CategoricalCrossentropy. '
             'This can be better for distillation as it treats each class independently.'
    )
    parser.add_argument(
        '--train_steps',
        type=int,
        default=None,
        help='Total number of steps to train the model for'
    )
    parser.add_argument(
        '--shuffle_buffer_size',
        type=int,
        default=25000,
        help='Size of the shuffle buffer for the training dataset (default: 25000)'
    )
    parser.add_argument(
        '--eval_files',
        type=str,
        nargs='*',
        default=None,
        help='File patterns matching TFRecord files in evaluation dataset(s)'
    )
    parser.add_argument(
        '--astro_ids_file',
        type=str,
        default=None,
        help='File containing Astro IDs to exclude from training'
    )
    parser.add_argument(
        '--save_format',
        type=str,
        choices=['keras', 'h5'],
        default='h5',
        help='Format for saving the trained model (default: h5)'
    )

    args = parser.parse_args()

    # Validate config arguments
    if bool(args.config_name) == bool(args.config_file):
        raise ValueError("Exactly one of --config_name and --config_file is required")

    logging.info('Starting student model training with soft labels')

    # Track training flags
    train_flags = {
        "model": args.model,
        "train_files": args.train_files,
        "eval_files": args.eval_files,
        "shuffle_buffer_size": args.shuffle_buffer_size,
        "soft_label_weight": args.soft_label_weight,
        "hard_label_weight": args.hard_label_weight,
        "temperature": args.temperature,
        "astro_ids_file": args.astro_ids_file,
    }

    # Load config
    if args.config_name:
        config = models.get_model_config(args.model, args.config_name)
        train_flags["config_name"] = args.config_name
        expt_name = f"{args.model}_{args.config_name}"
    else:
        config = config_util.load_config(args.config_file)
        train_flags["config_file"] = args.config_file
        logging.info(f"Loaded config from {args.config_file}")
        expt_name = args.model

    # Apply config overrides
    if args.config_overrides:
        overrides = config_util.parse_config_str(args.config_overrides)
        train_flags["config_overrides"] = args.config_overrides
        config_util.update(config, overrides)
        logging.info(f"Updated config with overrides {overrides}")

    # Set training steps
    if args.train_steps:
        config["train_steps"] = args.train_steps
        logging.info(f"Set config.train_steps to {args.train_steps}")
    if not config.get("train_steps"):
        raise ValueError(
            "train_steps must be set in the config or via --train_steps")

    # Load astro IDs to exclude
    exclude_astro_ids = set()
    if args.astro_ids_file:
        exclude_astro_ids = set(pd.read_csv(args.astro_ids_file, header=None).iloc[:,0].tolist())
        logging.info(f"Loaded {len(exclude_astro_ids)} Astro IDs to exclude from training.")

    # Build model
    model_class = models.get_model_class(args.model)
    model = model_class(config)

    # Create model directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(args.model_dir, f"{expt_name}_student_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    # Save configs
    config_util.save_config(train_flags, model_dir, basename="train_flags")
    config_util.save_config(config, model_dir)

    logging.info('Starting training: %d steps, shuffle_buffer=%d',
                 config['train_steps'], args.shuffle_buffer_size)
    logging.info("Model summary:")
    model.summary()

    # Train model
    history, student_model = train_student(
        model=model,
        config=config,
        train_files=args.train_files,
        shuffle_buffer_size=args.shuffle_buffer_size,
        exclude_astro_ids=exclude_astro_ids,
        soft_label_weight=args.soft_label_weight,
        hard_label_weight=args.hard_label_weight,
        temperature=args.temperature,
        use_binary_loss_for_hard_labels=args.use_binary_loss_for_hard_labels
    )

    # Save the base model (not the wrapper)
    base_model = student_model.base_model
    models.save_model(base_model, model_dir, args.save_format)
    logging.info(f"Model saved to {model_dir}")

    # Compile base model for evaluation (needed for model.evaluate())
    # Use the same loss function that would be used in standard training
    n_labels = len(config.inputs.label_columns)
    if n_labels > 1 and config.inputs.get("exclusive_labels", False):
        eval_loss = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=config.hparams.get("label_smoothing", 0.0))
    else:
        eval_loss = tf.keras.losses.BinaryCrossentropy()

    base_model.compile(
        optimizer=tf.keras.optimizers.Adam(),  # Dummy optimizer for evaluation
        loss=eval_loss,
        metrics=[
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
            ThresholdPrecision(threshold=0.3),
            ThresholdRecall(threshold=0.3)
        ]
    )

    # Evaluate if eval_files provided
    if args.eval_files:
        eval_dir = os.path.join(model_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)

        all_metrics = {}
        for eval_file in args.eval_files:
            if ":" in eval_file:
                name, file_pattern = eval_file.split(":", 1)
            elif len(args.eval_files) == 1:
                name = "eval"
            else:
                raise ValueError("Multiple evaluation datasets must be named with format 'name:file_patterns'")

            logging.info(f"\n{'='*70}")
            logging.info(f"Evaluating on {name} dataset")
            logging.info(f"{'='*70}")

            # Use base_model for evaluation (now compiled)
            metrics, labels, predictions, astro_ids = evaluation.evaluate_model(
                base_model, config.inputs, file_pattern, config.hparams.batch_size, threshold=0.215
            )
            all_metrics[name] = metrics

            # Print metrics to console
            logging.info(f"\nMetrics for {name}:")
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    logging.info(f"  {metric_name}: {metric_value:.6f}")
                else:
                    logging.info(f"  {metric_name}: {metric_value}")

            # Save numpy arrays
            labels_path = os.path.join(eval_dir, f"{name}_label.npy")
            pred_path = os.path.join(eval_dir, f"{name}_pred.npy")
            astro_ids_path = os.path.join(eval_dir, f"{name}_astro_ids.npy")
            results_path = os.path.join(eval_dir, f"{name}_exodash_results.csv")

            np.save(labels_path, labels)
            np.save(pred_path, predictions)
            np.save(astro_ids_path, astro_ids)

            # Export dash file (existing format)
            evaluation.export_dash_file(
                labels=labels, predictions=predictions, astro_ids=astro_ids,
                results_path=results_path
            )

            # Create detailed predictions CSV with astro_id, true labels, and predictions
            label_cols = config.inputs.label_columns
            pred_cols = [f"pred_{col}" for col in label_cols]
            true_cols = [f"true_{col}" for col in label_cols]

            # Create DataFrame
            predictions_df = pd.DataFrame({
                'astro_id': astro_ids
            })

            # Add true labels
            for i, col in enumerate(true_cols):
                predictions_df[col] = labels[:, i]

            # Add predictions
            for i, col in enumerate(pred_cols):
                predictions_df[col] = predictions[:, i]

            # Add predicted class (argmax)
            predictions_df['predicted_class'] = np.argmax(predictions, axis=1)
            predictions_df['true_class'] = np.argmax(labels, axis=1)

            # Add class names if available
            if len(label_cols) == 4:  # Standard vetting labels
                class_names = ['Planet', 'Eclipsing Binary', 'Not Sure', 'Junk']
                predictions_df['predicted_class_name'] = predictions_df['predicted_class'].map(
                    lambda x: class_names[x] if x < len(class_names) else f'Class_{x}'
                )
                predictions_df['true_class_name'] = predictions_df['true_class'].map(
                    lambda x: class_names[x] if x < len(class_names) else f'Class_{x}'
                )

            # Save detailed predictions CSV
            detailed_pred_path = os.path.join(eval_dir, f"{name}_detailed_predictions.csv")
            predictions_df.to_csv(detailed_pred_path, index=False)

            logging.info(f"\nSaved evaluation results for {name}:")
            logging.info(f"  - Labels: {labels_path}")
            logging.info(f"  - Predictions: {pred_path}")
            logging.info(f"  - Astro IDs: {astro_ids_path}")
            logging.info(f"  - Dash results: {results_path}")
            logging.info(f"  - Detailed predictions: {detailed_pred_path}")

        # Save all metrics
        evaluation.save_metrics(all_metrics, eval_dir)
        logging.info(f"\n{'='*70}")
        logging.info(f"All metrics saved to {eval_dir}/metrics.json")
        logging.info(f"{'='*70}\n")


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    main()
