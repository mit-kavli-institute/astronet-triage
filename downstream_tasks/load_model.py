"""Functions for loading trained AstroNet models."""

import os
import tensorflow as tf
from absl import logging

from astronet import models
from astronet.util import config_util
from astronet.util.configdict import ConfigDict


def load_model_from_checkpoint(model_dir: str, compile_model: bool = True) -> tuple[tf.keras.Model, ConfigDict]:
    """
    Load a trained model from a checkpoint directory.

    Args:
        model_dir: Path to directory containing model checkpoint. Must contain:
            - config.json: Model configuration
            - train_flags.json: Training flags (contains model name)
            - model.weights.h5 or saved_model.pb: Model weights (either format)
        compile_model: Whether to compile the model after loading. Defaults to True.

    Returns:
        A tuple (model, config) where:
            - model: The loaded Keras model
            - config: The model configuration as a ConfigDict

    Example:
        >>> model, config = load_model_from_checkpoint("/path/to/model/checkpoint")
        >>> predictions = model.predict(dataset)
    """
    # Validate model directory
    if not os.path.isdir(model_dir):
        raise ValueError(f"Model directory does not exist: {model_dir}")

    config_path = os.path.join(model_dir, "config.json")
    train_flags_path = os.path.join(model_dir, "train_flags.json")

    if not os.path.isfile(config_path):
        raise ValueError(f"config.json not found in {model_dir}")
    if not os.path.isfile(train_flags_path):
        raise ValueError(f"train_flags.json not found in {model_dir}")

    # Load configuration files
    logging.info(f"Loading config from {config_path}")
    config = config_util.load_config(config_path)

    logging.info(f"Loading train flags from {train_flags_path}")
    train_flags = config_util.load_config(train_flags_path)

    # Get model name from train flags
    model_name = train_flags.get("model")
    if not model_name:
        raise ValueError(f"train_flags.json must contain 'model' field")

    # Load the model
    logging.info(f"Loading model '{model_name}' from {model_dir}")
    model = models.load_model(model_name, model_dir)

    # Optionally compile the model (needed for evaluation/prediction)
    if compile_model:
        logging.info("Compiling model...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
            ],
        )

    logging.info("Model loaded successfully")
    return model, config
