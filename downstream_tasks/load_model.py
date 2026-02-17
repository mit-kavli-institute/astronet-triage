"""Functions for loading trained AstroNet models."""

import os
import numpy as np
import tensorflow as tf
from absl import logging

from astronet import models
from astronet.util import config_util
from astronet.util.configdict import ConfigDict
from astronet.astro_cnn_model import input_ds

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


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    logging.set_stderrthreshold(logging.INFO)

    path = "/pdo/astronet-data/models/vetting/experimental/pablomer/dec2025_cad_scat_v5_duration24/20251217/pablomer-2k-nopretrained/AstroCNNModelVetting_pablomer_20251217_134151/" #512
    # path = '/pdo/astronet-data/models/vetting/experimental/pablomer/dec2025_cad_scat_v5_duration24/20260203/pablomer-2k-nopretrained-z_dim32/AstroCNNModelVetting_pablomer_20260203_193940/' #32
    # path = '/pdo/astronet-data/models/vetting/experimental/pablomer/dec2025_cad_scat_v5_duration24/20260203/pablomer-2k-nopretrained-z_dim64/AstroCNNModelVetting_pablomer_20260203_194912/' #64


    z_dim = 512
    model, config = load_model_from_checkpoint(path)
    # print(model)
    # print(config)
    print('Loaded model from ', path)

    # files = "/pdo/astronet-data/data/tfrecords/sector-82-scatter/*"
    sectors = range(73, 84)  # 84 is exclusive, so this gives 73-83
    files = [f"/pdo/astronet-data/data/tfrecords/sector-{s}-scatter/*" for s in sectors]
    # Use build_eval_dataset for inference (matches predict.py pattern)
    # - No shuffling (shuffle_values_buffer=0 by default)
    # - No data augmentation
    # - No repeat (single pass through data)
    # Set include_identifiers=True if you need astro_ids, include_labels=False for inference
    ds = input_ds.build_eval_dataset(
        file_pattern=files,
        input_config=config.inputs,
        batch_size=config.hparams.batch_size,  # Use config batch size instead of hardcoded
        include_identifiers=True,  # Include astro_ids to track predictions
        include_labels=False,  # Not needed for inference
    )

    # Collect all embeddings and astro_ids
    logging.info("Extracting embeddings for all examples...")
    all_embeddings = []
    all_astro_ids = []

    batch_count = 0
    for batch in ds:
        # When include_identifiers=True and include_labels=False, batch is (features, astro_id)
        # features is a dict of tensors, astro_id is a tensor
        features, astro_ids = batch

        # Print batch info for first batch only
        if batch_count == 0:
            logging.info('Batch structure:')
            logging.info(f'  Features (dict keys): {list(features.keys())}')
            logging.info(f'  Astro IDs shape: {astro_ids.shape}')
            first_feature_key = list(features.keys())[0]
            logging.info(f'  First feature "{first_feature_key}" shape: {features[first_feature_key].shape}')

        # Pass features dict to get_embeddings (not the whole batch tuple)
        embeddings = model.get_embeddings(features, training=False)

        # Store embeddings and astro_ids (convert to numpy for storage)
        all_embeddings.append(embeddings.numpy())
        all_astro_ids.append(astro_ids.numpy())

        batch_count += 1
        if batch_count % 10 == 0:
            logging.info(f"Processed {batch_count} batches...")

    # Concatenate all batches
    logging.info(f"Concatenating {batch_count} batches...")
    embeddings_array = np.concatenate(all_embeddings, axis=0)
    astro_ids_array = np.concatenate(all_astro_ids, axis=0)

    logging.info(f"Extracted embeddings for {len(astro_ids_array)} examples")
    logging.info(f"Embeddings shape: {embeddings_array.shape}")
    logging.info(f"Astro IDs shape: {astro_ids_array.shape}")

    # Save embeddings to file
    output_path = f"embeddings_zdim{z_dim}.npz"
    logging.info(f"Saving embeddings to {output_path}")
    np.savez(output_path, embeddings=embeddings_array, astro_ids=astro_ids_array)
    logging.info(f"✅ Saved embeddings for {len(astro_ids_array)} examples to {output_path}")
