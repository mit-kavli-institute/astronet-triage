"""Functions for evaluating a trained model."""

import numpy as np

from astronet.astro_cnn_model import input_ds


def evaluate_model(model, input_config, file_pattern, batch_size):
  """Calculates a trained model's metrics over a dataset."""
  ds = input_ds.build_dataset(
      file_pattern=file_pattern,
      input_config=input_config,
      batch_size=batch_size)
  results = model.evaluate(ds)
  return dict(zip(model.metrics_names, results))
