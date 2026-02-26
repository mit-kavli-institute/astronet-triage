# Copyright 2018 The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Library of AstroNet models and configurations."""

import os

import tensorflow as tf
from absl import logging

from astronet.astro_cnn_model import (astro_cnn_model, configurations,
                                      configurations_vetting)
from ablation_studies.models import (
    astro_cnn_model_global_local_ablation,
    astro_cnn_model_mixed_ablation,
    astro_cnn_model_scalar_ablation,
    configurations_vetting_global_local_ablation,
    configurations_vetting_mixed_ablation,
    configurations_vetting_scalar_ablation,
)
from astronet.util import config_util, configdict

# Filename used when saving model weights.
MODEL_WEIGHTS_FILENAME = "model.weights.h5"

# Dictionary of model name to (model_class, configuration_module).
_MODELS = {
    "AstroCNNModel": (astro_cnn_model.AstroCNNModel, configurations),
    "AstroCNNModelVetting":
        (astro_cnn_model.AstroCNNModel, configurations_vetting),
    "AstroCNNModelVettingGlobalLocalAblation":
        (
            astro_cnn_model_global_local_ablation.AstroCNNModelGlobalLocalAblation,
            configurations_vetting_global_local_ablation,
        ),
    "AstroCNNModelVettingScalarAblation":
        (
            astro_cnn_model_scalar_ablation.AstroCNNModelScalarAblation,
            configurations_vetting_scalar_ablation,
        ),
    "AstroCNNModelVettingMixedAblation":
        (
            astro_cnn_model_mixed_ablation.AstroCNNModelMixedAblation,
            configurations_vetting_mixed_ablation,
        ),
}


def get_model_class(model_name):
  """Looks up a model class by name.

  Args:
    model_name: Name of the model class.

  Returns:
    model_class: The requested model class.

  Raises:
    ValueError: If model_name is unrecognized.
  """
  if model_name not in _MODELS:
    raise ValueError(f"Unrecognized model name: {model_name}")

  return _MODELS[model_name][0]


def get_model_config(model_name, config_name):
  """Looks up a model configuration by name.

  Args:
    model_name: Name of the model class.
    config_name: Name of a configuration-builder function from the model's
        configurations module.

  Returns:
    model_class: The requested model class.
    config: The requested configuration.

  Raises:
    ValueError: If model_name or config_name is unrecognized.
  """
  if model_name not in _MODELS:
    raise ValueError(f"Unrecognized model name: {model_name}")

  config_module = _MODELS[model_name][1]
  try:
    config = getattr(config_module, config_name)()
    config = configdict.ConfigDict(config)
    return config
  except AttributeError as e:
    raise ValueError(
        f"Config name '{config_name}' not found in configuration module: "
        f"{config_module.__name__}") from e


def get_weights_filename(model_dir):
  return os.path.join(model_dir, MODEL_WEIGHTS_FILENAME)


def load_from_weights(model_name, model_dir):
  """Loads a model from a weights h5 file."""
  model_class = get_model_class(model_name)
  config = config_util.load_config(model_dir)

  model = model_class(config)
  weights_filename = get_weights_filename(model_dir)
  model.load_weights(weights_filename)
  logging.info(f"Loaded weights from {weights_filename}")

  return model


def load_model(model_name, model_dir, save_format="auto"):
  """Loads a model saved in either Keras or h5 format."""
  if save_format == "auto":
    weights_filename = get_weights_filename(model_dir)
    save_format = "h5" if os.path.exists(weights_filename) else "keras"

  if save_format == "keras":
    return tf.keras.models.load_model(model_dir)

  if save_format == "h5":
    return load_from_weights(model_name, model_dir)

  raise ValueError(save_format)


def save_model(model, model_dir, save_format="keras"):
  """Saves a model saved in either Keras or h5 format."""
  if save_format == "keras":
    model.save(model_dir)
  elif save_format == "h5":
    weights_filename = get_weights_filename(model_dir)
    model.save_weights(weights_filename)
  else:
    raise ValueError(save_format)
