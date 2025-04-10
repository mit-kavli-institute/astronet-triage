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
"""Utility functions for configurations."""

import json
import os.path
from contextlib import suppress

import tensorflow as tf
from absl import logging

from astronet.util import configdict


def validate(config):
  """Validates a configuration."""
  # Make sure all model features are present in the input config.
  ts_hidden_config = config.hparams.time_series_hidden
  aux_inputs = config.hparams.aux_inputs
  feature_names = (set(ts_hidden_config) | set(aux_inputs))
  input_features = set(config.inputs.features)
  for name in feature_names:
    if name not in input_features:
      raise ValueError(
          f"Feature '{name}' is present in hparams but not the input config.")

  return config


def add_nested_param(config, path, value, overwrite=False):
  """Adds a nested parameter to config.
  
  Args:
    config: Configuration to add the parameter to.
    path: Nested parameter specification of the form 'a.b.c'
    value: Parameter value
    """
  if not isinstance(config, dict):
    raise ValueError(f"Config has type {type(config)}")
  if "." not in path:
    # Adding a leaf node.
    if path in config and not overwrite:
      raise ValueError(f"'{path}' already exists and overwrite=False")
    config[path] = value
  else:
    # Add nested parameter.
    key, subpath = path.split(".", 1)
    if key not in config:
      config[key] = {}
    add_nested_param(config[key], subpath, value, overwrite)


def unflatten(flat_config):
  """Unflattens a flattened config.
  
  E.g., {"a.b.c": 123, "a.d": 25} becomes {"a": {"b": {"c": 123}}, "d": 25}.
  """
  config = {}
  for key, value in flat_config.items():
    add_nested_param(config, key, value)
  return config


def flatten(config):
  """Flattens a nested config."""
  flat_config = {}
  for key, subconfig_or_value in config.items():
    if isinstance(subconfig_or_value, dict):
      flat_subconfig = flatten(subconfig_or_value)
      for subkey, value in flat_subconfig.items():
        flat_config[f"{key}.{subkey}"] = value
    else:
      flat_config[key] = subconfig_or_value
  return flat_config


def update(base, source):
  """Replaces parameters from base with source.
  
  Args:
    base: Base configuration to modify.
    source: Configuration parameters to update in base. Parameters must already
      be in base.
  """
  if not (isinstance(base, dict) and isinstance(source, dict)):
    raise ValueError(f"base is '{type(base)}', source is '{type(source)}'")
  for key, value in source.items():
    if key not in base:
      raise KeyError(key)
    if isinstance(value, dict):
      update(base[key], value)
    else:
      base[key] = value


def _parse_override_value(value):
  if value[0] == "'" and value[-1] == "'":
    return value[1:-1]
  if value[0] == "[" and value[-1] == "]":
    return [_parse_override_value(v) for v in value[1:-1].split(";") if v]
  with suppress(ValueError):
    return int(value)
  with suppress(ValueError):
    return float(value)
  if value.lower() == "true":
    return True
  if value.lower() == "false":
    return False
  return value


def parse_config_str(config_str):
  """Parses a string specifying configuration options."""
  overrides = {}
  while config_str:
    key, remainder = config_str.split("=", 1)
    if remainder.startswith("'"):
      split_i = remainder.index("'", 1)  # Closing quote
      value = remainder[:split_i + 1]
      config_str = remainder[split_i + 2:]  # Also omit comma
    else:
      split = remainder.split(",", 1)
      value = split[0]
      config_str = split[1] if len(split) == 2 else ""
    overrides[key] = _parse_override_value(value)
  return unflatten(overrides)


def merge_configs(base, source):
  """Adds new parameters from source to base.
  
  Args:
    base: Base configuration to modify.
    source: Configuration parameters to add to base. Parameters already in base
      will be ignored.
  """
  if not (isinstance(base, dict) and isinstance(source, dict)):
    raise ValueError(f"base is '{type(base)}', source is '{type(source)}'")
  for key, value in source.items():
    if key not in base:
      base[key] = value
    elif isinstance(value, dict):
      merge_configs(base[key], value)


def config_filename(output_dir, basename="config"):
  """Returns the filepath of a configuration object."""
  return os.path.join(output_dir, f"{basename}.json")


def save_config(config, output_dir, basename="config"):
  """Writes a JSON-serializable configuration object.
  Args:
    config: A JSON-serializable object.
    output_dir: Destination directory.
    basename: Base name of the output JSON file.
  """
  if hasattr(config, "to_json") and callable(config.to_json):
    config_json = config.to_json(indent=2)
  else:
    config_json = json.dumps(config, indent=2)

  tf.io.gfile.makedirs(output_dir)
  with tf.io.gfile.GFile(config_filename(output_dir, basename), "w") as f:
    f.write(config_json)


def load_config(config_dir_or_filename, basename="config"):
  """Parses values from a JSON file.
  Args:
    config_dir_or_filename: Either the path to a JSON file or a directory
      containing a JSON file.
    basename: Base name of the JSON file.
  Returns:
    A dictionary; the parsed JSON.
  """
  if tf.io.gfile.isdir(config_dir_or_filename):
    filename = config_filename(config_dir_or_filename, basename)
  else:
    filename = config_dir_or_filename
  with tf.io.gfile.GFile(filename, "r") as f:
    config = configdict.ConfigDict(json.load(f))
  logging.info(f"Loaded config from {filename}")
  return config
