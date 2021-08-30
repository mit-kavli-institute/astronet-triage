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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os.path


from absl import logging
from astronet.util import configdict


import tensorflow as tf


def merge_configs(base, source):
    if not isinstance(source, dict) and isinstance(base, dict):
        raise ValueError(f'source is {type(soure)}, but base is {type(base)}')
    for k in source:
        if k not in base:
            base[k] = source[k]
        elif isinstance(source[k], dict):
            merge_configs(base[k], source[k])               


def config_file(output_dir):
  return os.path.join(output_dir, "config.json")


def load_config(output_dir):
  """Parses values from a JSON file.

  Args:
    json_file: The path to a JSON file.

  Returns:
    A dictionary; the parsed JSON.
  """
  with tf.io.gfile.GFile(config_file(output_dir), 'r') as f:
    return configdict.ConfigDict(json.loads(f.read()))


def log_and_save_config(config, output_dir):
  """Logs and writes a JSON-serializable configuration object.

  Args:
    config: A JSON-serializable object.
    output_dir: Destination directory.
  """
  if hasattr(config, "to_json") and callable(config.to_json):
    config_json = config.to_json(indent=2)
  else:
    config_json = json.dumps(config, indent=2)

  tf.io.gfile.makedirs(output_dir)
  with tf.io.gfile.GFile(config_file(output_dir), "w") as f:
    f.write(config_json)


def unflatten(flat_config):
  """Transforms a flat configuration dictionary into a nested dictionary.

  Example:
    {
      "a": 1,
      "b.c": 2,
      "b.d.e": 3,
      "b.d.f": 4,
    }
  would be transformed to:
    {
      "a": 1,
      "b": {
        "c": 2,
        "d": {
          "e": 3,
          "f": 4,
        }
      }
    }

  Args:
    flat_config: A dictionary with strings as keys where nested configuration
        parameters are represented with period-separated names.

  Returns:
    A dictionary nested according to the keys of the input dictionary.
  """
  config = {}
  for path, value in flat_config.items():
    path = path.split(".")
    final_key = path.pop()
    nested_config = config
    for key in path:
      nested_config = nested_config.setdefault(key, {})
    nested_config[final_key] = value
  return config
