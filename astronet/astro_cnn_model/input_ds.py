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
"""Functions to build an input pipeline that reads from TFRecord files."""

import tensorflow as tf
from absl import logging


class ExampleParser:
  """Function to parse a single tf.Example into feature and label tensors."""

  def __init__(self, config, include_labels=False, include_identifiers=False):
    if include_labels and include_identifiers:
      raise ValueError("Cannot set both include_labels and include_identifiers")

    self.config = config
    self.include_labels = include_labels
    self.include_identifiers = include_identifiers

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

  def _extract_labels(self, parsed_features):
    """Extracts labels from raw parsed features."""
    label_features = [
        parsed_features.pop(name) for name in self.config.label_columns
    ]
    label_features = tf.cast(tf.stack(label_features), tf.float32)

    is_single_label = len(self.config.label_columns) == 1
    exclusive_labels = self.config.get("exclusive_labels", False)
    if is_single_label or not exclusive_labels:
      # Each element of the label vector can be 0 or 1 independently.
      labels = tf.squeeze(tf.minimum(label_features, 1))
    else:
      # Label vector is a probability distribution that sums to 1.
      labels = label_features / tf.reduce_sum(label_features)

    weight = 1.0
    if self.config.get("uncertainty_weight"):
      if len(self.config.label_columns) == 1:
        raise ValueError("uncertainty_weight requires multiple labels")
      weight = tf.reduce_max(label_features) / tf.maximum(
          tf.reduce_sum(label_features), 1.0)

    downweight_factor = self.config.get("non_primary_downweight_factor", 2.0)
    primary_class = 0 if is_single_label else self.config.primary_class
    if downweight_factor and label_features[primary_class] < 1:
      weight /= downweight_factor

    return labels, weight

  def __call__(self, serialized_example):
    """Parses a single tf.Example into feature and label tensors."""
    data_fields = {
        feature_name: tf.io.FixedLenFeature(feature.shape, tf.float32)
        for feature_name, feature in self.config.features.items()
    }
    if self.include_labels:
      for name in self.config.label_columns:
        data_fields[name] = tf.io.FixedLenFeature([], tf.int64)
    if self.include_identifiers:
      assert "astro_id" not in data_fields
      data_fields["astro_id"] = tf.io.FixedLenFeature([], tf.int64)

    parsed_features = tf.io.parse_single_example(
        serialized_example, features=data_fields)

    features = self._extract_features(parsed_features)

    if self.include_labels:
      labels, weight = self._extract_labels(parsed_features)
      return features, labels, weight

    if self.include_identifiers:
      identifiers = parsed_features.pop("astro_id")
      return features, identifiers

    return features


def build_dataset(file_pattern,
                  input_config,
                  batch_size,
                  include_labels=True,
                  shuffle_filenames=False,
                  shuffle_values_buffer=0,
                  repeat=1,
                  include_identifiers=False,
                  use_cache=True):
  """Builds a Tensorflow Dataset from TFrecord files."""
  filenames = tf.io.gfile.glob(file_pattern)
  if not filenames:
    raise ValueError(f"Found no files matching '{file_pattern}'")
  ds = tf.data.Dataset.from_tensor_slices(filenames)
  if shuffle_filenames:
    ds = ds.shuffle(ds.cardinality())
  ds = ds.flat_map(tf.data.TFRecordDataset)
  example_parser = ExampleParser(input_config, include_labels,
                                 include_identifiers)
  ds = ds.map(example_parser)
  if use_cache:
    # Cache the dataset in memory to avoid re-reading it over the network.
    ds = ds.cache()
    if shuffle_filenames:
      logging.warning("Both shuffle_filenames and use_cache are set to true. "
                      "Filenames will only be shuffled once, not each epoch.")

  if shuffle_values_buffer > 0:
    ds = ds.shuffle(shuffle_values_buffer)
  if repeat != 1:
    # Calling repeat() after shuffle() ensures that examples are shuffled
    # within each epoch, but not between epochs.
    ds = ds.repeat(repeat)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(10)

  return ds
