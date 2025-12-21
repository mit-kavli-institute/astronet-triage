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

    label_scheme = self.config.get("label_scheme", "binary")
    is_single_label = len(self.config.label_columns) == 1
    if is_single_label and label_scheme != "binary":
      raise ValueError("Single label requires label_scheme=binary")
    if label_scheme == "binary":
      # Each element of the label vector is 0 or 1 independently.
      labels = tf.squeeze(tf.minimum(label_features, 1))
    elif label_scheme == "distribution":
      # Label vector is a probability distribution that sums to 1.
      labels = label_features / tf.reduce_sum(label_features)
    elif label_scheme == "maximum":
      # Set the maximum element(s) to 1 and all others to 0.
      labels = tf.floor(label_features / tf.reduce_max(label_features))
      # Account for ties.
      labels /= tf.reduce_sum(labels)
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


class TimeSeriesRandomReverser:

  def __init__(self, feature_config, prob):
    self.feature_config = feature_config
    self.prob = prob

  def __call__(self, *args):
    features = args[0]
    aux_inputs = args[1:]

    if tf.random.uniform(shape=[]) < self.prob:
      new_features = {}
      for name, value in features.items():
        if self.feature_config[name]["is_time_series"]:
          value = tf.reverse(value, axis=[-1])
        new_features[name] = value
      features = new_features

    if aux_inputs:
      return (features,) + aux_inputs
    return features


def build_dataset(file_pattern,
                  input_config,
                  batch_size,
                  include_labels=True,
                  shuffle_filenames=False,
                  shuffle_values_buffer=0,
                  repeat=1,
                  include_identifiers=False,
                  weight_table=None,
                  upsample_table=None):

    def parse_example(serialized_example):
        """Parses a single tf.Example into feature and label tensors."""
        
        data_fields = {
            feature_name: tf.io.FixedLenFeature(feature.shape, tf.float32)
            for feature_name, feature in input_config.features.items()
        }
        if include_labels:
            for n in input_config.label_columns:
                data_fields[n] = tf.io.FixedLenFeature([], tf.int64)
        if include_identifiers:
            assert "astro_id" not in data_fields
            data_fields["astro_id"] = tf.io.FixedLenFeature([], tf.int64)


        parsed_features = tf.io.parse_single_example(serialized_example, features=data_fields)


        if include_labels:
            label_features = [parsed_features.pop(name) for name in input_config.label_columns]
            labels = tf.stack(label_features)
            labels_f = tf.cast(labels, tf.float32)
            labels = tf.cast(tf.minimum(labels, 1), tf.float32)

            weights = tf.reduce_max(labels_f) / tf.maximum(tf.reduce_sum(labels_f), 1.0)
            if labels[input_config.primary_class] < 1:
                weights /= 2.0

            if (weight_table is not None) and (astro_id is not None):
                # lookup returns -1.0 for "no override"
                extra_weight = weight_table.lookup(astro_id)
                # if extra_weight > 0, use it; otherwise fall back to base_weight
                weights = tf.where(extra_weight > 0.0, extra_weight, base_weight)
            else:
                weights = base_weight

        if include_identifiers:
            identifiers = parsed_features.pop("astro_id")
        else:
            assert "astro_id" not in parsed_features

        features = {}
        assert set(parsed_features.keys()) == set(input_config.features.keys())
        for name, value in parsed_features.items():
            cfg = input_config.features[name]
            if not cfg.is_time_series:
                if getattr(cfg, "scale", None) == "log":
                    value = tf.cast(value, tf.float64)
                    value = tf.maximum(value, cfg.min_val)
                    value = tf.minimum(value, cfg.max_val)
                    value = value - cfg.min_val + 1
                    value = tf.math.log(value) / tf.math.log(tf.constant(cfg.max_val, tf.float64))
                    value = tf.cast(value, tf.float32)
                elif getattr(cfg, "scale", None) == "norm":
                    value = (value - cfg["mean"]) / cfg["std"]
            features[name] = value
        
        if include_labels:
            if (upsample_table is not None) and (astro_id is not None):
                up_factor = upsample_table.lookup(astro_id)
            else:
                up_factor = tf.constant(1, tf.int32)
            return features, labels, weights, up_factor
        elif include_identifiers:
            return features, identifiers
        return features


    filenames = tf.constant(tf.io.gfile.glob(file_pattern), dtype=tf.string)
    ds = tf.data.Dataset.from_tensor_slices(filenames)
    ds = ds.flat_map(tf.data.TFRecordDataset)
    ds = ds.map(parse_example)

    if include_labels and (upsample_table is not None):
        # expand each example into `up_factor` copies
        def expand(features, labels, weights, up_factor):
            single = tf.data.Dataset.from_tensors((features, labels, weights))
            # ensure up_factor >= 1
            up_factor = tf.maximum(up_factor, 1)
            return single.repeat(up_factor)

        ds = ds.flat_map(expand)

    if repeat != 1:
        ds = ds.cache()

    if shuffle_values_buffer > 0:
        ds = ds.shuffle(shuffle_values_buffer)
    if repeat != 1:
        ds = ds.repeat(repeat)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(10)

    return ds

def build_train_dataset(file_pattern,
                        input_config,
                        batch_size,
                        shuffle_values_buffer=2500):
  """Builds a dataset for training."""
  return build_dataset(
      file_pattern,
      input_config,
      batch_size,
      shuffle_values_buffer=shuffle_values_buffer,
      repeat=None,
      use_cache=True,
      apply_data_augmentation=True)

def build_eval_dataset(file_pattern, input_config, batch_size, include_identifiers, include_labels):
  """Builds a dataset for evaluation."""
  return build_dataset(file_pattern, input_config, batch_size, include_identifiers=include_identifiers, include_labels=include_labels)
