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
      logging.warning("Both 'include_labels' and 'include_identifiers' are set. This is discouraged and may cause unexpected behavior.")
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

    if self.include_labels and self.include_identifiers:
       labels, weight = self._extract_labels(parsed_features)
       identifiers = parsed_features.pop("astro_id")
       return features, labels, weight, identifiers
    elif self.include_labels:
        labels, weight = self._extract_labels(parsed_features)
        return features, labels, weight
    elif self.include_identifiers:
        identifiers = parsed_features.pop("astro_id")
        return features, identifiers
    else:
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
                  use_cache=False,
                  apply_data_augmentation=False,
                  exclude_astro_ids=None):
  """Builds a Tensorflow Dataset from TFrecord files."""
  filenames = tf.io.gfile.glob(file_pattern)
  if not filenames:
    raise ValueError(f"Found no files matching '{file_pattern}'")
  ds = tf.data.Dataset.from_tensor_slices(filenames)
  if shuffle_filenames:
    ds = ds.shuffle(ds.cardinality())
  ds = ds.flat_map(tf.data.TFRecordDataset)

  # If we need to filter, always parse identifiers
  parse_identifiers = include_identifiers or (exclude_astro_ids is not None)
  example_parser = ExampleParser(input_config, include_labels, parse_identifiers)
  ds = ds.map(example_parser)

  # Filtering step
  if exclude_astro_ids is not None:
      exclude_astro_ids_tf = tf.constant(list(exclude_astro_ids), dtype=tf.int64)

      def filter_fn(*args):
          astro_id = args[-1]
          is_excluded = tf.reduce_any(tf.equal(astro_id, exclude_astro_ids_tf))
          tf.cond(
              is_excluded,
              lambda: tf.print("Excluding astro_id:", astro_id),
              lambda: tf.no_op()
          )
          return ~is_excluded

      ds = ds.filter(filter_fn)
      logging.info(f"Filtered out {len(exclude_astro_ids)} astro_ids")

      # If identifiers were not originally requested, remove them
      if not include_identifiers:
          def strip_identifiers(*args):
              return args[:-1]  # drop the last element (astro_id)
          ds = ds.map(strip_identifiers)

  if use_cache:
    # Cache the dataset in memory to avoid re-reading it over the network.
    ds = ds.cache()
    if shuffle_filenames:
      logging.warning("Both shuffle_filenames and use_cache are set to true. "
                      "Filenames will only be shuffled once, not each epoch.")

  if apply_data_augmentation and input_config.get("random_reverse_time_series"):
    ds = ds.map(TimeSeriesRandomReverser(input_config.features, prob=0.5))

  if shuffle_values_buffer > 0:
    ds = ds.shuffle(shuffle_values_buffer)
  if repeat != 1:
    # Calling repeat() after shuffle() ensures that examples are shuffled
    # within each epoch, but not between epochs.
    ds = ds.repeat(repeat)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(10)

  return ds


def build_train_dataset(file_pattern,
                        input_config,
                        batch_size,
                        shuffle_values_buffer=2500,
                        exclude_astro_ids=None):
  """Builds a dataset for training."""
  return build_dataset(
      file_pattern,
      input_config,
      batch_size,
      shuffle_values_buffer=shuffle_values_buffer,
      repeat=None,
      use_cache=True,
      apply_data_augmentation=True,
      exclude_astro_ids=exclude_astro_ids)


<<<<<<< HEAD
def build_eval_dataset(file_pattern, input_config, batch_size,include_identifiers=False,include_labels=True):
  """Builds a dataset for evaluation."""
  return build_dataset(file_pattern, input_config, batch_size, use_cache=False,include_identifiers=include_identifiers,include_labels=include_labels)
=======
def build_eval_dataset(file_pattern,
                       input_config,
                       batch_size,
                       include_identifiers=False,
                       include_labels=True):
  """Builds a dataset for evaluation."""
  return build_dataset(
      file_pattern,
      input_config,
      batch_size,
      include_identifiers=include_identifiers,
      include_labels=include_labels,
      use_cache=False)
>>>>>>> origin/main
