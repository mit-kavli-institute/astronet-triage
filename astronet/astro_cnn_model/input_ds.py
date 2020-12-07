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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import collections
import six
import tensorflow as tf


def build_dataset(file_pattern,
                  input_config,
                  batch_size,
                  include_labels=True,
                  shuffle_filenames=False,
                  shuffle_values_buffer=0,
                  repeat=1,
                  include_identifiers=False):

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
            assert "tic_id" not in data_fields
            data_fields["tic_id"] = tf.io.FixedLenFeature([], tf.int64)


        parsed_features = tf.io.parse_single_example(serialized_example, features=data_fields)


        if include_labels:
            labels = tf.stack(
                [parsed_features.pop(name) for name in input_config.label_columns])
            labels = tf.cast(tf.minimum(labels, 1), tf.float32)

            labels_f = tf.cast(labels, tf.float32)
            weights = tf.reduce_max(labels_f) / tf.maximum(tf.reduce_sum(labels_f), 1.0)
            if labels[input_config.primary_class] < 1:
                weights /= 2.0

        if include_identifiers:
            identifiers = parsed_features.pop("tic_id")

        features = {}
        assert set(parsed_features.keys()) == set(input_config.features.keys())
        assert not any(k.endswith("present") for k in input_config.features.keys())
        for name, value in parsed_features.items():
            cfg = input_config.features[name]
            if not cfg.is_time_series:
                use_value = True
                if cfg.has_nans:
                    if tf.math.is_nan(value):
                        use_value = False
                        mask = tf.zeros_like(value)
                    else:
                        mask = tf.ones_like(value)
                    features[f"{name}_present"] = mask
                if use_value:
                    if getattr(cfg, "log_scale", False):
                        value = tf.cast(value, tf.float64)
                        value = tf.maximum(value, cfg.min_val)
                        value = tf.minimum(value, cfg.max_val)
                        value = value - cfg.min_val + 1
                        value = tf.math.log(value) / tf.math.log(tf.constant(cfg.max_val, tf.float64))
                        value = tf.cast(value, tf.float32)
                    else:
                        value = (value - cfg["mean"]) / cfg["std"]
                else:
                    value = tf.zeros_like(value)
            features[name] = value
        if include_labels:
            return features, labels, weights
        elif include_identifiers:
            return features, identifiers
        return features


    filenames = tf.io.gfile.glob(file_pattern)
    ds = tf.data.Dataset.from_tensor_slices(filenames)
    ds = ds.flat_map(tf.data.TFRecordDataset)
    ds = ds.map(parse_example)
    if repeat != 1:
        ds = ds.cache()

    if shuffle_values_buffer > 0:
        ds = ds.shuffle(shuffle_values_buffer)
    if repeat != 1:
        ds = ds.repeat(repeat)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(10)

    return ds
