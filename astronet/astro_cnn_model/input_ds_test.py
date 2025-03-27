"""Unit tests for input_ds.py"""

import numpy as np
import tensorflow as tf
from absl.testing import absltest

from astronet.astro_cnn_model import input_ds
from astronet.util import configdict, example_util


class ExampleParserTest(absltest.TestCase):

  def test_extract_features(self):
    ex = tf.train.Example()
    example_util.set_float_feature(ex, "global_view", np.arange(10))
    example_util.set_float_feature(ex, "Transit_Depth", [1e6])
    example_util.set_float_feature(ex, "period", [25])
    example_util.set_float_feature(ex, "period_present", [1])
    serialized_example = ex.SerializeToString()
    del ex

    input_config = configdict.ConfigDict({
        "features": {
            # Time series feature.
            "global_view": {
                "shape": [10],
                "is_time_series": True,
            },
            # Feature with log scaling.
            "Transit_Depth": {
                "shape": [1],
                "is_time_series": False,
                "scale": "log",
                "min_val": 0,
                "max_val": 1e10,
            },
            # Feature with norm scaling.
            "period": {
                "shape": [1],
                "is_time_series": False,
                "scale": "norm",
                "mean": 5,
                "std": 2,
            },
            # Feature with no scaling.
            "period_present": {
                "shape": [1],
                "is_time_series": False,
            },
        }
    })

    parser = input_ds.ExampleParser(input_config)
    features = parser(serialized_example)
    np.testing.assert_almost_equal(features.pop("global_view"), np.arange(10))
    np.testing.assert_almost_equal(
        features.pop("Transit_Depth"),
        np.log(1e6 + 1) / np.log(1e10))
    np.testing.assert_almost_equal(features.pop("period"), 10)
    np.testing.assert_almost_equal(features.pop("period_present"), 1)
    self.assertEmpty(features)

  def test_extract_labels_multiclass_binary(self):
    ex = tf.train.Example()
    example_util.set_int64_feature(ex, "a", [0])
    example_util.set_int64_feature(ex, "b", [3])
    example_util.set_int64_feature(ex, "c", [1])
    example_util.set_int64_feature(ex, "d", [0])
    serialized_example = ex.SerializeToString()
    del ex

    # Multi-class binary, uncertainty_weight=True, primary class.
    input_config = configdict.ConfigDict({
        "label_columns": ["a", "b", "c", "d"],
        "exclusive_labels": False,
        "uncertainty_weight": True,
        "non_primary_downweight_factor": 2.0,
        "primary_class": 1,
        "features": {}
    })
    parser = input_ds.ExampleParser(input_config, include_labels=True)
    features, labels, weight = parser(serialized_example)
    self.assertEmpty(features)
    np.testing.assert_almost_equal(labels, [0, 1, 1, 0])
    np.testing.assert_almost_equal(weight, 0.75)

    # Multi-class binary, uncertainty_weight=True, non-primary class.
    input_config.primary_class = 0
    parser = input_ds.ExampleParser(input_config, include_labels=True)
    features, labels, weight = parser(serialized_example)
    self.assertEmpty(features)
    np.testing.assert_almost_equal(labels, [0, 1, 1, 0])
    np.testing.assert_almost_equal(weight, 0.375)

    # Multi-class binary, uncertainty_weight=False, non-primary class.
    input_config.uncertainty_weight = False
    parser = input_ds.ExampleParser(input_config, include_labels=True)
    features, labels, weight = parser(serialized_example)
    self.assertEmpty(features)
    np.testing.assert_almost_equal(labels, [0, 1, 1, 0])
    np.testing.assert_almost_equal(weight, 0.5)

    # Multi-class binary, uncertainty_weight=False, no downweighting.
    input_config.non_primary_downweight_factor = 0
    parser = input_ds.ExampleParser(input_config, include_labels=True)
    features, labels, weight = parser(serialized_example)
    self.assertEmpty(features)
    np.testing.assert_almost_equal(labels, [0, 1, 1, 0])
    np.testing.assert_almost_equal(weight, 1)

  def test_extract_labels_single_binary(self):
    ex = tf.train.Example()
    example_util.set_int64_feature(ex, "a", [0])
    example_util.set_int64_feature(ex, "b", [3])
    example_util.set_int64_feature(ex, "c", [1])
    example_util.set_int64_feature(ex, "d", [0])
    serialized_example = ex.SerializeToString()
    del ex

    # Single-class binary, positive_example.
    input_config = configdict.ConfigDict({
        "label_columns": ["b"],
        "exclusive_labels": False,
        "uncertainty_weight": False,
        "non_primary_downweight_factor": 2.0,
        "features": {}
    })
    parser = input_ds.ExampleParser(input_config, include_labels=True)
    features, labels, weight = parser(serialized_example)
    self.assertEmpty(features)
    np.testing.assert_almost_equal(labels, 1)
    np.testing.assert_almost_equal(weight, 1)

    # Single-class binary, negative example.
    input_config.label_columns = ["a"]
    parser = input_ds.ExampleParser(input_config, include_labels=True)
    features, labels, weight = parser(serialized_example)
    self.assertEmpty(features)
    np.testing.assert_almost_equal(labels, 0)
    np.testing.assert_almost_equal(weight, 0.5)

    # Single-class binary, negative example, no downweighting.
    input_config.label_columns = ["a"]
    parser = input_ds.ExampleParser(input_config, include_labels=True)
    features, labels, weight = parser(serialized_example)
    self.assertEmpty(features)
    np.testing.assert_almost_equal(labels, 0)
    np.testing.assert_almost_equal(weight, 0.5)

  def test_extract_labels_multiclass_categorical(self):
    ex = tf.train.Example()
    example_util.set_int64_feature(ex, "a", [0])
    example_util.set_int64_feature(ex, "b", [3])
    example_util.set_int64_feature(ex, "c", [1])
    example_util.set_int64_feature(ex, "d", [0])
    serialized_example = ex.SerializeToString()
    del ex

    # Multi-class categorical, uncertainty_weight=True, primary class.
    input_config = configdict.ConfigDict({
        "label_columns": ["a", "b", "c", "d"],
        "exclusive_labels": True,
        "uncertainty_weight": True,
        "non_primary_downweight_factor": 2.0,
        "primary_class": 1,
        "features": {}
    })
    parser = input_ds.ExampleParser(input_config, include_labels=True)
    features, labels, weight = parser(serialized_example)
    self.assertEmpty(features)
    np.testing.assert_almost_equal(labels, [0, 0.75, 0.25, 0])
    np.testing.assert_almost_equal(weight, 0.75)

    # Multi-class categorical, uncertainty_weight=True, non-primary class.
    input_config.primary_class = 0
    parser = input_ds.ExampleParser(input_config, include_labels=True)
    features, labels, weight = parser(serialized_example)
    self.assertEmpty(features)
    np.testing.assert_almost_equal(labels, [0, 0.75, 0.25, 0])
    np.testing.assert_almost_equal(weight, 0.375)

    # Multi-class categorical, uncertainty_weight=False, non-primary class.
    input_config.uncertainty_weight = False
    parser = input_ds.ExampleParser(input_config, include_labels=True)
    features, labels, weight = parser(serialized_example)
    self.assertEmpty(features)
    np.testing.assert_almost_equal(labels, [0, 0.75, 0.25, 0])
    np.testing.assert_almost_equal(weight, 0.5)

    # Multi-class categorical, uncertainty_weight=False, no downweighting.
    input_config.non_primary_downweight_factor = 0
    parser = input_ds.ExampleParser(input_config, include_labels=True)
    features, labels, weight = parser(serialized_example)
    self.assertEmpty(features)
    np.testing.assert_almost_equal(labels, [0, 0.75, 0.25, 0])
    np.testing.assert_almost_equal(weight, 1.0)

  def test_extract_identifiers(self):
    ex = tf.train.Example()
    example_util.set_float_feature(ex, "global_view", np.arange(10))
    example_util.set_int64_feature(ex, "astro_id", [12345])
    serialized_example = ex.SerializeToString()
    del ex

    input_config = configdict.ConfigDict({
        "features": {
            "global_view": {
                "shape": [10],
                "is_time_series": True,
            },
        }
    })

    parser = input_ds.ExampleParser(input_config, include_identifiers=True)
    features, identifiers = parser(serialized_example)
    np.testing.assert_almost_equal(features.pop("global_view"), np.arange(10))
    self.assertEmpty(features)
    self.assertEqual(identifiers, 12345)


class TimeSeriesRandomReverserTest(absltest.TestCase):

  @property
  def feature_config(self):
    return configdict.ConfigDict({
        # Time series features.
        "time_series_1": {
            "shape": [10],
            "is_time_series": True,
        },
        "time_series_2": {
            "shape": [2, 10],
            "is_time_series": True,
        },
        # Non-time series features.
        "non_time_series_1": {
            "shape": [5],
            "is_time_series": False,
        },
        "non_time_series_2": {
            "shape": [1],
            "is_time_series": False,
        },
    })

  @property
  def input_features(self):
    return {
        "time_series_1": np.arange(10),
        "time_series_2": np.stack([np.arange(10), 5 * np.arange(10)]),
        "non_time_series_1": np.arange(5),
        "non_time_series_2": 100.0,
    }

  def test_no_reverse(self):
    # Probability 0: no-op.
    reverser = input_ds.TimeSeriesRandomReverser(self.feature_config, prob=0.0)
    features = reverser(self.input_features)
    np.testing.assert_almost_equal(features.pop("time_series_1"), np.arange(10))
    np.testing.assert_almost_equal(
        features.pop("time_series_2"),
        np.stack([np.arange(10), 5 * np.arange(10)]))
    np.testing.assert_almost_equal(
        features.pop("non_time_series_1"), np.arange(5))
    np.testing.assert_almost_equal(features.pop("non_time_series_2"), 100)
    self.assertEmpty(features)

  def test_reverse(self):
    # Probability 1: time series are reversed.
    reverser = input_ds.TimeSeriesRandomReverser(self.feature_config, prob=1.0)

    labels = np.array([1, 2, 3], dtype=float)
    weight = 0.5
    inputs = (self.input_features, labels, weight)
    features, labels, weight = reverser(inputs)
    np.testing.assert_almost_equal(
        features.pop("time_series_1"), np.flip(np.arange(10)))
    np.testing.assert_almost_equal(
        features.pop("time_series_2"),
        np.stack([np.flip(np.arange(10)), 5 * np.flip(np.arange(10))]))
    np.testing.assert_almost_equal(
        features.pop("non_time_series_1"), np.arange(5))
    np.testing.assert_almost_equal(features.pop("non_time_series_2"), 100)
    self.assertEmpty(features)

    np.testing.assert_almost_equal(labels, [1, 2, 3])
    np.testing.assert_almost_equal(weight, 0.5)


if __name__ == "__main__":
  absltest.main()
