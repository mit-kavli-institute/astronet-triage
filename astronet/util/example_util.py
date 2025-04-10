"""Helper functions for constructing tf.train.Example protos."""

import numpy as np


def set_float_feature(ex, name, value):
  """Sets the value of a float feature in a tf.train.Example proto."""
  if name in ex.features.feature:
    raise ValueError(f"Duplicate feature: {name}")
  if isinstance(value, np.ndarray):
    value = value.reshape((-1,))
  values = [float(v) for v in value]
  if any(np.isnan(values)):
    raise ValueError(f"NaNs in {name}")
  ex.features.feature[name].float_list.value.extend(values)


def set_bytes_feature(ex, name, value):
  """Sets the value of a bytes feature in a tf.train.Example proto."""
  if name in ex.features.feature:
    raise ValueError(f"Duplicate feature: {name}")
  ex.features.feature[name].bytes_list.value.extend(
      [str(v).encode("latin-1") for v in value])


def set_int64_feature(ex, name, value):
  """Sets the value of an int64 feature in a tf.train.Example proto."""
  if name in ex.features.feature:
    raise ValueError(f"Duplicate feature: {name}")
  ex.features.feature[name].int64_list.value.extend([int(v) for v in value])
