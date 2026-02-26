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
"""Mixed ablation model: global/local views + scalar aux inputs only."""

from astronet.astro_cnn_model.astro_cnn_model import AstroCNNModel


class AstroCNNModelMixedAblation(AstroCNNModel):
  """Ablation model with only global/local TS and scalar auxiliaries."""

  _EXPECTED_TIME_SERIES = {"global_view", "local_view"}

  def __init__(self, config):
    ts_block_names = set(config.hparams.time_series_hidden.keys())
    if ts_block_names != self._EXPECTED_TIME_SERIES:
      raise ValueError(
          "AstroCNNModelMixedAblation expects exactly these ts blocks: "
          f"{sorted(self._EXPECTED_TIME_SERIES)}; got {sorted(ts_block_names)}")

    feature_names = set(config.inputs.features.keys())
    aux_names = set(config.hparams.get("aux_inputs", []))
    expected_feature_names = self._EXPECTED_TIME_SERIES | aux_names
    if feature_names != expected_feature_names:
      raise ValueError(
          "AstroCNNModelMixedAblation feature mismatch. "
          f"Expected {sorted(expected_feature_names)}, got {sorted(feature_names)}")

    for name in self._EXPECTED_TIME_SERIES:
      if not config.inputs.features[name].get("is_time_series"):
        raise ValueError(f"Expected '{name}' to be time-series.")
    for name in aux_names:
      if config.inputs.features[name].get("is_time_series"):
        raise ValueError(f"Expected aux input '{name}' to be scalar/non-time-series.")

    super().__init__(config)
