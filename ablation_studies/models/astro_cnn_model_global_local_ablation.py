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
"""Global+local-view-only ablation model for vetting."""

from astronet.astro_cnn_model.astro_cnn_model import AstroCNNModel


class AstroCNNModelGlobalLocalAblation(AstroCNNModel):
  """Ablation model restricted to global_view and local_view features."""

  _EXPECTED_TIME_SERIES = {"global_view", "local_view"}

  def __init__(self, config):
    feature_names = set(config.inputs.features.keys())
    if feature_names != self._EXPECTED_TIME_SERIES:
      raise ValueError(
          "AstroCNNModelGlobalLocalAblation expects exactly these features: "
          f"{sorted(self._EXPECTED_TIME_SERIES)}; got {sorted(feature_names)}")

    ts_block_names = set(config.hparams.time_series_hidden.keys())
    if ts_block_names != self._EXPECTED_TIME_SERIES:
      raise ValueError(
          "AstroCNNModelGlobalLocalAblation expects exactly these ts blocks: "
          f"{sorted(self._EXPECTED_TIME_SERIES)}; got {sorted(ts_block_names)}")

    if config.hparams.get("aux_inputs"):
      raise ValueError(
          "AstroCNNModelGlobalLocalAblation expects no aux_inputs, got: "
          f"{config.hparams.aux_inputs}")

    super().__init__(config)
