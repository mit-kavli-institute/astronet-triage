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
"""Time-series-only (no scalar aux inputs) vetting ablation model."""

from astronet.astro_cnn_model.astro_cnn_model import AstroCNNModel


class AstroCNNModelScalarAblation(AstroCNNModel):
  """Ablation model that removes scalar auxiliary inputs."""

  def __init__(self, config):
    if config.hparams.get("aux_inputs"):
      raise ValueError(
          "AstroCNNModelScalarAblation expects no aux_inputs, got: "
          f"{config.hparams.aux_inputs}")

    for feature_name, feature_spec in config.inputs.features.items():
      if not feature_spec.get("is_time_series"):
        raise ValueError(
            "AstroCNNModelScalarAblation expects time-series-only features, "
            f"found non-time-series feature '{feature_name}'")

    super().__init__(config)
