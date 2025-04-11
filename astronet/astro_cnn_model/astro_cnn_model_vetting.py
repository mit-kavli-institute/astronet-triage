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
"""A convolutional model for classifying light curves for TESS vetting."""

from astronet.astro_cnn_model import astro_cnn_model, base


class AstroCNNModelVetting(astro_cnn_model.AstroCNNModel):
  """A convolutional model for classifying light curves for TESS vetting."""

  def __init__(self, config, triage_model):
    super().__init__(config)
    self.triage_model = triage_model

  def call(self, inputs, training=None):
    y = self.triage_model.apply_ts_blocks(inputs, training)
    y.extend(self.apply_ts_blocks(inputs, training))
    aux_inputs = base.unpack_aux_features(inputs, self.config.hparams)
    y.extend([aux_inputs[k] for k in sorted(aux_inputs.keys())])
    y = base.apply_block(self.final, y, training)
    return y
