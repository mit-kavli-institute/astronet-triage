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

import tensorflow as tf

from astronet.astro_cnn_model import base


class AstroCNNModelVetting(tf.keras.Model):
  """A convolutional model for classifying light curves for TESS vetting."""

  def __init__(self, config, triage_model):
    super(AstroCNNModelVetting, self).__init__()
    # TODO(cshallue): use config_util.validate(), but make sure to validate both
    # the triage and vetting configs.
    self.config = config

    self.triage_model = triage_model
    if not config.vetting_hparams.use_preds_layer:
      self.triage_model.make_embeds_only()
    self.ts_blocks = base.create_ts_blocks(config.vetting_hparams)
    self.final = base.build_final_fc_layers(config.inputs,
                                            config.vetting_hparams)

  def call(self, inputs, training=None):
    ts_inputs, aux_inputs = base.unpack_inputs(inputs, self.config.hparams)
    y = []
    for k in sorted(ts_inputs.keys()):
      v = ts_inputs[k]
      y.append(base.apply_block(self.ts_blocks[k], v, training))
    y.extend([aux_inputs[k] for k in sorted(aux_inputs.keys())])
    y.append(self.triage_model(inputs, training=training))  # Triage embedding.
    y = base.apply_block(self.final, y, training)

    return y
