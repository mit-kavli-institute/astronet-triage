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

from astronet.astro_cnn_model import astro_cnn_model, base


class AstroCNNModelVetting(tf.keras.Model):
  """A convolutional model for classifying light curves for TESS vetting."""

  def __init__(self, config, triage_model):
    super(AstroCNNModelVetting, self).__init__()
    self.config = config

    embeds_only = not config.vetting_hparams.use_preds_layer
    self.triage_model = astro_cnn_model.AstroCNNModel(
        config, triage_model, embeds_only=embeds_only)
    self.ts_blocks = base.create_ts_blocks(config.hparams)
    self.final = base.build_final_fc_layers(config.inputs,
                                            config.vetting_hparams)

  def call(self, inputs, training=None):

    def is_vetting_input(k):
      if k.endswith('_present'):
        k = k[:-len('_present')]
      # The dataset makes them lowercase. We should change things to lowercase
      # throughout.
      if k not in self.config.inputs.features:
        k, = tuple(
            ck for ck in self.config.inputs.features.keys() if ck.lower() == k)
      return self.config.inputs.features[k].get('vetting_only', False)

    triage_inputs = {k: v for k, v in inputs.items() if not is_vetting_input(k)}
    vetting_inputs = {k: v for k, v in inputs.items() if k in self.ts_blocks}

    triage_embedding = self.triage_model(triage_inputs, training=training)

    ts_inputs = {}
    aux_inputs = {}
    for k, v in vetting_inputs.items():
      if k in self.config.vetting_hparams.time_series_hidden:
        c = self.config.vetting_hparams.time_series_hidden[k]
        chans = [v]
        for extra in getattr(c, 'extra_channels', []):
          chans.append(inputs[extra])
        if getattr(c, 'multichannel', False):
          ts_inputs[k] = tf.concat(chans, axis=-1)
        else:
          ts_inputs[k] = tf.stack(chans, axis=-1)
      elif k in self.config.hparams.aux_inputs:
        aux_inputs[k] = v

    y = [triage_embedding]
    for k in sorted(ts_inputs.keys()):
      v = ts_inputs[k]
      y.append(base.apply_block(self.ts_blocks[k], v, training))
    y.extend([aux_inputs[k] for k in sorted(aux_inputs.keys())])
    y = base.apply_block(self.final, y, training)

    return y
