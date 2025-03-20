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
"""A model for classifying light curves using a convolutional neural network.

See the base class (in astro_model.py) for a description of the general
framework of AstroModel and its subclasses.

The architecture of this model is:


                                     predictions
                                          ^
                                          |
                                       logits
                                          ^
                                          |
                                (fully connected layers)
                                          ^
                                          |
                                   pre_logits_concat
                                          ^
                                          |
                                    (concatenate)

              ^                           ^                          ^
              |                           |                          |
   (convolutional blocks 1)  (convolutional blocks 2)   ...          |
              ^                           ^                          |
              |                           |                          |
     time_series_feature_1     time_series_feature_2    ...     aux_features
"""

import tensorflow as tf

from astronet.astro_cnn_model import base


class AstroCNNModel(tf.keras.Model):
  """A convolutional model for classifying light curves."""

  def __init__(self, config, pretrain_model=None, embeds_only=False):
    super(AstroCNNModel, self).__init__()

    self.config = config
    self.embeds_only = embeds_only

    if pretrain_model is not None:
      self.ts_blocks = pretrain_model.ts_blocks
      if self.embeds_only:
        self.final = pretrain_model.final[:-1]
      else:
        self.final = pretrain_model.final
    else:
      self.ts_blocks = base.create_ts_blocks(config.hparams)
      self.final = base.build_final_fc_layers(config.inputs, config.hparams)

  def call(self, inputs, training=None):
    ts_inputs, aux_inputs = base.unpack_inputs(inputs, self.config.hparams)
    y = []
    for k in sorted(ts_inputs.keys()):
      v = ts_inputs[k]
      y.append(base.apply_block(self.ts_blocks[k], v, training))
    y.extend([aux_inputs[k] for k in sorted(aux_inputs.keys())])
    y = base.apply_block(self.final, y, training)

    return y
