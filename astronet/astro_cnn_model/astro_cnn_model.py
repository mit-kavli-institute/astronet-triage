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
from astronet.util import config_util


class AstroCNNModel(tf.keras.Model):
  """A convolutional model for classifying light curves."""

  def __init__(self, config):
    super().__init__()
    self.config = config_util.validate(config)
    self.embeds_only = False
    self.ts_blocks = base.create_ts_blocks(config.hparams)
    self.final = base.build_final_fc_layers(config.inputs, config.hparams)

  def make_embeds_only(self):
    """Removes the final output layer that generates probabilities.
    
    Although this could be implemented in the constructor, we generally only
    want to do this for a pretrained model, in which case we should load the
    full model first and then call this function.
    """
    if self.embeds_only:
      raise ValueError("Final embedding layer has already been removed")
    self.final = self.final[:-1]
    self.embeds_only = True

  def call(self, inputs, training=None):
    ts_inputs, aux_inputs = base.unpack_inputs(inputs, self.config.hparams)
    y = []
    for k in sorted(ts_inputs.keys()):
      v = ts_inputs[k]
      y.append(base.apply_block(self.ts_blocks[k], v, training))
    y.extend([aux_inputs[k] for k in sorted(aux_inputs.keys())])
    y = base.apply_block(self.final, y, training)

    return y
