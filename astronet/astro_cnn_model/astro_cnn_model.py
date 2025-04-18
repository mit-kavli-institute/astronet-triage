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

  def __init__(self, config, extra_embedding_layer=None):
    super().__init__()
    self.config = config_util.validate(config)
    self.extra_embedding_layer = extra_embedding_layer
    self.ts_blocks = base.TimeSeriesConvBlocks(
        config.hparams.time_series_hidden)
    self.dense_block = base.DenseBlock(
        num_layers=config.hparams.num_pre_logits_hidden_layers,
        layer_size=config.hparams.pre_logits_hidden_layer_size,
        use_batch_norm=config.hparams.use_batch_norm,
        dropout_rate=config.hparams.pre_logits_dropout_rate)
    self.output_layer = base.OutputLayer(
        n_labels=len(config.inputs.label_columns),
        exclusive_labels=config.inputs.get('exclusive_labels'))
    self.build()  # We know the input shapes so we might as well build now.

  def build(self, input_shape=None):
    del input_shape  # Keras default argument; unused here.
    input_layer = {}
    for feature_name, feature_spec in self.config.inputs.features.items():
      input_layer[feature_name] = tf.keras.Input(shape=feature_spec.shape)
    self.call(input_layer, training=True)  # Builds all the layers.
    self.built = True

  def call(self, inputs, training):
    y = self.ts_blocks(inputs, training)
    if self.extra_embedding_layer is not None:
      y.extend(self.extra_embedding_layer(inputs, training))
    y.extend([inputs[key] for key in sorted(self.config.hparams.aux_inputs)])
    y = self.dense_block(tf.concat(y, axis=-1), training)
    return self.output_layer(y)
