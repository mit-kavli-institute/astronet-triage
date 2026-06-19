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
    self.ts_blocks = {
        name: base.ConvBlock(name, spec)
        for name, spec in config.hparams.time_series_hidden.items()
    }
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

  def unpack_ts_inputs(self, inputs):
    ts_inputs = {}
    for key, block_params in self.config.hparams.time_series_hidden.items():
      chans = [inputs[key]]
      for extra in block_params.get('extra_channels', []):
        chans.append(inputs[extra])
      if block_params.get('multichannel'):
        ts_inputs[key] = tf.concat(chans, axis=-1)
      else:
        ts_inputs[key] = tf.stack(chans, axis=-1)
    return ts_inputs

  def apply_ts_blocks(self, inputs, training):
    ts_inputs = self.unpack_ts_inputs(inputs)
    return [
        self.ts_blocks[name](ts_inputs[name], training)
        for name in sorted(self.ts_blocks)
    ]

  def backbone(self, inputs, training=False):
    """Extracts backbone features (convolutional blocks + aux inputs).

    Args:
      inputs: Dictionary of input features.
      training: Whether the model is in training mode.

    Returns:
      Concatenated feature vector from all time-series blocks and aux inputs.
    """
    y = self.apply_ts_blocks(inputs, training)
    y.extend([inputs[key] for key in self.config.hparams.aux_inputs])
    return tf.concat(y, axis=-1)

  def head(self, features, training=False):
    """Applies the fully connected head to backbone features.

    Args:
      features: Concatenated feature vector from backbone.
      training: Whether the model is in training mode.

    Returns:
      Model predictions (logits/probabilities).
    """
    y = self.dense_block(features, training)
    return self.output_layer(y)

  def get_embeddings(self, inputs, training=False):
      """Returns the latent space embeddings (output of dense block)."""
      # 1. Get the concatenated features (CNN + Aux)
      backbone_feats = self.backbone(inputs, training=training)

      # 2. Pass through the dense block (The Fusion/Mixing happen here)
      embeddings = self.dense_block(backbone_feats, training=training)

      return embeddings

  def call(self, inputs, training):
    y = self.backbone(inputs, training)
    return self.head(y, training)
