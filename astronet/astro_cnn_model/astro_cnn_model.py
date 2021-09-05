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


class AstroCNNModel(tf.keras.Model):

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
            self.ts_blocks = self._create_ts_blocks(config)

            self.final = [
              tf.keras.layers.Concatenate()
            ]

            hps = config.hparams
            for i in range(hps.num_pre_logits_hidden_layers):
                self.final.append(tf.keras.layers.Dense(units=hps.pre_logits_hidden_layer_size, activation='relu'))
                if hps.use_batch_norm:
                    self.final.append(tf.keras.layers.BatchNormalization())
                self.final.append(tf.keras.layers.Dropout(hps.pre_logits_dropout_rate))

            self.final.append(tf.keras.layers.Dense(units=len(config.inputs.label_columns), activation='sigmoid'))

    def _create_conv_block(self, config, name):
        block_params = config.hparams.time_series_hidden[name]
        layers = []
        for i in range(block_params.cnn_num_blocks):
            block_name = '{}_block_{}'.format(name, i + 1)
            num_filters = int(block_params.cnn_initial_num_filters *
                              block_params.cnn_block_filter_factor ** i)
            for j in range(block_params.cnn_block_size):
                layers.append(tf.keras.layers.Conv1D(
                    filters=num_filters,
                    kernel_size=block_params.cnn_kernel_size,
                    padding=block_params.convolution_padding,
                    activation='relu',
                    name='{}_conv_{}'.format(block_name, j + 1)))
            if block_params.pool_size:
                layers.append(tf.keras.layers.MaxPool1D(
                    pool_size=block_params.pool_size,
                    strides=block_params.pool_strides,
                    name='{}_pool'.format(block_name)))
        layers.append(tf.keras.layers.Flatten())
        return layers

    def _create_ts_blocks(self, config):
        blocks = {}
        for key in config.hparams.time_series_hidden:
            blocks[key] = self._create_conv_block(config, key)
        return blocks

    def _apply_block(self, block, input_, training):
        y = input_
        for layer in block:
            y = layer(y, training=training)
        return y

    def call(self, inputs, training=None):
        ts_inputs = {}
        aux_inputs = {}
        for k, v in inputs.items():
            if k in self.config.hparams.time_series_hidden:
                c = self.config.hparams.time_series_hidden[k]
                chans = [v]
                for extra in getattr(c, 'extra_channels', []):
                    chans.append(inputs[extra])
                if getattr(c, 'multichannel', False):
                    ts_inputs[k] = tf.concat(chans, axis=-1)
                else:
                    ts_inputs[k] = tf.stack(chans, axis=-1)
            elif k in self.config.hparams.aux_inputs:
                aux_inputs[k] = v
        y = []
        for k, v in ts_inputs.items():
            y_k = self._apply_block(self.ts_blocks[k], v, training)
            y.append(y_k)
        y.extend(aux_inputs.values())
        y = self._apply_block(self.final, y, training)

        return y
