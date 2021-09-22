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

import tensorflow as tf

from astronet.astro_cnn_model import astro_cnn_model



class AstroCNNModelVetting(tf.keras.Model):

    def __init__(self, config, triage_model):
        super(AstroCNNModelVetting, self).__init__()
        
        self.triage_model = astro_cnn_model.AstroCNNModel(config, triage_model, embeds_only=True)
        self.config = config
        
        self.ts_blocks = self._create_ts_blocks(config)

        self.final = [
          tf.keras.layers.Concatenate()
        ]
        hps = config.vetting_hparams
        for i in range(hps.num_pre_logits_hidden_layers):
            self.final.append(
                tf.keras.layers.Dense(units=hps.pre_logits_hidden_layer_size, activation='relu'))
            self.final.append(tf.keras.layers.Dropout(hps.pre_logits_dropout_rate))
        self.final.append(
            tf.keras.layers.Dense(units=len(config.inputs.label_columns), activation='sigmoid'))

    def _create_conv_block(self, config, name):
        block_params = config.vetting_hparams.time_series_hidden[name]
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
        for key in config.vetting_hparams.time_series_hidden:
            blocks[key] = self._create_conv_block(config, key)
        return blocks

    def _apply_block(self, block, input_, training):
        y = input_
        for layer in block:
            y = layer(y, training=training)
        return y

    def call(self, inputs, training=None):
        def is_vetting_input(k):
            if k.endswith('_present'):
                k = k[:-len('_present')]
            return self.config.inputs.features[k].get('vetting_only', False)
        
        triage_inputs = {k:v for k, v in inputs.items() if not is_vetting_input(k)}
        vetting_inputs = {k:v for k, v in inputs.items() if is_vetting_input(k)}
        
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
            y_k = self._apply_block(self.ts_blocks[k], v, training)
            y.append(y_k)
        y.extend([aux_inputs[k] for k in sorted(aux_inputs.keys())])
        y = self._apply_block(self.final, y, training)
        
        return y
