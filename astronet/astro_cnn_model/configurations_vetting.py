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

from astronet.astro_cnn_model import configurations
from astronet.util import config_util


def base():
    config = {
        "inputs": {
            "label_columns": ["disp_p", "disp_e", "disp_n"],
            "exclusive_labels": True,
            "features": {
                "local_aperture_s": {
                    "shape": [61],
                    "is_time_series": True,
                    "vetting_only": True,
                },
                "local_aperture_m": {
                    "shape": [61],
                    "is_time_series": True,
                    "vetting_only": True,
                },
                "local_aperture_l": {
                    "shape": [61],
                    "is_time_series": True,
                    "vetting_only": True,
                },
            },
        },
        "vetting_hparams": {
            "num_pre_logits_hidden_layers": 3,
            "pre_logits_hidden_layer_size": 256,
            "pre_logits_dropout_rate": 0.1,

            "time_series_hidden": {
                "local_aperture_s": {
                    "cnn_num_blocks": 3,
                    "cnn_block_size": 1,
                    "cnn_initial_num_filters": 17,
                    "cnn_block_filter_factor": 2,
                    "cnn_kernel_size": 3,
                    "convolution_padding": "same",
                    "pool_size": 7,
                    "pool_strides": 1,
                    "extra_channels": [
                        "local_aperture_m",
                        "local_aperture_l",
                    ],
                },
            },
            "aux_inputs": [],
        },
        
        "tune_params": [
            {
                'parameter': 'exclusive_labels', 'type': 'CATEGORICAL',
                'categorical_value_spec' : {'values': ['True', 'False']}},
            {
                'parameter': 'learning_rate', 'type': 'DOUBLE',
                'double_value_spec' : {'min_value': 1e-6, 'max_value': 1e-3},
                'scale_type': 'UNIT_LOG_SCALE'},
            {
                'parameter': 'one_minus_adam_beta_1', 'type': 'DOUBLE',
                'double_value_spec' : {'min_value': 1e-2, 'max_value': 0.9},
                'scale_type': 'UNIT_LOG_SCALE'},
            {
                'parameter': 'one_minus_adam_beta_2', 'type': 'DOUBLE',
                'double_value_spec' : {'min_value': 1e-4, 'max_value': 0.9},
                'scale_type': 'UNIT_LOG_SCALE'},
            {
                'parameter': 'adam_epsilon', 'type': 'DOUBLE',
                'double_value_spec' : {'min_value': 1e-8, 'max_value': 1e-5},
                'scale_type': 'UNIT_LOG_SCALE'},
            {
                'parameter': 'batch_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 8, 'max_value' : 128}},
            {
                'parameter': 'train_steps', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 10, 'max_value' : 6000}},
            {
                'parameter': 'num_pre_logits_hidden_layers', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 4}},
            {
                'parameter': 'pre_logits_hidden_layer_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 32, 'max_value' : 1024}},
            {
                'parameter': 'pre_logits_dropout_rate', 'type' : 'DOUBLE',
                'double_value_spec' : {'min_value' : 0.0, 'max_value' : 0.4}},
            {
                'parameter': 'cnn_block_filter_factor', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 2}},
            {
                'parameter': 'cnn_block_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 4}},
            {
                'parameter': 'cnn_initial_num_filters', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 4, 'max_value' : 32}},
            {
                'parameter': 'cnn_kernel_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}},
            {
                'parameter': 'cnn_num_blocks', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 0, 'max_value' : 5}},
            {
                'parameter': 'pool_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 7}},
            {
                'parameter': 'pool_strides', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}}
        ],
    }
    
    config_util.merge_configs(config, configurations.extended())
    
    return config


def vrevised():
    config = {
        "inputs": {
            "label_columns": ["disp_p", "disp_e", "disp_n"],
            "exclusive_labels": True,
            "features": {
                "local_aperture_s": {
                    "shape": [61],
                    "is_time_series": True,
                    "vetting_only": True,
                },
                "local_aperture_m": {
                    "shape": [61],
                    "is_time_series": True,
                    "vetting_only": True,
                },
                "local_aperture_l": {
                    "shape": [61],
                    "is_time_series": True,
                    "vetting_only": True,
                },
            },
        },
        "train_steps": 1000,
        "hparams": {
            "batch_size": 100,

            "learning_rate": 1e-05,
            "optimizer": "adam",
            "one_minus_adam_beta_1": 0.1,
            "one_minus_adam_beta_2": 0.00,
            "adam_epsilon": 1e-07,
        },
        "vetting_hparams": {
            "use_batch_norm": False,

            "num_pre_logits_hidden_layers": 3,
            "pre_logits_hidden_layer_size": 256,
            "pre_logits_dropout_rate": 0.1,

            "time_series_hidden": {
                "local_aperture_s": {
                    "cnn_num_blocks": 3,
                    "cnn_block_size": 1,
                    "cnn_initial_num_filters": 8,
                    "cnn_block_filter_factor": 2,
                    "cnn_kernel_size": 3,
                    "convolution_padding": "valid",
                    "pool_size": 4,
                    "pool_strides": 2,
                    "separable": True,
                    "extra_channels": [
                        "local_aperture_m",
                        "local_aperture_l",
                    ],
                },
            },
            "aux_inputs": [],
        },
        
        "tune_params": [            
            {
                'parameter': 'train_steps', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 10, 'max_value' : 20000}},
            {
                'parameter': 'learning_rate', 'type': 'DOUBLE',
                'double_value_spec' : {'min_value': 1e-7, 'max_value': 1e-1},
                'scale_type': 'UNIT_LOG_SCALE'},
            {
                'parameter': 'batch_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 4, 'max_value' : 1024}},

            {
                'parameter': 'use_batch_norm', 'type': 'CATEGORICAL',
                'categorical_value_spec' : {'values': ['True', 'False']}},
            {
                'parameter': 'exclusive_labels', 'type': 'CATEGORICAL',
                'categorical_value_spec' : {'values': ['True', 'False']}},
            {
                'parameter': 'one_minus_adam_beta_1', 'type': 'DOUBLE',
                'double_value_spec' : {'min_value': 1e-2, 'max_value': 0.9},
                'scale_type': 'UNIT_LOG_SCALE'},
            {
                'parameter': 'one_minus_adam_beta_2', 'type': 'DOUBLE',
                'double_value_spec' : {'min_value': 1e-4, 'max_value': 0.9},
                'scale_type': 'UNIT_LOG_SCALE'},
            {
                'parameter': 'adam_epsilon', 'type': 'DOUBLE',
                'double_value_spec' : {'min_value': 1e-8, 'max_value': 1e-5},
                'scale_type': 'UNIT_LOG_SCALE'},

            {
                'parameter': 'num_pre_logits_hidden_layers', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 0, 'max_value' : 4}},
            {
                'parameter': 'pre_logits_hidden_layer_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 4, 'max_value' : 2048}},
            {
                'parameter': 'pre_logits_dropout_rate', 'type' : 'DOUBLE',
                'double_value_spec' : {'min_value' : 0.0, 'max_value' : 0.9}},

            {
                'parameter': 'cnn_num_blocks', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 0, 'max_value' : 5}},
            {
                'parameter': 'cnn_block_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}},
            {
                'parameter': 'cnn_initial_num_filters', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 512}},
            {
                'parameter': 'cnn_block_filter_factor', 'type' : 'DOUBLE',
                'double_value_spec' : {'min_value' : 1.0, 'max_value' : 3.0}},
            {
                'parameter': 'cnn_kernel_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 7}},
            {
                'parameter': 'pool_strides', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 7}},
            {
                'parameter': 'pool_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 7}},
            {
                'parameter': 'convolution_padding', 'type': 'CATEGORICAL',
                'categorical_value_spec' : {'values': ['valid', 'same']}},
            {
                'parameter': 'separable', 'type': 'CATEGORICAL',
                'categorical_value_spec' : {'values': ['True', 'False']}},
        ],
    }
    
    config_util.merge_configs(config, configurations.revised_tuned())
    
    return config

def vrevised_tuned():
    # projects/mdan-playground/locations/us-central1/studies/4_vrevised_1_vrevised
    config = vrevised()
    config['train_steps'] = 9560
    config['vetting_hparams'] = {
 'aux_inputs': [],
 'num_pre_logits_hidden_layers': 1,
 'pre_logits_dropout_rate': 0.4,
 'pre_logits_hidden_layer_size': 1024,
 'time_series_hidden': {'local_aperture_s': {'cnn_block_filter_factor': 1,
                                             'cnn_block_size': 1,
                                             'cnn_initial_num_filters': 127,
                                             'cnn_kernel_size': 5,
                                             'cnn_num_blocks': 1,
                                             'convolution_padding': 'same',
                                             'extra_channels': ['local_aperture_m',
                                                                'local_aperture_l'],
                                             'pool_size': 6,
                                             'pool_strides': 2,
                                             'separable': False}},
 'use_batch_norm': False}
    config['hparams'] = {
 'adam_epsilon': 1.0049552666495116e-08,
 'aux_inputs': ['Period',
                'Duration',
                'Transit_Depth',
                'Tmag',
                'star_mass',
                'star_mass_present',
                'star_rad',
                'star_rad_present',
                'n_folds',
                'local_scale',
                'local_scale_0.3',
                'local_scale_0.5',
                'local_scale_present',
                'local_scale_present_0.3',
                'local_scale_present_0.5',
                'secondary_scale',
                'secondary_scale_0.3',
                'secondary_scale_0.5',
                'secondary_scale_present',
                'secondary_scale_present_0.3',
                'secondary_scale_present_0.5'],
 'batch_size': 6,
 'learning_rate': 8.029020875920591e-06,
 'num_pre_logits_hidden_layers': 1,
 'one_minus_adam_beta_1': 0.01016827925279647,
 'one_minus_adam_beta_2': 0.02330777378281221,
 'optimizer': 'adam',
 'pre_logits_dropout_rate': 0.21823767222245816,
 'pre_logits_hidden_layer_size': 452,
 'time_series_hidden': {'global_view': {'cnn_block_filter_factor': 1,
                                        'cnn_block_size': 2,
                                        'cnn_initial_num_filters': 16,
                                        'cnn_kernel_size': 6,
                                        'cnn_num_blocks': 3,
                                        'convolution_padding': 'valid',
                                        'extra_channels': ['global_view_0.3',
                                                           'global_view_5.0',
                                                           'global_std',
                                                           'global_mask',
                                                           'global_transit_mask'],
                                        'pool_size': 3,
                                        'pool_strides': 1,
                                        'separable': True},
                        'local_view': {'cnn_block_filter_factor': 1,
                                       'cnn_block_size': 1,
                                       'cnn_initial_num_filters': 4,
                                       'cnn_kernel_size': 5,
                                       'cnn_num_blocks': 2,
                                       'convolution_padding': 'valid',
                                       'extra_channels': ['local_view_0.3',
                                                          'local_view_5.0',
                                                          'local_view_odd',
                                                          'local_view_even',
                                                          'local_std',
                                                          'local_std_odd',
                                                          'local_std_even',
                                                          'local_view_half_period_std',
                                                          'local_mask'],
                                       'pool_size': 3,
                                       'pool_strides': 2,
                                       'separable': True},
                        'sample_segments_local_view': {'cnn_block_filter_factor': 1,
                                                       'cnn_block_size': 2,
                                                       'cnn_initial_num_filters': 35,
                                                       'cnn_kernel_size': 3,
                                                       'cnn_num_blocks': 2,
                                                       'convolution_padding': 'valid',
                                                       'extra_channels': ['sample_segments_local_view_0.3',
                                                                          'sample_segments_local_view_5.0'],
                                                       'multichannel': True,
                                                       'pool_size': 6,
                                                       'pool_strides': 3,
                                                       'separable': True},
                        'secondary_view': {'cnn_block_filter_factor': 1,
                                           'cnn_block_size': 1,
                                           'cnn_initial_num_filters': 128,
                                           'cnn_kernel_size': 8,
                                           'cnn_num_blocks': 2,
                                           'convolution_padding': 'valid',
                                           'extra_channels': ['secondary_std',
                                                              'secondary_view_0.3',
                                                              'secondary_view_5.0',
                                                              'secondary_mask'],
                                           'pool_size': 2,
                                           'pool_strides': 3,
                                           'separable': True}},
 'use_batch_norm': False}
    return config
