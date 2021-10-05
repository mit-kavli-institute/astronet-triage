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
                'integer_value_spec' : {'min_value' : 4, 'max_value' : 1024}},
            {
                'parameter': 'pre_logits_dropout_rate', 'type' : 'DOUBLE',
                'double_value_spec' : {'min_value' : 0.0, 'max_value' : 0.4}},

            {
                'parameter': 'cnn_num_blocks', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 0, 'max_value' : 5}},
            {
                'parameter': 'cnn_block_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}},
            {
                'parameter': 'cnn_initial_num_filters', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 128}},
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
