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

"""Configurations for model building, training and evaluation.

Available configurations:
  * base: One time series feature per input example. Default is "global_view".
  * local_global: Two time series features per input example.
      - A "global" view of the entire orbital period.
      - A "local" zoomed-in view of the transit event.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def local_global_new():
    config = {
        "inputs": {
            "label_columns": ["disp_E", "disp_N", "disp_J", "disp_S", "disp_B"],
            "primary_class": 0,

            "features": {
                "local_view": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "global_view": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "secondary_view": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "Period": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 8.071377,
                    "std": 11.233816,
                    "has_nans": False,
                },
                "Duration": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.196459,
                    "std": 0.172065,
                    "has_nans": False,
                },
                "Transit_Depth": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 3.847200e+05,
                    "std": 3.220359e+07,
                    "has_nans": False,
                },
                "Tmag": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 10.162480,
                    "std": 1.225660,
                    "has_nans": False,
                },
                "star_mass": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 1.382456,
                    "std": 0.387535,
                    "has_nans": True,
                },
                "star_rad": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 11.881122,
                    "std": 19.495874,
                    "has_nans": True,
                },
            },
        },

        "hparams": {
            "prediction_threshold": 0.5,

            "batch_size": 64,

            "learning_rate": 1e-3,
            "clip_gradient_norm": None,
            "optimizer": "adam",
            "one_minus_adam_beta_1": 0.1,
            "one_minus_adam_beta_2": 0.001,
            "adam_epsilon": 1e-7,
            
            "use_batch_norm": False,
          
            "num_pre_logits_hidden_layers": 4,
            "pre_logits_hidden_layer_size": 512,
            "pre_logits_dropout_rate": 0.0,
            
            "aux_inputs": [
                "Period",
                "Duration",
                "Transit_Depth",
                "Tmag",
                "star_mass",
                "star_mass_present",
                "star_rad",
                "star_rad_present",
            ],
          
            "time_series_hidden": {
                "global_view": {
                    "cnn_num_blocks": 5,
                    "cnn_block_size": 2,
                    "cnn_initial_num_filters": 16,
                    "cnn_block_filter_factor": 2,
                    "cnn_kernel_size": 5,
                    "convolution_padding": "same",
                    "pool_size": 5,
                    "pool_strides": 2,
                },
                "local_view": {
                    "cnn_num_blocks": 2,
                    "cnn_block_size": 2,
                    "cnn_initial_num_filters": 16,
                    "cnn_block_filter_factor": 2,
                    "cnn_kernel_size": 5,
                    "convolution_padding": "same",
                    "pool_size": 7,
                    "pool_strides": 2,
                },
                "secondary_view": {
                    "cnn_num_blocks": 2,
                    "cnn_block_size": 2,
                    "cnn_initial_num_filters": 16,
                    "cnn_block_filter_factor": 2,
                    "cnn_kernel_size": 5,
                    "convolution_padding": "same",
                    "pool_size": 7,
                    "pool_strides": 2,
                },
            },
        },
        "tune_params": [
            {
                'parameter': 'use_batch_norm', 'type': 'CATEGORICAL',
                'categorical_value_spec' : {'values': ['True', 'False']}},
            {
                'parameter': 'prediction_threshold', 'type': 'DOUBLE',
                'double_value_spec' : {'min_value': 0.1, 'max_value': 0.5}},
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
                'parameter': 'cnn_num_blocks_global', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 0, 'max_value' : 5}},
            {
                'parameter': 'cnn_num_blocks_local', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 0, 'max_value' : 5}},
            {
                'parameter': 'pool_size_global', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 7}},
            {
                'parameter': 'pool_size_local', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 7}},
            {
                'parameter': 'pool_strides', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}}
        ],
    }

    return config


def local_global_new_tuned():
  config = local_global_new()

  # studies/and_yet_another_2_local_global_new
  config['hparams'] = {'adam_epsilon': 2.5037055725611666e-07,
     'batch_size': 83,
     'clip_gradient_norm': None,
     'learning_rate': 5.203528044134961e-06,
     'num_pre_logits_hidden_layers': 4,
     'one_minus_adam_beta_1': 0.16168028483420177,
     'one_minus_adam_beta_2': 0.022674419033475692,
     'optimizer': 'adam',
     'pre_logits_dropout_rate': 0.1690298097832756,
     'pre_logits_hidden_layer_size': 482,
     'prediction_threshold': 0.2152499407880693,
     'time_series_hidden': {'global_view': {'cnn_block_filter_factor': 2,
                                            'cnn_block_size': 1,
                                            'cnn_initial_num_filters': 17,
                                            'cnn_kernel_size': 2,
                                            'cnn_num_blocks': 3,
                                            'convolution_padding': 'same',
                                            'pool_size': 5,
                                            'pool_strides': 1},
                            'local_view': {'cnn_block_filter_factor': 2,
                                           'cnn_block_size': 1,
                                           'cnn_initial_num_filters': 17,
                                           'cnn_kernel_size': 2,
                                           'cnn_num_blocks': 3,
                                           'convolution_padding': 'same',
                                           'pool_size': 6,
                                           'pool_strides': 1},
                            'secondary_view': {'cnn_block_filter_factor': 2,
                                               'cnn_block_size': 1,
                                               'cnn_initial_num_filters': 17,
                                               'cnn_kernel_size': 2,
                                               'cnn_num_blocks': 3,
                                               'convolution_padding': 'same',
                                               'pool_size': 6,
                                               'pool_strides': 1}},
     'use_batch_norm': False}

  return config


def extended():
    config = {
        "train_steps": 20000,

        "inputs": {
            "label_columns": ["disp_E", "disp_N", "disp_J", "disp_S", "disp_B"],
            "primary_class": 0,

            "features": {
                "global_view": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "global_std": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "global_mask": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "global_transit_mask": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "global_view_0.3": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "global_view_5.0": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "local_view": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_std": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_mask": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_view_0.3": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_mask_0.3": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_view_5.0": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_mask_5.0": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "secondary_view": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "secondary_std": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "secondary_mask": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "secondary_view_0.3": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "secondary_mask_0.3": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "secondary_view_5.0": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "secondary_mask_5.0": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "global_view_half_period": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "global_view_half_period_std": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "global_view_double_period": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "global_view_double_period_std": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "sample_segments_view": {
                    "shape": [201, 14],
                    "is_time_series": True,
                },
                "sample_segments_view_0.3": {
                    "shape": [201, 14],
                    "is_time_series": True,
                },
                "sample_segments_view_5.0": {
                    "shape": [201, 14],
                    "is_time_series": True,
                },
                "Period": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 16.839051,
                    "std": 28.872394,
                    "has_nans": False,
                },
                "Duration": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.249504,
                    "std": 0.405356,
                    "has_nans": False,
                },
                "Transit_Depth": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 2.981201e+07,
                    "std": 2.953274e+09,
                    "has_nans": False,
                },
                "Tmag": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 9.000078,
                    "std": 1.480743,
                    "has_nans": False,
                },
                "star_mass": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.676770,
                    "std": 0.824015,
                    "has_nans": True,
                },
                "star_rad": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 7.415899,
                    "std": 10.706470,
                    "has_nans": True,
                },
                "n_folds": {
                    "shape": [1],
                    "is_time_series": False,
                    "log_scale": True,
                    "min_val": 0,
                    "max_val": 100,
                    "has_nans": False,
                },
                "n_points": {
                    "shape": [1],
                    "is_time_series": False,
                    "log_scale": True,
                    "min_val": 0,
                    "max_val": 1000,
                    "has_nans": False,
                },
            },
        },

        "hparams": {
            "prediction_threshold": 0.2152499407880693,

            "batch_size": 83,

            "learning_rate": 5.203528044134961e-06,
            "clip_gradient_norm": None,
            "optimizer": "adam",
            "one_minus_adam_beta_1": 0.16168028483420177,
            "one_minus_adam_beta_2": 0.022674419033475692,
            "adam_epsilon": 2.5037055725611666e-07,
            
            "use_batch_norm": False,
          
            "num_pre_logits_hidden_layers": 4,
            "pre_logits_hidden_layer_size": 482,
            "pre_logits_dropout_rate": 0.1690298097832756,
            
            "aux_inputs": [
                "Period",
                "Duration",
                "Transit_Depth",
                "Tmag",
                "star_mass",
                "star_mass_present",
                "star_rad",
                "star_rad_present",
                "n_folds",
                "n_points",
                "local_scale",
                "local_scale_0.3",
                "local_scale_0.5",
                "local_scale_present",
                "local_scale_present_0.3",
                "local_scale_present_0.5",
                "secondary_scale",
                "secondary_scale_0.3",
                "secondary_scale_0.5",
                "secondary_scale_present",
                "secondary_scale_present_0.3",
                "secondary_scale_present_0.5",
            ],
          
            "time_series_hidden": {
                "global_view": {
                    "cnn_num_blocks": 3,
                    "cnn_block_size": 1,
                    "cnn_initial_num_filters": 17,
                    "cnn_block_filter_factor": 2,
                    "cnn_kernel_size": 3,
                    "convolution_padding": "same",
                    "pool_size": 5,
                    "pool_strides": 1,
                    "extra_channels": [
                        "global_std",
                        "global_mask",
                        "global_transit_mask",
                        "global_view_0.3",
                        "global_view_5.0",
                    ],
                },
                "global_view_double_period": {
                    "cnn_num_blocks": 3,
                    "cnn_block_size": 1,
                    "cnn_initial_num_filters": 17,
                    "cnn_block_filter_factor": 2,
                    "cnn_kernel_size": 3,
                    "convolution_padding": "same",
                    "pool_size": 7,
                    "pool_strides": 2,
                    "extra_channels": [
                        "global_view_double_period_std",
                    ],
                },
                "global_view_half_period": {
                    "cnn_num_blocks": 3,
                    "cnn_block_size": 1,
                    "cnn_initial_num_filters": 17,
                    "cnn_block_filter_factor": 2,
                    "cnn_kernel_size": 3,
                    "convolution_padding": "same",
                    "pool_size": 7,
                    "pool_strides": 2,
                    "extra_channels": [
                        "global_view_half_period_std",
                    ],
                },
                "local_view": {
                    "cnn_num_blocks": 3,
                    "cnn_block_size": 1,
                    "cnn_initial_num_filters": 17,
                    "cnn_block_filter_factor": 2,
                    "cnn_kernel_size": 3,
                    "convolution_padding": "same",
                    "pool_size": 7,
                    "pool_strides": 1,
                    "extra_channels": [
                        "local_std",
                        "local_mask",
                        "local_view_0.3",
                        "local_mask_0.3",
                        "local_view_5.0",
                        "local_mask_5.0",
                    ],
                },
                "secondary_view": {
                    "cnn_num_blocks": 3,
                    "cnn_block_size": 1,
                    "cnn_initial_num_filters": 17,
                    "cnn_block_filter_factor": 2,
                    "cnn_kernel_size": 3,
                    "convolution_padding": "same",
                    "pool_size": 7,
                    "pool_strides": 1,
                    "extra_channels": [
                        "secondary_std",
                        "secondary_mask",
                        "secondary_view_0.3",
                        "secondary_mask_0.3",
                        "secondary_view_5.0",
                        "secondary_mask_5.0",
                    ],
                },
                "sample_segments_view": {
                    "cnn_num_blocks": 3,
                    "cnn_block_size": 1,
                    "cnn_initial_num_filters": 51,
                    "cnn_block_filter_factor": 2,
                    "cnn_kernel_size": 3,
                    "convolution_padding": "same",
                    "pool_size": 7,
                    "pool_strides": 2,
                    "multichannel": True,
                    "extra_channels": [
                        "sample_segments_view_0.3",
                        "sample_segments_view_5.0",
                    ],
                },
            },
        },

    }

    return config


def revised():
    config = {
        "train_steps": 20000,
        "inputs": {
            "label_columns": ["disp_E", "disp_N", "disp_J", "disp_S", "disp_B"],
            "primary_class": 0,

            "features": {
                "global_view": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "global_view_0.3": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "global_view_5.0": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "global_std": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "global_mask": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "global_transit_mask": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "local_view": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_view_0.3": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_view_5.0": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_view_odd": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_view_even": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_std": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_std_odd": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_std_even": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_view_half_period_std": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_mask": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_mask_even": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_mask_odd": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "secondary_view": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "secondary_std": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "secondary_view_0.3": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "secondary_view_5.0": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "secondary_mask": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "sample_segments_local_view": {
                    "shape": [61, 16],
                    "is_time_series": True,
                },
                "sample_segments_local_view_0.3": {
                    "shape": [61, 16],
                    "is_time_series": True,
                },
                "sample_segments_local_view_5.0": {
                    "shape": [61, 16],
                    "is_time_series": True,
                },
                "Period": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 16.839051,
                    "std": 28.872394,
                    "has_nans": False,
                },
                "Duration": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.249504,
                    "std": 0.405356,
                    "has_nans": False,
                },
                "Transit_Depth": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 2.981201e+07,
                    "std": 2.953274e+09,
                    "has_nans": False,
                },
                "Tmag": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 9.000078,
                    "std": 1.480743,
                    "has_nans": False,
                },
                "star_mass": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.676770,
                    "std": 0.824015,
                    "has_nans": True,
                },
                "star_rad": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 7.415899,
                    "std": 10.706470,
                    "has_nans": True,
                },
                "n_folds": {
                    "shape": [1],
                    "is_time_series": False,
                    "log_scale": True,
                    "min_val": 0,
                    "max_val": 100,
                    "has_nans": False,
                },
                "secondary_phase": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.5,
                    "std": 0.2,
                    "has_nans": True,
                },
                "secondary_phase_0.3": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.5,
                    "std": 0.2,
                    "has_nans": True,
                },
                "secondary_phase_5.0": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.5,
                    "std": 0.2,
                    "has_nans": True,
                },
                "local_scale": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.016534,
                    "std": 0.092118,
                    "has_nans": True,
                },
                "local_scale_0.3": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.016534,
                    "std": 0.092118,
                    "has_nans": True,
                },
                "local_scale_5.0": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.016534,
                    "std": 0.092118,
                    "has_nans": True,
                },
                "secondary_scale": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.005585,
                    "std": 0.029651,
                    "has_nans": True,
                },
                "secondary_scale_0.3": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.005585,
                    "std": 0.029651,
                    "has_nans": True,
                },
                "secondary_scale_5.0": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.005585,
                    "std": 0.029651,
                    "has_nans": True,
                },
            },
        },

        "hparams": {
            "batch_size": 100,

            "learning_rate": 1e-05,
            "optimizer": "adam",
            "one_minus_adam_beta_1": 0.1,
            "one_minus_adam_beta_2": 0.00,
            "adam_epsilon": 1e-07,
            
            "use_batch_norm": False,
          
            "num_pre_logits_hidden_layers": 3,
            "pre_logits_hidden_layer_size": 250,
            "pre_logits_dropout_rate": 0.15,
            
            "aux_inputs": [
                "Period",
                "Duration",
                "Transit_Depth",
                "Tmag",
                "star_mass",
                "star_mass_present",
                "star_rad",
                "star_rad_present",
                "n_folds",
                "local_scale",
                "local_scale_0.3",
                "local_scale_0.5",
                "local_scale_present",
                "local_scale_present_0.3",
                "local_scale_present_0.5",
                "secondary_scale",
                "secondary_scale_0.3",
                "secondary_scale_0.5",
                "secondary_scale_present",
                "secondary_scale_present_0.3",
                "secondary_scale_present_0.5",
            ],
          
            "time_series_hidden": {
                "global_view": {
                    "cnn_num_blocks": 3,
                    "cnn_block_size": 1,
                    "cnn_initial_num_filters": 10,
                    "cnn_block_filter_factor": 2,
                    "cnn_kernel_size": 3,
                    "convolution_padding": "valid",
                    "pool_size": 3,
                    "pool_strides": 2,
                    "separable": True,
                    "extra_channels": [
                        "global_view_0.3",
                        "global_view_5.0",
                        "global_std",
                        "global_mask",
                        "global_transit_mask",
                    ],
                },
                "local_view": {
                    "cnn_num_blocks": 3,
                    "cnn_block_size": 1,
                    "cnn_initial_num_filters": 10,
                    "cnn_block_filter_factor": 2,
                    "cnn_kernel_size": 3,
                    "convolution_padding": "valid",
                    "pool_size": 3,
                    "pool_strides": 2,
                    "separable": True,
                    "extra_channels": [
                        "local_view_0.3",
                        "local_view_5.0",
                        "local_view_odd",
                        "local_view_even",
                        "local_std",
                        "local_std_odd",
                        "local_std_even",
                        "local_view_half_period_std",
                        "local_mask",
                        'local_mask_odd',
                        'local_mask_even'
                    ],
                },
                "secondary_view": {
                    "cnn_num_blocks": 3,
                    "cnn_block_size": 1,
                    "cnn_initial_num_filters": 10,
                    "cnn_block_filter_factor": 2,
                    "cnn_kernel_size": 3,
                    "convolution_padding": "valid",
                    "pool_size": 3,
                    "pool_strides": 2,
                    "separable": True,
                    "extra_channels": [
                        "secondary_std",
                        "secondary_view_0.3",
                        "secondary_view_5.0",
                        "secondary_mask",
                    ],
                },
                "sample_segments_local_view": {
                    "cnn_num_blocks": 3,
                    "cnn_block_size": 1,
                    "cnn_initial_num_filters": 10,
                    "cnn_block_filter_factor": 2,
                    "cnn_kernel_size": 3,
                    "convolution_padding": "valid",
                    "pool_size": 3,
                    "pool_strides": 2,
                    "separable": True,
                    "multichannel": True,
                    "extra_channels": [
                        "sample_segments_local_view_0.3",
                        "sample_segments_local_view_5.0",
                    ],
                },
            },
        },

        "tune_params": [
            {
                'parameter': 'train_steps', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 4000, 'max_value' : 30000},
                'scale_type': 'UNIT_LOG_SCALE'},
            {
                'parameter': 'learning_rate', 'type': 'DOUBLE',
                'double_value_spec' : {'min_value': 1e-7, 'max_value': 1e-2},
                'scale_type': 'UNIT_LOG_SCALE'},
            {
                'parameter': 'batch_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 4, 'max_value' : 1024}},


            {
                'parameter': 'use_batch_norm', 'type': 'CATEGORICAL',
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
                'integer_value_spec' : {'min_value' : 16, 'max_value' : 1024}},
            {
                'parameter': 'pre_logits_dropout_rate', 'type' : 'DOUBLE',
                'double_value_spec' : {'min_value' : 0.0, 'max_value' : 0.4}},

            {
                'parameter': 'global_cnn_num_blocks', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}},
            {
                'parameter': 'global_cnn_block_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}},
            {
                'parameter': 'global_cnn_initial_num_filters', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 128}},
            {
                'parameter': 'global_cnn_block_filter_factor', 'type' : 'DOUBLE',
                'double_value_spec' : {'min_value' : 0.2, 'max_value' : 3.0}},
            {
                'parameter': 'global_cnn_kernel_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 8}},
            {
                'parameter': 'global_pool_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 8}},
            {
                'parameter': 'global_pool_strides', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 7}},
            {
                'parameter': 'global_separable', 'type': 'CATEGORICAL',
                'categorical_value_spec' : {'values': ['True', 'False']}},

            {
                'parameter': 'local_cnn_num_blocks', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}},
            {
                'parameter': 'local_cnn_block_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}},
            {
                'parameter': 'local_cnn_initial_num_filters', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 128}},
            {
                'parameter': 'local_cnn_block_filter_factor', 'type' : 'DOUBLE',
                'double_value_spec' : {'min_value' : 0.2, 'max_value' : 3.0}},
            {
                'parameter': 'local_cnn_kernel_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 8}},
            {
                'parameter': 'local_pool_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 8}},
            {
                'parameter': 'local_pool_strides', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 7}},
            {
                'parameter': 'local_separable', 'type': 'CATEGORICAL',
                'categorical_value_spec' : {'values': ['True', 'False']}},

            {
                'parameter': 'sec_cnn_num_blocks', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}},
            {
                'parameter': 'sec_cnn_block_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}},
            {
                'parameter': 'sec_cnn_initial_num_filters', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 512}},
            {
                'parameter': 'sec_cnn_block_filter_factor', 'type' : 'DOUBLE',
                'double_value_spec' : {'min_value' : 0.2, 'max_value' : 3.0}},
            {
                'parameter': 'sec_cnn_kernel_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 15}},
            {
                'parameter': 'sec_pool_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 8}},
            {
                'parameter': 'sec_pool_strides', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 7}},
            {
                'parameter': 'sec_separable', 'type': 'CATEGORICAL',
                'categorical_value_spec' : {'values': ['True', 'False']}},

            {
                'parameter': 'ind_cnn_num_blocks', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}},
            {
                'parameter': 'ind_cnn_block_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}},
            {
                'parameter': 'ind_cnn_initial_num_filters', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 256}},
            {
                'parameter': 'ind_cnn_block_filter_factor', 'type' : 'DOUBLE',
                'double_value_spec' : {'min_value' : 0.2, 'max_value' : 3.0}},
            {
                'parameter': 'ind_cnn_kernel_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 8}},
            {
                'parameter': 'ind_pool_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 8}},
            {
                'parameter': 'ind_pool_strides', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 7}},
            {
                'parameter': 'ind_separable', 'type': 'CATEGORICAL',
                'categorical_value_spec' : {'values': ['True', 'False']}},
        ],
    }

    return config


def revised_tuned():
    # projects/mdan-playground/locations/us-central1/studies/33_revised_1b_revised
    # (546 0.046330880373716354)
    config = revised()
    config['train_steps'] = 4033
    config['hparams'] = {
 'adam_epsilon': 6.634908638557719e-06,
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
 'batch_size': 304,
 'learning_rate': 0.00012430771320309707,
 'num_pre_logits_hidden_layers': 2,
 'one_minus_adam_beta_1': 0.6325014551176217,
 'one_minus_adam_beta_2': 0.04109274984689409,
 'optimizer': 'adam',
 'pre_logits_dropout_rate': 0.2716590871645748,
 'pre_logits_hidden_layer_size': 443,
 'time_series_hidden': {'global_view': {'cnn_block_filter_factor': 1.007563093344749,
                                        'cnn_block_size': 2,
                                        'cnn_initial_num_filters': 29,
                                        'cnn_kernel_size': 5,
                                        'cnn_num_blocks': 1,
                                        'convolution_padding': 'valid',
                                        'extra_channels': ['global_view_0.3',
                                                           'global_view_5.0',
                                                           'global_std',
                                                           'global_mask',
                                                           'global_transit_mask'],
                                        'pool_size': 7,
                                        'pool_strides': 2,
                                        'separable': True},
                        'local_view': {'cnn_block_filter_factor': 1.1900166187086434,
                                       'cnn_block_size': 1,
                                       'cnn_initial_num_filters': 19,
                                       'cnn_kernel_size': 6,
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
                                                          'local_mask',
                                                          'local_mask_odd',
                                                          'local_mask_even'],
                                       'pool_size': 4,
                                       'pool_strides': 3,
                                       'separable': False},
                        'sample_segments_local_view': {'cnn_block_filter_factor': 0.9827415177848933,
                                                       'cnn_block_size': 4,
                                                       'cnn_initial_num_filters': 38,
                                                       'cnn_kernel_size': 1,
                                                       'cnn_num_blocks': 1,
                                                       'convolution_padding': 'valid',
                                                       'extra_channels': ['sample_segments_local_view_0.3',
                                                                          'sample_segments_local_view_5.0'],
                                                       'multichannel': True,
                                                       'pool_size': 6,
                                                       'pool_strides': 2,
                                                       'separable': True},
                        'secondary_view': {'cnn_block_filter_factor': 0.7760375262452452,
                                           'cnn_block_size': 2,
                                           'cnn_initial_num_filters': 116,
                                           'cnn_kernel_size': 6,
                                           'cnn_num_blocks': 2,
                                           'convolution_padding': 'valid',
                                           'extra_channels': ['secondary_std',
                                                              'secondary_view_0.3',
                                                              'secondary_view_5.0',
                                                              'secondary_mask'],
                                           'pool_size': 2,
                                           'pool_strides': 1,
                                           'separable': True}},
 'use_batch_norm': False}
    return config


def rev2():
    config = {
        "train_steps": 20000,
        "inputs": {
            "label_columns": ["disp_E", "disp_N", "disp_J", "disp_S", "disp_B"],
            "primary_class": 0,

            "features": {
                "global_view": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "global_view_0.3": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "global_view_5.0": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "global_std": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "global_mask": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "global_transit_mask": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "local_view": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_view_0.3": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_view_5.0": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_view_odd": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_view_even": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_std": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_std_odd": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_std_even": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_view_half_period_std": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_mask": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_mask_even": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_mask_odd": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "secondary_view": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "secondary_std": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "secondary_view_0.3": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "secondary_view_5.0": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "secondary_mask": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "sample_segments_local_view": {
                    "shape": [61, 16],
                    "is_time_series": True,
                },
                "sample_segments_local_view_0.3": {
                    "shape": [61, 16],
                    "is_time_series": True,
                },
                "sample_segments_local_view_5.0": {
                    "shape": [61, 16],
                    "is_time_series": True,
                },
                "Period": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 9.987338,
                    "std": 12.119008,
                    "has_nans": False,
                },
                "Duration": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.249504,
                    "std": 0.405356,
                    "has_nans": False,
                },
                "Transit_Depth": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 2.981201e+07,
                    "std": 2.953274e+09,
                    "has_nans": False,
                },
                "Tmag": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 9.000078,
                    "std": 1.480743,
                    "has_nans": False,
                },
                "star_mass": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.676770,
                    "std": 0.824015,
                    "has_nans": True,
                },
                "star_rad": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 7.415899,
                    "std": 10.706470,
                    "has_nans": True,
                },
                "n_folds": {
                    "shape": [1],
                    "is_time_series": False,
                    "log_scale": True,
                    "min_val": 0,
                    "max_val": 100,
                    "has_nans": False,
                },
                "secondary_phase": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.5,
                    "std": 0.2,
                    "has_nans": True,
                },
                "secondary_phase_0.3": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.5,
                    "std": 0.2,
                    "has_nans": True,
                },
                "secondary_phase_5.0": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.5,
                    "std": 0.2,
                    "has_nans": True,
                },
                "local_scale": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.016534,
                    "std": 0.092118,
                    "has_nans": True,
                },
                "local_scale_0.3": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.016534,
                    "std": 0.092118,
                    "has_nans": True,
                },
                "local_scale_5.0": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.016534,
                    "std": 0.092118,
                    "has_nans": True,
                },
                "secondary_scale": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.005585,
                    "std": 0.029651,
                    "has_nans": True,
                },
                "secondary_scale_0.3": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.005585,
                    "std": 0.029651,
                    "has_nans": True,
                },
                "secondary_scale_5.0": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.005585,
                    "std": 0.029651,
                    "has_nans": True,
                },
            },
        },

        "hparams": {
            "batch_size": 100,

            "learning_rate": 1e-05,
            "optimizer": "adam",
            "one_minus_adam_beta_1": 0.1,
            "one_minus_adam_beta_2": 0.00,
            "adam_epsilon": 1e-07,
            
            "use_batch_norm": False,
          
            "num_pre_logits_hidden_layers": 3,
            "pre_logits_hidden_layer_size": 250,
            "pre_logits_dropout_rate": 0.15,
            
            "aux_inputs": [
                "Period",
                "Duration",
                "Transit_Depth",
                "Tmag",
                "star_mass",
                "star_mass_present",
                "star_rad",
                "star_rad_present",
                "n_folds",
                "local_scale",
                "local_scale_0.3",
                "local_scale_0.5",
                "local_scale_present",
                "local_scale_present_0.3",
                "local_scale_present_0.5",
                "secondary_scale",
                "secondary_scale_0.3",
                "secondary_scale_0.5",
                "secondary_scale_present",
                "secondary_scale_present_0.3",
                "secondary_scale_present_0.5",
            ],
          
            "time_series_hidden": {
                "global_view": {
                    "cnn_num_blocks": 3,
                    "cnn_block_size": 1,
                    "cnn_initial_num_filters": 10,
                    "cnn_block_filter_factor": 2,
                    "cnn_kernel_size": 3,
                    "convolution_padding": "valid",
                    "pool_size": 3,
                    "pool_strides": 2,
                    "separable": True,
                    "extra_channels": [
                        "global_view_0.3",
                        "global_view_5.0",
                        "global_std",
                        "global_mask",
                        "global_transit_mask",
                    ],
                },
                "local_view": {
                    "cnn_num_blocks": 3,
                    "cnn_block_size": 1,
                    "cnn_initial_num_filters": 10,
                    "cnn_block_filter_factor": 2,
                    "cnn_kernel_size": 3,
                    "convolution_padding": "valid",
                    "pool_size": 3,
                    "pool_strides": 2,
                    "separable": True,
                    "extra_channels": [
                        "local_view_0.3",
                        "local_view_5.0",
                        "local_view_odd",
                        "local_view_even",
                        "local_std",
                        "local_std_odd",
                        "local_std_even",
                        "local_view_half_period_std",
                        "local_mask",
                        'local_mask_odd',
                        'local_mask_even'
                    ],
                },
                "secondary_view": {
                    "cnn_num_blocks": 3,
                    "cnn_block_size": 1,
                    "cnn_initial_num_filters": 10,
                    "cnn_block_filter_factor": 2,
                    "cnn_kernel_size": 3,
                    "convolution_padding": "valid",
                    "pool_size": 3,
                    "pool_strides": 2,
                    "separable": True,
                    "extra_channels": [
                        "secondary_std",
                        "secondary_view_0.3",
                        "secondary_view_5.0",
                        "secondary_mask",
                    ],
                },
                "sample_segments_local_view": {
                    "cnn_num_blocks": 3,
                    "cnn_block_size": 1,
                    "cnn_initial_num_filters": 10,
                    "cnn_block_filter_factor": 2,
                    "cnn_kernel_size": 3,
                    "convolution_padding": "valid",
                    "pool_size": 3,
                    "pool_strides": 2,
                    "separable": True,
                    "multichannel": True,
                    "extra_channels": [
                        "sample_segments_local_view_0.3",
                        "sample_segments_local_view_5.0",
                    ],
                },
            },
        },

        "tune_params": [
            {
                'parameter': 'train_steps', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 4000, 'max_value' : 30000},
                'scale_type': 'UNIT_LOG_SCALE'},
            {
                'parameter': 'learning_rate', 'type': 'DOUBLE',
                'double_value_spec' : {'min_value': 1e-7, 'max_value': 1e-2},
                'scale_type': 'UNIT_LOG_SCALE'},
            {
                'parameter': 'batch_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 4, 'max_value' : 1024}},


            {
                'parameter': 'use_batch_norm', 'type': 'CATEGORICAL',
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
                'integer_value_spec' : {'min_value' : 16, 'max_value' : 1024}},
            {
                'parameter': 'pre_logits_dropout_rate', 'type' : 'DOUBLE',
                'double_value_spec' : {'min_value' : 0.0, 'max_value' : 0.4}},

            {
                'parameter': 'global_cnn_num_blocks', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}},
            {
                'parameter': 'global_cnn_block_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}},
            {
                'parameter': 'global_cnn_initial_num_filters', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 128}},
            {
                'parameter': 'global_cnn_block_filter_factor', 'type' : 'DOUBLE',
                'double_value_spec' : {'min_value' : 0.2, 'max_value' : 3.0}},
            {
                'parameter': 'global_cnn_kernel_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 8}},
            {
                'parameter': 'global_pool_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 8}},
            {
                'parameter': 'global_pool_strides', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 7}},
            {
                'parameter': 'global_separable', 'type': 'CATEGORICAL',
                'categorical_value_spec' : {'values': ['True', 'False']}},

            {
                'parameter': 'local_cnn_num_blocks', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}},
            {
                'parameter': 'local_cnn_block_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}},
            {
                'parameter': 'local_cnn_initial_num_filters', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 128}},
            {
                'parameter': 'local_cnn_block_filter_factor', 'type' : 'DOUBLE',
                'double_value_spec' : {'min_value' : 0.2, 'max_value' : 3.0}},
            {
                'parameter': 'local_cnn_kernel_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 8}},
            {
                'parameter': 'local_pool_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 8}},
            {
                'parameter': 'local_pool_strides', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 7}},
            {
                'parameter': 'local_separable', 'type': 'CATEGORICAL',
                'categorical_value_spec' : {'values': ['True', 'False']}},

            {
                'parameter': 'sec_cnn_num_blocks', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}},
            {
                'parameter': 'sec_cnn_block_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}},
            {
                'parameter': 'sec_cnn_initial_num_filters', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 512}},
            {
                'parameter': 'sec_cnn_block_filter_factor', 'type' : 'DOUBLE',
                'double_value_spec' : {'min_value' : 0.2, 'max_value' : 3.0}},
            {
                'parameter': 'sec_cnn_kernel_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 15}},
            {
                'parameter': 'sec_pool_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 8}},
            {
                'parameter': 'sec_pool_strides', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 7}},
            {
                'parameter': 'sec_separable', 'type': 'CATEGORICAL',
                'categorical_value_spec' : {'values': ['True', 'False']}},

            {
                'parameter': 'ind_cnn_num_blocks', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}},
            {
                'parameter': 'ind_cnn_block_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}},
            {
                'parameter': 'ind_cnn_initial_num_filters', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 256}},
            {
                'parameter': 'ind_cnn_block_filter_factor', 'type' : 'DOUBLE',
                'double_value_spec' : {'min_value' : 0.2, 'max_value' : 3.0}},
            {
                'parameter': 'ind_cnn_kernel_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 8}},
            {
                'parameter': 'ind_pool_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 8}},
            {
                'parameter': 'ind_pool_strides', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 7}},
            {
                'parameter': 'ind_separable', 'type': 'CATEGORICAL',
                'categorical_value_spec' : {'values': ['True', 'False']}},
        ],
    }

    return config


def final_alpha():
    config = {
        "train_steps": 20000,

        "inputs": {
            "label_columns": ["disp_E", "disp_N", "disp_J", "disp_S", "disp_B"],
            "primary_class": 0,

            "features": {
                "global_view": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "global_std": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "global_mask": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "global_transit_mask": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "global_view_0.3": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "global_view_5.0": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "local_view": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_std": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_mask": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_view_0.3": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_view_5.0": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_view_odd": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_std_odd": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_view_even": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_std_even": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "local_view_half_period_std": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "secondary_view": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "secondary_std": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "secondary_mask": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "secondary_view_0.3": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "secondary_view_5.0": {
                    "shape": [61],
                    "is_time_series": True,
                },
                "global_view_double_period": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "global_view_double_period_0.3": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "global_view_double_period_5.0": {
                    "shape": [201],
                    "is_time_series": True,
                },
                "sample_segments_view": {
                    "shape": [201, 14],
                    "is_time_series": True,
                },
                "sample_segments_view_0.3": {
                    "shape": [201, 14],
                    "is_time_series": True,
                },
                "sample_segments_view_5.0": {
                    "shape": [201, 14],
                    "is_time_series": True,
                },
                "sample_segments_local_view": {
                    "shape": [61, 16],
                    "is_time_series": True,
                },
                "sample_segments_local_view_0.3": {
                    "shape": [61, 16],
                    "is_time_series": True,
                },
                "sample_segments_local_view_5.0": {
                    "shape": [61, 16],
                    "is_time_series": True,
                },
                "Period": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 16.425697,
                    "std": 27.911264,
                    "has_nans": False,
                },
                "Duration": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.263291,
                    "std": 0.741017,
                    "has_nans": False,
                },
                "Transit_Depth": {
                    "shape": [1],
                    "is_time_series": False,
                    "log_scale": True,
                    "min_val": 0,
                    "max_val": 3.879001e+11,
                    "has_nans": False,
                },
                "Tmag": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 9.204857,
                    "std": 1.604575,
                    "has_nans": False,
                },
                "star_mass": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.741020,
                    "std": 0.815089,
                    "has_nans": True,
                },
                "star_rad": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 6.826552,
                    "std": 10.290280,
                    "has_nans": True,
                },
                "n_folds": {
                    "shape": [1],
                    "is_time_series": False,
                    "log_scale": True,
                    "min_val": 0,
                    "max_val": 100,
                    "has_nans": False,
                },
                "secondary_phase": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.497270,
                    "std": 0.200908,
                    "has_nans": True,
                },
                "secondary_phase_0.3": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.497270,
                    "std": 0.200908,
                    "has_nans": True,
                },
                "secondary_phase_5.0": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.497270,
                    "std": 0.200908,
                    "has_nans": True,
                },
                "local_scale": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.019245,
                    "std": 0.114780,
                    "has_nans": True,
                },
                "local_scale_0.3": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.019245,
                    "std": 0.114780,
                    "has_nans": True,
                },
                "local_scale_5.0": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.019245,
                    "std": 0.114780,
                    "has_nans": True,
                },
                "secondary_scale": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.006294,
                    "std": 0.032557,
                    "has_nans": True,
                },
                "secondary_scale_0.3": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.006294,
                    "std": 0.032557,
                    "has_nans": True,
                },
                "secondary_scale_5.0": {
                    "shape": [1],
                    "is_time_series": False,
                    "mean": 0.006294,
                    "std": 0.032557,
                    "has_nans": True,
                },
            },
        },

        "hparams": {
            "batch_size": 83,

            "learning_rate": 5.203528044134961e-06,
            "clip_gradient_norm": None,
            "optimizer": "adam",
            "one_minus_adam_beta_1": 0.16168028483420177,
            "one_minus_adam_beta_2": 0.022674419033475692,
            "adam_epsilon": 2.5037055725611666e-07,
            
            "use_batch_norm": False,
          
            "num_pre_logits_hidden_layers": 4,
            "pre_logits_hidden_layer_size": 482,
            "pre_logits_dropout_rate": 0.1690298097832756,
            
            "aux_inputs": [
                "Period",
                "Duration",
                "Transit_Depth",
                "Tmag",
                "star_mass",
                "star_mass_present",
                "star_rad",
                "star_rad_present",
                "n_folds",
                "local_scale",
                "local_scale_0.3",
                "local_scale_0.5",
                "local_scale_present",
                "local_scale_present_0.3",
                "local_scale_present_0.5",
                "secondary_scale",
                "secondary_scale_0.3",
                "secondary_scale_0.5",
                "secondary_scale_present",
                "secondary_scale_present_0.3",
                "secondary_scale_present_0.5",
            ],
          
            "time_series_hidden": {
                "global_view": {
                    "cnn_num_blocks": 3,
                    "cnn_block_size": 1,
                    "cnn_initial_num_filters": 17,
                    "cnn_block_filter_factor": 2,
                    "cnn_kernel_size": 3,
                    "convolution_padding": "same",
                    "pool_size": 5,
                    "pool_strides": 1,
                    "extra_channels": [
                        "global_std",
                        "global_mask",
                        "global_transit_mask",
                        "global_view_0.3",
                        "global_view_5.0",
                    ],
                },
                "global_view_double_period": {
                    "cnn_num_blocks": 3,
                    "cnn_block_size": 1,
                    "cnn_initial_num_filters": 17,
                    "cnn_block_filter_factor": 2,
                    "cnn_kernel_size": 3,
                    "convolution_padding": "same",
                    "pool_size": 7,
                    "pool_strides": 2,
                    "extra_channels": [
                        "global_view_double_period_0.3",
                        "global_view_double_period_5.0",
                    ],
                },
                "local_view": {
                    "cnn_num_blocks": 3,
                    "cnn_block_size": 1,
                    "cnn_initial_num_filters": 17,
                    "cnn_block_filter_factor": 2,
                    "cnn_kernel_size": 3,
                    "convolution_padding": "same",
                    "pool_size": 7,
                    "pool_strides": 1,
                    "extra_channels": [
                        "local_std",
                        "local_mask",
                        "local_view_0.3",
                        "local_view_5.0",
                        "local_view_odd",
                        "local_std_odd",
                        "local_view_even",
                        "local_std_even",
                        "local_half_period_std",
                    ],
                },
                "secondary_view": {
                    "cnn_num_blocks": 3,
                    "cnn_block_size": 1,
                    "cnn_initial_num_filters": 17,
                    "cnn_block_filter_factor": 2,
                    "cnn_kernel_size": 3,
                    "convolution_padding": "same",
                    "pool_size": 7,
                    "pool_strides": 1,
                    "extra_channels": [
                        "secondary_std",
                        "secondary_mask",
                        "secondary_view_0.3",
                        "secondary_view_5.0",
                    ],
                },
                "sample_segments_view": {
                    "cnn_num_blocks": 3,
                    "cnn_block_size": 1,
                    "cnn_initial_num_filters": 51,
                    "cnn_block_filter_factor": 2,
                    "cnn_kernel_size": 3,
                    "convolution_padding": "same",
                    "pool_size": 7,
                    "pool_strides": 2,
                    "multichannel": True,
                    "extra_channels": [
                        "sample_segments_view_0.3",
                        "sample_segments_view_5.0",
                    ],
                },
                "sample_segments_local_view": {
                    "cnn_num_blocks": 3,
                    "cnn_block_size": 1,
                    "cnn_initial_num_filters": 51,
                    "cnn_block_filter_factor": 2,
                    "cnn_kernel_size": 3,
                    "convolution_padding": "same",
                    "pool_size": 7,
                    "pool_strides": 2,
                    "multichannel": True,
                    "extra_channels": [
                        "sample_segments_local_view_0.3",
                        "sample_segments_local_view_5.0",
                    ],
                },
            },
        },

        "tune_params": [
            {
                'parameter': 'learning_rate', 'type': 'DOUBLE',
                'double_value_spec' : {'min_value': 1e-7, 'max_value': 1e-5},
                'scale_type': 'UNIT_LOG_SCALE'},
            {
                'parameter': 'batch_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 50, 'max_value' : 200}},


            {
                'parameter': 'one_minus_adam_beta_1', 'type': 'DOUBLE',
                'double_value_spec' : {'min_value': 0.01, 'max_value': 0.4},
                'scale_type': 'UNIT_LOG_SCALE'},
            {
                'parameter': 'one_minus_adam_beta_2', 'type': 'DOUBLE',
                'double_value_spec' : {'min_value': 0.001, 'max_value': 0.1},
                'scale_type': 'UNIT_LOG_SCALE'},
            {
                'parameter': 'adam_epsilon', 'type': 'DOUBLE',
                'double_value_spec' : {'min_value': 1e-8, 'max_value': 1e-6},
                'scale_type': 'UNIT_LOG_SCALE'},

            {
                'parameter': 'num_pre_logits_hidden_layers', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 2, 'max_value' : 5}},
            {
                'parameter': 'pre_logits_hidden_layer_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 200, 'max_value' : 600}},
            {
                'parameter': 'pre_logits_dropout_rate', 'type' : 'DOUBLE',
                'double_value_spec' : {'min_value' : 0.0, 'max_value' : 0.4}},

            {
                'parameter': 'global_cnn_num_blocks', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 0, 'max_value' : 4}},
            {
                'parameter': 'global_cnn_block_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 2}},
            {
                'parameter': 'global_cnn_initial_num_filters', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 10, 'max_value' : 50}},
            {
                'parameter': 'global_cnn_block_filter_factor', 'type' : 'DOUBLE',
                'double_value_spec' : {'min_value' : 0.1, 'max_value' : 3.0}},
            {
                'parameter': 'global_cnn_kernel_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 2, 'max_value' : 8}},
            {
                'parameter': 'global_pool_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 3, 'max_value' : 7}},
            {
                'parameter': 'global_pool_strides', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}},
            {
                'parameter': 'global_separable', 'type': 'CATEGORICAL',
                'categorical_value_spec' : {'values': ['True', 'False']}},

            {
                'parameter': 'globald_cnn_num_blocks', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 0, 'max_value' : 4}},
            {
                'parameter': 'globald_cnn_block_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 2}},
            {
                'parameter': 'globald_cnn_initial_num_filters', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 10, 'max_value' : 50}},
            {
                'parameter': 'globald_cnn_block_filter_factor', 'type' : 'DOUBLE',
                'double_value_spec' : {'min_value' : 0.1, 'max_value' : 3.0}},
            {
                'parameter': 'globald_cnn_kernel_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 2, 'max_value' : 8}},
            {
                'parameter': 'globald_pool_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 5, 'max_value' : 9}},
            {
                'parameter': 'globald_pool_strides', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}},
            {
                'parameter': 'globald_separable', 'type': 'CATEGORICAL',
                'categorical_value_spec' : {'values': ['True', 'False']}},

            {
                'parameter': 'local_cnn_num_blocks', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 0, 'max_value' : 4}},
            {
                'parameter': 'local_cnn_block_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 3}},
            {
                'parameter': 'local_cnn_initial_num_filters', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 10, 'max_value' : 50}},
            {
                'parameter': 'local_cnn_block_filter_factor', 'type' : 'DOUBLE',
                'double_value_spec' : {'min_value' : 0.2, 'max_value' : 3.0}},
            {
                'parameter': 'local_cnn_kernel_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}},
            {
                'parameter': 'local_pool_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 5, 'max_value' : 9}},
            {
                'parameter': 'local_pool_strides', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}},
            {
                'parameter': 'local_separable', 'type': 'CATEGORICAL',
                'categorical_value_spec' : {'values': ['True', 'False']}},

            {
                'parameter': 'sec_cnn_num_blocks', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 0, 'max_value' : 4}},
            {
                'parameter': 'sec_cnn_block_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 2}},
            {
                'parameter': 'sec_cnn_initial_num_filters', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 10, 'max_value' : 50}},
            {
                'parameter': 'sec_cnn_block_filter_factor', 'type' : 'DOUBLE',
                'double_value_spec' : {'min_value' : 0.2, 'max_value' : 3.0}},
            {
                'parameter': 'sec_cnn_kernel_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}},
            {
                'parameter': 'sec_pool_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 5, 'max_value' : 9}},
            {
                'parameter': 'sec_pool_strides', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}},
            {
                'parameter': 'sec_separable', 'type': 'CATEGORICAL',
                'categorical_value_spec' : {'values': ['True', 'False']}},

            {
                'parameter': 'ind_cnn_num_blocks', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 0, 'max_value' : 4}},
            {
                'parameter': 'ind_cnn_block_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 2}},
            {
                'parameter': 'ind_cnn_initial_num_filters', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 10, 'max_value' : 80}},
            {
                'parameter': 'ind_cnn_block_filter_factor', 'type' : 'DOUBLE',
                'double_value_spec' : {'min_value' : 0.2, 'max_value' : 3.0}},
            {
                'parameter': 'ind_cnn_kernel_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}},
            {
                'parameter': 'ind_pool_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 5, 'max_value' : 9}},
            {
                'parameter': 'ind_pool_strides', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 3}},
            {
                'parameter': 'ind_separable', 'type': 'CATEGORICAL',
                'categorical_value_spec' : {'values': ['True', 'False']}},

            {
                'parameter': 'lind_cnn_num_blocks', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 0, 'max_value' : 4}},
            {
                'parameter': 'lind_cnn_block_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 2}},
            {
                'parameter': 'lind_cnn_initial_num_filters', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 10, 'max_value' : 80}},
            {
                'parameter': 'lind_cnn_block_filter_factor', 'type' : 'DOUBLE',
                'double_value_spec' : {'min_value' : 0.2, 'max_value' : 3.0}},
            {
                'parameter': 'lind_cnn_kernel_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 5}},
            {
                'parameter': 'lind_pool_size', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 5, 'max_value' : 9}},
            {
                'parameter': 'lind_pool_strides', 'type' : 'INTEGER',
                'integer_value_spec' : {'min_value' : 1, 'max_value' : 3}},
            {
                'parameter': 'lind_separable', 'type': 'CATEGORICAL',
                'categorical_value_spec' : {'values': ['True', 'False']}},
        ],
    }

    return config
