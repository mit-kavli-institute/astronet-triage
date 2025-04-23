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
"""Configurations for model building, training and evaluation."""

from astronet.astro_cnn_model import configurations
from astronet.util import config_util


def base():
  config = {
      "train_steps":
          20000,
      "inputs": {
          "label_columns": ["disp_p", "disp_e", "disp_n"],
          "primary_class": 0,
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
      "hparams": {
          'prediction_threshold': 0.2152499407880693,
          'batch_size': 83,
          'learning_rate': 5.203528044134961e-06,
          'clip_gradient_norm': None,
          'optimizer': 'adam',
          'one_minus_adam_beta_1': 0.16168028483420177,
          'one_minus_adam_beta_2': 0.022674419033475692,
          'adam_epsilon': 2.5037055725611666e-07,
          'use_batch_norm': False,
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
      "tune_params": [{
          "parameter": "exclusive_labels",
          "type": "CATEGORICAL",
          "categorical_value_spec": {
              "values": ["True", "False"]
          }
      }, {
          "parameter": "learning_rate",
          "type": "DOUBLE",
          "double_value_spec": {
              "min_value": 1e-6,
              "max_value": 1e-3
          },
          "scale_type": "UNIT_LOG_SCALE"
      }, {
          "parameter": "one_minus_adam_beta_1",
          "type": "DOUBLE",
          "double_value_spec": {
              "min_value": 1e-2,
              "max_value": 0.9
          },
          "scale_type": "UNIT_LOG_SCALE"
      }, {
          "parameter": "one_minus_adam_beta_2",
          "type": "DOUBLE",
          "double_value_spec": {
              "min_value": 1e-4,
              "max_value": 0.9
          },
          "scale_type": "UNIT_LOG_SCALE"
      }, {
          "parameter": "adam_epsilon",
          "type": "DOUBLE",
          "double_value_spec": {
              "min_value": 1e-8,
              "max_value": 1e-5
          },
          "scale_type": "UNIT_LOG_SCALE"
      }, {
          "parameter": "batch_size",
          "type": "INTEGER",
          "integer_value_spec": {
              "min_value": 8,
              "max_value": 128
          }
      }, {
          "parameter": "train_steps",
          "type": "INTEGER",
          "integer_value_spec": {
              "min_value": 10,
              "max_value": 6000
          }
      }, {
          "parameter": "num_pre_logits_hidden_layers",
          "type": "INTEGER",
          "integer_value_spec": {
              "min_value": 1,
              "max_value": 4
          }
      }, {
          "parameter": "pre_logits_hidden_layer_size",
          "type": "INTEGER",
          "integer_value_spec": {
              "min_value": 32,
              "max_value": 1024
          }
      }, {
          "parameter": "pre_logits_dropout_rate",
          "type": "DOUBLE",
          "double_value_spec": {
              "min_value": 0.0,
              "max_value": 0.4
          }
      }, {
          "parameter": "cnn_block_filter_factor",
          "type": "DOUBLE",
          "integer_value_spec": {
              "min_value": 0.2,
              "max_value": 5.0
          }
      }, {
          "parameter": "cnn_block_size",
          "type": "INTEGER",
          "integer_value_spec": {
              "min_value": 1,
              "max_value": 4
          }
      }, {
          "parameter": "cnn_initial_num_filters",
          "type": "INTEGER",
          "integer_value_spec": {
              "min_value": 4,
              "max_value": 32
          }
      }, {
          "parameter": "cnn_kernel_size",
          "type": "INTEGER",
          "integer_value_spec": {
              "min_value": 1,
              "max_value": 5
          }
      }, {
          "parameter": "cnn_num_blocks",
          "type": "INTEGER",
          "integer_value_spec": {
              "min_value": 0,
              "max_value": 5
          }
      }, {
          "parameter": "pool_size",
          "type": "INTEGER",
          "integer_value_spec": {
              "min_value": 1,
              "max_value": 7
          }
      }, {
          "parameter": "pool_strides",
          "type": "INTEGER",
          "integer_value_spec": {
              "min_value": 1,
              "max_value": 5
          }
      }],
  }

  triage_config = configurations.extended()
  config_util.merge_configs(config["inputs"]["features"],
                            triage_config["inputs"]["features"])

  return config


def vrevised():
  config = {
      "inputs": {
          "primary_class": 0,
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
              # "centroid_dist": {
              #     "shape": [1],
              #     "is_time_series": False,
              #     "vetting_only": True,
              #     "has_nans": True,
              # },
          },
      },
      "train_steps":
          1000,
      "hparams": {
          "batch_size":
              100,
          "learning_rate":
              1e-05,
          "optimizer":
              "adam",
          "one_minus_adam_beta_1":
              0.1,
          "one_minus_adam_beta_2":
              0.00,
          "adam_epsilon":
              1e-07,
          "use_batch_norm":
              False,
          "num_pre_logits_hidden_layers":
              3,
          "pre_logits_hidden_layer_size":
              256,
          "pre_logits_dropout_rate":
              0.1,
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
          "aux_inputs": [
              "Period",
              "Duration",
              "Transit_Depth",
              "star_mass",
              "star_mass_present",
              "star_rad",
              "star_rad_present",
              "local_scale",
              "secondary_scale",
          ],
      },
      "tune_params": [
          {
              "parameter": "train_steps",
              "type": "INTEGER",
              "integer_value_spec": {
                  "min_value": 10,
                  "max_value": 40000
              }
          },
          {
              "parameter": "learning_rate",
              "type": "DOUBLE",
              "double_value_spec": {
                  "min_value": 1e-7,
                  "max_value": 1e-2
              },
              "scale_type": "UNIT_LOG_SCALE"
          },
          {
              "parameter": "batch_size",
              "type": "INTEGER",
              "integer_value_spec": {
                  "min_value": 4,
                  "max_value": 256
              }
          },
          {
              "parameter": "one_minus_adam_beta_1",
              "type": "DOUBLE",
              "double_value_spec": {
                  "min_value": 1e-2,
                  "max_value": 0.9
              },
              "scale_type": "UNIT_LOG_SCALE"
          },
          {
              "parameter": "one_minus_adam_beta_2",
              "type": "DOUBLE",
              "double_value_spec": {
                  "min_value": 1e-4,
                  "max_value": 0.9
              },
              "scale_type": "UNIT_LOG_SCALE"
          },
          {
              "parameter": "adam_epsilon",
              "type": "DOUBLE",
              "double_value_spec": {
                  "min_value": 1e-8,
                  "max_value": 1e-5
              },
              "scale_type": "UNIT_LOG_SCALE"
          },
          {
              "parameter": "num_pre_logits_hidden_layers",
              "type": "INTEGER",
              "integer_value_spec": {
                  "min_value": 0,
                  "max_value": 4
              }
          },
          {
              "parameter": "pre_logits_hidden_layer_size",
              "type": "INTEGER",
              "integer_value_spec": {
                  "min_value": 4,
                  "max_value": 1024
              }
          },
          {
              "parameter": "pre_logits_dropout_rate",
              "type": "DOUBLE",
              "double_value_spec": {
                  "min_value": 0.0,
                  "max_value": 0.4
              }
          },
          {
              "parameter": "cnn_num_blocks",
              "type": "INTEGER",
              "integer_value_spec": {
                  "min_value": 0,
                  "max_value": 3
              }
          },
          {
              "parameter": "cnn_block_size",
              "type": "INTEGER",
              "integer_value_spec": {
                  "min_value": 1,
                  "max_value": 3
              }
          },
          {
              "parameter": "cnn_initial_num_filters",
              "type": "INTEGER",
              "integer_value_spec": {
                  "min_value": 1,
                  "max_value": 256
              }
          },
          {
              "parameter": "cnn_block_filter_factor",
              "type": "DOUBLE",
              "double_value_spec": {
                  "min_value": 0.2,
                  "max_value": 2.0
              }
          },
          {
              "parameter": "cnn_kernel_size",
              "type": "INTEGER",
              "integer_value_spec": {
                  "min_value": 1,
                  "max_value": 13
              }
          },
          {
              "parameter": "pool_strides",
              "type": "INTEGER",
              "integer_value_spec": {
                  "min_value": 1,
                  "max_value": 7
              }
          },
          {
              "parameter": "pool_size",
              "type": "INTEGER",
              "integer_value_spec": {
                  "min_value": 1,
                  "max_value": 9
              }
          },
          {
              "parameter": "convolution_padding",
              "type": "CATEGORICAL",
              "categorical_value_spec": {
                  "values": ["valid", "same"]
              }
          },
          {
              "parameter": "separable",
              "type": "CATEGORICAL",
              "categorical_value_spec": {
                  "values": ["True", "False"]
              }
          },
      ],
  }

  triage_config = configurations.revised_tuned()
  config_util.merge_configs(config["inputs"]["features"],
                            triage_config["inputs"]["features"])

  return config


def vrevised_tuned():
  # projects/mdan-playground/locations/us-central1/studies/
  # 6_vrevised_1b_vrevised
  config = vrevised()
  config["train_steps"] = 20000
  config["hparams"] = {
      "adam_epsilon": 6.1506818994358365e-06,
      "batch_size": 34,
      "learning_rate": 2.930120708847796e-06,
      "one_minus_adam_beta_1": 0.1813578657256338,
      "one_minus_adam_beta_2": 0.2230986830387786,
      "optimizer": "adam",
      "aux_inputs": [],
      "num_pre_logits_hidden_layers": 2,
      "pre_logits_dropout_rate": 0.1680582825717817,
      "pre_logits_hidden_layer_size": 630,
      "time_series_hidden": {
          "local_aperture_s": {
              "cnn_block_filter_factor": 0.9802741759212847,
              "cnn_block_size": 1,
              "cnn_initial_num_filters": 179,
              "cnn_kernel_size": 10,
              "cnn_num_blocks": 1,
              "convolution_padding": "same",
              "extra_channels": ["local_aperture_m", "local_aperture_l"],
              "pool_size": 4,
              "pool_strides": 5,
              "separable": False
          }
      },
      "use_batch_norm": False,
  }
  return config


def direct():
  config = vrevised_tuned()
  config["inputs"]["features"]["secondary_view"] = {
      "shape": [61],
      "is_time_series": True,
  }
  config["inputs"]["features"]["local_view"] = {
      "shape": [61],
      "is_time_series": True,
  }
  config["vetting_hparams"] = {
      "adam_epsilon": 6.1506818994358365e-06,
      "batch_size": 34,
      "learning_rate": 2.930120708847796e-06,
      "one_minus_adam_beta_1": 0.1813578657256338,
      "one_minus_adam_beta_2": 0.2230986830387786,
      "optimizer": "adam",
      "aux_inputs": [],
      "num_pre_logits_hidden_layers": 2,
      "pre_logits_dropout_rate": 0.1680582825717817,
      "pre_logits_hidden_layer_size": 630,
      "time_series_hidden": {
          "local_aperture_s": {
              "cnn_block_filter_factor": 0.9802741759212847,
              "cnn_block_size": 1,
              "cnn_initial_num_filters": 179,
              "cnn_kernel_size": 10,
              "cnn_num_blocks": 1,
              "convolution_padding": "same",
              "extra_channels": ["local_aperture_m", "local_aperture_l"],
              "pool_size": 4,
              "pool_strides": 5,
              "separable": False
          },
          "secondary_view": {
              "cnn_block_filter_factor": 0.8944552332537086,
              "cnn_block_size": 3,
              "cnn_initial_num_filters": 88,
              "cnn_kernel_size": 7,
              "cnn_num_blocks": 2,
              "convolution_padding": "valid",
              "extra_channels": ["secondary_mask"],
              "pool_size": 3,
              "pool_strides": 2,
              "separable": True
          },
          "local_view": {
              "cnn_block_filter_factor": 0.9307389694001378,
              "cnn_block_size": 1,
              "cnn_initial_num_filters": 14,
              "cnn_kernel_size": 8,
              "cnn_num_blocks": 2,
              "convolution_padding": "valid",
              "extra_channels": [
                  "local_view_0.3", "local_view_5.0", "local_view_odd",
                  "local_view_even", "local_std", "local_std_odd",
                  "local_std_even", "local_view_half_period_std", "local_mask",
                  "local_mask_odd", "local_mask_even"
              ],
              "pool_size": 4,
              "pool_strides": 2,
              "separable": True
          },
      },
      "use_batch_norm": False,
  }
  return config


def base_new():
  config = {
      "train_steps": 20000,
      "inputs": {
          "label_columns": ["disp_p", "disp_e", "disp_n", "disp_j"],
          "primary_class": 0,
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
      "hparams": {
          'adam_epsilon': 6.816179817920565e-07,
          'batch_size': 187,
          'clip_gradient_norm': None,
          'learning_rate': 9.967886235128946e-06,
          'one_minus_adam_beta_1': 0.1671901607461752,
          'one_minus_adam_beta_2': 0.0016810682046840494,
          'optimizer': 'adam',
          "aux_inputs": [],
          "num_pre_logits_hidden_layers": 2,
          "pre_logits_dropout_rate": 0.1680582825717817,
          "pre_logits_hidden_layer_size": 630,
          "time_series_hidden": {
              "local_aperture_s": {
                  "cnn_block_filter_factor": 0.9802741759212847,
                  "cnn_block_size": 1,
                  "cnn_initial_num_filters": 179,
                  "cnn_kernel_size": 10,
                  "cnn_num_blocks": 1,
                  "convolution_padding": "same",
                  "extra_channels": ["local_aperture_m", "local_aperture_l"],
                  "pool_size": 4,
                  "pool_strides": 5,
                  "separable": False
              },
              "secondary_view": {
                  "cnn_block_filter_factor": 0.8944552332537086,
                  "cnn_block_size": 3,
                  "cnn_initial_num_filters": 88,
                  "cnn_kernel_size": 7,
                  "cnn_num_blocks": 2,
                  "convolution_padding": "valid",
                  "extra_channels": ["secondary_mask"],
                  "pool_size": 3,
                  "pool_strides": 2,
                  "separable": True
              },
              "local_view": {
                  "cnn_block_filter_factor": 0.9307389694001378,
                  "cnn_block_size": 1,
                  "cnn_initial_num_filters": 14,
                  "cnn_kernel_size": 8,
                  "cnn_num_blocks": 2,
                  "convolution_padding": "valid",
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
                      #   "local_mask_odd",
                      #   "local_mask_even"
                  ],
                  "pool_size": 4,
                  "pool_strides": 2,
                  "separable": True
              },
          },
          "use_batch_norm": False,
      },
  }

  triage_config = configurations.final_alpha_1_tuned()
  config_util.merge_configs(config["inputs"]["features"],
                            triage_config["inputs"]["features"])
  config["tune_params"] = triage_config["tune_params"]

  return config


def cshallue():
  triage_config = configurations.cshallue()

  config = {
      "train_steps": 1000,
      "init_from_pretrained_model": True,
      "freeze_pretrained_params": False,
      "inputs": {
          "label_columns": ["disp_p", "disp_e", "disp_n", "disp_j"],
          "exclusive_labels": True,
          "label_scheme": "binary",
          "uncertainty_weight": False,
          "non_primary_downweight_factor": 2.0,
          "primary_class": 0,
          "random_reverse_time_series": False,
          "features": triage_config["inputs"]["features"],
      },
      "hparams": {
          "batch_size": 512,
          "learning_rate": 1e-5,
          "clip_gradient_norm": None,
          "optimizer": "adam",
          "one_minus_adam_beta_1": 0.1,
          "one_minus_adam_beta_2": 0.001,
          "adam_epsilon": 1e-07,
          "weight_decay": 0.005,
          "label_smoothing": 0.0,
          "use_batch_norm": True,
          "num_pre_logits_hidden_layers": 4,
          "pre_logits_hidden_layer_size": 512,
          "pre_logits_dropout_rate": 0.2,
          "aux_inputs": triage_config["hparams"]["aux_inputs"],
          "time_series_hidden": triage_config["hparams"]["time_series_hidden"],
      },
  }

  # Add extra vetting time series.
  config["inputs"]["features"].update({
      "local_aperture_s": {
          "shape": [61],
          "is_time_series": True,
      },
      "local_aperture_m": {
          "shape": [61],
          "is_time_series": True,
      },
      "local_aperture_l": {
          "shape": [61],
          "is_time_series": True,
      }
  })
  config["hparams"]["time_series_hidden"].update({
      "local_aperture_s": {
          "cnn_num_blocks": 1,
          "cnn_block_size": 3,
          "cnn_initial_num_filters": 16,
          "cnn_block_filter_factor": 2,
          "cnn_kernel_size": 5,
          "convolution_padding": "same",
          "pool_size": 5,
          "pool_strides": 2,
          "separable": False,
          "extra_channels": ["local_aperture_m", "local_aperture_l"],
      }
  })

  return config


def pablomer():
  triage_config = configurations.cshallue()

  config = {
      "train_steps": 1000,
      "inputs": {
        #   "label_columns": ["disp_p", "disp_e", "disp_n", "disp_j"],
          "label_columns": ["disp_p", "disp_e", "disp_j"],
          "exclusive_labels": True,
          "label_scheme": "maximum",
          "uncertainty_weight": False,
          "non_primary_downweight_factor": 2.0,
          "primary_class": 0,
          "random_reverse_time_series": False,
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
      "hparams": {
          "batch_size": 512,
          "learning_rate": 0.001,
          "clip_gradient_norm": None,
          "optimizer": "adam",
          "one_minus_adam_beta_1": 0.1,
          "one_minus_adam_beta_2": 0.001,
          "adam_epsilon": 1e-07,
          "weight_decay": 0.4375,
          "label_smoothing": 0.0,
          "use_batch_norm": False,
          "num_pre_logits_hidden_layers": 4,
          "pre_logits_hidden_layer_size": 512,
          "pre_logits_dropout_rate": 0.45,
          "aux_inputs": triage_config["hparams"]["aux_inputs"],
          "time_series_hidden": {
              "local_aperture_s": {
                  "cnn_num_blocks": 1,
                  "cnn_block_size": 3,
                  "cnn_initial_num_filters": 16,
                  "cnn_block_filter_factor": 2,
                  "cnn_kernel_size": 5,
                  "convolution_padding": "same",
                  "pool_size": 5,
                  "pool_strides": 2,
                  "separable": False,
                  "extra_channels": ["local_aperture_m", "local_aperture_l"],
              },
              "secondary_view": {
                  "cnn_num_blocks": 1,
                  "cnn_block_size": 3,
                  "cnn_initial_num_filters": 16,
                  "cnn_block_filter_factor": 2,
                  "cnn_kernel_size": 5,
                  "convolution_padding": "same",
                  "pool_size": 5,
                  "pool_strides": 2,
                  "separable": False,
                  "extra_channels": ["secondary_mask"],
              },
              "local_view": {
                  "cnn_num_blocks":
                      1,
                  "cnn_block_size":
                      3,
                  "cnn_initial_num_filters":
                      16,
                  "cnn_block_filter_factor":
                      2,
                  "cnn_kernel_size":
                      5,
                  "convolution_padding":
                      "same",
                  "pool_size":
                      5,
                  "pool_strides":
                      2,
                  "separable":
                      False,
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
                  ],
              },
          },
          "use_batch_norm": False,
      },
  }

  config_util.merge_configs(config["inputs"]["features"],
                            triage_config["inputs"]["features"])

  return config
