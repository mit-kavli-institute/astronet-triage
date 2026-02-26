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
"""Vetting configs for mixed ablation: global/local + scalar inputs only."""

import copy

from astronet.astro_cnn_model import configurations


def pablomer():
  triage_config = configurations.pablomer()
  aux_inputs = copy.deepcopy(triage_config["hparams"]["aux_inputs"])

  scalar_features = {
      name: spec
      for name, spec in triage_config["inputs"]["features"].items()
      if (not spec.get("is_time_series")) and (name in aux_inputs)
  }

  config = {
      "train_steps": 1000,
      "init_from_pretrained_model": False,
      "freeze_pretrained_params": False,
      "inputs": {
          "label_columns": ["disp_p", "disp_e", "disp_n", "disp_j"],
          "exclusive_labels": True,
          "label_scheme": "binary",
          "uncertainty_weight": False,
          "non_primary_downweight_factor": 2.0,
          "primary_class": 0,
          "random_reverse_time_series": False,
          "features": {
              "global_view": {
                  "shape": [201],
                  "is_time_series": True,
              },
              "local_view": {
                  "shape": [61],
                  "is_time_series": True,
              },
              **copy.deepcopy(scalar_features),
          },
      },
      "hparams": {
          "batch_size": 512,
          "learning_rate": 1e-5,
          "learning_rate_schedule": "constant",
          "learning_rate_warmup_frac": 0.0,
          "learning_rate_decay_alpha": 0.01,
          "one_minus_momentum": 0.1,
          "clip_gradient_norm": None,
          "optimizer": "adam",
          "one_minus_adam_beta_1": 0.1,
          "one_minus_adam_beta_2": 0.001,
          "adam_epsilon": 1e-7,
          "weight_decay": 0.005,
          "label_smoothing": 0.0,
          "use_batch_norm": True,
          "num_pre_logits_hidden_layers": 4,
          "pre_logits_hidden_layer_size": 512,
          "pre_logits_dropout_rate": 0.2,
          "aux_inputs": aux_inputs,
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
          },
      },
  }

  return config
