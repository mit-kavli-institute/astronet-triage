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
            "num_pre_logits_hidden_layers": 2,
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
                    "pool_strides": 2,
                    "extra_channels": [
                        "local_aperture_m",
                        "local_aperture_l",
                    ],
                },
            },
        },
    }
    
    config_util.merge_configs(config, configurations.extended())

    return config
