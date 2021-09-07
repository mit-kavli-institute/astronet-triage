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

"""Utility functions for configurations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os.path


from absl import logging
from astronet.util import configdict


import tensorflow as tf


def merge_configs(base, source):
    if not isinstance(source, dict) and isinstance(base, dict):
        raise ValueError(f'source is {type(soure)}, but base is {type(base)}')
    for k in source:
        if k not in base:
            base[k] = source[k]
        elif isinstance(source[k], dict):
            merge_configs(base[k], source[k])               
