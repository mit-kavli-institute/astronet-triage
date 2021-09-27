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

"""Script for training an AstroNet model."""

import argparse
import datetime
import os
import sys

from absl import app
from absl import logging

import tensorflow as tf

from astronet import models
from astronet.astro_cnn_model import input_ds
from astronet.astro_cnn_model import astro_cnn_model
from astronet.astro_cnn_model import astro_cnn_model_vetting
from astronet.astro_cnn_model import configurations
from astronet.astro_cnn_model import configurations_vetting
from astronet.util import config_util
from astronet.util import configdict

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model", type=str, required=True, help="Name of the model class.")

parser.add_argument(
    "--config_name",
    type=str,
    required=True,
    help="Name of the model and training configuration.")

parser.add_argument(
    "--train_files",
    type=str,
    required=True,
    help="Comma-separated list of file patterns matching the TFRecord files in "
    "the training dataset.")

parser.add_argument(
    "--eval_files",
    type=str,
    help="Comma-separated list of file patterns matching the TFRecord files in "
    "the validation dataset.")

parser.add_argument(
    "--model_dir",
    type=str,
    default="",
    help="Directory for model checkpoints and summaries.")

parser.add_argument(
    "--pretrain_model_dir",
    type=str,
    default="",
    help="Directory for pretrained model checkpoints.")

parser.add_argument(
    "--train_steps",
    type=int,
    default=12000,
    help="Total number of steps to train the model for.")

parser.add_argument(
    "--train_epochs",
    type=int,
    default=1,
    help="Total number of epochs to train the model for.")

parser.add_argument(
    "--shuffle_buffer_size",
    type=int,
    default=25000,
    help="Size of the shuffle buffer for the training dataset.")


def train(model, config):
    if FLAGS.model_dir:
        dir_name = "{}/{}_{}_{}".format(
            FLAGS.model_dir,
            FLAGS.model,
            FLAGS.config_name,
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        config_util.log_and_save_config(config, dir_name)

    ds = input_ds.build_dataset(
        file_pattern=FLAGS.train_files,
        input_config=config.inputs,
        batch_size=config.hparams.batch_size,
        include_labels=True,
        shuffle_filenames=True,
        shuffle_values_buffer=FLAGS.shuffle_buffer_size,
        repeat=None)

    if FLAGS.eval_files:
        eval_ds = input_ds.build_dataset(
            file_pattern=FLAGS.eval_files,
            input_config=config.inputs,
            batch_size=config.hparams.batch_size,
            include_labels=True,
            shuffle_filenames=False,
            repeat=1)
    else:
        eval_ds = None

    assert config.hparams.optimizer == 'adam'
    lr = config.hparams.learning_rate
    beta_1 = 1.0 - config.hparams.one_minus_adam_beta_1
    beta_2 = 1.0 - config.hparams.one_minus_adam_beta_2
    epsilon = config.hparams.adam_epsilon
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

    if config.inputs.get('exclusive_labels', False):
        loss = tf.keras.losses.CategoricalCrossentropy()
    else:
        loss = tf.keras.losses.BinaryCrossentropy()

    metrics = [
        tf.keras.metrics.Recall(
            name='r',
            class_id=config.inputs.primary_class,
            thresholds=0.2,
        ),
        tf.keras.metrics.Precision(
            name='p',
            class_id=config.inputs.primary_class,
            thresholds=0.2,
        ),
    ]
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    if getattr(config.hparams, 'decreasing_lr', False):
        def scheduler(epoch, lr):
            if epoch > 1:
                return lr / 10
            else:
                return lr
        callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler)]
    else:
        callbacks = []

    history = model.fit(
        ds,
        epochs=FLAGS.train_epochs,
        steps_per_epoch=FLAGS.train_steps,
        validation_data=eval_ds)

    if FLAGS.model_dir:
        model.save(dir_name)

    return history


def main(_):
    config = models.get_model_config(FLAGS.model, FLAGS.config_name)
    model_class = models.get_model_class(FLAGS.model) 

    if FLAGS.pretrain_model_dir:
        pretrain_model = tf.keras.models.load_model(
            os.path.join(FLAGS.pretrain_model_dir,
                         os.listdir(FLAGS.pretrain_model_dir + '/')[0]))
        model = model_class(config, pretrain_model)
    else:
        model = model_class(config)

    train(model, config)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)
