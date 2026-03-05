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

import datetime
import os
import numpy as np
import tensorflow as tf
from absl import app, flags, logging
from astronet.astro_cnn_model.astro_cnn_model import AstroCNNModel
import pprint
import pandas as pd
from astronet import evaluation, models, training
from astronet.astro_cnn_model import input_ds
from astronet.util import config_util
import numpy as np

flags.DEFINE_string("astro_ids_file", None, "File containing the Astro IDs to exclude from training.")

flags.DEFINE_string("model", None, "Name of the model class.", required=True)

flags.DEFINE_string("config_name", None,
                    "Name of the model and training configuration.")

flags.DEFINE_string("config_file", None,
                    "File containing the model and training configuration.")

flags.DEFINE_string("config_overrides", None,
                    "Overrides to the base configuration.")

flags.DEFINE_string(
    "train_files",
    None,
    "Comma-separated list of file patterns matching the TFRecord files in "
    "the training dataset.",
    required=True,
)

flags.DEFINE_multi_string(
    "eval_files", None,
    "File patterns matching the TFRecord files in the evaluation dataset(s). "
    "Multiple evaluation datasets are allowed. The training set does not need "
    "to be specified. Each evaluation dataset can be named with the format "
    "name:file_patterns.")

flags.DEFINE_string(
    "model_dir",
    None,
    "Directory for model checkpoints and summaries.",
    required=True)

flags.DEFINE_enum("save_format", "h5", ["keras", "h5"],
                  "Format for saving the trained model.")

flags.DEFINE_string("pretrain_model_dir", None,
                    "Directory for pretrained model checkpoints.")

flags.DEFINE_integer("train_steps", None,
                     "Total number of steps to train the model for.")

flags.DEFINE_integer("shuffle_buffer_size", 25000,
                     "Size of the shuffle buffer for the training dataset.")

flags.DEFINE_bool(
    "dump_block_weights",
    False,
    "If True, log and save ts_blocks weights at start and end of training.",
)

flags.DEFINE_bool(
    "log_training_history",
    False,
    "If True, log and save per-step training and validation metrics to training_history.json.",
)

flags.DEFINE_integer(
    "early_stopping_patience",
    None,
    "Stop training if validation loss does not improve for this many validation checks.",
)

FLAGS = flags.FLAGS

def dump_block_weights(model, filepath):
    """
    Logs each block’s weights (with their Keras names) and saves them into a .npz archive.
    """
    weights_dict = {}
    for block_name, block in model.ts_blocks.items():
        # block.weights is a list of tf.Variable, each has a .name and a .numpy() value
        for var in block.weights:
            # e.g. var.name == "local_aperture_s_block_1_conv_1/kernel:0"
            clean_name = var.name.replace(":", "_").replace("/", "_")
            weights_dict[clean_name] = var.numpy()

    # make sure parent dir of filepath exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    np.savez(filepath, **weights_dict)
    logging.info(f"Saved block weights archive to {filepath}")


def main(_):
  logging.info('Entered train.py')
  # Keep track of training flags for record-keeping purposes.
  train_flags = {
      "model": FLAGS.model,
      "train_files": FLAGS.train_files,
      "eval_files": FLAGS.eval_files,
      "shuffle_buffer_size": FLAGS.shuffle_buffer_size,
      "astro_ids_file": FLAGS.astro_ids_file,
  }

  # Load the config.
  if bool(FLAGS.config_name) == bool(FLAGS.config_file):
    raise ValueError("Exactly one of config_name and config_file is required")
  if FLAGS.config_name:
    config = models.get_model_config(FLAGS.model, FLAGS.config_name)
    train_flags["config_name"] = FLAGS.config_name
    expt_name = f"{FLAGS.model}_{FLAGS.config_name}"
  else:
    config = config_util.load_config(FLAGS.config_file)
    train_flags["config_file"] = FLAGS.config_file
    logging.info(f"Loaded config from {FLAGS.config_file}")
    expt_name = FLAGS.model
  if FLAGS.config_overrides:
    overrides = config_util.parse_config_str(FLAGS.config_overrides)
    train_flags["config_overrides"] = overrides
    config_util.update(config, overrides)
    logging.info(f"Updated config with overrides {overrides}")

  # Set the number of training steps.
  if FLAGS.train_steps:
    config["train_steps"] = FLAGS.train_steps
    logging.info(f"Set config.train_steps to {FLAGS.train_steps}")
  if not config["train_steps"]:
    raise ValueError(
        "train_steps must be set in the config or via --train_steps")

  # Set the astro ids to exclude from training
  exclude_astro_ids = set()
  if FLAGS.astro_ids_file:
      exclude_astro_ids = set(pd.read_csv(FLAGS.astro_ids_file, header=None).iloc[:,0].tolist())
      logging.info(f"Loaded {len(exclude_astro_ids)} Astro IDs to exclude from training.")

# Build the model.
  model_class = models.get_model_class(FLAGS.model)
  model = model_class(config)
  init_from_pretrained_model = config.get("init_from_pretrained_model")
  # 1) Hard error: asked to init but no dir supplied
  if init_from_pretrained_model and not bool(FLAGS.pretrain_model_dir):
      logging.error(
          "init_from_pretrained_model=%r but --pretrain_model_dir=%r is not set",
          init_from_pretrained_model, FLAGS.pretrain_model_dir
      )
      raise ValueError(
          "init_from_pretrained_model=True requires --pretrain_model_dir to be set"
      )

  # 2) Gentle warning: dir supplied but not using it
  if not init_from_pretrained_model and bool(FLAGS.pretrain_model_dir):
      logging.warning(
          "Got --pretrain_model_dir=%r but init_from_pretrained_model=False; ignoring it.",
          FLAGS.pretrain_model_dir
      )
  if init_from_pretrained_model:
    pretrain_config = config_util.load_config(FLAGS.pretrain_model_dir)
    config_util.validate_pretrain_config(config, pretrain_config)
    pretrain_model = models.load_model("AstroCNNModel",
                                       FLAGS.pretrain_model_dir)
    train_flags["pretrain_model_dir"] = FLAGS.pretrain_model_dir
    for name, block in model.ts_blocks.items():
      pretrain_block = pretrain_model.ts_blocks.get(name)
      if pretrain_block is not None:
        block.set_weights(pretrain_block.get_weights())
        logging.info(f"Block '{name}': set params from pretrained model")
        if config.freeze_pretrained_params:
          block.trainable = False
        logging.info(f"Block '{name}': set trainable={block.trainable}")
        # if block.trainable:
        #   trainable_preloaded_blocks.append(name)
      else:
        logging.info(f"Block '{name}': no such block in pretrained model")
        # blocks_not_in_pretrained_model.append(name)

  # Make model directory and save the configs.
  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  model_dir = os.path.join(FLAGS.model_dir, f"{expt_name}_{timestamp}")
  os.makedirs(model_dir)
  if FLAGS.pretrain_model_dir:
    train_flags["pretrain_model_dir"] = FLAGS.pretrain_model_dir
  config_util.save_config(train_flags, model_dir, basename="train_flags")
  config_util.save_config(config, model_dir)

  logging.info('Starting training: %d steps shuffle_buffer=%d', config['train_steps'], FLAGS.shuffle_buffer_size)

  if FLAGS.dump_block_weights:
    init_dump_path = os.path.join(model_dir, "initial_block_weights.npz")
    dump_block_weights(model, init_dump_path)
    logging.info(f"Saved initial block weights to {init_dump_path}")

  # Before training, print the model summary.
  logging.info("Model summary:")
  model.summary()

  # Build validation dataset if eval_files are provided
  validation_data = None
  validation_steps = None
  if FLAGS.eval_files:
    # Use the first eval dataset for validation during training
    val_file_pattern = FLAGS.eval_files[0]
    # Remove name prefix if present (format: "name:file_pattern")
    if ":" in val_file_pattern:
      _, val_file_pattern = val_file_pattern.split(":", 1)
    logging.info(f"Building validation dataset from {val_file_pattern}")
    validation_data = input_ds.build_eval_dataset(
        file_pattern=val_file_pattern,
        input_config=config.inputs,
        batch_size=config.hparams.batch_size,
        include_identifiers=False,
        include_labels=True
    )
    # Optionally set validation_steps (None means use all validation data)
    # You could set this to a fixed number like 100 to speed up validation
    validation_steps = config.get("validation_steps", None)

  # Train and save model.
  history, step_metrics, val_step_metrics = training.train(
      model,
      config,
      train_files=FLAGS.train_files,
      shuffle_buffer_size=FLAGS.shuffle_buffer_size,
      exclude_astro_ids=exclude_astro_ids,  # pass it here
      validation_data=validation_data,
      validation_steps=validation_steps,
      log_training_history=FLAGS.log_training_history,
      early_stopping_patience=FLAGS.early_stopping_patience
  )

  # Save training history (per-step metrics) only if logging is enabled
  if FLAGS.log_training_history:
    # step_metrics is a dict mapping metric names to lists of values per training step
    # Convert to JSON-serializable format
    step_metrics_dict = {}
    for metric_name, values in step_metrics.items():
      # Values are already floats from the callback, but ensure they're serializable
      step_metrics_dict[metric_name] = [float(v) for v in values]

    # Add validation metrics if available (these are per-step, evaluated after each training step)
    if val_step_metrics:
      for metric_name, values in val_step_metrics.items():
        # Add 'val_' prefix to distinguish from training metrics
        val_key = f"val_{metric_name}"
        step_metrics_dict[val_key] = [float(v) for v in values]
      logging.info(f"Validation metrics collected per step: {list(val_step_metrics.keys())}")

    config_util.save_config(step_metrics_dict, model_dir, basename="training_history")
    n_train_steps = len(step_metrics_dict.get('loss', []))
    n_val_steps = len(step_metrics_dict.get('val_loss', []))
    logging.info(f"Saved training history ({n_train_steps} train steps, {n_val_steps} val steps) to {os.path.join(model_dir, 'training_history.json')}")

  # Also save per-epoch history for reference (though it only has 1 epoch)
  epoch_history_dict = {}
  for metric_name, values in history.history.items():
    epoch_history_dict[metric_name] = [float(v) for v in values]
  config_util.save_config(epoch_history_dict, model_dir, basename="training_history_epoch")

  # Save the model in the specified format.
  models.save_model(model, model_dir, FLAGS.save_format)

  if FLAGS.dump_block_weights:
    final_dump_path = os.path.join(model_dir, "final_block_weights.npz")
    dump_block_weights(model, final_dump_path)
    logging.info(f"Saved final block weights to {final_dump_path}")

  # Construct evaluation datasets.
  # This includes the training set and possibly additional datasets.
  eval_datasets = [("train", FLAGS.train_files)]
  for file_pattern in FLAGS.eval_files:
    if ":" in file_pattern:
      name, file_pattern = file_pattern.split(":")
    elif len(FLAGS.eval_files) == 1:
      # If there is only a single evaluation dataset, default name is "eval".
      name = "eval"
    else:
      raise ValueError("Multiple evaluation datasets must be named with format "
                       "name:file_patterns")
    eval_datasets.append((name, file_pattern))

  # Generate predictions on the evaluation datasets and save the output files.
  eval_dir = os.path.join(model_dir, "evaluation")
  if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)
  all_metrics = {}
  for name, file_pattern in eval_datasets:
    metrics, labels, predictions, astro_ids = evaluation.evaluate_model(
        model, config.inputs, file_pattern, config.hparams.batch_size, threshold=0.215)
    all_metrics[name] = metrics
    labels_path = os.path.join(eval_dir, f"{name}_label.npy")
    pred_path = os.path.join(eval_dir, f"{name}_pred.npy")
    astro_ids_path = os.path.join(eval_dir, f"{name}_astro_ids.npy")
    results_path = os.path.join(eval_dir, f"{name}_exodash_results.csv")
    np.save(labels_path, labels)
    np.save(pred_path, predictions)
    np.save(astro_ids_path, astro_ids)
    evaluation.export_dash_file(labels=labels, predictions=predictions, astro_ids=astro_ids, results_path=results_path)
    logging.info(f"Saved labels to {labels_path}")
    logging.info(f"Saved predictions to {pred_path}")
    logging.info(f"Saved astro_ids to {astro_ids_path}")
    logging.info(f"Saved results to {results_path}")
  evaluation.save_metrics(all_metrics, eval_dir)
  logging.info(f"Saved metrics to {eval_dir}")

if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  app.run(main)
