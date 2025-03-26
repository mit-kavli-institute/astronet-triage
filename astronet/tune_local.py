"""Script for tuning an AstroNet model locally."""

import tensorflow as tf
from absl import app, flags, logging

from astronet.tuning import study
from astronet.util import config_util

flags.DEFINE_string(
    "config_file",
    None,
    "Path of JSON file containing the study configuration.",
    required=True)

flags.DEFINE_string(
    "study_dir", None, "Directory to write study results.", required=True)

flags.DEFINE_bool("overwrite", False,
                  "Whether to overwrite existing study directory.")

flags.DEFINE_integer("n_trials", None, "Maximum number of trials to run.")

flags.DEFINE_integer("gpu", None, "Index of GPU devide to run on.")

FLAGS = flags.FLAGS


def main(_):
  if FLAGS.gpu is not None:
    gpu_devices = tf.config.get_visible_devices("GPU")
    tf.config.set_visible_devices([gpu_devices[FLAGS.gpu]], "GPU")
    logging.info(
        f"Set logical GPU devices to {tf.config.list_logical_devices('GPU')}")

  study_config = config_util.load_config(FLAGS.config_file)

  # TODO(cshallue): support vetting model with pretrain_model_dir.
  study.run_tuning_study(
      study_config,
      study_dir=FLAGS.study_dir,
      n_trials=FLAGS.n_trials,
      overwrite=FLAGS.overwrite)


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  app.run(main)
