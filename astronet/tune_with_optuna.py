import optuna
import optuna.visualization as vis
from optuna.integration import TFKerasPruningCallback
import datetime
import os

import tensorflow as tf
from absl import app, flags, logging

from astronet.astro_cnn_model.astro_cnn_model import AstroCNNModel
from astronet import models, training, evaluation
from astronet.astro_cnn_model.input_ds import build_train_dataset, build_eval_dataset
from astronet.util import config_util

import numpy as np

import math

import json

import pandas as pd

# 1) Define command-line flags
flags.DEFINE_string("model", None, "Name of the model class.", required=True)
flags.DEFINE_string("config_name", None, "Name of the model configuration.", required=True)
flags.DEFINE_string("config_file", None, "File containing the model configuration.")
flags.DEFINE_string("config_overrides", None, "Overrides to the base configuration.")
flags.DEFINE_string("train_files", None, "TFRecord patterns for training.", required=True)
flags.DEFINE_string("eval_files", None, "TFRecord patterns for evaluation.", required=True)
flags.DEFINE_string("model_dir", "./optuna_tuning", "Directory to save tuning results.")
flags.DEFINE_integer("n_trials", 20, "Number of Optuna trials to run.")
flags.DEFINE_integer("n_runs",   1,  "Number of independent runs per trial to average.")
flags.DEFINE_string("pretrain_model_dir", None, "Directory of pretrained model to initialize from.")
FLAGS = flags.FLAGS


#Helper functions that define the parameters and ranges to tune
def sample_phase1(trial, config):
    """
    Phase 1: quick sweep over the highest‐impact knobs.
    """
    # core optimizer hyperparams
    config["hparams"]["learning_rate"] = trial.suggest_float(
        "learning_rate", 1e-6, 1e-2, log=True
    )
    config["hparams"]["weight_decay"] = trial.suggest_float(
        "weight_decay", 1e-6, 1e-1, log=True
    )
    config["hparams"]["pre_logits_dropout_rate"] = trial.suggest_float(
        "pre_logits_dropout_rate", 0.0, 0.5
    )

    # dense head size/depth
    config["hparams"]["num_pre_logits_hidden_layers"] = trial.suggest_int(
        "num_pre_logits_hidden_layers", 1, 6
    )
    config["hparams"]["pre_logits_hidden_layer_size"] = trial.suggest_categorical(
        "pre_logits_hidden_layer_size", [128, 256, 512, 1024]
    )

    # warm‐start switches
    # config["init_from_pretrained_model"] = trial.suggest_categorical(
    #     "init_from_pretrained_model", [True, False]
    # )
    # config["freeze_pretrained_params"] = trial.suggest_categorical(
    #     "freeze_pretrained_params", [True, False]
    # )

def sample_phase2(trial, config):
    """
    Phase 2: expand to include regularization, optimizer choices, etc.
    Builds on phase1.
    """
    # always include the core phase1 space
    sample_phase1(trial, config)

    # # batch size
    # config["hparams"]["batch_size"] = trial.suggest_categorical(
    #     "batch_size", [128, 256, 512, 1024]
    # )

    # # optimizer family
    # config["hparams"]["optimizer"] = trial.suggest_categorical(
    #     "optimizer", ["adam", "sgd"]
    # )

    # Adam‐specific betas and eps
    config["hparams"]["one_minus_adam_beta_1"] = trial.suggest_float(
        "one_minus_adam_beta_1", 0.01, 0.5
    )
    config["hparams"]["one_minus_adam_beta_2"] = trial.suggest_float(
        "one_minus_adam_beta_2", 1e-4, 0.1, log=True
    )
    config["hparams"]["adam_epsilon"] = trial.suggest_float(
        "adam_epsilon", 1e-9, 1e-6, log=True
    )

    # gradient clipping
    config["hparams"]["clip_gradient_norm"] = trial.suggest_categorical(
        "clip_gradient_norm", [None, 1.0, 5.0, 10.0]
    )

    # regularization / smoothing
    config["hparams"]["label_smoothing"] = trial.suggest_float(
        "label_smoothing", 0.0, 0.2
    )

    # batch‐norm toggle
    config["hparams"]["use_batch_norm"] = trial.suggest_categorical(
        "use_batch_norm", [True, False]
    )

    # data‐augmentation toggle
    config["inputs"]["random_reverse_time_series"] = trial.suggest_categorical(
        "random_reverse_time_series", [True, False]
    )



# Trial is a optuna.trial.Trial object
def objective(trial):

    # 2) Sample hyperparameters
    config = models.get_model_config(FLAGS.model, FLAGS.config_name)
    if FLAGS.config_file:
        config = config_util.load_config_dict_from_file(FLAGS.config_file)
    if FLAGS.config_overrides:
        overrides = config_util.parse_config_str(FLAGS.config_overrides)
        config_util.update(config, overrides)
    if FLAGS.pretrain_model_dir:
        config["pretrain_model_dir"] = FLAGS.pretrain_model_dir

    # Hyperparameter sampling
    sample_phase1(trial, config)
    # sample_phase2(trial, config)

    # 3) Build model
    model_class = models.get_model_class(FLAGS.model)
    model = model_class(config)

    # 4) Compile with new hyperparameters
    training.compile_model(model, config)

    # 5) Prepare datasets
    train_ds = build_train_dataset(
        FLAGS.train_files,
        config["inputs"],
        batch_size=config["hparams"]["batch_size"]
    )
    eval_ds = build_eval_dataset(
        FLAGS.eval_files,
        config["inputs"],
        batch_size=config["hparams"]["batch_size"]
    )

    # 6) Set up callbacks: TensorBoard + pruning
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_dir = os.path.join(FLAGS.model_dir, f"trial_{trial.number}_{timestamp}")
    os.makedirs(trial_dir, exist_ok=True)

    # Define logdir once per trial
    tb_logdir = os.path.join(trial_dir, "tb_logs")
    pruning_cb = TFKerasPruningCallback(trial, "pr_auc")

    # 7) Optionally repeat training+eval multiple times for stability
    n_runs = FLAGS.n_runs
    pr_scores = []

    for run_idx in range(n_runs):
        # reproducible seed per run
        tf.random.set_seed(run_idx)

        # rebuild & compile fresh model each run
        model = model_class(config)
        training.compile_model(model, config)

        # # # Original:
        # # train & (prune on) this run
        # model.fit(
        #     train_ds,
        #     validation_data=eval_ds,
        #     epochs=1,  # or tune `epochs` if you like
        #     steps_per_epoch=config["train_steps"],
        #     callbacks=[tensorboard_cb, pruning_cb],
        #     validation_steps=100
        # )


        # # Option 1: Cardinality
        # # 1) figure out how many batches make one full pass
        # batches_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
        # # if your dataset is finite, this will be a positive integer

        # # 2) total number of steps you want
        # total_steps = config["train_steps"]

        # # 3) compute how many full passes (epochs) you need
        # n_epochs = math.ceil(total_steps / batches_per_epoch)

        # # 4) now call fit with true “full‐pass” epochs
        # model.fit(
        #     train_ds,
        #     validation_data=eval_ds,
        #     epochs=n_epochs,
        #     steps_per_epoch=batches_per_epoch,
        #     callbacks=[tensorboard_cb, pruning_cb],
        #     validation_steps=100,
        # )

        # Option 2: Log every batch to a single tb_logdir
        tensorboard_cb = tf.keras.callbacks.TensorBoard(
            log_dir=tb_logdir,
            update_freq=10,       # or an integer like 10 to log every 10 batches, or 'batch' to update every batch
        )

        model.fit(
            train_ds,
            validation_data=eval_ds,
            steps_per_epoch=config["train_steps"],
            epochs=1,
            callbacks=[tensorboard_cb, pruning_cb],
            validation_steps=100,
        )



        # evaluate
        metrics = model.evaluate(eval_ds, return_dict=True)
        pr = metrics.get("pr_auc")
        if pr is None:
            # fallback if key isn’t there
            pr = model.history.history["pr_auc"][-1]
        pr_scores.append(pr)

    # 8) return the average across runs
    return sum(pr_scores) / len(pr_scores)


def main(_):
    logging.info("Starting Optuna tuning with %d trials", FLAGS.n_trials)
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=FLAGS.n_trials)

    # Save best hyperparameters
    best = study.best_trial
    print("Best trial:")
    print(f"  Value: {best.value}")
    print("  Params:")
    for key, val in best.params.items():
        print(f"    {key}: {val}")

    # Plot and save hyperparameter importances
    fig = vis.plot_param_importances(study)
    outpath = os.path.join(FLAGS.model_dir, "param_importances.html")
    fig.write_html(outpath)
    print(f"Saved hyperparameter importances to {outpath}")

        # ————————————————————————————————
    # 1) Write out *all* trials to CSV
    rows = []
    for t in study.trials:
        row = {"trial_number": t.number, "value": t.value, "state": t.state.name}
        row.update(t.params)
        rows.append(row)
    df = pd.DataFrame(rows)
    trials_csv = os.path.join(FLAGS.model_dir, "optuna_trials.csv")
    df.to_csv(trials_csv, index=False)
    print(f"Saved all trial parameters to {trials_csv}")

    # 2) Save best params to JSON (or .txt)
    best_json = os.path.join(FLAGS.model_dir, "best_params.json")
    with open(best_json, "w") as f:
        json.dump({"value": best.value, "params": best.params}, f, indent=2)
    print(f"Saved best trial to      {best_json}")

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(main)
