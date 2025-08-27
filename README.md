This is a variation of the Liang model found in https://github.com/yuliang419/Astronet-Vetting.

# Usage

## Basic commands

Generate new input data:
```
python astronet/data/generate_input_records.py --input_tce_csv_file=astronet/tces-new+old.
csv --tess_data_dir=/home/${USER}/lc --output_dir=astronet/tfrecords-new+old --num_worker_processes=8 --make_test_set
```

Train ensemble (modify the .sh file to match the output_dir above):
```
./astronet/ensemble_train.sh
```

Tune (requires some setup, see Tune.ipynb):
```
python astronet/tune.py --model=AstroCNNModel --config_name=local_global_new --train_files=astronet/tfrecords-new\+old/test-0000[0-5]* --eval_files=astronet/tfrecords-new\+old/test-0000[6-6]* --train_steps=7000 --tune_trials=1000 --client_secrets=${HOME}/client_secrets.json --study_id=a_unique_string_id
```

Run predictions (or use Predict.ipynb for one-offs):
```
python astronet/predict.py --model_dir=/tmp/astronet/AstroCNNModel_local_global_multiclass_20200222_154634 --data_files=astronet/tfrecords-new\+old/* --output_file=/home/${USER}/predictions.csv
```
# Style and linting

* This repository follows the [Google Python style guide](https://google.github.io/styleguide/pyguide.html), with the following exceptions:
  1. Indentation is 2 spaces, not 4. The Google style guide says 4 to be consistent with PEP8, but Google's own internal and open source repositories use 2 spaces.
* Formatting is done automatically using the `yapf` formatter with `style=yapf`. This produces formatting consistent with the style guide. `yapf` can be installed via pip. Developers must run the following command before committing any code:
```bash
yapf -ri --style=yapf astronet/
```
* Import statements are sorted using `isort`. `isort` can be installed via pip. Developers must run the following command before committing any code:
```bash
isort  .
```
* Code is linted using `pylint`. `pylint` can be installed via pip. The configuration is stored in the `pylint` file in this directory. It is based on the file provided in the Google Python style guide. Developers must run the following command before committing any code, ensuring that there are no lint errors:
```bash
pylint --recursive=y astronet
```

For developers using VScode, the above tools (`yapf`, `isort`, `pylint`) can all be installed as extensions and run automatically while editing using following workspace settings:
```json
"[python]": {
    "editor.rulers": [
        80
    ],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": "explicit"
    },
    "editor.formatOnType": true,
    "editor.insertSpaces": true,
    "editor.tabSize": 2,
    "editor.defaultFormatter": "eeyore.yapf",
}
```