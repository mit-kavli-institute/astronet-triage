# Installing / updating astronet

> **For the QLP pipeline you usually do nothing.** Production QLP imports the shared
> `/sw/astronet` (a swadm-managed symlink). When that symlink is moved to a release,
> every QLP run uses it automatically. The steps below are for your **own dev/training
> clone** of this repo, or for testing a version before it is promoted.

This repository is **not** a pip package (there is no `setup.py`/`pyproject.toml`).
`import astronet` works only when the **repo root is on your `PYTHONPATH`**.
Third-party dependencies (TensorFlow < 2.15, absl-py, statsmodels, pydl,
`tensorboard.plugins.hparams`, typing_extensions, …) come from your conda/venv env.

## Update an existing clone
```bash
cd <your astronet clone>           # clone of git@github.com:mit-kavli-institute/astronet.git
git fetch origin
git checkout v3.1.0                # release matching the production vetting ensemble
#   (or: git checkout main && git pull   to track latest main)
```

## Fresh install
```bash
git clone git@github.com:mit-kavli-institute/astronet.git
cd astronet
git checkout v3.1.0
# use an env that already has the deps (e.g. the QLP venv or a training env),
# or create one and install TF<2.15 + the imports listed above.
```

## Put this clone on the path (pick one)
```bash
# (1) per session:
export PYTHONPATH=$(pwd):$PYTHONPATH
# (2) persistent for a conda env (recommended) — drop a .pth into site-packages:
echo "$(pwd)" > "$(python -c 'import site; print(site.getsitepackages()[0])')/astronet-repo.pth"
# (3) or always run your scripts from the repo root.
```

## Verify you are on the right version
```bash
python -c "import astronet; print(astronet.__file__)"   # -> this clone
git describe --tags                                     # -> v3.1.0
```

## Testing a model against this code without deploying to production
Point `--model-dir` at any ensemble dir and prepend this clone to `PYTHONPATH`
(prepend, so `qlp`/`k2astronet` stay resolvable when running inside the QLP env):
```bash
PYTHONPATH=$(pwd):$PYTHONPATH \
  qlp estools astronet --vetting --model-dir <ensemble-dir> -i <list>.ls -s <SECTOR> -n 72 -o out.csv
```

## Rollback
```bash
git checkout v3.0.1
# and ask swadm to point /sw/astronet back to astronet-3.0.1, and restore the
# cshallue model from /pdo/astronet-data/models/vetting/archive/  (see CHANGELOG.md).
```
