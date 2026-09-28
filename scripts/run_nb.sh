#!/usr/bin/env bash
# Convert a jupytext .py notebook to .ipynb and execute it so outputs are stored.
# Usage: scripts/run_nb.sh notebooks/01_preprocessing.py [timeout_seconds]
set -euo pipefail
src="$1"; timeout="${2:-1200}"
nb="${src%.py}.ipynb"
export PYTHONHASHSEED=0   # gensim seeds word vectors with hash(); fix it for repeatable runs
.venv/bin/jupytext --quiet --to ipynb "$src" -o "$nb"
start=$(date +%s)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout="$timeout" --ExecutePreprocessor.kernel_name=python3 "$nb"
echo "executed $nb in $(( $(date +%s) - start ))s"
