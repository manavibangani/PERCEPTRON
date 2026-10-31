#!/usr/bin/env bash
set -euo pipefail

task_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
task_venv="$task_repo_root/../.venv"

if [[ ! -x "$task_venv/bin/python" ]]; then
  python3 -m venv "$task_venv"
fi

"$task_venv/bin/python" -m pip install --upgrade pip setuptools wheel
# Resolve together: e.g. datasets and MLflow constrain PyArrow differently.
# Only the Python agent runtime is needed; city-server binaries require separate
# private credentials and are deliberately not downloaded by this installer.
AGENTSOCIETY_SKIP_BINARIES=1 "$task_venv/bin/python" -m pip install \
  -r "$task_repo_root/examples/advertising_ope/requirements.txt" -e "$task_repo_root"
"$task_venv/bin/python" "$task_repo_root/examples/advertising_ope/repair_grpc_metadata.py"
"$task_venv/bin/python" -m pip check

cd "$task_repo_root"
"$task_venv/bin/python" - <<'PY'
import datasets
import matplotlib
import numpy
import ray
import sklearn
import agentsociety

print("Criteo and AgentSociety dependencies installed; imports verified.")
PY
