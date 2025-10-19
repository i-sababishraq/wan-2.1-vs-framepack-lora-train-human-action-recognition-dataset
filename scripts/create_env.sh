#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=wan21
PYTHON_VERSION=3.10

echo "Creating conda environment '$ENV_NAME' with python=$PYTHON_VERSION..."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
echo "Done. To activate: conda activate $ENV_NAME"
