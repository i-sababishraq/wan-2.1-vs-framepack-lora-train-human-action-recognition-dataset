#!/usr/bin/env bash
set -euo pipefail

# hf_auth.sh - authenticate huggingface-cli using HF_TOKEN in .env
# Usage: Fill HF_TOKEN in .env, then run:
#   bash scripts/hf_auth.sh

ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  set -o allexport
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +o allexport
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set in $ENV_FILE. Please add your Hugging Face token to HF_TOKEN in the file or export it in your shell." >&2
  exit 1
fi

echo "Logging into Hugging Face CLI using token from $ENV_FILE..."

# Use conda run to ensure the correct environment has huggingface-cli installed
if conda info --envs | grep -q "^wan21\s"; then
  echo "Using conda env 'wan21' to run huggingface-cli"
  conda run -n wan21 huggingface-cli login --token "$HF_TOKEN"
else
  echo "Conda env 'wan21' not found or not listed. Attempting to run huggingface-cli from PATH."
  huggingface-cli login --token "$HF_TOKEN"
fi

echo "Hugging Face CLI login attempted. If no errors, token is configured for huggingface-cli." 
