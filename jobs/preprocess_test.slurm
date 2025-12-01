#!/bin/bash
#SBATCH --job-name=wan21_preprocess_test
#SBATCH --output=logs/preprocess_test_%j.out
#SBATCH --error=logs/preprocess_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=H100
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --partition=ai
#SBATCH --account=soc250046-ai

echo ""

# Quick validation run for preprocessing pipeline

set -euo pipefail

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd /anvil/projects/x-soc250046/x-sishraq/WAN\ 2.1
mkdir -p logs

# Activate conda environment (robust)
if [ -f "/anvil/projects/x-soc250046/x-sishraq/.conda/etc/profile.d/conda.sh" ]; then
    source /anvil/projects/x-soc250046/x-sishraq/.conda/etc/profile.d/conda.sh
    conda activate /anvil/projects/x-soc250046/x-sishraq/.conda/envs/wan21
elif [ -f "/anvil/projects/x-soc250046/x-sishraq/anaconda3/etc/profile.d/conda.sh" ]; then
    source /anvil/projects/x-soc250046/x-sishraq/anaconda3/etc/profile.d/conda.sh
    conda activate wan21
else
    echo "Warning: conda init not found; trying 'conda activate wan21' directly"
    conda activate wan21 || { echo "Conda activation failed" >&2; exit 1; }
fi
export PYTHONUSERBASE=/anvil/projects/x-soc250046/x-sishraq/.local
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "Starting preprocessing TEST run..."
echo "Processing at most 1 video per label"

test_output_dir=data/processed_test
rm -rf "$test_output_dir"
mkdir -p "$test_output_dir"

python scripts/preprocess.py \
        --input_dir "data/raw/Human Activity Recognition - Video Dataset" \
        --output_dir "$test_output_dir" \
        --clip_len 81 \
        --stride 40 \
        --width 832 \
        --height 480 \
        --max_videos_per_label 1

EXIT_CODE=$?

echo ""
echo "Test preprocessing finished with exit code: $EXIT_CODE"
echo "Job ended at $(date)"

echo ""
echo "=== Test Summary ==="
echo "Output dir: $test_output_dir"
echo "Manifest entries: $(wc -l < $test_output_dir/train.jsonl 2>/dev/null || echo 0)"
echo "Total clips: $(find $test_output_dir/clips/ -name '*.npz' 2>/dev/null | wc -l)"

exit $EXIT_CODE
