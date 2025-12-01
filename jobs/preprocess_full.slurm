#!/bin/bash
#SBATCH --job-name=wan21_preprocess
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=H100
#SBATCH --mem=64GB
#SBATCH --time=20:00:00
#SBATCH --partition=ai
#SBATCH --account=soc250046-ai

# Preprocessing job for Wan 2.1 HAR dataset
# This extracts clips from all videos in data/raw/ and creates train.jsonl manifest

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Navigate to project directory
cd /anvil/projects/x-soc250046/x-sishraq/WAN\ 2.1

# Create logs directory if it doesn't exist
mkdir -p logs

set -euo pipefail

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

# Print environment info
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Run preprocessing
echo ""
echo "Starting preprocessing of full dataset..."
echo "Input: data/raw/Human Activity Recognition - Video Dataset"
echo "Output: data/processed/"
echo "Clip length: 81 frames (4*20+1)"
echo "Stride: 40 frames"
echo "Resolution: 832x480"
echo ""

mkdir -p data/processed_full
python scripts/preprocess.py \
    --input_dir "data/raw/Human Activity Recognition - Video Dataset" \
    --output_dir data/processed_full \
    --clip_len 81 \
    --stride 40 \
    --width 832 \
    --height 480 \
    --max_videos_per_label 0

EXIT_CODE=$?

echo ""
echo "Preprocessing finished with exit code: $EXIT_CODE"
echo "Job ended at $(date)"

# Print summary statistics
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=== Preprocessing Summary ==="
    echo "Clips directory: data/processed_full/clips/"
    echo "Manifest: data/processed_full/train.jsonl"
    echo ""
    echo "Number of clips per activity:"
    for dir in data/processed_full/clips/*/; do
        activity=$(basename "$dir")
        count=$(find "$dir" -name "*.npz" 2>/dev/null | wc -l)
        echo "  $activity: $count clips"
    done
    echo ""
    echo "Total clips: $(find data/processed_full/clips/ -name "*.npz" 2>/dev/null | wc -l)"
    echo "Total manifest entries: $(wc -l < data/processed_full/train.jsonl 2>/dev/null || echo 0)"
fi

exit $EXIT_CODE
