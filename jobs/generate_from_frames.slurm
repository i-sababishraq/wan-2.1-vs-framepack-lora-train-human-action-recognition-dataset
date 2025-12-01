#!/bin/bash
#SBATCH --job-name=wan_from_frames
#SBATCH --output=logs/generate_from_frames_%j.out
#SBATCH --error=logs/generate_from_frames_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=ai
#SBATCH --account=soc250046-ai

set -euo pipefail

# Activate conda environment
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

# Print GPU info
nvidia-smi

echo "Starting video generation from extracted frames..."
echo "Processing 70 frames (10 per activity × 7 activities)"

# Run video generation from extracted frames using full python path to avoid PATH issues
/anvil/projects/x-soc250046/x-sishraq/.conda/envs/wan21/bin/python3.10 scripts/generate_t2v_from_frames.py \
  --frames_dir data/first_frames \
  --output_dir generated_videos/wan_from_frames \
  --lora_path checkpoints/lora_full/lora_final_step13000.pt \
  --num_inference_steps 150 \
  --guidance_scale 9.0 \
  --height 720 \
  --width 1280 \
  --num_frames 20 \
  --device cuda \
  --seed 42

echo "Video generation complete!"
echo "Output directory: generated_videos/wan_from_frames"
echo "Total frames processed: 70 (10 per activity × 7 activities)"
