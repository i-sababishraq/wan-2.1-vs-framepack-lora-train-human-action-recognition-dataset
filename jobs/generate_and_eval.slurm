#!/bin/bash
#SBATCH --job-name=wan_eval_gen
#SBATCH --output=logs/generate_and_eval_%j.out
#SBATCH --error=logs/generate_and_eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=40:00:00
#SBATCH --partition=ai
#SBATCH --account=soc250046-ai

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
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=8
fi

# Ensure output directory exists
mkdir -p generated_videos/full

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

MODEL_ID="models/Wan2.1-T2V-1.3B-Local"
LORA_CKPT="checkpoints/lora_full/lora_final_step10000.pt"
MANIFEST="data/processed_full/train.jsonl"
OUT_DIR="generated_videos/full"

python scripts/eval_compare.py \
  --model_id "$MODEL_ID" \
  --lora_checkpoint "$LORA_CKPT" \
  --manifest "$MANIFEST" \
  --activities "Meet and Split" "Walking While Reading Book" "Standing Still" "Clapping" "Sitting" "Walking While Using Phone" "Walking" \
  --num_per_activity 10 \
  --num_inference_steps 100 \
  --seeds_count 8 \
  --seeds_base 1000 \
  --output_dir "$OUT_DIR" \
  --height 480 --width 832 --num_frames 81 \
  --guidance_scale 7.5

echo "Job finished"
