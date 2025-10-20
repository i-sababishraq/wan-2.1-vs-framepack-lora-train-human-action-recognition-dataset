#!/bin/bash
#SBATCH --job-name=wan21_train_lora_test
#SBATCH --output=logs/train_lora_test_%j.out
#SBATCH --error=logs/train_lora_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --constraint=H100
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --partition=ai
#SBATCH --account=soc250046-ai

# Lightweight test training job for LoRA (small dataset / fast run)

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd /anvil/projects/x-soc250046/x-sishraq/WAN\ 2.1
mkdir -p logs
mkdir -p checkpoints/lora_test
mkdir -p cache/latents_test

# Activate conda env
# Use the project's Anaconda installation (anaconda3) rather than a non-existent ".conda" prefix.
# This ensures `conda activate wan21` works inside the job.
if [ -f "/anvil/projects/x-soc250046/x-sishraq/anaconda3/etc/profile.d/conda.sh" ]; then
    source /anvil/projects/x-soc250046/x-sishraq/anaconda3/etc/profile.d/conda.sh
    conda activate wan21
else
    # Fallback: try to use conda if available on PATH
    echo "Warning: expected conda.sh not found at anaconda3; trying 'conda activate wan21' directly"
    conda activate wan21 || {
        echo "Conda activation failed; ensure the 'wan21' environment exists and conda is installed." >&2
        exit 1
    }
fi
export PYTHONUSERBASE=/anvil/projects/x-soc250046/x-sishraq/.local

# Env info
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "GPUs:"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Test training configuration
MODEL_ID="models/Wan2.1-T2V-1.3B-Local"
MANIFEST="data/processed_test/train.jsonl"
OUTPUT_DIR="checkpoints/lora_test"
LATENT_CACHE_DIR="cache/latents_test"
MAX_STEPS=20
EPOCHS=2
BATCH_SIZE=1
LR=1e-4
LORA_RANK=32
LORA_ALPHA=32
FRAME_SUBSAMPLE=2
RESIZE_W=512
RESIZE_H=320
SAVE_EVERY=10

echo "Starting quick training test..."

# Launch on a single GPU with accelerate (1 process)
python -m accelerate.commands.launch \
    --num_processes 1 \
    training/train_lora.py \
    --model_id "$MODEL_ID" \
    --manifest "$MANIFEST" \
    --output_dir "$OUTPUT_DIR" \
    --latent_cache_dir "$LATENT_CACHE_DIR" \
    --max_steps $MAX_STEPS \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --frame_subsample $FRAME_SUBSAMPLE \
    --resize_width $RESIZE_W \
    --resize_height $RESIZE_H \
    --save_every $SAVE_EVERY

EXIT_CODE=$?

echo "Test training finished with exit code: $EXIT_CODE"
echo "Job ended at $(date)"

exit $EXIT_CODE
