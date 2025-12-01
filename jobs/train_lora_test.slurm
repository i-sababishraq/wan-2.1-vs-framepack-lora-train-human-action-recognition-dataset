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
        echo "Conda activation failed; ensure the 'wan21' environment exists and conda is installed." >&2
        exit 1
    }

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
# Augmentation for short test
LORA_RANK=64
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

LORA_ALPHA=32
# Augmentation flags (empty to disable)
RANDOM_FLIP_FLAG="--random_flip"
COLOR_JITTER=0.08

echo "Starting quick training test..."

# Build CLI args reliably (prevents word splitting issues when flags are empty)
PYARGS=(
    training/train_lora.py
    --model_id "$MODEL_ID"
    --manifest "$MANIFEST"
    --output_dir "$OUTPUT_DIR"
    --latent_cache_dir "$LATENT_CACHE_DIR"
    --max_steps $MAX_STEPS
    --epochs $EPOCHS
    --batch_size $BATCH_SIZE
    --lr $LR
    --lora_rank $LORA_RANK
    --lora_alpha $LORA_ALPHA
    --frame_subsample $FRAME_SUBSAMPLE
    --resize_width $RESIZE_W
    --resize_height $RESIZE_H
    --save_every $SAVE_EVERY
)

# Optionally add augment flags
if [ "$RANDOM_FLIP_FLAG" != "" ]; then
    PYARGS+=("$RANDOM_FLIP_FLAG")
fi
if [ "$COLOR_JITTER" != "" ]; then
    PYARGS+=("--color_jitter" "$COLOR_JITTER")
fi

# Export effective runtime variables so the training script can record them reliably
export LORA_RANK="$LORA_RANK"
export LORA_ALPHA="$LORA_ALPHA"
if [ "$RANDOM_FLIP_FLAG" != "" ]; then
    export RANDOM_FLIP=1
else
    export RANDOM_FLIP=0
fi
export COLOR_JITTER="$COLOR_JITTER"

# Launch on a single GPU with accelerate (1 process)
python -m accelerate.commands.launch --num_processes 1 "${PYARGS[@]}"

EXIT_CODE=$?

echo "Test training finished with exit code: $EXIT_CODE"
echo "Job ended at $(date)"

exit $EXIT_CODE
