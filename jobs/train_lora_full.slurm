#!/bin/bash
#SBATCH --job-name=wan21_train_lora
#SBATCH --output=logs/train_lora_%j.out
#SBATCH --error=logs/train_lora_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --constraint=H100
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
#SBATCH --partition=ai
#SBATCH --account=soc250046-ai

# LoRA training job for Wan2.1 on processed_full dataset

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd /anvil/projects/x-soc250046/x-sishraq/WAN\ 2.1
mkdir -p logs
mkdir -p checkpoints/lora_full
mkdir -p cache/latents_full

# Activate conda env
# Prefer the project's Anaconda installation path so `conda activate wan21` works inside the job.
if [ -f "/anvil/projects/x-soc250046/x-sishraq/anaconda3/etc/profile.d/conda.sh" ]; then
    source /anvil/projects/x-soc250046/x-sishraq/anaconda3/etc/profile.d/conda.sh
    conda activate wan21
else
    echo "Warning: expected conda.sh not found at anaconda3; trying 'conda activate wan21' directly"
    conda activate wan21 || {
        echo "Conda activation failed; ensure the 'wan21' environment exists and conda is installed." >&2
        exit 1
    }
fi
export PYTHONUSERBASE=/anvil/projects/x-soc250046/x-sishraq/.local

# Reduce tokenizer spurious messages and avoid oversubscription of CPU threads
export TOKENIZERS_PARALLELISM=false
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
    export OMP_NUM_THREADS=12
fi

# Environment info
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "GPUs:"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# --- Training hyperparameters (edit below before submitting if desired) ---
MODEL_ID="models/Wan2.1-T2V-1.3B-Local"
MANIFEST="data/processed_full/train.jsonl"
OUTPUT_DIR="checkpoints/lora_full"
LATENT_CACHE_DIR="cache/latents_full"
MAX_STEPS=1000
EPOCHS=10
BATCH_SIZE=1
LR=1e-4
LORA_RANK=32
LORA_ALPHA=32
FRAME_SUBSAMPLE=2
RESIZE_W=512
RESIZE_H=320
SAVE_EVERY=100
# -----------------------------------------------------------------------

echo "Starting training with the following config:"
echo " model: $MODEL_ID"
echo " manifest: $MANIFEST"
echo " output: $OUTPUT_DIR"
echo " max_steps: $MAX_STEPS, epochs: $EPOCHS, batch_size: $BATCH_SIZE"
echo " frame_subsample: $FRAME_SUBSAMPLE, resize: ${RESIZE_W}x${RESIZE_H}"

echo "Launching accelerate with 2 processes..."

# Launch using Accelerate on 2 processes (GPUs on the same node)
python -m accelerate.commands.launch \
    --num_processes 2 \
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

echo "Training finished with exit code: $EXIT_CODE"
echo "Job ended at $(date)"

exit $EXIT_CODE
