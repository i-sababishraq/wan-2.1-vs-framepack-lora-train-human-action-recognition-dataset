#!/bin/bash
#SBATCH --job-name=wan21_gen_test
#SBATCH --output=logs/generate_videos_test_%j.out
#SBATCH --error=logs/generate_videos_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --constraint=H100
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --partition=ai
#SBATCH --account=soc250046-ai

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd /anvil/projects/x-soc250046/x-sishraq/WAN\ 2.1
mkdir -p logs
mkdir -p generated_videos/test

# Activate conda env (use the project anaconda3 path if present)
if [ -f "/anvil/projects/x-soc250046/x-sishraq/anaconda3/etc/profile.d/conda.sh" ]; then
    source /anvil/projects/x-soc250046/x-sishraq/anaconda3/etc/profile.d/conda.sh
    conda activate wan21
else
    echo "Warning: expected conda.sh not found at anaconda3; trying 'conda activate wan21' directly"
    conda activate wan21 || { echo "Conda activation failed" >&2; exit 1; }
fi
export PYTHONUSERBASE=/anvil/projects/x-soc250046/x-sishraq/.local
export TOKENIZERS_PARALLELISM=false
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
    export OMP_NUM_THREADS=6
fi

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "GPUs:"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Generation config (quick smoke test)
MODEL_ID="models/Wan2.1-T2V-1.3B-Local"
LORA_CHECKPOINT="checkpoints/lora_test/lora_final_step20.pt"
OUTPUT_DIR="generated_videos/test"
NUM_PER_ACTIVITY=2
NUM_FRAMES=21
HEIGHT=320
WIDTH=512
NUM_INFERENCE_STEPS=20
GUIDANCE_SCALE=7.5
SEED=1234

echo "Generating videos (quick test): $NUM_PER_ACTIVITY per activity, $NUM_FRAMES frames each"

python scripts/generate_videos.py \
    --model_id "$MODEL_ID" \
    --lora_checkpoint "$LORA_CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --num_per_activity $NUM_PER_ACTIVITY \
    --num_frames $NUM_FRAMES \
    --height $HEIGHT \
    --width $WIDTH \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --seed $SEED

EXIT_CODE=$?

echo "Generation finished with exit code: $EXIT_CODE"
echo "Job ended at $(date)"

exit $EXIT_CODE
