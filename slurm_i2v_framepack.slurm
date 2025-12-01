#!/bin/bash
#SBATCH --job-name=framepack_i2v
#SBATCH --output=logs/slurm_%j_framepack.out
#SBATCH --error=logs/slurm_%j_framepack.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=H100
#SBATCH --mem=64GB
#SBATCH --partition=ai
#SBATCH --account=soc250046-ai

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $HOSTNAME"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Navigate to project directory
cd "/anvil/projects/x-soc250046/x-sishraq/WAN 2.1" || exit 1

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
echo "Activating conda environment..."
source /anvil/projects/x-soc250046/x-sishraq/anaconda3/etc/profile.d/conda.sh
conda activate /anvil/projects/x-soc250046/x-sishraq/.conda/envs/wan-2.1

# Set environment variables
export PYTHONUSERBASE=/anvil/projects/x-soc250046/x-sishraq/.local
unset PYTHONNOUSERSITE
export PATH=/anvil/projects/x-soc250046/x-sishraq/.local/bin:$PATH

# Load HF token from .env file
set -a
[ -f .env ] && . .env
set +a
[ -n "$HF_TOKEN" ] && export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

# Show GPU info
echo "GPU Information:"
nvidia-smi
echo ""

# Set number of samples to process (can be overridden)
export NUM_SAMPLES=${NUM_SAMPLES:-70}  # Default to all 70
export NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-20}  #
export NUM_FRAMES=${NUM_FRAMES:-81}  # Default 81 frames
export HEIGHT=${HEIGHT:-480}
export WIDTH=${WIDTH:-832}
export GUIDANCE_SCALE=${GUIDANCE_SCALE:-9.0}
export FPS=${FPS:-8}

echo "Configuration:"
echo "  NUM_SAMPLES: $NUM_SAMPLES"
echo "  NUM_INFERENCE_STEPS: $NUM_INFERENCE_STEPS"
echo "  NUM_FRAMES: $NUM_FRAMES"
echo "  RESOLUTION: ${HEIGHT}x${WIDTH}"
echo "  GUIDANCE_SCALE: $GUIDANCE_SCALE"
echo "  FPS: $FPS"
echo ""

# Create timestamp for this run
TIMESTAMP=$(date +%s)
LOG_FILE="logs/framepack_i2v_${TIMESTAMP}.log"

echo "Starting Framepack I2V generation..."
echo "Log file: $LOG_FILE"
echo ""

# Run the Framepack generation script
python scripts/generate_i2v_framepack.py 2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE

