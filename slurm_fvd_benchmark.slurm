#!/bin/bash
#SBATCH --job-name=fvd_benchmark
#SBATCH --partition=ai
#SBATCH --account=soc250046-ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_%j_fvd_benchmark.out
#SBATCH --error=logs/slurm_%j_fvd_benchmark.err

echo "=========================================="
echo "FVD Benchmark Job Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="
echo ""

# Change to project directory
cd "/anvil/projects/x-soc250046/x-sishraq/WAN 2.1" || exit 1

# Create logs directory
mkdir -p logs
mkdir -p fvd_results
mkdir -p models/i3d

echo "Current directory: $(pwd)"
echo ""

# Activate conda environment
echo "Activating conda environment..."
source /anvil/projects/x-soc250046/x-sishraq/anaconda3/etc/profile.d/conda.sh
conda activate /anvil/projects/x-soc250046/x-sishraq/.conda/envs/wan-2.1

# Set Python user base
export PYTHONUSERBASE=/anvil/projects/x-soc250046/x-sishraq/.local
export PATH=/anvil/projects/x-soc250046/x-sishraq/.local/bin:$PATH

# Load HuggingFace token
set -a
[ -f .env ] && . .env
set +a
[ -n "$HF_TOKEN" ] && export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

# Show GPU info
echo "GPU Information:"
nvidia-smi
echo ""

# Set parameters (can be overridden via sbatch --export)
export REFERENCE_DIR=${REFERENCE_DIR:-"data/reference_videos_mp4"}
export WAN_VIDEOS=${WAN_VIDEOS:-"generated_videos/i2v_wan14b_480p"}
export FRAMEPACK_VIDEOS=${FRAMEPACK_VIDEOS:-"generated_videos/i2v_framepack_480p"}
export OUTPUT_DIR=${OUTPUT_DIR:-"fvd_results"}
export MODEL_PATH=${MODEL_PATH:-"models/i3d"}
export BATCH_SIZE=${BATCH_SIZE:-4}
export NUM_FRAMES=${NUM_FRAMES:-16}

echo "Configuration:"
echo "  REFERENCE_DIR: $REFERENCE_DIR"
echo "  WAN_VIDEOS: $WAN_VIDEOS"
echo "  FRAMEPACK_VIDEOS: $FRAMEPACK_VIDEOS"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  NUM_FRAMES: $NUM_FRAMES"
echo ""

# Run the benchmark
echo "=========================================="
echo "Starting FVD Benchmark"
echo "=========================================="
echo ""

python scripts/benchmark_fvd.py \
    --reference_dir "$REFERENCE_DIR" \
    --wan_videos "$WAN_VIDEOS" \
    --framepack_videos "$FRAMEPACK_VIDEOS" \
    --output_dir "$OUTPUT_DIR" \
    --model_path "$MODEL_PATH" \
    --batch_size $BATCH_SIZE \
    --num_frames $NUM_FRAMES \
    --device cuda

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "FVD Benchmark Job Completed"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE

