#!/bin/bash
#SBATCH --job-name=wan_i2v_14b
#SBATCH --output=logs/slurm_%j_i2v14b.out
#SBATCH --error=logs/slurm_%j_i2v14b.err
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
NUM_SAMPLES=${NUM_SAMPLES:-70}  # Default to all 70, override with --export=NUM_SAMPLES=3
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-10}  # Default to 10 steps

echo "Configuration:"
echo "  NUM_SAMPLES: $NUM_SAMPLES"
echo "  NUM_INFERENCE_STEPS: $NUM_INFERENCE_STEPS"
echo ""

# Create timestamp for this run
TIMESTAMP=$(date +%s)
LOG_FILE="logs/i2v14b_480p_${TIMESTAMP}.log"

echo "Starting I2V generation..."
echo "Log file: $LOG_FILE"
echo ""

# Run the I2V generation script
python - <<'PY' 2>&1 | tee "$LOG_FILE"
import json, os, numpy as np
from pathlib import Path
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from transformers import CLIPVisionModel
from diffusers.utils import load_image
import torch, cv2

manifest = "data/starting_frames/starting_frames_manifest.jsonl"
out_root = Path("generated_videos/i2v_wan14b_480p")
out_root.mkdir(parents=True, exist_ok=True)

model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
print(f"Loading I2V model: {model_id}")
image_encoder = CLIPVisionModel.from_pretrained(
    model_id, 
    subfolder="image_encoder", 
    torch_dtype=torch.float32
)
vae = AutoencoderKLWan.from_pretrained(
    model_id, 
    subfolder="vae", 
    torch_dtype=torch.float32
)
pipe = WanImageToVideoPipeline.from_pretrained(
    model_id, 
    vae=vae, 
    image_encoder=image_encoder, 
    torch_dtype=torch.bfloat16
)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
pipe.to(device)
print("Model loaded successfully!")
print("")

# Get number of samples from environment
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "70"))
STEPS = int(os.environ.get("NUM_INFERENCE_STEPS", "10"))

def save_video_cv2(frames, out_path, fps=8):
    """Save video frames using OpenCV with uint8 conversion."""
    frames_u8 = []
    for f in frames:
        arr = np.asarray(f)
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        frames_u8.append(arr)
    
    h, w = frames_u8[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {out_path}")
    
    for f in frames_u8:
        if f.ndim == 2:
            f = np.stack([f]*3, axis=-1)
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    
    vw.release()

def gen(image_path, prompt, out_path, num_frames=81, steps=STEPS, guidance=5.0, seed=42):
    """Generate video from image."""
    image = load_image(image_path)
    max_area = 480 * 832
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))
    
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    video = pipe(
        image=image,
        prompt=prompt,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).frames[0]
    
    save_video_cv2(video, out_path, fps=8)

# Load manifest
entries = []
with open(manifest, "r") as f:
    for line in f:
        entries.append(json.loads(line))

# Limit to NUM_SAMPLES
entries = entries[:NUM_SAMPLES]

print(f"Generating {len(entries)} videos to {out_root} with steps={STEPS}...")
print("")

success_count = 0
error_count = 0

for i, e in enumerate(entries, 1):
    act = (e.get("activity") or "unknown").replace(" ", "_")
    prompt = e.get("prompt") or ""
    frame = e.get("frame_path")
    
    if not frame or not prompt:
        print(f"[{i}/{len(entries)}] SKIP: Missing frame or prompt")
        continue
    
    out_dir = out_root / act
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(frame).stem
    out_path = out_dir / f"{name}_i2v14b.mp4"
    
    try:
        print(f"[{i}/{len(entries)}] Processing: {act}")
        print(f"  Image: {frame}")
        print(f"  Prompt: {prompt}")
        gen(frame, prompt, out_path, seed=42 + i)
        print(f"  ✓ Saved to: {out_path}")
        success_count += 1
    except Exception as ex:
        print(f"  ✗ ERROR: {ex}")
        error_count += 1
    print("")

print("========================================")
print("Generation Complete!")
print(f"  Success: {success_count}")
print(f"  Errors: {error_count}")
print(f"  Total: {len(entries)}")
print("========================================")
PY

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE

