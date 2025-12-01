# WAN 2.1 — LoRA fine-tuning, generation, and evaluation pipeline

This repository contains tools to fine-tune a LoRA adapter on the Wan 2.1 (T2V 1.3B) video diffusion model, generate videos for Human Activity Recognition (HAR) activities, and compute quantitative metrics comparing base vs LoRA outputs (LPIPS, frame-level FID, proxy FVD). The project is organized as scripts, SLURM job wrappers, checkpoints, and generated outputs.

This README explains the end-to-end pipeline, how to run components, and the role of the key files you asked about.

## Quick overview — end-to-end pipeline
- Preprocess dataset -> produce `data/processed_full/train.jsonl` and frame/video assets (preprocess.py)
- Train LoRA adapters with `training/train_lora.py` (via `jobs/train_lora_full.slurm`) → saves `checkpoints/lora_full/lora_final_step{N}.pt`
- Generate videos (baseline & LoRA) with `scripts/generate_videos.py` or `scripts/eval_compare.py` (SLURM helper: `jobs/generate_and_eval.slurm`)
- Compute metrics with `scripts/compute_metrics.py` → per-activity LPIPS/FID/fvd_approx and per-video LPIPS
- Produce plots and reports via `scripts/plot_report.py` and merge metrics with `scripts/merge_metrics_all.py`

## Pipeline diagram
Below is a visual representation of the full pipeline. 

```
[Download model & dataset]
            |
            v
[Preprocess (preprocess.py)]
            |
            v
[Create manifest / cache (data/processed_full/train.jsonl)]
            |
            v
[Train LoRA (training/train_lora.py)] --(saves)-> [checkpoints/lora_full/]
            |
            v
[Generate videos (scripts/generate_videos.py / scripts/eval_compare.py)]
            |
            v
[Generated videos (generated_videos/*/)]
            |
            v
[Compute metrics (scripts/compute_metrics.py)]


SLURM helpers:
- `jobs/train_lora_full.slurm` -> wraps training + post-eval
- `jobs/generate_and_eval.slurm` -> robust generation + metrics wrapper
```


## How to run (examples)

Prerequisites
- Create/activate a conda env (recommended name `wan21`) and install packages listed in `requirements` below. Ensure CUDA appropriate `torch` wheel is installed.

Example: generate LoRA videos for a single activity (local run):
```bash
conda activate wan21
cd /path/to/WAN\ 2.1
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
python scripts/generate_videos.py \
  --model_id "models/Wan2.1-T2V-1.3B-Local" \
  --lora_checkpoint "checkpoints/lora_full/lora_final_step13000.pt" \
  --activities "Walking While Reading Book" \
  --num_per_activity 10 \
  --num_inference_steps 100 \
  --guidance_scale 8.5 \
  --output_dir generated_videos/walking_reading_book_lora
```

To run training on the cluster, use the SLURM wrapper (example already provided):
```bash
sbatch jobs/train_lora_full.slurm
```
This script will use `training/train_lora.py` and (if a checkpoint exists) resume from it. After training completes, `train_lora_full.slurm` also runs generation and metrics automatically.

To run evaluation/generation for a set of activities on the cluster:
```bash
sbatch jobs/generate_and_eval.slurm
```

## Requirements (packages)
- python >= 3.9
- torch (CUDA wheel matching the machine)
- torchvision
- diffusers
- transformers
- accelerate
- peft
- opencv-python-headless
- imageio, imageio-ffmpeg
- lpips (optional, for LPIPS)
- torchmetrics (optional, for FID)
- scipy, matplotlib

Install example (adjust torch index for CUDA):
```bash
conda create -n wan21 python=3.10 -y
conda activate wan21
# install torch for your CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate peft opencv-python-headless imageio imageio-ffmpeg lpips torchmetrics matplotlib scipy
```

## Where outputs live
- Checkpoints: `checkpoints/lora_full/`
- Generated videos: `generated_videos/` (activity subfolders)
- Merged metrics and per-video LPIPS: `generated_videos/full/` (merged json files and reports)

---
See `REPORT.md` for a per-file explanation and `INSIGHTS.md` for a concise summary of metrics, results, and challenges.

## Downloading the base model and dataset
If you don't already have the base Wan2.1 model and/or dataset locally, you can fetch them from the Hugging Face Hub (or other hosting). Two common options are shown below.

Download the model (Hugging Face Hub example)
```bash
# install huggingface_hub if needed
pip install huggingface-hub

# Python one-liner to snapshot the repo to a local folder
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", local_dir="models/Wan2.1-T2V-1.3B-Local")
print('Model downloaded to models/Wan2.1-T2V-1.3B-Local')
PY
```

Alternatively, if the model is stored as a Git LFS repo you can use `git lfs`:
```bash
git lfs install
git clone https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers models/Wan2.1-T2V-1.3B-Local
```

Download dataset (example placeholder)
```bash
# If the dataset is hosted as a tarball or on a public URL, a simple curl/wget + extract works
wget -O dataset.tar.gz "https://example.com/path/to/har_dataset.tar.gz"
tar -xzf dataset.tar.gz -C data/raw

# If the dataset is provided via Hugging Face Datasets or another API you can use a short Python snippet
python - <<'PY'
from datasets import load_dataset
ds = load_dataset('your_dataset_repo_or_name')
ds.save_to_disk('data/raw/har_dataset')
print('Dataset saved to data/raw/har_dataset')
PY
```

Replace the example URLs/repo names with the actual dataset/model sources you have access to.

## Generating for all activities
To run generation for every activity defined in `scripts/generate_videos.py` (the script's `ACTIVITY_PROMPTS`), you can either pass all activity names on the command line or loop through them. Two options:

# 1 Pass all activities in one command (space-separated):
```bash
python scripts/generate_videos.py \
  --model_id "models/Wan2.1-T2V-1.3B-Local" \
  --lora_checkpoint "checkpoints/lora_full/lora_final_step13000.pt" \
  --activities "Clapping" "Meet and Split" "Sitting" "Standing Still" "Walking" "Walking While Reading Book" "Walking While Using Phone" \
  --num_per_activity 10 \
  --num_inference_steps 100 \
  --guidance_scale 8.5 \
  --output_dir generated_videos/full
```

# 2 Loop (useful for per-activity logging / rate-limiting):
```bash
activities=("Clapping" "Meet and Split" "Sitting" "Standing Still" "Walking" "Walking While Reading Book" "Walking While Using Phone")
for a in "${activities[@]}"; do
  echo "Generating activity: $a"
  python scripts/generate_videos.py \
    --model_id "models/Wan2.1-T2V-1.3B-Local" \
    --lora_checkpoint "checkpoints/lora_full/lora_final_step13000.pt" \
    --activities "$a" \
    --num_per_activity 10 \
    --num_inference_steps 100 \
    --guidance_scale 8.5 \
    --output_dir generated_videos/full
done
```

When running on a cluster, prefer separate SLURM submissions per activity (one job per activity) so failures or walltime limits for one activity do not affect the others.

---
If you want, I can also produce a small `scripts/download_model.py` and `scripts/download_dataset.py` that wrap the Hugging Face `snapshot_download` and dataset extraction commands shown above — say the word and I will add them.
