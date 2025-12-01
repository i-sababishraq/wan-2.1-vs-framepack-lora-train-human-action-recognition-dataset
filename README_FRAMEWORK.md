# FVD Benchmark Framework - Usage Guide

This document provides comprehensive instructions for running the FVD (Fréchet Video Distance) benchmark framework to compare video generation methods.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Detailed Pipeline Steps](#detailed-pipeline-steps)
6. [Configuration Options](#configuration-options)
7. [Troubleshooting](#troubleshooting)
8. [Understanding Results](#understanding-results)

---

## Overview

This framework compares image-to-video (I2V) generation methods using the official FVD metric from the paper ["Towards Accurate Generative Models of Video"](https://openreview.net/pdf?id=rylgEULtdN).

**What it does**:
1. Extracts starting frames from reference videos
2. Generates videos using multiple I2V methods
3. Computes FVD scores using I3D feature extraction
4. Produces comparison reports

**Supported Methods**:
- Wan2.1 I2V (14B, 480P model)
- Framepack I2V (HunyuanVideo)
- Extensible to other I2V methods

---

## Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA GPU with 40GB+ VRAM (tested on H100 80GB)
- **RAM**: 64GB+ recommended
- **Storage**: 100GB+ free space
- **Compute**: SLURM cluster (or adapt for local execution)

### Software Requirements

- **Python**: 3.10+
- **CUDA**: 12.0+
- **Conda**: For environment management
- **Git**: For version control

---

## Installation

### Step 1: Clone Repository

```bash
cd /path/to/your/workspace
git clone <repository-url>
cd "WAN 2.1"
```

### Step 2: Create Conda Environment

```bash
# Create environment
conda create -n wan-2.1 python=3.10 -y
conda activate wan-2.1
```

### Step 3: Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install requirements
pip install -r requirements.txt
```

**requirements.txt** includes:
```
torch>=2.1.0
diffusers>=0.24.0
transformers>=4.30.0
accelerate>=0.22.0
peft>=0.4.0
opencv-python
tqdm
numpy
torchvision
huggingface-hub
scipy>=1.9.0
```

### Step 4: Set Up HuggingFace Token

Create a `.env` file in the project root:

```bash
echo "HF_TOKEN=your_huggingface_token_here" > .env
```

Get your token from: https://huggingface.co/settings/tokens

### Step 5: Download I3D Model (Optional)

The I3D model will be downloaded automatically when first running FVD computation. To download manually:

```bash
mkdir -p models/i3d
cd models/i3d
wget https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_imagenet.pt -O i3d_rgb.pt
cd ../..
```

---

## Quick Start

### Option 1: Run Complete Pipeline (SLURM)

```bash
# 1. Extract starting frames (if not already done)
python scripts/extract_starting_frames.py \
    --manifest data/processed_full/manifest.jsonl \
    --output_dir data/starting_frames \
    --samples_per_category 10

# 2. Convert reference videos
python scripts/convert_reference_videos.py \
    --manifest data/starting_frames/starting_frames_manifest.jsonl \
    --output_dir data/reference_videos_mp4

# 3. Generate Wan2.1 videos
sbatch slurm_i2v_wan14b.slurm

# 4. Generate Framepack videos
sbatch slurm_i2v_framepack.slurm

# 5. Run FVD benchmark
sbatch slurm_fvd_benchmark.slurm
```

### Option 2: Run Locally (Interactive)

```bash
# Activate environment
conda activate wan-2.1
export PYTHONUSERBASE=/path/to/.local
export PATH=/path/to/.local/bin:$PATH

# Run FVD computation directly
python scripts/compute_fvd.py \
    --real_videos data/reference_videos_mp4 \
    --generated_videos generated_videos/i2v_wan14b_480p \
    --output fvd_results/fvd_wan2.1_test.json \
    --batch_size 4 \
    --num_frames 16 \
    --device cuda
```

---

## Detailed Pipeline Steps

### Step 1: Prepare Reference Videos

#### 1.1 Extract Starting Frames

Extract the first frame from reference videos to use as conditioning for I2V generation.

```bash
python scripts/extract_starting_frames.py \
    --manifest data/processed_full/manifest.jsonl \
    --output_dir data/starting_frames \
    --samples_per_category 10
```

**Parameters**:
- `--manifest`: Path to video manifest (JSONL format)
- `--output_dir`: Where to save extracted frames
- `--samples_per_category`: Number of videos per activity category

**Output**:
- `data/starting_frames/<category>/<video_id>_frame0.png`
- `data/starting_frames/starting_frames_manifest.jsonl`

#### 1.2 Convert Reference Videos to MP4

Convert preprocessed .npz clips to standard MP4 format for FVD computation.

```bash
python scripts/convert_reference_videos.py \
    --manifest data/starting_frames/starting_frames_manifest.jsonl \
    --output_dir data/reference_videos_mp4 \
    --fps 8
```

**Parameters**:
- `--manifest`: Starting frames manifest
- `--output_dir`: Output directory for MP4 videos
- `--fps`: Frames per second (default: 8)

**Output**:
- `data/reference_videos_mp4/<category>/<video_id>.mp4`

---

### Step 2: Generate Videos with Wan2.1 I2V

#### 2.1 Submit SLURM Job

```bash
sbatch slurm_i2v_wan14b.slurm
```

#### 2.2 Custom Configuration

Edit `slurm_i2v_wan14b.slurm` or override via environment variables:

```bash
sbatch --export=ALL,NUM_SAMPLES=70,NUM_INFERENCE_STEPS=20 slurm_i2v_wan14b.slurm
```

**Environment Variables**:
- `NUM_SAMPLES`: Number of videos to generate (default: 70)
- `NUM_INFERENCE_STEPS`: Diffusion steps (default: 20)

**Monitor Progress**:
```bash
# Check job status
squeue -u $USER

# Watch output log
tail -f logs/slurm_<JOB_ID>_i2v14b.out
```

**Output**:
- Videos saved to: `generated_videos/i2v_wan14b_480p/<category>/<video_id>_i2v14b.mp4`

---

### Step 3: Generate Videos with Framepack I2V

#### 3.1 Submit SLURM Job

```bash
sbatch slurm_i2v_framepack.slurm
```

#### 3.2 Custom Configuration

```bash
sbatch --export=ALL,NUM_SAMPLES=70,NUM_INFERENCE_STEPS=20,GUIDANCE_SCALE=9.0 \
    slurm_i2v_framepack.slurm
```

**Environment Variables**:
- `NUM_SAMPLES`: Number of videos to generate (default: 70)
- `NUM_INFERENCE_STEPS`: Diffusion steps (default: 20)
- `NUM_FRAMES`: Frames per video (default: 81)
- `HEIGHT`: Video height (default: 480)
- `WIDTH`: Video width (default: 832)
- `GUIDANCE_SCALE`: Classifier-free guidance (default: 9.0)
- `FPS`: Frames per second (default: 8)

**Output**:
- Videos saved to: `generated_videos/i2v_framepack_480p/<category>/<video_id>_framepack.mp4`

---

### Step 4: Compute FVD Benchmark

#### 4.1 Submit SLURM Job

```bash
sbatch slurm_fvd_benchmark.slurm
```

#### 4.2 Custom Paths

```bash
sbatch --export=ALL,REFERENCE_DIR=data/reference_videos_mp4,\
WAN_VIDEOS=generated_videos/i2v_wan14b_480p,\
FRAMEPACK_VIDEOS=generated_videos/i2v_framepack_480p,\
BATCH_SIZE=8 \
slurm_fvd_benchmark.slurm
```

**Environment Variables**:
- `REFERENCE_DIR`: Path to reference videos
- `WAN_VIDEOS`: Path to Wan2.1 generated videos
- `FRAMEPACK_VIDEOS`: Path to Framepack generated videos
- `OUTPUT_DIR`: Results directory (default: `fvd_results`)
- `BATCH_SIZE`: Videos per batch (default: 4)
- `NUM_FRAMES`: Frames to sample per video (default: 16)

#### 4.3 Run Locally (Python)

For individual FVD computation:

```bash
python scripts/compute_fvd.py \
    --real_videos data/reference_videos_mp4 \
    --generated_videos generated_videos/i2v_wan14b_480p \
    --model_path models/i3d \
    --batch_size 4 \
    --num_frames 16 \
    --device cuda \
    --output fvd_results/fvd_wan2.1.json
```

For full benchmark comparison:

```bash
python scripts/benchmark_fvd.py \
    --reference_dir data/reference_videos_mp4 \
    --wan_videos generated_videos/i2v_wan14b_480p \
    --framepack_videos generated_videos/i2v_framepack_480p \
    --output_dir fvd_results \
    --batch_size 4
```

**Output Files**:
- `fvd_results/fvd_wan2.1_<timestamp>.json`
- `fvd_results/fvd_framepack_<timestamp>.json`
- `fvd_results/benchmark_summary_<timestamp>.json`

---

## Configuration Options

### Video Generation Parameters

| Parameter | Wan2.1 Default | Framepack Default | Description |
|-----------|---------------|-------------------|-------------|
| `num_inference_steps` | 20 | 20 | Diffusion denoising steps |
| `num_frames` | 81 | 81 | Frames per video |
| `height` | 480 | 480 | Video height in pixels |
| `width` | 832 | 832 | Video width in pixels |
| `guidance_scale` | 5.0 | 9.0 | Classifier-free guidance |
| `fps` | 8 | 8 | Frames per second |

### FVD Computation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 4 | Videos processed simultaneously |
| `num_frames` | 16 | Frames sampled per video for I3D |
| `target_size` | (224, 224) | I3D input resolution |
| `device` | cuda | Device for computation |

---

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Reduce batch size:
   ```bash
   sbatch --export=ALL,BATCH_SIZE=2 slurm_fvd_benchmark.slurm
   ```

2. Use smaller videos:
   ```bash
   export NUM_FRAMES=61  # Instead of 81
   ```

3. Request more GPU memory in SLURM:
   ```bash
   #SBATCH --gpus-per-node=1
   #SBATCH --constraint="h100"  # Request H100 with 80GB
   ```

### Issue 2: I3D Model Not Loading

**Symptoms**:
```
FileNotFoundError: models/i3d/i3d_rgb.pt not found
```

**Solution**:
```bash
mkdir -p models/i3d
wget https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_imagenet.pt \
    -O models/i3d/i3d_rgb.pt
```

### Issue 3: HuggingFace Token Error

**Symptoms**:
```
OSError: Access to model requires authentication token
```

**Solution**:
1. Create `.env` file with HF token
2. Or export directly:
   ```bash
   export HUGGINGFACE_HUB_TOKEN="your_token_here"
   ```

### Issue 4: SLURM Job Fails Immediately

**Check logs**:
```bash
# Find your job ID
squeue -u $USER

# Check output
cat logs/slurm_<JOB_ID>_*.out

# Check errors
cat logs/slurm_<JOB_ID>_*.err
```

**Common fixes**:
1. Wrong partition: Change `#SBATCH --partition=ai`
2. Wrong account: Change `#SBATCH --account=<your-account>`
3. Conda not activated: Verify paths in SLURM script

### Issue 5: Video Format Errors

**Symptoms**:
```
cv2.error: OpenCV cannot open video file
```

**Solution**:
Ensure ffmpeg is installed:
```bash
conda install ffmpeg -c conda-forge
```

---

## Understanding Results

### FVD Score Interpretation

According to the [official paper](https://openreview.net/pdf?id=rylgEULtdN):

| FVD Range | Quality | Description |
|-----------|---------|-------------|
| **< 50** | Exceptional | Nearly indistinguishable from real videos |
| **50-100** | Excellent | Very close to real videos |
| **100-300** | Good | Reasonable similarity |
| **300-500** | Moderate | Noticeable differences |
| **> 500** | Poor | Significant differences |

**Lower FVD = Better Quality**

### Reading Result Files

#### Individual FVD Result (JSON)

Example for Wan2.1:

```json
{
  "fvd": 40.73,
  "num_real_videos": 70,
  "num_generated_videos": 70,
  "real_features_shape": [70, 1024],
  "generated_features_shape": [70, 1024],
  "real_videos_dir": "data/reference_videos_mp4",
  "generated_videos_dir": "generated_videos/i2v_wan14b_480p"
}
```

**Key Fields**:
- `fvd`: The FVD score (lower is better)
- `num_*_videos`: Number of videos processed
- `*_features_shape`: Dimensions of extracted I3D features (1024 for I3D)

#### Benchmark Summary (JSON)

```json
{
  "timestamp": "20251106_090712",
  "reference_dir": "data/reference_videos_mp4",
  "methods": {
    "wan2.1_i2v": {
      "fvd": 40.73,
      "num_videos": 70,
      "output_file": "fvd_results/fvd_wan2.1_20251106_090712.json"
    },
    "framepack_i2v": {
      "fvd": 32.49,
      "num_videos": 70,
      "output_file": "fvd_results/fvd_framepack_20251106_090712.json"
    }
  }
}
```

**Comparison**:
- **Framepack I2V: 32.49**  Winner (20.2% better)
- Wan2.1 I2V: 40.73

Both models achieve excellent quality (FVD < 50), but Framepack demonstrates superior temporal coherence and frame conditioning.

---

## Advanced Usage

### Adding a New Video Generation Method

1. **Create generation script**:
   ```python
   # scripts/generate_i2v_mymethod.py
   from your_method import load_pipeline
   
   # Load model
   pipe = load_pipeline("model/path")
   
   # Generate videos
   for entry in manifest:
       video = pipe(
           image=entry["frame_path"],
           prompt=entry["prompt"],
           num_frames=81
       )
       save_video(video, output_path)
   ```

2. **Create SLURM script**:
   ```bash
   # slurm_i2v_mymethod.slurm
   #!/bin/bash
   #SBATCH --job-name=mymethod_i2v
   # ... other SLURM directives ...
   
   python scripts/generate_i2v_mymethod.py
   ```

3. **Update benchmark script**:
   ```python
   # In scripts/benchmark_fvd.py, add:
   mymethod_results = run_fvd_computation(
       real_videos=args.reference_dir,
       generated_videos="generated_videos/i2v_mymethod",
       output_file=str(output_dir / f"fvd_mymethod_{timestamp}.json"),
       ...
   )
   ```

### Batch Processing Multiple Experiments

```bash
# Create experiment script
cat > run_experiments.sh << 'EOF'
#!/bin/bash

for steps in 10 15 20 30; do
    echo "Running with $steps inference steps..."
    sbatch --export=ALL,NUM_INFERENCE_STEPS=$steps,OUTPUT_SUFFIX="_steps${steps}" \
        slurm_i2v_wan14b.slurm
done
EOF

chmod +x run_experiments.sh
./run_experiments.sh
```

### Computing FVD for Subset of Videos

```bash
# Create temporary directory with subset
mkdir -p data/subset_videos
find data/reference_videos_mp4 -name "*.mp4" | head -20 | xargs -I {} cp {} data/subset_videos/

# Run FVD on subset
python scripts/compute_fvd.py \
    --real_videos data/subset_videos \
    --generated_videos generated_videos/i2v_wan14b_480p \
    --output fvd_results/fvd_subset.json
```

---

## Performance Optimization

### GPU Utilization

- **Batch Size**: Increase for better GPU utilization
  ```bash
  export BATCH_SIZE=8  # If you have 80GB GPU
  ```

- **Mixed Precision**: Enable in generation scripts
  ```python
  pipe.enable_model_cpu_offload()
  pipe.enable_vae_tiling()
  ```

### Storage Optimization

- **Compression**: Use higher compression for videos
  ```python
  # In video saving function
  writer = imageio.get_writer(
      str(out_path),
      codec='libx264',
      ffmpeg_params=['-crf', '23']  # Lower = better quality
  )
  ```

### Parallel Execution

Run multiple independent jobs:
```bash
# Generate all methods in parallel
sbatch slurm_i2v_wan14b.slurm
sbatch slurm_i2v_framepack.slurm

# FVD will be run after both complete
```

---

## Citation

If you use this framework, please cite:

```bibtex
@article{fvd2018,
  title={Towards Accurate Generative Models of Video: A New Metric \& Challenges},
  author={Unterthiner, Thomas and van Steenkiste, Sjoerd and Kurach, Karol and 
          Marinier, Rapha{\"e}l and Michalski, Marcin and Gelly, Sylvain},
  journal={arXiv preprint arXiv:1812.01717},
  year={2018}
}
```

---

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review SLURM logs in `logs/` directory
3. Verify environment setup with `test_fvd_setup.py`
4. Check disk space and GPU availability

---

## License

This framework is provided for research and educational purposes. Please respect the licenses of individual models and datasets used.

---

**Last Updated**: November 6, 2025  
**Framework Version**: 1.0  
**Tested On**: NVIDIA H100, PyTorch 2.8.0, CUDA 12.8

