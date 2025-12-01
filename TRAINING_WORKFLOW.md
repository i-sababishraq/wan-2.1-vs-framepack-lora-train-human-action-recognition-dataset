# Training Workflow: From Processed Clips to LoRA Fine-tuned Model

## Overview
This document explains how to use the preprocessed clips to train LoRA adapters for Wan 2.1 T2V-1.3B.

## Current Status
✅ **Preprocessed clips ready** (222 clips from test run)
- Format: `.npz` files with 81 frames (480×832, uint8)
- Manifest: `data/processed/train.jsonl` with prompts
- Storage: 14 GB (test) → ~400-500 GB (full dataset)

## Training Pipeline

### Step 1: Verify Dataloader
Test that the dataset loads correctly and normalizes to [-1, 1]:

```python
from training.dataset import ClipDataset, collate_fn
from torch.utils.data import DataLoader

# Load dataset
dataset = ClipDataset("data/processed/train.jsonl")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# Test one batch
batch = next(iter(dataloader))
print(f"Frames shape: {batch['frames'].shape}")  # Should be [1, 81, 3, 480, 832]
print(f"Value range: [{batch['frames'].min():.2f}, {batch['frames'].max():.2f}]")  # Should be ~[-1, 1]
print(f"Prompt: {batch['prompts'][0]}")
```

### Step 2: LoRA Training Architecture

#### A. Load Wan 2.1 Model (Diffusers)
```python
from diffusers import WanPipeline, AutoencoderKLWan
import torch

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

# Load VAE (for encoding videos to latents)
vae = AutoencoderKLWan.from_pretrained(
    model_id, 
    subfolder="vae", 
    torch_dtype=torch.float32
)

# Load full pipeline (includes transformer/DiT, text encoder, scheduler)
pipe = WanPipeline.from_pretrained(
    model_id, 
    vae=vae, 
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")
```

#### B. Apply LoRA to Transformer
```python
from peft import LoraConfig, get_peft_model

# LoRA configuration
lora_config = LoraConfig(
    r=32,  # Rank
    lora_alpha=32,  # Alpha (scaling factor)
    target_modules=[
        "to_q", "to_k", "to_v", "to_out.0",  # Attention layers
        "proj_in", "proj_out"  # Projection layers
    ],
    lora_dropout=0.0,
    bias="none",
)

# Apply LoRA to the transformer (DiT)
transformer = pipe.transformer
transformer = get_peft_model(transformer, lora_config)
transformer.print_trainable_parameters()
```

#### C. Training Loop
```python
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm import tqdm

# Setup
accelerator = Accelerator(mixed_precision="bf16")
optimizer = AdamW(transformer.parameters(), lr=1e-4)

# Prepare with accelerator
transformer, optimizer, dataloader = accelerator.prepare(
    transformer, optimizer, dataloader
)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for step, batch in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(transformer):
            # 1. Encode frames to latents with VAE
            frames = batch["frames"]  # [B, T, C, H, W]
            with torch.no_grad():
                latents = vae.encode(frames).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
            
            # 2. Add noise (diffusion forward process)
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],))
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            
            # 3. Encode prompts
            prompt_embeds = pipe.encode_prompt(batch["prompts"])
            
            # 4. Predict noise with transformer
            model_pred = transformer(
                noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds
            ).sample
            
            # 5. Calculate loss
            loss = F.mse_loss(model_pred, noise)
            
            # 6. Backward and optimize
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
        if step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
    
    # Save checkpoint
    transformer.save_pretrained(f"checkpoints/lora_epoch_{epoch}")
```

### Step 3: Implementation Status

#### Already Implemented ✅
- `training/dataset.py` - Dataset loader with normalization
- `scripts/preprocess.py` - Video → clips preprocessing
- Environment setup with PyTorch, Diffusers, PEFT

#### Needs Implementation 🔧
1. **Complete training script** (`training/train_lora.py`)
   - VAE encoding step
   - Noise scheduler integration
   - Prompt encoding with T5
   - Full diffusion loss calculation
   - Checkpointing and logging

2. **Evaluation script** (`scripts/evaluate.py`)
   - Load trained LoRA weights
   - Generate videos per activity
   - Compare with base model

3. **Inference script** (`scripts/generate.py`)
   - Load LoRA adapters
   - Generate videos from prompts
   - Save as MP4 files

## Recommended Next Steps

### Option 1: Quick Smoke Test (Recommended)
Use the current 222 clips to validate the pipeline:
```bash
# 1. Test dataloader
python -c "from training.dataset import ClipDataset; ds = ClipDataset('data/processed/train.jsonl'); print(f'Loaded {len(ds)} clips'); print(ds[0]['frames'].shape)"

# 2. Run 50-step training (once train_lora.py is complete)
python training/train_lora.py \
    --manifest data/processed/train.jsonl \
    --output_dir outputs/smoke_test \
    --max_steps 50 \
    --batch_size 1 \
    --gradient_accumulation_steps 4

# 3. Generate test video
python scripts/generate.py \
    --lora_weights outputs/smoke_test \
    --prompt "A person clapping their hands" \
    --output test_generation.mp4
```

### Option 2: Full Dataset Training
After smoke test passes:
```bash
# 1. Process full dataset
python scripts/preprocess.py  # Process all 1,113 videos (~10 hours)

# 2. Full training
python training/train_lora.py \
    --manifest data/processed/train.jsonl \
    --output_dir outputs/full_training \
    --num_epochs 10 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --save_steps 500

# 3. Generate videos (10 per activity)
for activity in "clapping" "sitting" "walking"; do
    python scripts/generate.py \
        --lora_weights outputs/full_training \
        --prompt "A realistic video showing a person $activity" \
        --num_videos 10 \
        --output_dir outputs/generated/$activity
done
```

## Memory & Compute Requirements

### H100 80GB Estimates:
- **Preprocessing**: CPU-bound, ~30 sec/video
- **Training**:
  - Batch size 1: ~25-30 GB VRAM (bf16)
  - With gradient accumulation (4-8): Effective batch 4-8
  - Throughput: ~5-10 sec/step
  - Full training (10 epochs, 8K clips): ~10-15 hours

### Disk Space:
- Preprocessed clips: ~400-500 GB
- Model checkpoints: ~5-10 GB per checkpoint
- Generated videos: ~100 MB per video

## What You Can Do Right Now

### 1. Test Current Implementation
```bash
# Verify dataloader works
python3 -c "
from training.dataset import ClipDataset, collate_fn
from torch.utils.data import DataLoader

ds = ClipDataset('data/processed/train.jsonl')
loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
batch = next(iter(loader))
print(f'✅ Loaded batch: {batch[\"frames\"].shape}')
print(f'✅ Value range: [{batch[\"frames\"].min():.2f}, {batch[\"frames\"].max():.2f}]')
print(f'✅ Prompts: {batch[\"prompts\"]}')
"
```

### 2. Complete Training Script
I can help you complete `training/train_lora.py` with:
- Proper VAE encoding
- Wan 2.1 specific noise scheduler
- T5 text encoding
- Full training loop with checkpointing

### 3. Or Process Full Dataset First
If preprocessing looks good, we can start the full dataset processing in background while we work on the training script.

## Which would you like to do next?
1. **Test the dataloader** (verify everything loads correctly)
2. **Complete the training script** (implement full LoRA training)
3. **Process full dataset** (start 10-hour preprocessing in background)
4. **All of the above** (I'll do them in sequence)
