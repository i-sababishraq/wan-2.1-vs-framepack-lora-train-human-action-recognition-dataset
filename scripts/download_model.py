"""Download Wan 2.1 model to local cache for faster training."""
from diffusers import WanPipeline
import torch
import os

# Set cache directory
cache_dir = "models/Wan2.1-T2V-1.3B-Diffusers"
os.makedirs(cache_dir, exist_ok=True)

print(f"Downloading Wan 2.1 model to {cache_dir}...")
print("This will take several minutes (~30GB download)...\n")

# Download and save the model
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    torch_dtype=torch.bfloat16,
    cache_dir=cache_dir
)

# Save to local directory
local_path = "models/Wan2.1-T2V-1.3B-Local"
print(f"\nSaving to {local_path}...")
pipe.save_pretrained(local_path)

print(f"\n✅ Model downloaded and saved to {local_path}")
print(f"Use '--model_id models/Wan2.1-T2V-1.3B-Local' in training commands")
