"""Minimal smoke test with just 5 clips and 10 steps."""
import sys
import os
sys.path.insert(0, 'training')

from dataset import ClipDataset
import json

# Create a mini manifest with just 5 clips
manifest_path = "data/processed/train.jsonl"
mini_manifest_path = "data/processed/train_mini.jsonl"

print("Creating mini dataset with 5 clips...")
with open(manifest_path, 'r') as f:
    lines = f.readlines()

# Take first 5 clips
with open(mini_manifest_path, 'w') as f:
    for line in lines[:5]:
        f.write(line)

print(f"Created {mini_manifest_path} with 5 clips")

# Now run training
os.system(f"python training/train_lora.py --max_steps 10 --batch_size 1 --save_every 5 --output_dir checkpoints/mini_test --manifest {mini_manifest_path}")
