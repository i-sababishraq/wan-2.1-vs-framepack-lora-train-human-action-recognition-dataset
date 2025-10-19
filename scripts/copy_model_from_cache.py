"""Copy Wan 2.1 model from HF cache to local directory."""
import os
import shutil
from pathlib import Path

# HuggingFace cache location
hf_cache = Path.home() / ".cache/huggingface/hub"

# Find the Wan model in cache
print("Searching for Wan 2.1 model in HuggingFace cache...")
wan_cache = None
for path in hf_cache.glob("models--Wan-AI--*"):
    if "Wan2.1-T2V-1.3B-Diffusers" in str(path):
        wan_cache = path
        break

if not wan_cache:
    print("Model not found in cache. Checking alternate locations...")
    # Check if it's in the conda env cache
    conda_cache = Path("/anvil/projects/x-soc250046/x-sishraq/.cache/huggingface/hub")
    for path in conda_cache.glob("models--Wan-AI--*"):
        if "Wan2.1-T2V-1.3B-Diffusers" in str(path):
            wan_cache = path
            break

if wan_cache:
    print(f"Found model in cache: {wan_cache}")
    
    # Create local directory
    local_path = Path("models/Wan2.1-T2V-1.3B-Local")
    local_path.mkdir(parents=True, exist_ok=True)
    
    # Find the snapshots directory
    snapshots_dir = wan_cache / "snapshots"
    if snapshots_dir.exists():
        # Get the latest snapshot
        snapshots = list(snapshots_dir.iterdir())
        if snapshots:
            latest = max(snapshots, key=os.path.getctime)
            print(f"Copying from snapshot: {latest.name}")
            
            # Copy all files
            for item in latest.iterdir():
                dest = local_path / item.name
                if item.is_file():
                    print(f"  Copying {item.name}...")
                    shutil.copy2(item, dest)
                elif item.is_dir():
                    print(f"  Copying directory {item.name}...")
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
            
            print(f"\n✅ Model copied to {local_path}")
            print(f"Use '--model_id {local_path}' in training commands")
        else:
            print("No snapshots found in cache")
    else:
        print("Snapshots directory not found")
else:
    print("Model not found in cache.")
    print("The model should have been cached during the training run.")
    print("Please check the cache location or wait for the rate limit to expire.")
