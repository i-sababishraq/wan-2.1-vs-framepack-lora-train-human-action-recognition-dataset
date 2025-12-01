"""
Quick test to verify FVD computation setup.
"""

import sys
from pathlib import Path
import torch

print("="*80)
print("FVD Setup Test")
print("="*80)

# 1. Check Python packages
print("\n1. Checking Python packages...")
packages = {
    "torch": None,
    "numpy": None,
    "scipy": None,
    "cv2": None,
    "tqdm": None
}

for pkg in packages:
    try:
        if pkg == "cv2":
            import cv2
            packages[pkg] = cv2.__version__
        else:
            mod = __import__(pkg)
            packages[pkg] = getattr(mod, "__version__", "installed")
        print(f"  ✓ {pkg}: {packages[pkg]}")
    except ImportError:
        print(f"  ✗ {pkg}: NOT INSTALLED")
        packages[pkg] = None

# 2. Check CUDA
print("\n2. Checking CUDA...")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU count: {torch.cuda.device_count()}")
    print(f"  GPU name: {torch.cuda.get_device_name(0)}")

# 3. Check directories
print("\n3. Checking directories...")
dirs_to_check = [
    "data/clips",
    "generated_videos/i2v_wan14b_480p",
    "generated_videos/i2v_framepack_480p",
]

for dir_path in dirs_to_check:
    p = Path(dir_path)
    if p.exists():
        video_count = len(list(p.rglob("*.mp4")))
        print(f"  ✓ {dir_path}: {video_count} videos")
    else:
        print(f"  ✗ {dir_path}: NOT FOUND")

# 4. Check scripts
print("\n4. Checking scripts...")
scripts = [
    "scripts/compute_fvd.py",
    "scripts/benchmark_fvd.py"
]

for script in scripts:
    p = Path(script)
    if p.exists():
        print(f"  ✓ {script}")
    else:
        print(f"  ✗ {script}: NOT FOUND")

# 5. Try to import I3D (optional)
print("\n5. Checking I3D model...")
try:
    from pytorch_i3d import InceptionI3d
    print("  ✓ pytorch_i3d installed")
except ImportError:
    print("  ⚠ pytorch_i3d not installed (will install when needed)")

# 6. Check model directory
print("\n6. Checking model directory...")
model_dir = Path("models/i3d")
model_dir.mkdir(parents=True, exist_ok=True)
model_file = model_dir / "i3d_rgb.pt"
if model_file.exists():
    print(f"  ✓ I3D model exists: {model_file}")
else:
    print(f"  ⚠ I3D model not found (will download when needed)")

print("\n" + "="*80)
print("Setup test complete!")
print("="*80)

# Check if everything is ready
all_packages = all(v is not None for v in packages.values())
cuda_available = torch.cuda.is_available()
all_dirs = all(Path(d).exists() for d in dirs_to_check)

if all_packages and cuda_available and all_dirs:
    print("\n✅ All checks passed! Ready to run FVD benchmark.")
    sys.exit(0)
else:
    print("\n⚠️ Some checks failed. Review the output above.")
    sys.exit(1)

