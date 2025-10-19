# Preprocess and LoRA training (smoke test)

This document shows minimal commands to preprocess the Kaggle HAR videos into clips and run a smoke test training script that wires LoRA adapters to Wan2.1 UNet.

1) Install dependencies (prefer a virtualenv or conda env)

```bash
pip install -r requirements.txt
```

2) Arrange your raw dataset under `data/raw/<label>/*.mp4`.

3) Preprocess (example)

```bash
python scripts/preprocess.py --input_dir data/raw --output_dir data/processed --clip_len 16 --stride 8 --width 832 --height 480
```

This creates `data/processed/clips/<label>/*.npz` and `data/processed/train.jsonl` manifest.

4) Run a smoke test training (this is a minimal skeleton and does not implement the full diffusion loss yet):

```bash
python training/train_lora.py --manifest data/processed/train.jsonl --output_dir checkpoints --max_steps 10 --batch_size 1
```

Notes:
- The training script is a starting point and demonstrates how to apply LoRA to the UNet using PEFT. It does not yet implement the diffusion forward/backward pipeline or VAE latent encoding. Use community examples (DiffSynth-Studio, EchoShot) to adapt full training steps.
- If you see OOM, try `--batch_size 1`, use `--offload_model True` (for Wan CLI usage) or run on a larger GPU.
