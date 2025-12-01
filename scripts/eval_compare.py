#!/usr/bin/env python3
"""Generate baseline vs LoRA videos, enrich prompts at generation, and compute LPIPS + proxy FID.

This script re-uses utilities in `scripts/generate_videos.py` to load the Wan pipeline
and generate videos. It then attempts to compute LPIPS (frame-wise) and a proxy FID
(frame-wise Inception/FID via torchmetrics) per activity. If required metric packages
are not installed the script will print instructions and skip the metric.

Notes:
- Uses the processed manifest to sample reference (ground-truth) clips for metrics.
- Does NOT re-run preprocessing or retraining; generates from the provided checkpoints.
"""
import argparse
import json
import math
import random
from pathlib import Path
import sys

import numpy as np
import torch

# Import shared helpers from the generate script
from scripts.generate_videos import (
    load_pipeline_with_lora,
    save_video_frames_to_mp4,
    ACTIVITY_PROMPTS,
)


def enrich_prompt(base_prompt, variant_idx=0):
    """Create a small set of prompt variants for generation-time enrichment.

    This is intentionally simple: a few templates that prepend or append style hints.
    For stronger paraphrases you can integrate an external paraphraser or a small
    hand-curated template file.
    """
    templates = [
        "A cinematic, high quality video showing {}",
        "A realistic, well-lit video showing {}",
        "A close-up cinematic shot of a person {}",
        "A documentary-style realistic video showing {}",
        "A clear, high-resolution video showing {}",
    ]
    t = templates[variant_idx % len(templates)]
    return t.format(base_prompt.replace('A realistic video showing ', ''))


def load_manifest(manifest_path):
    entries = []
    with open(manifest_path, "r") as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def sample_reference_clips(manifest_entries, activity, k=10):
    # Filter entries by activity label
    filtered = [e for e in manifest_entries if e.get("activity") == activity]
    if not filtered:
        return []
    if k >= len(filtered):
        return filtered
    return random.sample(filtered, k)


def frames_to_tensor(frames):
    # frames: (T,H,W,3) uint8 or float
    arr = np.asarray(frames)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    # Convert to (T,3,H,W)
    arr = arr.transpose(0, 3, 1, 2)
    return torch.from_numpy(arr)


def compute_lpips_pairs(lpips_model, gen_video_path, ref_video_path, device):
    import cv2

    # Read videos frame-by-frame and compute average LPIPS over corresponding frames
    def read_frames(path):
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # BGR->RGB
            frames.append(frame[:, :, ::-1])
        cap.release()
        return np.asarray(frames)

    gen_frames = read_frames(gen_video_path)
    ref_frames = read_frames(ref_video_path)
    if len(gen_frames) == 0 or len(ref_frames) == 0:
        return None

    # Align frame counts by cropping to min length
    L = min(len(gen_frames), len(ref_frames))
    gen_frames = gen_frames[:L]
    ref_frames = ref_frames[:L]

    # Convert and compute per-frame LPIPS
    total = 0.0
    n = 0
    for i in range(L):
        g = gen_frames[i].astype(np.float32) / 255.0
        r = ref_frames[i].astype(np.float32) / 255.0
        # LPIPS expects tensors in [-1,1]
        gt = torch.from_numpy(r).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
        gen = torch.from_numpy(g).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
        with torch.no_grad():
            score = lpips_model(gen, gt)
        total += float(score.cpu().item())
        n += 1

    return total / n if n > 0 else None


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs LoRA generation + metrics")
    parser.add_argument("--model_id", default="models/Wan2.1-T2V-1.3B-Local")
    parser.add_argument("--lora_checkpoint", default="checkpoints/lora_full/lora_final_step10000.pt")
    parser.add_argument("--manifest", default="data/processed_full/train.jsonl")
    parser.add_argument("--activities", nargs="+", default=None,
                        help="Activities to evaluate (default: all in ACTIVITY_PROMPTS)")
    parser.add_argument("--num_per_activity", type=int, default=10,
                        help="Number of videos to generate per activity per model")
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--seeds_base", type=int, default=1000,
                        help="Base seed; final seeds will be base + i")
    parser.add_argument("--seeds_count", type=int, default=8,
                        help="Number of distinct seeds to use per activity (will cycle if num_per_activity > seeds_count)")
    parser.add_argument("--output_dir", default="generated_videos/full")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Validate
    if args.num_inference_steps < 50:
        args.num_inference_steps = 50

    # Load manifest
    manifest_entries = load_manifest(args.manifest)

    activities = args.activities or list(ACTIVITY_PROMPTS.keys())

    # Load pipelines
    print("Loading baseline pipeline (no LoRA)...")
    baseline_pipe = load_pipeline_with_lora(args.model_id, lora_checkpoint=None, device=device, dtype=torch.float32)
    print("Loading LoRA pipeline...")
    lora_pipe = load_pipeline_with_lora(args.model_id, lora_checkpoint=args.lora_checkpoint, device=device, dtype=torch.float32)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Try to import LPIPS and FID
    have_lpips = True
    have_fid = True
    try:
        import lpips
        lpips_model = lpips.LPIPS(net='vgg').to(device)
        lpips_model.eval()
    except Exception:
        print("LPIPS not available. Install with: pip install lpips")
        have_lpips = False
        lpips_model = None

    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        fid_module = FrechetInceptionDistance(feature=64).to(device)
    except Exception:
        print("torchmetrics FID not available. Install with: pip install torchmetrics[image]")
        have_fid = False
        fid_module = None

    results = {}

    for activity in activities:
        print(f"\n--- Evaluating activity: {activity} ---")
        activity_dir = out_root / activity.replace(" ", "_")
        baseline_dir = activity_dir / "baseline"
        lora_dir = activity_dir / "lora"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        lora_dir.mkdir(parents=True, exist_ok=True)

        # Prepare seeds
        seeds = [args.seeds_base + i for i in range(args.seeds_count)]

        # For metrics: sample reference clips from manifest (up to num_per_activity)
        ref_clips = sample_reference_clips(manifest_entries, activity, k=args.num_per_activity)

        per_activity_metrics = {
            "activity": activity,
            "generated": {
                "baseline": [],
                "lora": []
            },
            "lpips": {"baseline": None, "lora": None},
            "fid": {"baseline": None, "lora": None},
        }

        # Generate for baseline and LoRA, keeping seed alignment
        for model_name, pipe in [("baseline", baseline_pipe), ("lora", lora_pipe)]:
            target_dir = baseline_dir if model_name == "baseline" else lora_dir

            generated_paths = []
            for i in range(args.num_per_activity):
                seed_idx = i % args.seeds_count
                current_seed = seeds[seed_idx]
                variant_idx = i % 5
                prompt = enrich_prompt(ACTIVITY_PROMPTS.get(activity, activity), variant_idx=variant_idx)

                generator = torch.Generator(device=pipe.device).manual_seed(current_seed)
                out_path = target_dir / f"{activity.replace(' ', '_')}_{i:03d}_s{current_seed}.mp4"

                print(f"Generating [{model_name}] {i+1}/{args.num_per_activity} seed={current_seed} prompt='{prompt[:80]}...'")
                video = pipe(
                    prompt=prompt,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                ).frames[0]

                save_video_frames_to_mp4(video, str(out_path), fps=8)
                generated_paths.append(str(out_path))

            per_activity_metrics["generated"][model_name] = generated_paths

        # Compute LPIPS per model against reference clips when available
        if have_lpips and len(ref_clips) > 0:
            for model_name in ("baseline", "lora"):
                scores = []
                gen_paths = per_activity_metrics["generated"][model_name]
                for idx, gen_path in enumerate(gen_paths):
                    # Match with reference clip idx if available, else cycle
                    ref_idx = idx % len(ref_clips)
                    ref_path = ref_clips[ref_idx].get("video_path")
                    try:
                        s = compute_lpips_pairs(lpips_model, gen_path, ref_path, device)
                        if s is not None:
                            scores.append(s)
                    except Exception as e:
                        print(f"LPIPS compute error for {gen_path} vs {ref_path}: {e}")
                per_activity_metrics["lpips"][model_name] = float(np.mean(scores)) if len(scores) > 0 else None

        # Compute proxy FID (frame-wise FID) if available
        if have_fid:
            for model_name in ("baseline", "lora"):
                gen_paths = per_activity_metrics["generated"][model_name]
                fid_module.reset()
                for idx, gen_path in enumerate(gen_paths):
                    # load frames and add to module
                    import cv2
                    gen_cap = cv2.VideoCapture(str(gen_path))
                    while True:
                        ret, frame = gen_cap.read()
                        if not ret:
                            break
                        # frame is BGR uint8
                        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                        fid_module.update(img_t, real=False)
                    gen_cap.release()

                # Add reference frames (use sampled ref_clips)
                for rc in ref_clips:
                    ref_path = rc.get("video_path")
                    cap = __import__('cv2').VideoCapture(str(ref_path))
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        img = __import__('cv2').cvtColor(frame, __import__('cv2').COLOR_BGR2RGB)
                        img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                        fid_module.update(img_t, real=True)
                    cap.release()

                try:
                    fid_val = float(fid_module.compute())
                except Exception as e:
                    print(f"FID compute error for activity {activity}, model {model_name}: {e}")
                    fid_val = None
                per_activity_metrics["fid"][model_name] = fid_val

        results[activity] = per_activity_metrics

        # Save intermediate results per activity
        with open(activity_dir / "metrics.json", "w") as f:
            json.dump(per_activity_metrics, f, indent=2)

    # Save global results
    with open(out_root / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nEvaluation complete. Results saved to:", out_root)


if __name__ == "__main__":
    main()
