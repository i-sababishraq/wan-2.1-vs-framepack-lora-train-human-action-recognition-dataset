#!/usr/bin/env python3
"""
Generate videos from images using WAN 2.1 Image-to-Video pipeline.

This script loads first frames and generates videos using the WAN I2V model,
optionally with LoRA adapters.

Usage:
    python scripts/generate_i2v_wan.py --frames_dir data/first_frames --output_dir generated_videos/i2v_wan --lora_path checkpoints/lora_full/lora_final_step13000.pt --num_inference_steps 150
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import WanImageToVideoPipeline

try:
    from peft import PeftModel, LoraConfig
except ImportError:
    print("WARNING: peft not available, LoRA loading will not work")
    PeftModel = None


def load_pipeline_with_lora(lora_path=None, device="cuda", dtype=torch.bfloat16):
    """Load WAN Image-to-Video pipeline with optional LoRA."""
    print("Loading WAN Image-to-Video pipeline...")
    
    # Note: WAN 2.1 primarily supports T2V. For I2V, we can use T2V with image conditioning
    # or check if a dedicated I2V pipeline is available
    try:
        pipeline = WanImageToVideoPipeline.from_pretrained(
            DEFAULT_MODEL,
            torch_dtype=dtype
        ).to(device)
        print("Loaded WAN 2.1 I2V pipeline")
    except Exception as e:
        print(f"I2V model not found, trying T2V model with image conditioning...")
        # Fallback: Use T2V model - may need custom image conditioning
        from diffusers import WanPipeline
        pipeline = WanPipeline.from_pretrained(
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            torch_dtype=dtype
        ).to(device)
        print("Loaded WAN 2.1 T2V pipeline (I2V mode not available)")
    
    if lora_path and PeftModel:
        print(f"Loading LoRA weights from {lora_path}...")
        # Apply LoRA to transformer
        transformer = pipeline.transformer
        
        # Load LoRA state dict
        lora_state_dict = torch.load(lora_path, map_location=device)
        
        # Filter keys for transformer
        transformer_keys = {k: v for k, v in lora_state_dict.items() if "transformer" in k}
        if transformer_keys:
            # Remove "transformer." prefix if present
            transformer_keys = {k.replace("transformer.", ""): v for k, v in transformer_keys.items()}
            transformer.load_state_dict(transformer_keys, strict=False)
            print(f"Loaded {len(transformer_keys)} LoRA parameters")
        else:
            # Try loading directly
            transformer.load_state_dict(lora_state_dict, strict=False)
            print("Loaded LoRA weights directly")
    
    return pipeline


def save_video_frames(frames, output_path, fps=8):
    """Save video frames as MP4."""
    import imageio
    
    # Convert tensors to numpy if needed
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()
    
    # Ensure uint8 format
    if frames.dtype != np.uint8:
        frames = (frames * 255).astype(np.uint8)
    
    # Save video
    imageio.mimsave(output_path, frames, fps=fps, codec='libx264', quality=8)


def main():
    parser = argparse.ArgumentParser(description="Generate videos from images using WAN I2V")
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory containing first frames")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated videos")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-I2V-14B-480P", help="WAN I2V model id or local path")
    parser.add_argument("--lora_path", type=str, help="Path to LoRA checkpoint")
    parser.add_argument("--num_inference_steps", type=int, default=100, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--num_frames", type=int, default=20, help="Number of frames to generate")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load pipeline
    global DEFAULT_MODEL
    DEFAULT_MODEL = args.model_id
    pipeline = load_pipeline_with_lora(lora_path=args.lora_path, device=args.device)
    
    # Load frames metadata
    frames_dir = Path(args.frames_dir)
    metadata_path = frames_dir / "frames_metadata.json"
    
    if not metadata_path.exists():
        print(f"ERROR: Metadata not found at {metadata_path}")
        print("Run extract_first_frames.py first")
        return
    
    with open(metadata_path, 'r') as f:
        frames_metadata = json.load(f)
    
    print(f"Found {len(frames_metadata)} frames to process")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate videos
    results = []
    
    for i, frame_info in enumerate(frames_metadata):
        frame_path = Path(frame_info["frame_path"])
        activity = frame_info["activity"]
        
        print(f"\n[{i+1}/{len(frames_metadata)}] Processing {frame_path.name}...")
        
        # Load image
        image = Image.open(frame_path).convert("RGB")
        
        # Generate video
        with torch.no_grad():
            output = pipeline(
                image=image,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator(device=args.device).manual_seed(args.seed + i)
            )
        
        frames = output.frames[0]  # Get first (and only) video
        
        # Save video
        activity_safe = activity.replace(" ", "_")
        output_video_dir = output_dir / activity_safe
        output_video_dir.mkdir(parents=True, exist_ok=True)
        
        video_filename = f"{activity_safe}_{frame_info['index']:03d}.mp4"
        video_path = output_video_dir / video_filename
        
        save_video_frames(frames, str(video_path), fps=args.fps)
        
        print(f"  Saved: {video_path}")
        
        results.append({
            "input_frame": str(frame_path),
            "output_video": str(video_path),
            "activity": activity,
            "original_video": frame_info["original_video"]
        })
    
    # Save results
    results_path = output_dir / "generation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nGenerated {len(results)} videos")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
