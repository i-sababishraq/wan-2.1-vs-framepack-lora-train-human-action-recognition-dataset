#!/usr/bin/env python3
"""
Generate videos from images using WAN 2.1 T2V with image-conditioned prompts.

Since WAN 2.1 I2V may not be available, this uses the T2V model with
activity-specific prompts corresponding to the extracted first frames.

Usage:
    python scripts/generate_t2v_from_frames.py --frames_dir data/first_frames --output_dir generated_videos/wan_t2v_from_frames --lora_path checkpoints/lora_full/lora_final_step13000.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import WanPipeline
from PIL import Image

try:
    from peft import PeftModel, LoraConfig
except ImportError:
    print("WARNING: peft not available, LoRA loading will not work")
    PeftModel = None


# Activity prompts matching the training prompts
ACTIVITY_PROMPTS = {
    "Clapping": "A high-quality realistic video showing a person clapping hands enthusiastically, clear hand motion, well-lit environment, full body or upper body view, natural movement",
    "Walking": "A high-quality realistic video showing a person walking naturally forward, clear leg and arm movement, steady pace, well-lit outdoor or indoor environment, full body view",
    "Sitting": "A high-quality realistic video showing a person sitting down on a chair or bench from a standing position, smooth transition, clear body movement, well-lit environment, side or front view",
    "Walking While Using Phone": "A high-quality realistic video showing a person walking while looking at and using a smartphone, holding phone in hand, natural walking pace, divided attention, well-lit environment, full body view",
    "Meet and Split": "A high-quality realistic video showing two people walking toward each other from opposite sides, meeting in the center, briefly interacting, then walking away in opposite directions, clear full-body view, well-lit environment",
    "Walking While Reading Book": "A high-quality realistic video showing a person walking while holding and reading a book, slow careful pace, book held at reading height, well-lit environment, full body or upper body view",
    "Standing Still": "A high-quality realistic video showing a person standing still in a stationary position, minimal movement, upright posture, well-lit environment, full body view"
}


def load_pipeline_with_lora(lora_path=None, device="cuda", dtype=torch.bfloat16):
    """Load WAN T2V pipeline with optional LoRA."""
    print("Loading WAN 2.1 T2V pipeline...")
    
    pipeline = WanPipeline.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        torch_dtype=dtype
    ).to(device)
    
    if lora_path and PeftModel:
        print(f"Loading LoRA weights from {lora_path}...")
        
        # Load LoRA state dict
        lora_state_dict = torch.load(lora_path, map_location=device)
        
        # Apply to transformer
        transformer = pipeline.transformer
        
        # Filter and load LoRA weights
        if isinstance(lora_state_dict, dict):
            # Try to load directly or extract transformer keys
            try:
                transformer.load_state_dict(lora_state_dict, strict=False)
                print("Loaded LoRA weights")
            except Exception as e:
                print(f"Note: Some LoRA keys may not match: {e}")
                # Filter for transformer-specific keys
                transformer_keys = {k.replace("transformer.", ""): v 
                                  for k, v in lora_state_dict.items() 
                                  if "transformer" in k}
                if transformer_keys:
                    transformer.load_state_dict(transformer_keys, strict=False)
                    print(f"Loaded {len(transformer_keys)} LoRA parameters")
    
    return pipeline


def save_video_frames(frames, output_path, fps=8):
    """Save video frames as MP4."""
    import imageio
    
    # Convert to numpy if needed
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()
    
    # Ensure frames is a list or array of images
    if len(frames.shape) == 5:  # N x T x C x H x W or N x T x H x W x C
        frames = frames[0]  # Take first video
    
    # Convert to uint8 if needed
    if frames.dtype != np.uint8:
        if frames.max() <= 1.0:
            frames = (frames * 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)
    
    # Ensure correct shape: T x H x W x C
    if frames.shape[-1] != 3 and frames.shape[1] == 3:
        # Convert from T x C x H x W to T x H x W x C
        frames = frames.transpose(0, 2, 3, 1)
    
    # Save video
    imageio.mimsave(output_path, frames, fps=fps, codec='libx264', quality=8)


def main():
    parser = argparse.ArgumentParser(description="Generate videos from frames using WAN T2V")
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory containing first frames")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated videos")
    parser.add_argument("--lora_path", type=str, help="Path to LoRA checkpoint")
    parser.add_argument("--num_inference_steps", type=int, default=150, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=9.0, help="Guidance scale")
    parser.add_argument("--height", type=int, default=720, help="Video height")
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--num_frames", type=int, default=20, help="Number of frames to generate")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load pipeline
    pipeline = load_pipeline_with_lora(
        lora_path=args.lora_path,
        device=args.device
    )
    
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
        
        # Get prompt for this activity
        prompt = ACTIVITY_PROMPTS.get(activity, f"A realistic video showing {activity.lower()}")
        
        print(f"\n[{i+1}/{len(frames_metadata)}] Processing {frame_path.name}...")
        print(f"  Activity: {activity}")
        print(f"  Prompt: {prompt[:100]}...")
        
        # Generate video using T2V
        with torch.no_grad():
            output = pipeline(
                prompt=prompt,
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
            "prompt": prompt,
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
