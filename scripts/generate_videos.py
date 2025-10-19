"""Generate videos using Wan 2.1 with LoRA fine-tuned weights.

This script generates synthetic videos for human activity recognition
using the base Wan 2.1 model with optional LoRA adapters.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import WanPipeline
from peft import LoraConfig, get_peft_model


# Activity prompts for HAR dataset
ACTIVITY_PROMPTS = {
    "Clapping": "A realistic video showing a person clapping their hands",
    "Meet and Split": "A realistic video showing two people meeting and then splitting apart",
    "Sitting": "A realistic video showing a person sitting down",
    "Standing Still": "A realistic video showing a person standing still",
    "Walking": "A realistic video showing a person walking",
    "Walking While Reading Book": "A realistic video showing a person walking while reading a book",
    "Walking While Using Phone": "A realistic video showing a person walking while using a phone",
}


def save_video_frames_to_mp4(frames, path, fps=8):
    """Persist a sequence of RGB frames to disk as an MP4 file."""
    frames_np = frames.detach().cpu().numpy() if isinstance(frames, torch.Tensor) else np.asarray(frames)

    if frames_np.ndim == 5:
        frames_np = frames_np[0]

    if frames_np.ndim != 4 or frames_np.shape[-1] not in (1, 3):
        raise ValueError(f"Expected frames with shape (T,H,W,3), received {frames_np.shape}.")

    if frames_np.dtype != np.uint8:
        # Handle potential [-1,1] or [0,1] ranges
        if frames_np.min() >= -1.0 and frames_np.max() <= 1.0:
            frames_np = (frames_np + 1.0) * 0.5
        frames_np = np.clip(frames_np, 0.0, 1.0)
        frames_np = (frames_np * 255.0).round().astype(np.uint8)

    height, width = frames_np.shape[1], frames_np.shape[2]
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")

    try:
        for frame in frames_np:
            if frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def apply_lora_to_transformer(transformer, lora_rank=32, lora_alpha=32, target_modules=None):
    if target_modules is None:
        target_modules = [
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
            "ffn.net.0.proj",
            "ffn.net.2",
        ]

    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )

    return get_peft_model(transformer, config)


def load_pipeline_with_lora(model_id, lora_checkpoint=None, device="cuda", dtype=torch.bfloat16):
    """Load Wan pipeline with optional LoRA weights."""
    print(f"Loading base model: {model_id}")
    pipe = WanPipeline.from_pretrained(model_id, dtype=dtype)
    
    if lora_checkpoint:
        print(f"Loading LoRA checkpoint: {lora_checkpoint}")
        checkpoint = torch.load(lora_checkpoint, map_location="cpu")
        lora_state_dict = checkpoint.get("lora_state_dict", checkpoint)

        args_dict = checkpoint.get("args", {})
        lora_rank = args_dict.get("lora_rank", 32)
        lora_alpha = args_dict.get("lora_alpha", 32)

        pipe.transformer = apply_lora_to_transformer(
            pipe.transformer,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        pipe.transformer.load_state_dict(lora_state_dict, strict=False)
        pipe.transformer.eval()
        print(
            "LoRA weights loaded from step "
            f"{checkpoint.get('step', 'unknown')} (rank={lora_rank}, alpha={lora_alpha})"
        )
    
    pipe = pipe.to(device)
    return pipe


def generate_videos(
    pipe,
    output_dir,
    activities=None,
    num_per_activity=10,
    num_frames=81,
    height=480,
    width=832,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42,
):
    """Generate videos for specified activities."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if activities is None:
        activities = list(ACTIVITY_PROMPTS.keys())
    
    # Manifest for generated videos
    manifest = []
    
    for activity in activities:
        print(f"\n=== Generating videos for: {activity} ===")
        prompt = ACTIVITY_PROMPTS[activity]
        
        activity_dir = output_dir / activity.replace(" ", "_")
        activity_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(num_per_activity):
            current_seed = seed + i
            generator = torch.Generator(device=pipe.device).manual_seed(current_seed)
            
            print(f"  [{i+1}/{num_per_activity}] Generating with seed {current_seed}...")
            
            # Generate video
            video = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).frames[0]
            
            # Save video
            output_path = activity_dir / f"{activity.replace(' ', '_')}_{i:03d}.mp4"
            save_video_frames_to_mp4(video, str(output_path), fps=8)
            
            # Add to manifest
            manifest.append({
                "video_path": str(output_path),
                "activity": activity,
                "prompt": prompt,
                "seed": current_seed,
                "num_frames": num_frames,
                "height": height,
                "width": width,
            })
            
            print(f"    Saved to: {output_path}")
    
    # Save manifest
    manifest_path = output_dir / "generated_manifest.jsonl"
    with open(manifest_path, "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")
    
    print(f"\n✅ Generation complete!")
    print(f"   Videos saved to: {output_dir}")
    print(f"   Manifest saved to: {manifest_path}")
    print(f"   Total videos: {len(manifest)}")


def main():
    parser = argparse.ArgumentParser(description="Generate HAR videos with Wan 2.1 + LoRA")
    
    # Model arguments
    parser.add_argument("--model_id", type=str, 
                        default="models/Wan2.1-T2V-1.3B-Local",
                        help="Path to base Wan 2.1 model")
    parser.add_argument("--lora_checkpoint", type=str, default=None,
                        help="Path to LoRA checkpoint (.pt file)")
    
    # Generation arguments
    parser.add_argument("--output_dir", type=str, default="generated_videos",
                        help="Directory to save generated videos")
    parser.add_argument("--activities", type=str, nargs="+", default=None,
                        help="Activities to generate (default: all)")
    parser.add_argument("--num_per_activity", type=int, default=10,
                        help="Number of videos to generate per activity")
    
    # Video parameters
    parser.add_argument("--num_frames", type=int, default=81,
                        help="Number of frames to generate (must be 4n+1)")
    parser.add_argument("--height", type=int, default=480,
                        help="Video height (multiple of 16)")
    parser.add_argument("--width", type=int, default=832,
                        help="Video width (multiple of 16)")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    
    args = parser.parse_args()
    
    # Validate frame count
    if (args.num_frames - 1) % 4 != 0:
        parser.error(f"--num_frames must be 4n+1 (got {args.num_frames})")
    
    # Validate dimensions
    if args.height % 16 != 0 or args.width % 16 != 0:
        parser.error(f"--height and --width must be multiples of 16")
    
    # Load pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    
    pipe = load_pipeline_with_lora(
        args.model_id,
        args.lora_checkpoint,
        device=device,
        dtype=dtype
    )
    
    # Generate videos
    generate_videos(
        pipe=pipe,
        output_dir=args.output_dir,
        activities=args.activities,
        num_per_activity=args.num_per_activity,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
