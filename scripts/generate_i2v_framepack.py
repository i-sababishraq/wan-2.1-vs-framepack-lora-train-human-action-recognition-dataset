#!/usr/bin/env python3
"""
Generate videos from images using Framepack I2V pipeline.
Uses the same starting frames as Wan2.1 for fair comparison.
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
import cv2
from diffusers import HunyuanVideoFramepackPipeline, HunyuanVideoFramepackTransformer3DModel
from diffusers.utils import load_image, export_to_video
from transformers import SiglipImageProcessor, SiglipVisionModel


def save_video_cv2(frames, out_path, fps=8):
    """Save video frames using OpenCV with uint8 conversion."""
    frames_u8 = []
    for f in frames:
        arr = np.asarray(f)
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        frames_u8.append(arr)
    
    h, w = frames_u8[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {out_path}")
    
    for f in frames_u8:
        if f.ndim == 2:
            f = np.stack([f]*3, axis=-1)
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    
    vw.release()


def main():
    # Configuration
    manifest_path = "data/starting_frames/starting_frames_manifest.jsonl"
    output_root = Path("generated_videos/i2v_framepack_480p")
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Get configuration from environment
    num_samples = int(os.environ.get("NUM_SAMPLES", "70"))
    num_inference_steps = int(os.environ.get("NUM_INFERENCE_STEPS", "30"))
    num_frames = int(os.environ.get("NUM_FRAMES", "81"))
    height = int(os.environ.get("HEIGHT", "480"))
    width = int(os.environ.get("WIDTH", "832"))
    guidance_scale = float(os.environ.get("GUIDANCE_SCALE", "9.0"))
    fps = int(os.environ.get("FPS", "8"))
    
    print("=" * 60)
    print("Framepack I2V Generation")
    print("=" * 60)
    print(f"Model: lllyasviel/FramePackI2V_HY")
    print(f"Samples: {num_samples}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Num frames: {num_frames}")
    print(f"Resolution: {height}x{width}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"FPS: {fps}")
    print("=" * 60)
    print()
    
    # Load Framepack model
    print("Loading Framepack transformer...")
    transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
        "lllyasviel/FramePackI2V_HY",
        torch_dtype=torch.bfloat16
    )
    
    print("Loading feature extractor and image encoder...")
    feature_extractor = SiglipImageProcessor.from_pretrained(
        "lllyasviel/flux_redux_bfl",
        subfolder="feature_extractor"
    )
    image_encoder = SiglipVisionModel.from_pretrained(
        "lllyasviel/flux_redux_bfl",
        subfolder="image_encoder",
        torch_dtype=torch.float16
    )
    
    print("Loading pipeline...")
    pipe = HunyuanVideoFramepackPipeline.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        transformer=transformer,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
    )
    
    # Enable memory optimizations
    print("Enabling memory optimizations...")
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    
    print("Model loaded successfully!")
    print()
    
    # Load manifest
    print(f"Loading manifest from {manifest_path}...")
    entries = []
    with open(manifest_path, "r") as f:
        for line in f:
            entries.append(json.loads(line))
    
    # Limit to NUM_SAMPLES
    entries = entries[:num_samples]
    
    print(f"Processing {len(entries)} videos...")
    print()
    
    success_count = 0
    error_count = 0
    
    for i, entry in enumerate(entries, 1):
        activity = entry.get("activity") or entry.get("label") or "unknown"
        activity_clean = activity.replace(" ", "_")
        prompt = entry.get("prompt", "")
        frame_path = entry.get("frame_path", "")
        
        if not frame_path or not prompt:
            print(f"[{i}/{len(entries)}] SKIP: Missing frame or prompt")
            continue
        
        output_dir = output_root / activity_clean
        output_dir.mkdir(parents=True, exist_ok=True)
        
        frame_name = Path(frame_path).stem
        output_path = output_dir / f"{frame_name}_framepack.mp4"
        
        try:
            print(f"[{i}/{len(entries)}] Processing: {activity}")
            print(f"  Image: {frame_path}")
            print(f"  Prompt: {prompt}")
            
            # Load image
            image = load_image(frame_path)
            
            # Generate video
            output = pipe(
                image=image,
                prompt=prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator().manual_seed(42 + i),
                sampling_type="inverted_anti_drifting",
            ).frames[0]
            
            # Save video
            save_video_cv2(output, output_path, fps=fps)
            
            print(f"  ✓ Saved to: {output_path}")
            success_count += 1
            
        except Exception as ex:
            print(f"  ✗ ERROR: {ex}")
            error_count += 1
        
        print()
    
    print("=" * 60)
    print("Generation Complete!")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total: {len(entries)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
