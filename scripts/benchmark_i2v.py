#!/usr/bin/env python3
"""
Benchmark WAN 2.1 I2V vs FramePack I2V using FVD metric.

This script:
1. Loads first frames from reference videos
2. Generates videos using both WAN I2V and FramePack I2V
3. Computes FVD comparing each method to ground truth
4. Reports comparative results

Usage:
    python scripts/benchmark_i2v.py --frames_dir data/first_frames --output_dir generated_videos/i2v_benchmark --wan_lora checkpoints/lora_full/lora_final_step13000.pt
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


def run_frame_extraction(args):
    """Step 1: Extract first frames from reference videos."""
    print("\n" + "="*80)
    print("STEP 1: Extracting first frames from reference videos")
    print("="*80)
    
    cmd = [
        sys.executable, "scripts/extract_first_frames.py",
        "--input_dir", args.input_dir,
        "--output_dir", args.frames_dir,
        "--manifest", args.manifest,
        "--num_per_activity", str(args.num_per_activity),
        "--seed", str(args.seed)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    
    if result.returncode != 0:
        print("ERROR: Frame extraction failed")
        sys.exit(1)
    
    print("✓ Frame extraction complete")


def run_wan_i2v_generation(args):
    """Step 2: Generate videos using WAN T2V (with prompts from activity labels)."""
    print("\n" + "="*80)
    print("STEP 2: Generating videos with WAN 2.1 T2V")
    print("="*80)
    
    output_dir = Path(args.output_dir) / "wan_t2v"
    
    # Use T2V generation script since WAN 2.1 uses Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    cmd = [
        sys.executable, "scripts/generate_i2v_wan.py",
        "--frames_dir", args.frames_dir,
        "--output_dir", str(output_dir),
        "--model_id", "Wan-AI/Wan2.1-I2V-14B-480P",
        "--num_inference_steps", str(args.num_inference_steps),
        "--guidance_scale", str(args.guidance_scale),
        "--height", str(args.height),
        "--width", str(args.width),
        "--num_frames", str(args.num_frames),
        "--device", args.device,
        "--seed", str(args.seed)
    ]
    
    if args.wan_lora:
        cmd.extend(["--lora_path", args.wan_lora])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    
    if result.returncode != 0:
        print("ERROR: WAN T2V generation failed")
        sys.exit(1)
    
    print("✓ WAN T2V generation complete")
    return output_dir


def run_framepack_i2v_generation(args):
    """Step 3: Generate videos using FramePack I2V."""
    print("\n" + "="*80)
    print("STEP 3: Generating videos with FramePack I2V")
    print("="*80)
    
    # Check if FramePack is available
    try:
        import framepack
        print("FramePack found")
    except ImportError:
        print("WARNING: FramePack not installed")
        print("To install FramePack, follow instructions at: https://github.com/pschaldenbrand/FramePack")
        print("Skipping FramePack generation...")
        return None
    
    output_dir = Path(args.output_dir) / "framepack_i2v"
    
    # TODO: Implement FramePack I2V generation
    # Use helper script we just added
    cmd = [
        sys.executable, "scripts/generate_i2v_framepack.py",
        "--frames_dir", args.frames_dir,
        "--output_dir", str(output_dir),
        "--num_frames", str(args.num_frames),
        "--num_inference_steps", str(args.num_inference_steps),
        "--guidance_scale", str(args.guidance_scale),
        "--height", str(args.height),
        "--width", str(args.width),
        "--device", args.device,
        "--fps", "8",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("ERROR: FramePack generation failed")
        sys.exit(1)

    print("✓ FramePack generation complete")
    return output_dir


def compute_fvd_for_method(method_name, generated_dir, reference_videos, args):
    """Compute FVD for a specific I2V method."""
    print(f"\nComputing FVD for {method_name}...")
    
    # Group reference videos by activity
    from collections import defaultdict
    activity_refs = defaultdict(list)
    
    frames_metadata_path = Path(args.frames_dir) / "frames_metadata.json"
    with open(frames_metadata_path, 'r') as f:
        frames_metadata = json.load(f)
    
    for frame_info in frames_metadata:
        activity = frame_info["activity"]
        activity_refs[activity].append(frame_info["original_video"])
    
    # Compute FVD per activity
    results = {}
    
    for activity, ref_vids in activity_refs.items():
        activity_safe = activity.replace(" ", "_")
        gen_activity_dir = generated_dir / activity_safe
        
        if not gen_activity_dir.exists():
            print(f"  WARNING: No generated videos for {activity}")
            continue
        
        gen_vids = list(gen_activity_dir.glob("*.mp4"))
        
        if len(gen_vids) == 0:
            print(f"  WARNING: No generated videos found in {gen_activity_dir}")
            continue
        
        print(f"  Computing FVD for {activity} ({len(ref_vids)} ref, {len(gen_vids)} gen)...")
        
        # Call FVD computation
        fvd_output = args.output_dir / f"fvd_{method_name}_{activity_safe}.json"
        
        cmd = [
            sys.executable, "scripts/compute_fvd.py",
            "--real_videos"] + [str(v) for v in ref_vids[:len(gen_vids)]] + [
            "--generated_videos"] + [str(v) for v in gen_vids] + [
            "--out", str(fvd_output),
            "--device", args.device,
            "--num_frames", str(args.num_frames)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            with open(fvd_output, 'r') as f:
                fvd_data = json.load(f)
            results[activity] = fvd_data["fvd"]
            print(f"    FVD: {fvd_data['fvd']:.4f}")
        else:
            print(f"    ERROR: FVD computation failed")
            print(result.stderr)
    
    return results


def run_fvd_benchmark(wan_dir, framepack_dir, args):
    """Step 4: Compute FVD for both methods."""
    print("\n" + "="*80)
    print("STEP 4: Computing FVD metrics")
    print("="*80)
    
    # Load reference videos
    frames_metadata_path = Path(args.frames_dir) / "frames_metadata.json"
    with open(frames_metadata_path, 'r') as f:
        frames_metadata = json.load(f)
    
    reference_videos = [info["original_video"] for info in frames_metadata]
    
    # Compute FVD for WAN I2V
    wan_fvd = compute_fvd_for_method("wan", wan_dir, reference_videos, args)
    
    # Compute FVD for FramePack I2V (if available)
    framepack_fvd = {}
    if framepack_dir and framepack_dir.exists():
        framepack_fvd = compute_fvd_for_method("framepack", framepack_dir, reference_videos, args)
    
    return wan_fvd, framepack_fvd


def generate_report(wan_fvd, framepack_fvd, args):
    """Step 5: Generate comparison report."""
    print("\n" + "="*80)
    print("STEP 5: Generating benchmark report")
    print("="*80)
    
    report = {
        "benchmark": "WAN 2.1 I2V vs FramePack I2V",
        "metric": "FVD (Fréchet Video Distance)",
        "wan_i2v": {
            "per_activity": wan_fvd,
            "mean": sum(wan_fvd.values()) / len(wan_fvd) if wan_fvd else None
        },
        "framepack_i2v": {
            "per_activity": framepack_fvd,
            "mean": sum(framepack_fvd.values()) / len(framepack_fvd) if framepack_fvd else None
        },
        "parameters": {
            "num_inference_steps": args.num_inference_steps,
            "num_frames": args.num_frames,
            "resolution": f"{args.height}x{args.width}",
            "guidance_scale": args.guidance_scale,
            "num_per_activity": args.num_per_activity
        }
    }
    
    # Save report
    report_path = Path(args.output_dir) / "benchmark_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    print("\nWAN 2.1 I2V:")
    if wan_fvd:
        for activity, fvd in sorted(wan_fvd.items()):
            print(f"  {activity:30s}: {fvd:8.4f}")
        print(f"  {'MEAN':30s}: {report['wan_i2v']['mean']:8.4f}")
    else:
        print("  No results")
    
    print("\nFramePack I2V:")
    if framepack_fvd:
        for activity, fvd in sorted(framepack_fvd.items()):
            print(f"  {activity:30s}: {fvd:8.4f}")
        print(f"  {'MEAN':30s}: {report['framepack_i2v']['mean']:8.4f}")
    else:
        print("  Not available (FramePack not installed or generation failed)")
    
    if wan_fvd and framepack_fvd:
        print("\nComparison:")
        wan_mean = report['wan_i2v']['mean']
        fp_mean = report['framepack_i2v']['mean']
        diff = wan_mean - fp_mean
        print(f"  WAN - FramePack: {diff:+.4f}")
        if diff < 0:
            print(f"  WAN is BETTER (lower FVD by {abs(diff):.4f})")
        else:
            print(f"  FramePack is BETTER (lower FVD by {diff:.4f})")
    
    print(f"\nReport saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark WAN I2V vs FramePack I2V")
    
    # Data paths
    parser.add_argument("--input_dir", type=str, default="data/processed_full/clips",
                      help="Directory containing reference videos")
    parser.add_argument("--manifest", type=str, default="data/processed_full/train.jsonl",
                      help="Path to manifest JSONL")
    parser.add_argument("--frames_dir", type=str, default="data/first_frames",
                      help="Directory to save/load first frames")
    parser.add_argument("--output_dir", type=str, default="generated_videos/i2v_benchmark",
                      help="Output directory for benchmark results")
    
    # Model paths
    parser.add_argument("--wan_lora", type=str,
                      help="Path to WAN LoRA checkpoint")
    
    # Generation parameters
    parser.add_argument("--num_per_activity", type=int, default=10,
                      help="Number of videos per activity")
    parser.add_argument("--num_inference_steps", type=int, default=150,
                      help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=9.0,
                      help="Guidance scale")
    parser.add_argument("--height", type=int, default=720,
                      help="Video height")
    parser.add_argument("--width", type=int, default=1280,
                      help="Video width")
    parser.add_argument("--num_frames", type=int, default=20,
                      help="Number of frames to generate")
    
    # Runtime parameters
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    parser.add_argument("--skip_extraction", action="store_true",
                      help="Skip frame extraction if already done")
    parser.add_argument("--skip_wan", action="store_true",
                      help="Skip WAN generation if already done")
    parser.add_argument("--skip_framepack", action="store_true",
                      help="Skip FramePack generation")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Extract frames (if needed)
    frames_metadata_path = Path(args.frames_dir) / "frames_metadata.json"
    if not args.skip_extraction or not frames_metadata_path.exists():
        run_frame_extraction(args)
    else:
        print("Skipping frame extraction (already done)")
    
    # Step 2: Generate with WAN I2V
    wan_dir = Path(args.output_dir) / "wan_t2v"
    if not args.skip_wan or not wan_dir.exists():
        wan_dir = run_wan_i2v_generation(args)
    else:
        print("Skipping WAN generation (already done)")
    
    # Step 3: Generate with FramePack I2V
    framepack_dir = None
    if not args.skip_framepack:
        framepack_dir = run_framepack_i2v_generation(args)
    else:
        print("Skipping FramePack generation")
    
    # Step 4: Compute FVD
    wan_fvd, framepack_fvd = run_fvd_benchmark(wan_dir, framepack_dir, args)
    
    # Step 5: Generate report
    generate_report(wan_fvd, framepack_fvd, args)
    
    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
