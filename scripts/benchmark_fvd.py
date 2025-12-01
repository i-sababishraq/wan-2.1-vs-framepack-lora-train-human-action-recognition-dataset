"""
Benchmark video generation methods using FVD metric.

This script compares multiple video generation methods against reference videos
using the Fréchet Video Distance (FVD) metric.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import subprocess


def run_fvd_computation(
    real_videos: str,
    generated_videos: str,
    output_file: str,
    model_path: str = "models/i3d",
    batch_size: int = 4,
    num_frames: int = 16,
    device: str = "cuda"
) -> dict:
    """
    Run FVD computation for a pair of video directories.
    
    Args:
        real_videos: Directory with reference videos
        generated_videos: Directory with generated videos
        output_file: Path to save results JSON
        model_path: Path to I3D model
        batch_size: Batch size for processing
        num_frames: Number of frames to sample
        device: Device to use (cuda/cpu)
    
    Returns:
        Dictionary with FVD results
    """
    cmd = [
        "python", "scripts/compute_fvd.py",
        "--real_videos", real_videos,
        "--generated_videos", generated_videos,
        "--model_path", model_path,
        "--batch_size", str(batch_size),
        "--num_frames", str(num_frames),
        "--device", device,
        "--output", output_file
    ]
    
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"Error running FVD computation: {result.returncode}")
        return None
    
    # Load results
    if Path(output_file).exists():
        with open(output_file, "r") as f:
            return json.load(f)
    
    return None


def create_reference_video_list(data_dir: str, starting_frames_manifest: str) -> list:
    """
    Create a list of reference videos that correspond to the generated videos.
    
    Args:
        data_dir: Directory containing preprocessed clips
        starting_frames_manifest: Path to manifest with starting frames info
    
    Returns:
        List of paths to reference video files
    """
    reference_videos = []
    
    with open(starting_frames_manifest, "r") as f:
        for line in f:
            entry = json.loads(line)
            # Extract original video info from frame path
            # e.g., "Walking/Walking (123)_clip1_frame0.png" -> "Walking/Walking (123)_clip1.mp4"
            frame_path = entry.get("frame_path", "")
            video_id = entry.get("video_id", "")
            
            if video_id:
                # Construct reference video path
                activity = entry.get("activity", "").replace(" ", "_")
                ref_path = Path(data_dir) / "clips" / activity / f"{video_id}.mp4"
                if ref_path.exists():
                    reference_videos.append(str(ref_path))
                else:
                    print(f"Warning: Reference video not found: {ref_path}")
    
    return reference_videos


def copy_reference_videos_to_temp(reference_videos: list, temp_dir: str) -> str:
    """
    Copy reference videos to a temporary directory for FVD computation.
    
    Args:
        reference_videos: List of reference video paths
        temp_dir: Temporary directory to copy to
    
    Returns:
        Path to temporary directory
    """
    temp_path = Path(temp_dir)
    temp_path.mkdir(parents=True, exist_ok=True)
    
    import shutil
    
    for i, video_path in enumerate(reference_videos):
        src = Path(video_path)
        if src.exists():
            dst = temp_path / f"ref_{i:04d}_{src.name}"
            shutil.copy(src, dst)
    
    return str(temp_path)


def main():
    parser = argparse.ArgumentParser(description="Benchmark video generation methods with FVD")
    parser.add_argument("--reference_dir", type=str, default="data/clips",
                       help="Directory containing reference/real videos")
    parser.add_argument("--wan_videos", type=str, default="generated_videos/i2v_wan14b_480p",
                       help="Directory containing Wan2.1 generated videos")
    parser.add_argument("--framepack_videos", type=str, default="generated_videos/i2v_framepack_480p",
                       help="Directory containing Framepack generated videos")
    parser.add_argument("--output_dir", type=str, default="fvd_results",
                       help="Directory to save results")
    parser.add_argument("--model_path", type=str, default="models/i3d",
                       help="Path to I3D model directory")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for feature extraction")
    parser.add_argument("--num_frames", type=int, default=16,
                       help="Number of frames to sample per video")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Check that directories exist
    for name, path in [("Reference", args.reference_dir), 
                       ("Wan2.1", args.wan_videos),
                       ("Framepack", args.framepack_videos)]:
        if not Path(path).exists():
            print(f"Error: {name} directory not found: {path}")
            sys.exit(1)
    
    results_summary = {
        "timestamp": timestamp,
        "reference_dir": args.reference_dir,
        "methods": {}
    }
    
    print("\n" + "="*80)
    print("FVD BENCHMARK: Comparing Video Generation Methods")
    print("="*80)
    
    # 1. Compute FVD for Wan2.1 I2V
    print("\n>>> Benchmarking Wan2.1 I2V vs Reference Videos")
    wan_output = output_dir / f"fvd_wan2.1_{timestamp}.json"
    wan_results = run_fvd_computation(
        real_videos=args.reference_dir,
        generated_videos=args.wan_videos,
        output_file=str(wan_output),
        model_path=args.model_path,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        device=args.device
    )
    
    if wan_results:
        results_summary["methods"]["wan2.1_i2v"] = {
            "fvd": wan_results["fvd"],
            "num_videos": wan_results["num_generated_videos"],
            "output_file": str(wan_output)
        }
        print(f"\n✅ Wan2.1 I2V FVD: {wan_results['fvd']:.2f}")
    else:
        print("\n❌ Failed to compute FVD for Wan2.1")
    
    # 2. Compute FVD for Framepack I2V
    print("\n>>> Benchmarking Framepack I2V vs Reference Videos")
    framepack_output = output_dir / f"fvd_framepack_{timestamp}.json"
    framepack_results = run_fvd_computation(
        real_videos=args.reference_dir,
        generated_videos=args.framepack_videos,
        output_file=str(framepack_output),
        model_path=args.model_path,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        device=args.device
    )
    
    if framepack_results:
        results_summary["methods"]["framepack_i2v"] = {
            "fvd": framepack_results["fvd"],
            "num_videos": framepack_results["num_generated_videos"],
            "output_file": str(framepack_output)
        }
        print(f"\n✅ Framepack I2V FVD: {framepack_results['fvd']:.2f}")
    else:
        print("\n❌ Failed to compute FVD for Framepack")
    
    # Save summary
    summary_file = output_dir / f"benchmark_summary_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    
    # Print final comparison
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    
    if "wan2.1_i2v" in results_summary["methods"] and "framepack_i2v" in results_summary["methods"]:
        wan_fvd = results_summary["methods"]["wan2.1_i2v"]["fvd"]
        framepack_fvd = results_summary["methods"]["framepack_i2v"]["fvd"]
        
        print(f"\nWan2.1 I2V FVD:    {wan_fvd:.2f}")
        print(f"Framepack I2V FVD: {framepack_fvd:.2f}")
        print(f"\nDifference:        {abs(wan_fvd - framepack_fvd):.2f}")
        
        if wan_fvd < framepack_fvd:
            improvement = ((framepack_fvd - wan_fvd) / framepack_fvd) * 100
            print(f"\n🏆 Wan2.1 I2V is better by {improvement:.1f}%")
        else:
            improvement = ((wan_fvd - framepack_fvd) / wan_fvd) * 100
            print(f"\n🏆 Framepack I2V is better by {improvement:.1f}%")
    
    print(f"\n📊 Summary saved to: {summary_file}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

