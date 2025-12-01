"""
Convert reference .npz video clips to .mp4 format for FVD computation.

This script reads the starting_frames_manifest.jsonl and converts the corresponding
reference clips from .npz (numpy arrays) to .mp4 format.
"""

import json
import argparse
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm


def npz_to_mp4(npz_path: str, output_path: str, fps: int = 8) -> bool:
    """
    Convert a .npz video clip to .mp4 format.
    
    Args:
        npz_path: Path to input .npz file
        output_path: Path to output .mp4 file
        fps: Frames per second for output video
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load npz file
        data = np.load(npz_path)
        
        # Try different possible keys for video data
        if 'video' in data:
            frames = data['video']
        elif 'frames' in data:
            frames = data['frames']
        elif 'arr_0' in data:
            frames = data['arr_0']
        else:
            # Try the first array in the file
            keys = list(data.keys())
            if keys:
                frames = data[keys[0]]
            else:
                print(f"No valid data found in {npz_path}")
                return False
        
        # Ensure frames are in the right shape and format
        # Expected shape: (T, H, W, C) or (T, C, H, W)
        if frames.ndim != 4:
            print(f"Unexpected frame shape {frames.shape} in {npz_path}")
            return False
        
        # Convert to (T, H, W, C) if needed
        if frames.shape[1] == 3 or frames.shape[1] == 1:  # (T, C, H, W)
            frames = np.transpose(frames, (0, 2, 3, 1))
        
        T, H, W, C = frames.shape
        
        # Normalize to 0-255 uint8 if needed
        if frames.dtype == np.float32 or frames.dtype == np.float64:
            if frames.max() <= 1.0:
                frames = (frames * 255).astype(np.uint8)
            else:
                frames = frames.astype(np.uint8)
        elif frames.dtype != np.uint8:
            frames = frames.astype(np.uint8)
        
        # Handle grayscale
        if C == 1:
            frames = np.repeat(frames, 3, axis=-1)
        elif C != 3:
            print(f"Unexpected number of channels {C} in {npz_path}")
            return False
        
        # Write video using OpenCV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        
        writer.release()
        return True
        
    except Exception as e:
        print(f"Error converting {npz_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert reference npz clips to mp4")
    parser.add_argument("--manifest", type=str, 
                       default="data/starting_frames/starting_frames_manifest.jsonl",
                       help="Path to starting frames manifest")
    parser.add_argument("--output_dir", type=str,
                       default="data/reference_videos_mp4",
                       help="Output directory for mp4 files")
    parser.add_argument("--fps", type=int, default=8,
                       help="Frames per second for output videos")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read manifest
    entries = []
    with open(args.manifest, "r") as f:
        for line in f:
            entries.append(json.loads(line))
    
    print(f"Found {len(entries)} entries in manifest")
    print(f"Converting npz clips to mp4 in {output_dir}")
    
    # Convert each reference video
    successful = 0
    failed = 0
    
    for entry in tqdm(entries, desc="Converting videos"):
        npz_path = entry.get("video_path")
        activity = entry.get("activity", "unknown").replace(" ", "_")
        
        if not npz_path:
            print(f"No video_path in entry: {entry}")
            failed += 1
            continue
        
        npz_path = Path(npz_path)
        if not npz_path.exists():
            print(f"File not found: {npz_path}")
            failed += 1
            continue
        
        # Create output path maintaining directory structure
        output_subdir = output_dir / activity
        output_subdir.mkdir(parents=True, exist_ok=True)
        output_path = output_subdir / f"{npz_path.stem}.mp4"
        
        # Convert
        if npz_to_mp4(str(npz_path), str(output_path), fps=args.fps):
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

