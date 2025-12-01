#!/usr/bin/env python3
"""
Extract the first frame from each reference video for image-to-video generation.

This script processes videos from the dataset and saves the first frame
as a PNG file with metadata (activity label, video ID).

Usage:
    python scripts/extract_first_frames.py --input_dir data/processed_full/clips --output_dir data/first_frames --manifest data/processed_full/train.jsonl --num_per_activity 10
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


def read_manifest(manifest_path):
    """Read manifest JSONL file and return as list of dicts."""
    videos = []
    with open(manifest_path, 'r') as f:
        for line in f:
            videos.append(json.loads(line.strip()))
    return videos


def extract_first_frame(video_path):
    """Extract the first frame from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Extract first frames from reference videos")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing video files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save first frames")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest JSONL file")
    parser.add_argument("--num_per_activity", type=int, default=10, help="Number of frames per activity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Read manifest
    print(f"Reading manifest from {args.manifest}...")
    videos = read_manifest(args.manifest)
    print(f"Found {len(videos)} videos in manifest")
    
    # Group by activity
    activity_videos = {}
    for video in videos:
        # Check both "label" and "activity" keys (handle different manifest formats)
        label = video.get("label") or video.get("activity")
        if label:
            if label not in activity_videos:
                activity_videos[label] = []
            activity_videos[label].append(video)
    
    print(f"Found {len(activity_videos)} activities:")
    for activity, vids in activity_videos.items():
        print(f"  {activity}: {len(vids)} videos")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample and extract frames
    extracted_info = []
    
    for activity, vids in activity_videos.items():
        print(f"\nProcessing activity: {activity}")
        
        # Sample videos
        num_to_sample = min(args.num_per_activity, len(vids))
        sampled = random.sample(vids, num_to_sample)
        
        # Create activity directory
        activity_dir = output_dir / activity.replace(" ", "_")
        activity_dir.mkdir(parents=True, exist_ok=True)
        
        for i, video_info in enumerate(sampled):
            # Get video path - use source_video from manifest
            video_path = Path(video_info.get("source_video", ""))
            video_id = video_info.get("video_id") or video_info.get("id") or video_path.stem
            
            # If source_video not available, try constructing path
            if not video_path.exists():
                if video_id:
                    video_filename = f"{video_id}.mp4"
                    video_path = Path(args.input_dir) / activity.replace(" ", "_") / video_filename
            
            if not video_path.exists() or str(video_path) == "":
                print(f"  WARNING: Video not found: {video_path}")
                continue
            
            # Extract first frame
            frame = extract_first_frame(video_path)
            if frame is None:
                print(f"  WARNING: Could not extract frame from: {video_path}")
                continue
            
            # Save frame
            frame_filename = f"{activity.replace(' ', '_')}_{i:03d}.png"
            frame_path = activity_dir / frame_filename
            
            # Convert RGB to BGR for cv2.imwrite
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frame_path), frame_bgr)
            
            # Record info
            extracted_info.append({
                "frame_path": str(frame_path),
                "activity": activity,
                "video_id": video_id,
                "original_video": str(video_path),
                "index": i
            })
            
            print(f"  Saved: {frame_path} (shape: {frame.shape})")
    
    # Save metadata
    metadata_path = output_dir / "frames_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(extracted_info, f, indent=2)
    
    print(f"\nExtracted {len(extracted_info)} frames")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
