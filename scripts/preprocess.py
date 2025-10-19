"""Preprocess videos into clip npz files and a JSONL manifest for Wan 2.1 training.

Usage (example):
  python scripts/preprocess.py --input_dir "data/raw/Human Activity Recognition - Video Dataset" \
      --output_dir data/processed --clip_len 81 --stride 40 --width 832 --height 480

This script expects videos organized as:
  data/raw/<label>/*.mp4

Outputs:
  data/processed/clips/<label>/*.npz  -- each npz contains 'frames' uint8 (T,H,W,3)
  data/processed/train.jsonl          -- manifest with entries {npz, prompt, label}

Note: Wan 2.1 requires clip_len to be 4n+1 (e.g., 81, 85, 89 frames).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def extract_clips_from_video(video_path: Path, clip_len: int, stride: int, size: tuple[int, int]):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    ok, frame = cap.read()
    while ok:
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if size is not None:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        frames.append(frame)
        ok, frame = cap.read()
    cap.release()
    if len(frames) < clip_len:
        return []
    clips = []
    for start in range(0, len(frames) - clip_len + 1, stride):
        clip = np.stack(frames[start : start + clip_len], axis=0)  # (T,H,W,3)
        clips.append(clip)
    return clips


def build_prompt_from_label(label: str) -> str:
    """Generate a natural language prompt from the activity label.
    
    Maps folder names to descriptive prompts suitable for Wan 2.1 training.
    """
    # Clean up label formatting
    label_clean = label.lower().replace("_", " ").strip()
    
    # Map specific labels to better prompts
    prompt_map = {
        "clapping": "a person clapping their hands",
        "meet and split": "two people meeting and then walking in opposite directions",
        "sitting": "a person sitting on a chair",
        "standing still": "a person standing still",
        "walking": "a person walking",
        "walking while reading book": "a person walking while reading a book",
        "walking while using phone": "a person walking while using their phone",
    }
    
    activity = prompt_map.get(label_clean, f"a person {label_clean}")
    return f"A realistic video showing {activity}"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/raw/Human Activity Recognition - Video Dataset",
                        help="Root directory containing activity folders with videos")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--clip_len", type=int, default=81,
                        help="Number of frames per clip (must be 4n+1 for Wan 2.1, e.g., 81, 85, 89)")
    parser.add_argument("--stride", type=int, default=40,
                        help="Frame stride for sliding window clip extraction")
    parser.add_argument("--width", type=int, default=832,
                        help="Target width (must be multiple of 16 for Wan 2.1)")
    parser.add_argument("--height", type=int, default=480,
                        help="Target height (must be multiple of 16 for Wan 2.1)")
    parser.add_argument("--max_videos_per_label", type=int, default=0,
                        help="Limit the number of videos processed per label (0 = no limit)")
    args = parser.parse_args(argv)
    
    # Validate clip_len is 4n+1
    if (args.clip_len - 1) % 4 != 0:
        print(f"Warning: clip_len={args.clip_len} is not 4n+1. Wan 2.1 requires 4n+1 frames (e.g., 81, 85, 89).")
    
    # Validate dimensions are multiples of 16
    if args.width % 16 != 0 or args.height % 16 != 0:
        print(f"Warning: width={args.width} or height={args.height} is not a multiple of 16. "
              "Wan 2.1 requires dimensions divisible by 16.")

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "train.jsonl"
    if manifest_path.exists():
        print(f"Overwriting manifest at {manifest_path}")
        manifest_path.unlink()

    size = (args.width, args.height)

    labels = [p.name for p in input_dir.iterdir() if p.is_dir()]
    if not labels:
        # fallback: treat files directly under input_dir as videos with unknown label
        labels = ["unknown"]

    for label in labels:
        label_in = input_dir / label
        if not label_in.exists():
            # if label directory doesn't exist, skip
            continue
        out_label_dir = clips_dir / label
        out_label_dir.mkdir(parents=True, exist_ok=True)

        video_files = sorted([p for p in label_in.glob("**/*") if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")])
        if args.max_videos_per_label > 0:
            video_files = video_files[: args.max_videos_per_label]

        for vid_path in tqdm(video_files, desc=f"Processing {label}"):
            try:
                clips = extract_clips_from_video(vid_path, args.clip_len, args.stride, size)
            except Exception as e:
                print(f"Failed processing {vid_path}: {e}")
                continue

            base = vid_path.stem
            for i, clip in enumerate(clips):
                out_name = f"{base}_clip{i}.npz"
                out_path = out_label_dir / out_name
                # store as uint8 to save space; training script will convert to float and normalize
                np.savez_compressed(out_path, frames=clip.astype(np.uint8))
                prompt = build_prompt_from_label(label)
                manifest_entry = {
                    "npz": str(out_path),
                    "prompt": prompt,
                    "label": label,
                    "source_video": str(vid_path),
                }
                with open(manifest_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")

    print(f"Done. Clips stored in {clips_dir}, manifest at {manifest_path}")


if __name__ == "__main__":
    main()
