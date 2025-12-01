#!/usr/bin/env python3
"""Scan a folder tree of starting-frame PNGs (one sub-dir per activity)
   and write frames_metadata.json compatible with generate_*_from_frames scripts.

Usage:
    python scripts/build_frame_metadata.py \
        --frames_root data/starting_frames \
        --out data/starting_frames/frames_metadata.json [--seed 42]

Each metadata entry:
{
  "frame_path": <png>,
  "activity":   <activity dir name>,
  "original_video": "",  # unknown
  "index": <running index>
}
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames_root", required=True, help="Root dir containing activity sub-folders with PNGs")
    p.add_argument("--out", required=True, help="Output metadata json path")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed (for reproducibility)")
    args = p.parse_args()

    root = Path(args.frames_root)
    assert root.is_dir(), f"{root} not found"

    entries = []
    for png in root.rglob("*.png"):
        if not png.is_file():
            continue
        # activity = first dir after root
        try:
            activity = png.relative_to(root).parts[0]
        except ValueError:
            activity = "unknown"
        entries.append({
            "frame_path": str(png),
            "activity": activity,
            "original_video": "",
            "index": 0  # to be filled later
        })

    # group per activity and cap 10 per activity
    random.seed(args.seed)
    final = []
    grouped = {}
    for e in entries:
        grouped.setdefault(e["activity"], []).append(e)
    for act, lst in grouped.items():
        random.shuffle(lst)
        for i, rec in enumerate(lst[:10]):
            rec["index"] = i
            final.append(rec)

    print(f"Writing {len(final)} entries to {args.out}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(final, f, indent=2)


if __name__ == "__main__":
    main()
