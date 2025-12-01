"""Dataset helpers for loading preprocessed NPZ clips and prompts.

Produces PyTorch Dataset yielding dicts: {'frames': tensor(T,C,H,W), 'prompt': str}
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
import numpy as np


class ClipDataset(Dataset):
    def __init__(self, manifest_path: str, transform=None, random_flip: bool = False, color_jitter: float = 0.0):
        """Simple dataset for NPZ clips.

        Args:
            manifest_path: path to JSONL manifest
            transform: optional callable applied to the sample
            random_flip: if True, randomly horizontally flips clips (p=0.5)
            color_jitter: float in [0, 1] controlling brightness jitter magnitude
        """
        self.manifest_path = Path(manifest_path)
        self.entries = [json.loads(l) for l in open(self.manifest_path, "r", encoding="utf-8")]
        self.transform = transform
        self.random_flip = bool(random_flip)
        self.color_jitter = float(color_jitter or 0.0)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        npz_path = Path(e["npz"]).resolve()
        data = np.load(npz_path)
        frames = data["frames"]  # (T,H,W,3) uint8
        # convert to torch float tensor in CHW and normalize to [-1,1]
        frames = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 127.5 - 1.0
        cache_id = hashlib.sha1(str(npz_path).encode("utf-8")).hexdigest()
        # Apply simple augmentations in clip-space (temporal axis preserved)
        if self.random_flip:
            # flip horizontally with 50% probability
            if torch.rand(1).item() < 0.5:
                frames = frames.flip(-1)

        if self.color_jitter and self.color_jitter > 0.0:
            # apply a uniform brightness multiplier in [1-cj, 1+cj]
            cj = float(self.color_jitter)
            factor = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * cj
            frames = frames * factor
            frames = torch.clamp(frames, -1.0, 1.0)

        sample = {
            "frames": frames,
            "prompt": e.get("prompt", ""),
            "cache_id": cache_id,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def collate_fn(batch: List[dict]):
    # batch: list of samples with variable-length frames (but we expect fixed T)
    frames = torch.stack([b["frames"] for b in batch], dim=0)  # (B,T,C,H,W)
    prompts = [b["prompt"] for b in batch]
    cache_ids = [b.get("cache_id") for b in batch]
    return {"frames": frames, "prompts": prompts, "cache_ids": cache_ids}
