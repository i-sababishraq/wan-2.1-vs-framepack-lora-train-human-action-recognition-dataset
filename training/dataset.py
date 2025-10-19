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
    def __init__(self, manifest_path: str, transform=None):
        self.manifest_path = Path(manifest_path)
        self.entries = [json.loads(l) for l in open(self.manifest_path, "r", encoding="utf-8")]
        self.transform = transform

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
