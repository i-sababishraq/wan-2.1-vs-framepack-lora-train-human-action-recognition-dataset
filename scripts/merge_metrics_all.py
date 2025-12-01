#!/usr/bin/env python3
import json
from pathlib import Path

base = Path('generated_videos/full')
old = base / 'evaluation_metrics_postproc_20frames.json'
new = base / 'evaluation_metrics_postproc_new_20frames.json'
out = base / 'evaluation_metrics_all_20frames.json'
merged = {}
for p in (old, new):
    if p.exists():
        try:
            merged.update(json.load(p.open()))
        except Exception as e:
            print('Failed to read', p, e)

out.parent.mkdir(parents=True, exist_ok=True)
json.dump(merged, out.open('w'), indent=2)
print('Wrote merged metrics ->', out)

# Merge per-video LPIPS
pprev = base / 'per_video_lpips_prev.json'
pnew = base / 'per_video_lpips.json'
out_pp = base / 'per_video_lpips_all.json'
pervideo = {}
for p in (pprev, pnew):
    if p.exists():
        try:
            d = json.load(p.open())
            for k, v in d.items():
                pervideo.setdefault(k, []).extend(v)
        except Exception as e:
            print('Failed to read per-video', p, e)

json.dump(pervideo, out_pp.open('w'), indent=2)
print('Wrote merged per-video LPIPS ->', out_pp)
