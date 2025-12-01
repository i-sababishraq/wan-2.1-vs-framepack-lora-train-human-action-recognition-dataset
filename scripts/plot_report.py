#!/usr/bin/env python3
"""Plot a small report using evaluation metrics and per-video LPIPS.

Saves:
 - bar charts: fid_bar.png, lpips_bar.png
 - example frames: <activity>_example_low.png, _med, _high

Usage:
  python scripts/plot_report.py --metrics generated_videos/full/evaluation_metrics_postproc_20frames.json --per_video generated_videos/full/per_video_lpips.json --out generated_videos/full/report
"""
import argparse
import json
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio


def load_json(p):
    with open(p, 'r') as f:
        return json.load(f)


def save_bar(values, labels, title, outp):
    plt.figure(figsize=(8,4))
    x = np.arange(len(labels))
    plt.bar(x, values, color='C0')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outp, dpi=150)
    plt.close()


def extract_side_by_side_frame(bpath, lpath, frame_idx=0):
    # read first frame from each video (or specified index)
    rb = imageio.get_reader(bpath)
    rl = imageio.get_reader(lpath)
    try:
        fb = rb.get_data(frame_idx)
    except Exception:
        fb = rb.get_data(0)
    try:
        fl = rl.get_data(frame_idx)
    except Exception:
        fl = rl.get_data(0)
    # convert to PIL and resize to same height
    ib = Image.fromarray(fb).convert('RGB')
    il = Image.fromarray(fl).convert('RGB')
    h = 256
    ib = ib.resize((int(ib.width * h / ib.height), h))
    il = il.resize((int(il.width * h / il.height), h))
    # create combined image
    comb = Image.new('RGB', (ib.width + il.width + 10, h), (255,255,255))
    comb.paste(ib, (0,0))
    comb.paste(il, (ib.width + 10, 0))
    return comb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', required=True)
    parser.add_argument('--per_video', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    metrics = load_json(args.metrics)
    per_video = load_json(args.per_video)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    activities = list(metrics.keys())
    # FID bar (use fid values)
    fids = [metrics[a].get('fid', None) for a in activities]
    # LPIPS bar (use lpips values)
    lpips = [metrics[a].get('lpips', None) for a in activities]

    save_bar(fids, activities, 'Per-Activity Frame-FID', outdir / 'fid_bar.png')
    save_bar(lpips, activities, 'Per-Activity Frame-LPIPS', outdir / 'lpips_bar.png')

    # For each activity, pick low/med/high LPIPS videos
    for a in activities:
        pv = per_video.get(a, [])
        if len(pv) == 0:
            continue
        # filter entries with numeric lpips
        numeric = [p for p in pv if p['lpips'] is not None]
        if len(numeric) == 0:
            continue
        vals = np.array([p['lpips'] for p in numeric])
        idx_low = int(np.argmin(vals))
        idx_high = int(np.argmax(vals))
        idx_med = int(len(vals)//2)
        picks = [numeric[idx_low], numeric[idx_med], numeric[idx_high]]
        names = ['low','med','high']
        for pick, name in zip(picks, names):
            try:
                comb = extract_side_by_side_frame(pick['baseline'], pick['lora'], frame_idx=0)
                comb.save(outdir / f"{a.replace(' ','_')}_example_{name}.png")
            except Exception as e:
                print('Failed to save example for', a, name, e)

    print('Saved report to', str(outdir))

if __name__ == '__main__':
    main()
