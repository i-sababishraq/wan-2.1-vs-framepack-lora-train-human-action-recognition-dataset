#!/usr/bin/env python3
"""
Compute LPIPS and FID between baseline and LoRA generated videos per activity.
Saves results to an output JSON.

Usage:
  python scripts/compute_metrics.py --input generated_videos/full/evaluation_results.json --out generated_videos/full/evaluation_metrics_postproc.json --frames_per_video 5

Notes:
- LPIPS requires `lpips` package. If unavailable, LPIPS will be skipped.
- FID is computed using torchvision's Inception v3 pretrained model (no torchmetrics required).
- This script samples up to `frames_per_video` frames per video, uniformly across the clip.
"""
import argparse
import json
import os
import sys
from pathlib import Path

try:
    import imageio
except Exception as e:
    print("imageio is required: pip install imageio", file=sys.stderr)
    raise

import numpy as np
import torch
import torchvision.transforms as T
import torchvision
from scipy import linalg

# Try lpips
try:
    import lpips
    HAS_LPIPS = True
except Exception:
    HAS_LPIPS = False


def read_video_frames(path, max_frames=None):
    """Return list of RGB frames as uint8 HxWx3"""
    reader = imageio.get_reader(path)
    frames = []
    try:
        for frame in reader:
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        try:
            reader.close()
        except Exception:
            pass
    return frames


def sample_frames(frames, k):
    n = len(frames)
    if n == 0:
        return []
    if n <= k:
        return frames
    # uniform sample indices
    idx = np.linspace(0, n - 1, k).round().astype(int)
    return [frames[i] for i in idx]


def preprocess_for_inception(frames, device):
    # frames: list of HxWx3 uint8
    # returns tensor Nx3x299x299 float32
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((299, 299)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    out = [transform(frame) for frame in frames]
    if len(out) == 0:
        return torch.empty((0, 3, 299, 299), device=device)
    return torch.stack(out).to(device)


def get_inception_features(tensor_x, model, device, batch_size=32):
    # tensor_x: N x 3 x 299 x 299
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(tensor_x), batch_size):
            b = tensor_x[i:i+batch_size].to(device)
            # inception returns logits; we want pool3 features
            # torchvision's inception model has Mixed_7c as final feature stage; easier: use forward to get aux/logits not helpful
            # Instead, use model.forward and extract from model's Mixed_7c output by registering hook
            out = model(b)
            # model(b) returns logits when aux_logits=False; however we can use the pretrained model's fc features by replacing fc
            # Simpler approach: use AdaptiveAvgPool on final features via model.fc's input
            # We'll instead create a feature-extractor using torchvision.models.feature_extraction
    

def get_feature_extractor(device):
    # Use torchvision feature_extraction to get pool3 features
    from torchvision.models.feature_extraction import create_feature_extractor
    # Use the modern weights API; set aux_logits=True to match older torchvision expectations
    try:
        from torchvision.models import Inception_V3_Weights
        inception = torchvision.models.inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
    except Exception:
        # Fallback for older torchvision versions
        inception = torchvision.models.inception_v3(pretrained=True, aux_logits=True)
    inception.to(device)
    # Identify the node name for final pooling
    # Use 'avgpool' node which yields (N,2048,1,1)
    return create_feature_extractor(inception, return_nodes={'avgpool': 'feat'})


def compute_stats(feats):
    # feats: NxD numpy
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    # product might be nearly singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(fid)


def compute_fid_for_activity(paths_baseline, paths_lora, device, frames_per_video=5):
    # gather frames from all baseline videos and all lora videos
    frames_base = []
    frames_lora = []
    for p in paths_baseline:
        frames = read_video_frames(p, max_frames=None)
        frames = sample_frames(frames, frames_per_video)
        frames_base.extend(frames)
    for p in paths_lora:
        frames = read_video_frames(p, max_frames=None)
        frames = sample_frames(frames, frames_per_video)
        frames_lora.extend(frames)
    if len(frames_base) == 0 or len(frames_lora) == 0:
        return None
    # feature extractor
    feat_extractor = get_feature_extractor(device)
    feat_extractor.eval()
    # preprocess and compute features in batches
    def frames_to_feats(frames):
        xs = []
        batch = 64
        for i in range(0, len(frames), batch):
            sub = frames[i:i+batch]
            t = preprocess_for_inception(sub, device)
            with torch.no_grad():
                out = feat_extractor(t)
                # out['feat'] shape N x 2048 x 1 x 1
                f = out['feat'].squeeze(-1).squeeze(-1).cpu().numpy()
                xs.append(f)
        if len(xs) == 0:
            return np.zeros((0, 2048), dtype=np.float32)
        return np.concatenate(xs, axis=0)
    feats_b = frames_to_feats(frames_base)
    feats_l = frames_to_feats(frames_lora)
    mu1, sigma1 = compute_stats(feats_b)
    mu2, sigma2 = compute_stats(feats_l)
    fid = frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


def compute_fvd_for_activity(paths_baseline, paths_lora, device, frames_per_video=16):
    """Compute a video-level FVD-like score using a 3D CNN (r3d_18) features.
    This is an approximation to FVD which normally uses I3D features; r3d_18 from torchvision
    provides a reasonable video feature extractor for motion-aware embeddings.
    """
    # lazy import
    from torchvision.models.feature_extraction import create_feature_extractor
    # load r3d_18 pretrained on kinetics
    try:
        r3d = torchvision.models.video.r3d_18(weights=torchvision.models.video.R3D_18_Weights.KINETICS400_V1)
    except Exception:
        # fallback to older API
        r3d = torchvision.models.video.r3d_18(pretrained=True)
    r3d.to(device).eval()
    feat_extractor = create_feature_extractor(r3d, return_nodes={'avgpool': 'feat'})

    def video_to_feat(path):
        frames = read_video_frames(path, max_frames=None)
        if len(frames) == 0:
            return None
        sampled = sample_frames(frames, frames_per_video)
        # preprocess: resize to 112x112 (common for video models), convert to tensor shape CxTxxHxW
        resized = []
        for f in sampled:
            pil = T.ToPILImage()(f)
            pil = pil.resize((112,112))
            resized.append(T.ToTensor()(pil))
        # stack into tensor: T x C x H x W -> C x T x H x W
        t = torch.stack(resized, dim=1).unsqueeze(0).to(device)
        with torch.no_grad():
            out = feat_extractor(t)
            feat = out['feat'].squeeze()  # shape (512,1,1) or (512)
            if feat.ndim > 1:
                feat = feat.view(-1)
            return feat.cpu().numpy()

    feats_b = []
    feats_l = []
    for p in paths_baseline:
        f = video_to_feat(p)
        if f is not None:
            feats_b.append(f)
    for p in paths_lora:
        f = video_to_feat(p)
        if f is not None:
            feats_l.append(f)
    if len(feats_b) == 0 or len(feats_l) == 0:
        return None
    feats_b = np.stack(feats_b, axis=0)
    feats_l = np.stack(feats_l, axis=0)
    mu1, sigma1 = compute_stats(feats_b)
    mu2, sigma2 = compute_stats(feats_l)
    fid = frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


def compute_lpips_for_activity(paths_baseline, paths_lora, device, frames_per_video=5):
    if not HAS_LPIPS:
        return None
    loss_fn = lpips.LPIPS(net='vgg').to(device)
    vals = []
    for p_b, p_l in zip(paths_baseline, paths_lora):
        frames_b = read_video_frames(p_b, max_frames=None)
        frames_l = read_video_frames(p_l, max_frames=None)
        if len(frames_b) == 0 or len(frames_l) == 0:
            continue
        sb = sample_frames(frames_b, frames_per_video)
        sl = sample_frames(frames_l, frames_per_video)
        # ensure equal length
        n = min(len(sb), len(sl))
        if n == 0:
            continue
        per_video_vals = []
        for i in range(n):
            a = sb[i]
            b = sl[i]
            # convert to tensor - normalized [-1,1]
            ta = T.ToTensor()(a).unsqueeze(0).to(device) * 2 - 1
            tb = T.ToTensor()(b).unsqueeze(0).to(device) * 2 - 1
            with torch.no_grad():
                v = loss_fn(ta, tb)
            per_video_vals.append(float(v.cpu().numpy()))
        if len(per_video_vals) > 0:
            vals.append(float(np.mean(per_video_vals)))
    if len(vals) == 0:
        return None
    return float(np.mean(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="evaluation_results.json path")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--frames_per_video", type=int, default=5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(args.input, 'r') as f:
        data = json.load(f)

    results = {}
    per_video_lpips = {}
    for activity, info in data.items():
        paths_b = info['generated']['baseline']
        paths_l = info['generated']['lora']
        # normalize paths
        paths_b = [p.strip() for p in paths_b]
        paths_l = [p.strip() for p in paths_l]
        # resolve relative paths
        paths_b = [str(Path(p)) for p in paths_b]
        paths_l = [str(Path(p)) for p in paths_l]
        print(f"Computing metrics for activity {activity}: {len(paths_b)} baseline videos, {len(paths_l)} lora videos")
        # compute frame-based FID and LPIPS
        fid = compute_fid_for_activity(paths_b, paths_l, device, frames_per_video=args.frames_per_video)
        lp = compute_lpips_for_activity(paths_b, paths_l, device, frames_per_video=args.frames_per_video)
        # compute video-level FVD-like score using r3d_18 features
        fvd = compute_fvd_for_activity(paths_b, paths_l, device, frames_per_video=args.frames_per_video)

        results[activity] = {
            'fid': fid,
            'fvd_approx': fvd,
            'lpips': lp,
            'num_baseline_videos': len(paths_b),
            'num_lora_videos': len(paths_l)
        }
        print(f"  -> fid={fid}, fvd_approx={fvd}, lpips={lp}")

        # per-video LPIPS: compute LPIPS per paired baseline/lora video and store
        per_video = []
        for pb, pl in zip(paths_b, paths_l):
            val = None
            if HAS_LPIPS:
                try:
                    val = compute_lpips_for_activity([pb], [pl], device, frames_per_video=args.frames_per_video)
                except Exception:
                    val = None
            per_video.append({'baseline': pb, 'lora': pl, 'lpips': val})
        per_video_lpips[activity] = per_video

    # Save
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved metrics to', str(outp))
    # save per-video LPIPS
    pv = outp.parent / 'per_video_lpips.json'
    with open(pv, 'w') as f:
        json.dump(per_video_lpips, f, indent=2)
    print('Saved per-video LPIPS to', str(pv))

    # report missing packages
    if not HAS_LPIPS:
        print('\nLPIPS not available; install with: pip install lpips', file=sys.stderr)


if __name__ == '__main__':
    main()
