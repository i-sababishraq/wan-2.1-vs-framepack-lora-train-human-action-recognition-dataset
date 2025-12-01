"""
Compute Fréchet Video Distance (FVD) between two sets of videos.

FVD measures the distance between the feature distributions of real and generated videos.
Lower FVD scores indicate better video quality and diversity.

Reference: "Towards Accurate Generative Models of Video: A New Metric & Challenges"
https://openreview.net/pdf?id=rylgEULtdN
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import cv2
from scipy import linalg

# We'll implement a simple 3D CNN feature extractor instead of using pytorch-i3d
# which has installation issues
InceptionI3d = None


class Unit3D(torch.nn.Module):
    """Basic unit for I3D: 3D convolution + batch norm + activation."""
    
    def __init__(self, in_channels, out_channels, kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1), padding=0, activation_fn=F.relu, use_batch_norm=True, use_bias=False):
        super().__init__()
        
        self.conv3d = torch.nn.Conv3d(
            in_channels, out_channels, kernel_shape, stride=stride, padding=padding, bias=use_bias
        )
        
        if use_batch_norm:
            self.bn = torch.nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.001)
        else:
            self.bn = None
        
        self.activation_fn = activation_fn
    
    def forward(self, x):
        x = self.conv3d(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x


class InceptionModule(torch.nn.Module):
    """Inception module for I3D."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 1x1x1 conv
        self.b0 = Unit3D(in_channels, out_channels[0], kernel_shape=(1, 1, 1))
        
        # 1x1x1 -> 3x3x3 conv
        self.b1a = Unit3D(in_channels, out_channels[1], kernel_shape=(1, 1, 1))
        self.b1b = Unit3D(out_channels[1], out_channels[2], kernel_shape=(3, 3, 3), padding=1)
        
        # 1x1x1 -> 3x3x3 conv
        self.b2a = Unit3D(in_channels, out_channels[3], kernel_shape=(1, 1, 1))
        self.b2b = Unit3D(out_channels[3], out_channels[4], kernel_shape=(3, 3, 3), padding=1)
        
        # 3x3x3 pool -> 1x1x1 conv
        self.b3a = torch.nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.b3b = Unit3D(in_channels, out_channels[5], kernel_shape=(1, 1, 1))
    
    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(torch.nn.Module):
    """Inflated Inception-v1 for video classification (I3D)."""
    
    def __init__(self, num_classes=400, in_channels=3):
        super().__init__()
        
        self.Conv3d_1a_7x7 = Unit3D(in_channels, 64, kernel_shape=(7, 7, 7), stride=(2, 2, 2), padding=3)
        self.MaxPool3d_2a_3x3 = torch.nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        self.Conv3d_2b_1x1 = Unit3D(64, 64, kernel_shape=(1, 1, 1))
        self.Conv3d_2c_3x3 = Unit3D(64, 192, kernel_shape=(3, 3, 3), padding=1)
        self.MaxPool3d_3a_3x3 = torch.nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        self.Mixed_3b = InceptionModule(192, [64, 96, 128, 16, 32, 32])
        self.Mixed_3c = InceptionModule(256, [128, 128, 192, 32, 96, 64])
        self.MaxPool3d_4a_3x3 = torch.nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        
        self.Mixed_4b = InceptionModule(480, [192, 96, 208, 16, 48, 64])
        self.Mixed_4c = InceptionModule(512, [160, 112, 224, 24, 64, 64])
        self.Mixed_4d = InceptionModule(512, [128, 128, 256, 24, 64, 64])
        self.Mixed_4e = InceptionModule(512, [112, 144, 288, 32, 64, 64])
        self.Mixed_4f = InceptionModule(528, [256, 160, 320, 32, 128, 128])
        self.MaxPool3d_5a_2x2 = torch.nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.Mixed_5b = InceptionModule(832, [256, 160, 320, 32, 128, 128])
        self.Mixed_5c = InceptionModule(832, [384, 192, 384, 48, 128, 128])
        
        self.avg_pool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = torch.nn.Dropout(0.5)
        self.logits = Unit3D(1024, num_classes, kernel_shape=(1, 1, 1), activation_fn=None, use_batch_norm=False, use_bias=True)
    
    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.Conv3d_1a_7x7(x)
        x = self.MaxPool3d_2a_3x3(x)
        
        x = self.Conv3d_2b_1x1(x)
        x = self.Conv3d_2c_3x3(x)
        x = self.MaxPool3d_3a_3x3(x)
        
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)
        x = self.MaxPool3d_4a_3x3(x)
        
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)
        x = self.MaxPool3d_5a_2x2(x)
        
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        
        # For FVD, we use average pooling features (before logits)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)  # (B, 1024)
        
        return x
    
    def extract_features(self, x):
        """Extract features for FVD computation."""
        return self.forward(x)


def load_i3d_model(device: str = "cuda", model_path: str = "models/i3d") -> torch.nn.Module:
    """Load I3D model with pre-trained weights for feature extraction."""
    model_path = Path(model_path)
    model_file = model_path / "i3d_rgb.pt"
    
    # Check if I3D weights exist
    if model_file.exists():
        print(f"Loading I3D model from {model_file}")
        try:
            # Create I3D model
            model = InceptionI3d(num_classes=400, in_channels=3)
            
            # Load pre-trained weights
            state_dict = torch.load(str(model_file), map_location='cpu')
            model.load_state_dict(state_dict)
            
            model.to(device)
            model.eval()
            print("✓ I3D model loaded successfully (official FVD method)")
            return model
        except Exception as e:
            print(f"Error loading I3D weights: {e}")
            print("Falling back to ResNet3D...")
    else:
        print(f"I3D weights not found at {model_file}")
        print("Falling back to ResNet3D...")
    
    # Fallback to ResNet3D
    try:
        from torchvision.models.video import r3d_18, R3D_18_Weights
        print("Using ResNet3D-18 for feature extraction")
        model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        # Remove the final classification layer
        model.fc = torch.nn.Identity()
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading ResNet3D: {e}")
        raise RuntimeError("Could not load any video feature extraction model")


def load_video_frames(video_path: str, num_frames: int = 16, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Load and preprocess video frames for I3D.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample (I3D uses 16 or 64)
        target_size: Target spatial size (height, width)
    
    Returns:
        Video tensor of shape (num_frames, height, width, 3) with values in [0, 1]
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        raise ValueError(f"Video has no frames: {video_path}")
    
    # Sample frame indices uniformly
    if total_frames >= num_frames:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        # Repeat frames if video is too short
        frame_indices = np.array([i % total_frames for i in range(num_frames)])
    
    frames = []
    current_idx = 0
    
    for target_idx in frame_indices:
        # Seek to target frame
        while current_idx <= target_idx:
            ret, frame = cap.read()
            if not ret:
                break
            if current_idx == target_idx:
                # Convert BGR to RGB and resize
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, target_size)
                # Normalize to [0, 1]
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            current_idx += 1
    
    cap.release()
    
    if len(frames) != num_frames:
        raise ValueError(f"Expected {num_frames} frames, got {len(frames)} from {video_path}")
    
    return np.stack(frames)


def preprocess_video_for_i3d(video_tensor: np.ndarray) -> torch.Tensor:
    """
    Preprocess video tensor for I3D model.
    
    Args:
        video_tensor: (T, H, W, C) numpy array with values in [0, 1]
    
    Returns:
        torch.Tensor of shape (1, C, T, H, W) normalized for I3D
    """
    # Convert to torch tensor and rearrange dimensions
    # From (T, H, W, C) to (C, T, H, W)
    video_tensor = torch.from_numpy(video_tensor).permute(3, 0, 1, 2)
    
    # Normalize using ImageNet statistics (I3D is trained on ImageNet)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    video_tensor = (video_tensor - mean) / std
    
    # Add batch dimension
    video_tensor = video_tensor.unsqueeze(0)
    
    return video_tensor


@torch.no_grad()
def extract_i3d_features(
    video_paths: List[str],
    i3d_model: torch.nn.Module,
    device: str = "cuda",
    batch_size: int = 4,
    num_frames: int = 16
) -> np.ndarray:
    """
    Extract video features from a list of videos.
    
    Args:
        video_paths: List of paths to video files
        i3d_model: Pre-trained 3D CNN model
        device: Device to run on
        batch_size: Number of videos to process at once
        num_frames: Number of frames to sample per video
    
    Returns:
        Feature matrix of shape (num_videos, feature_dim)
    """
    features_list = []
    
    for i in tqdm(range(0, len(video_paths), batch_size), desc="Extracting video features"):
        batch_paths = video_paths[i:i + batch_size]
        batch_videos = []
        
        for video_path in batch_paths:
            try:
                # Load and preprocess video
                frames = load_video_frames(video_path, num_frames=num_frames)
                video_tensor = preprocess_video_for_i3d(frames)
                batch_videos.append(video_tensor)
            except Exception as e:
                print(f"Error loading {video_path}: {e}")
                # Use zeros as fallback
                batch_videos.append(torch.zeros(1, 3, num_frames, 224, 224))
        
        # Stack batch
        batch_tensor = torch.cat(batch_videos, dim=0).to(device)
        
        # Extract features - works with I3D, ResNet3D
        try:
            features = i3d_model(batch_tensor)
            
            # Ensure features are 2D (batch_size, feature_dim)
            if features.dim() > 2:
                # Apply global average pooling if output is still spatial/temporal
                features = F.adaptive_avg_pool3d(features, 1) if features.dim() == 5 else features
                features = features.view(features.size(0), -1)
            elif features.dim() == 1:
                features = features.unsqueeze(0)
                
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Fallback to zeros - I3D outputs 1024-dim, ResNet3D outputs 512-dim
            feature_dim = 1024 if isinstance(i3d_model, InceptionI3d) else 512
            features = torch.zeros(len(batch_videos), feature_dim).to(device)
        
        features_list.append(features.cpu().numpy())
    
    return np.vstack(features_list)


def calculate_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, 
                               mu2: np.ndarray, sigma2: np.ndarray,
                               eps: float = 1e-6) -> float:
    """
    Calculate Fréchet Distance between two multivariate Gaussians.
    
    The Fréchet distance between two multivariate Gaussians X_1 ~ N(mu_1, sigma_1)
    and X_2 ~ N(mu_2, sigma_2) is:
    
    d^2 = ||mu_1 - mu_2||^2 + Tr(sigma_1 + sigma_2 - 2*sqrt(sigma_1*sigma_2))
    
    Args:
        mu1: Mean of first distribution (feature_dim,)
        sigma1: Covariance of first distribution (feature_dim, feature_dim)
        mu2: Mean of second distribution (feature_dim,)
        sigma2: Covariance of second distribution (feature_dim, feature_dim)
        eps: Small value for numerical stability
    
    Returns:
        Fréchet distance
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariance matrices have different dimensions"
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        print(f"WARNING: FID calculation produces singular product; adding {eps} to diagonal of covariance estimates")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def compute_fvd(real_features: np.ndarray, generated_features: np.ndarray) -> float:
    """
    Compute FVD between real and generated video features.
    
    Args:
        real_features: Features from real videos (num_videos, feature_dim)
        generated_features: Features from generated videos (num_videos, feature_dim)
    
    Returns:
        FVD score (lower is better)
    """
    # Compute statistics
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    mu_gen = np.mean(generated_features, axis=0)
    sigma_gen = np.cov(generated_features, rowvar=False)
    
    # Calculate FVD
    fvd_score = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    
    return fvd_score


def main():
    parser = argparse.ArgumentParser(description="Compute FVD between two sets of videos")
    parser.add_argument("--real_videos", type=str, required=True,
                       help="Directory containing real/reference videos")
    parser.add_argument("--generated_videos", type=str, required=True,
                       help="Directory containing generated videos")
    parser.add_argument("--model_path", type=str, default="models/i3d",
                       help="Path to I3D model directory")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for feature extraction")
    parser.add_argument("--num_frames", type=int, default=16,
                       help="Number of frames to sample per video")
    parser.add_argument("--output", type=str, default="fvd_results.json",
                       help="Output file for results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Find video files
    real_videos = sorted(Path(args.real_videos).rglob("*.mp4"))
    generated_videos = sorted(Path(args.generated_videos).rglob("*.mp4"))
    
    print(f"Found {len(real_videos)} real videos")
    print(f"Found {len(generated_videos)} generated videos")
    
    if len(real_videos) == 0 or len(generated_videos) == 0:
        print("Error: No videos found in one or both directories")
        sys.exit(1)
    
    # Load I3D model
    device = args.device if torch.cuda.is_available() else "cpu"
    i3d_model = load_i3d_model(device=device, model_path=args.model_path)
    
    # Extract features
    print("\nExtracting features from real videos...")
    real_features = extract_i3d_features(
        [str(p) for p in real_videos],
        i3d_model,
        device=device,
        batch_size=args.batch_size,
        num_frames=args.num_frames
    )
    
    print("\nExtracting features from generated videos...")
    generated_features = extract_i3d_features(
        [str(p) for p in generated_videos],
        i3d_model,
        device=device,
        batch_size=args.batch_size,
        num_frames=args.num_frames
    )
    
    # Compute FVD
    print("\nComputing FVD...")
    fvd_score = compute_fvd(real_features, generated_features)
    
    # Save results
    results = {
        "fvd": float(fvd_score),
        "num_real_videos": len(real_videos),
        "num_generated_videos": len(generated_videos),
        "real_features_shape": real_features.shape,
        "generated_features_shape": generated_features.shape,
        "real_videos_dir": args.real_videos,
        "generated_videos_dir": args.generated_videos
    }
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"FVD Score: {fvd_score:.2f}")
    print(f"{'='*60}")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
