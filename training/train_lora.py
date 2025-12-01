"""LoRA training for Wan2.1 T2V 1.3B using Diffusers + PEFT.

This script implements full LoRA fine-tuning for Wan 2.1 with:
- VAE encoding to latent space
- Flow matching noise scheduling
- Text encoding with T5
- Mixed precision training (bf16)
- Gradient accumulation and checkpointing

Run a smoke test with a tiny subset first:
  python training/train_lora.py --max_steps 50 --batch_size 1
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Suppress tokenizers parallelism warning when forking processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime

try:
    from peft import LoraConfig, get_peft_model
except Exception:
    LoraConfig = None

from diffusers import WanPipeline, FlowMatchEulerDiscreteScheduler

try:
    from accelerate import Accelerator
except ImportError as exc:
    raise ImportError("Install 'accelerate' to enable distributed LoRA training.") from exc

# Import from same directory
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import ClipDataset, collate_fn


def apply_lora_to_transformer(transformer, r=32, lora_alpha=32, lora_dropout=0.0, target_modules=None):
    """Apply LoRA adapters to Wan 2.1 DiT transformer.
    
    Args:
        transformer: Wan DiT transformer model
        r: LoRA rank (16-64 typical for video models)
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout for LoRA layers
        target_modules: Which modules to apply LoRA to
    
    Returns:
        PEFT-wrapped transformer with trainable LoRA parameters
    """
    if LoraConfig is None:
        raise RuntimeError("PEFT is not installed. Install 'peft' to use LoRA.")
    
    if target_modules is None:
        # Target attention and projection layers in Wan DiT
        # Wan has blocks with attn1 (self-attention) and attn2 (cross-attention)
        target_modules = [
            "attn1.to_q",
            "attn1.to_k", 
            "attn1.to_v",
            "attn1.to_out.0",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
            "ffn.net.0.proj",  # Feed-forward projections
            "ffn.net.2",
        ]
    
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )
    
    peft_model = get_peft_model(transformer, config)
    peft_model.print_trainable_parameters()
    return peft_model


def train(args):
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    accelerator = Accelerator(mixed_precision="bf16" if use_bf16 else "no")
    device = accelerator.device
    dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32

    accelerator.print(f"Device: {device}, Dtype: {dtype}")

    latent_cache_dir = Path(args.latent_cache_dir).expanduser().resolve() if args.latent_cache_dir else None
    if latent_cache_dir:
        if accelerator.is_main_process:
            latent_cache_dir.mkdir(parents=True, exist_ok=True)
        accelerator.wait_for_everyone()

    # Load model components
    model_id = args.model_id
    accelerator.print(f"Loading base model {model_id}")

    pipe = WanPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to(device)

    # Keep VAE in float32 for stability
    pipe.vae = pipe.vae.to(device=device, dtype=torch.float32)
    pipe.vae.requires_grad_(False)
    pipe.vae.eval()
    vae = pipe.vae
    scaling_factor = getattr(vae.config, 'scaling_factor', 0.13025)
    
    # Freeze text encoder and keep VAE frozen
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder.eval()
    
    # Apply LoRA to transformer (DiT)
    accelerator.print("Applying LoRA to Transformer")
    pipe.transformer = apply_lora_to_transformer(
        pipe.transformer, 
        r=args.lora_rank, 
        lora_alpha=args.lora_alpha
    )
    pipe.transformer.train()
    
    # Load scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_id, 
        subfolder="scheduler"
    )

    def get_sigmas(indices_cpu: torch.Tensor, n_dim: int, dtype: torch.dtype) -> torch.Tensor:
        """Gather sigmas for the provided scheduler indices and reshape for broadcasting."""
        sigma = noise_scheduler.sigmas[indices_cpu].to(device=device, dtype=dtype)
        while sigma.ndim < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    # Prepare dataset and dataloader
    accelerator.print(f"Loading dataset from {args.manifest}")
    # Pass augmentation flags into the dataset (simple flip / color jitter)
    ds = ClipDataset(args.manifest, random_flip=getattr(args, 'random_flip', False), color_jitter=getattr(args, 'color_jitter', 0.0))
    dl = DataLoader(
        ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    accelerator.print(f"Dataset size: {len(ds)} clips")
    
    # Optimizer: only LoRA parameters
    trainable_params = [p for p in pipe.transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    pipe.transformer, optimizer, dl = accelerator.prepare(pipe.transformer, optimizer, dl)

    # Training loop
    num_epochs = args.epochs
    global_step = 0
    
    # Create output directory
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Apply environment-variable overrides (if any) so the saved config matches runtime
        try:
            env_rank = os.environ.get("LORA_RANK")
            if env_rank is not None:
                args.lora_rank = int(env_rank)
        except Exception:
            pass
        try:
            env_alpha = os.environ.get("LORA_ALPHA")
            if env_alpha is not None:
                args.lora_alpha = int(env_alpha)
        except Exception:
            pass
        # boolean-ish flags for aug
        if os.environ.get("RANDOM_FLIP") is not None:
            args.random_flip = os.environ.get("RANDOM_FLIP") in ("1", "true", "True", "yes")
        if os.environ.get("COLOR_JITTER") is not None:
            try:
                args.color_jitter = float(os.environ.get("COLOR_JITTER"))
            except Exception:
                pass

        # Save run config after all finalizations so it reflects the effective runtime values
        config_path = output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        accelerator.print(f"Config saved to {config_path}")
    accelerator.wait_for_everyone()

    # Training log
    log_file = output_dir / "training_log.jsonl"

    try:
        accelerator.print(f"\nStarting training: {num_epochs} epochs, max {args.max_steps} steps")
        accelerator.print(f"Batch size: {args.batch_size}, Learning rate: {args.lr}")
        accelerator.print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}\n")
    
        optimizer.zero_grad(set_to_none=True)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            transform_suffix = []
            if args.frame_subsample > 1:
                transform_suffix.append(f"fs{args.frame_subsample}")
            if args.resize_height and args.resize_width:
                transform_suffix.append(f"rs{args.resize_height}x{args.resize_width}")
            cache_suffix = ("_" + "_".join(transform_suffix)) if transform_suffix else ""

            pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not accelerator.is_local_main_process)
            for batch in pbar:
                frames = batch["frames"].to(device)  # (B, T, C, H, W)
                prompts = batch["prompts"]
                cache_ids = batch.get("cache_ids")
                
                batch_size = frames.shape[0]

                if args.frame_subsample > 1:
                    frames = frames[:, ::args.frame_subsample]

                if args.resize_height and args.resize_width:
                    bsz, frames_t, channels, height, width = frames.shape
                    frames = frames.view(bsz * frames_t, channels, height, width)
                    frames = F.interpolate(
                        frames,
                        size=(args.resize_height, args.resize_width),
                        mode="bilinear",
                        align_corners=False,
                    )
                    frames = frames.view(bsz, frames_t, channels, args.resize_height, args.resize_width)

                # Encode frames to latents with optional caching
                use_cache = latent_cache_dir is not None and cache_ids is not None
                if use_cache:
                    latents_list = [None] * batch_size
                    cache_paths = []
                    missing_indices = []

                    latent_channels = getattr(vae.config, "latent_channels", 16)

                    for idx in range(batch_size):
                        cache_id = cache_ids[idx]
                        cache_path = latent_cache_dir / f"{cache_id}{cache_suffix}.pt" if cache_id else None
                        cache_paths.append(cache_path)
                        if cache_path and cache_path.exists():
                            cached_latent = torch.load(cache_path, map_location="cpu")
                            cached_latent = cached_latent.to(device=device, dtype=dtype)
                            if cached_latent.ndim == 4 and cached_latent.shape[0] != latent_channels and cached_latent.shape[1] == latent_channels:
                                cached_latent = cached_latent.permute(1, 0, 2, 3)
                            latents_list[idx] = cached_latent
                        else:
                            missing_indices.append(idx)

                    if missing_indices:
                        frames_subset = frames[missing_indices].permute(0, 2, 1, 3, 4)
                        with torch.no_grad():
                            latents_subset = vae.encode(frames_subset.to(torch.float32)).latent_dist.sample()
                            latents_subset = latents_subset * scaling_factor
                            latents_subset = latents_subset.to(dtype)

                        for offset, idx in enumerate(missing_indices):
                            lat = latents_subset[offset]
                            latents_list[idx] = lat
                            cache_path = cache_paths[idx]
                            if cache_path and not cache_path.exists():
                                torch.save(lat.detach().to("cpu", dtype=torch.float16), cache_path)

                    latents = torch.stack(latents_list, dim=0)
                else:
                    frames_vae = frames.permute(0, 2, 1, 3, 4)
                    with torch.no_grad():
                        latents = vae.encode(frames_vae.to(torch.float32)).latent_dist.sample()
                        latents = latents * scaling_factor
                        latents = latents.to(dtype)

                with torch.no_grad():
                    prompt_embeds, _ = pipe.encode_prompt(
                        prompt=prompts,
                        device=device,
                        num_videos_per_prompt=1,
                        do_classifier_free_guidance=False
                    )
                
                step_completed = True

                with accelerator.accumulate(pipe.transformer):
                    # Sample noise
                    noise = torch.randn_like(latents)

                    # Sample random timesteps for flow matching
                    # Flow matching uses timesteps in range [0, 1] representing interpolation
                    u = torch.rand(batch_size, device=device)

                    # Scale to scheduler's timestep range
                    indices = (u * noise_scheduler.config.num_train_timesteps).long()
                    indices = torch.clamp(indices, 0, noise_scheduler.config.num_train_timesteps - 1)
                    indices_cpu = indices.cpu()
                    timesteps = noise_scheduler.timesteps[indices_cpu].to(device)

                    # Flow matching: use scheduler sigmas to blend data and noise (velocity target)
                    sigmas = get_sigmas(indices_cpu, n_dim=latents.ndim, dtype=latents.dtype)
                    noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

                    # Forward pass through transformer
                    model_pred = pipe.transformer(
                        hidden_states=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False
                    )[0]

                    # Compute loss
                    # For flow matching, the target is the velocity field: noise - data
                    target = noise - latents
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    loss_value = loss.item()

                    accelerator.backward(loss)

                    if accelerator.sync_gradients and args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(pipe.transformer.parameters(), args.max_grad_norm)

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    step_completed = accelerator.sync_gradients

                # Update metrics
                if step_completed:
                    global_step += 1
                    epoch_loss += loss_value
                    num_batches += 1

                    # Update progress bar
                    if accelerator.is_local_main_process:
                        pbar.set_postfix({
                            'loss': f'{loss_value:.4f}',
                            'avg_loss': f'{epoch_loss/num_batches:.4f}',
                            'step': global_step
                        })

                    # Log to file
                    if accelerator.is_main_process:
                        log_entry = {
                            'step': global_step,
                            'epoch': epoch + 1,
                            'loss': loss_value,
                            'timestamp': datetime.now().isoformat()
                        }
                        with open(log_file, 'a') as f:
                            f.write(json.dumps(log_entry) + '\n')

                    # Save checkpoint
                    if global_step % args.save_every == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            optimizer_state = optimizer.state_dict()
                            checkpoint_path = output_dir / f"lora_step{global_step}.pt"
                            accelerator.print(f"\nSaving checkpoint to {checkpoint_path}")

                            unwrapped = accelerator.unwrap_model(pipe.transformer)
                            lora_state_dict = {
                                k: v for k, v in unwrapped.state_dict().items()
                                if 'lora' in k
                            }

                            checkpoint = {
                                'lora_state_dict': lora_state_dict,
                                'step': global_step,
                                'epoch': epoch + 1,
                                'optimizer_state_dict': optimizer_state,
                                'loss': loss_value,
                                'args': vars(args)
                            }
                            accelerator.save(checkpoint, checkpoint_path)
                        accelerator.wait_for_everyone()

                    # Check max steps
                    if global_step >= args.max_steps:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            optimizer_state = optimizer.state_dict()
                            accelerator.print(f"\nReached max steps ({args.max_steps}), finishing training")

                            # Save final checkpoint
                            final_path = output_dir / f"lora_final_step{global_step}.pt"
                            unwrapped = accelerator.unwrap_model(pipe.transformer)
                            lora_state_dict = {
                                k: v for k, v in unwrapped.state_dict().items()
                                if 'lora' in k
                            }
                            checkpoint = {
                                'lora_state_dict': lora_state_dict,
                                'step': global_step,
                                'epoch': epoch + 1,
                                'optimizer_state_dict': optimizer_state,
                                'loss': loss_value,
                                'args': vars(args)
                            }
                            accelerator.save(checkpoint, final_path)
                            accelerator.print(f"Final checkpoint saved to {final_path}")
                        accelerator.wait_for_everyone()
                        return
            
            # End of epoch summary
            loss_tensor = torch.tensor([epoch_loss], device=device)
            batch_tensor = torch.tensor([num_batches], device=device)
            gathered_loss = accelerator.gather(loss_tensor)
            gathered_batches = accelerator.gather(batch_tensor)
            if accelerator.is_main_process:
                total_batches = gathered_batches.sum().item()
                avg_epoch_loss = gathered_loss.sum().item() / max(total_batches, 1)
                accelerator.print(f"\nEpoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}\n")

    finally:
        # Clean up distributed resources
        try:
            accelerator.end_training()
        except Exception:
            pass
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass

    accelerator.wait_for_everyone()


def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapter for Wan 2.1 T2V model")
    
    # Model and data
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                        help="HuggingFace model ID")
    parser.add_argument("--manifest", type=str, default="data/processed/train.jsonl",
                        help="Path to JSONL manifest file")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Training batch size (1 for 81-frame videos on 80GB GPU)")
    parser.add_argument("--latent_cache_dir", type=str, default=None,
                        help="Optional directory to cache VAE latents for faster subsequent runs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for optimizer")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs (will stop early if max_steps reached)")
    parser.add_argument("--max_steps", type=int, default=100,
                        help="Maximum training steps")
    parser.add_argument("--save_every", type=int, default=50,
                        help="Save checkpoint every N steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping (0 to disable)")
    parser.add_argument("--frame_subsample", type=int, default=1,
                        help="Take every Nth frame from each clip (>=1)")
    parser.add_argument("--resize_width", type=int, default=None,
                        help="Optional resize width for frames")
    parser.add_argument("--resize_height", type=int, default=None,
                        help="Optional resize height for frames")
    
    # LoRA configuration
    parser.add_argument("--lora_rank", type=int, default=32,
                        help="LoRA rank (8-64 typical)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (often equals rank)")
    
    # Data loading
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of dataloader workers")
    # Simple augmentations to improve robustness during LoRA training
    parser.add_argument("--random_flip", action='store_true',
                        help="Enable random horizontal flip augmentation (p=0.5)")
    parser.add_argument("--color_jitter", type=float, default=0.0,
                        help="Simple brightness jitter multiplier magnitude (0.0-1.0)")
    
    args = parser.parse_args()

    if args.frame_subsample < 1:
        parser.error("--frame_subsample must be >= 1")
    if (args.resize_width is None) != (args.resize_height is None):
        parser.error("Provide both --resize_width and --resize_height together")

    train(args)


if __name__ == "__main__":
    main()
