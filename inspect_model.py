"""Inspect Wan 2.1 transformer architecture to find correct LoRA target modules."""
import torch
from diffusers import WanPipeline

print("Loading Wan 2.1 pipeline...")
pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", torch_dtype=torch.bfloat16)

print("\n" + "="*80)
print("TRANSFORMER MODULE NAMES")
print("="*80)

# Print all named modules
for name, module in pipe.transformer.named_modules():
    if any(key in name for key in ['attn', 'attention', 'proj', 'to_q', 'to_k', 'to_v', 'to_out', 'ff', 'mlp']):
        print(f"{name}: {type(module).__name__}")

print("\n" + "="*80)
print("SAMPLE MODULE STRUCTURE (first few layers)")
print("="*80)

# Show structure more clearly
for i, (name, module) in enumerate(pipe.transformer.named_modules()):
    if i < 50:  # First 50 modules
        print(f"{name}: {type(module).__name__}")
