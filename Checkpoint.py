from safetensors.torch import load_file
import torch

# adjust filename if it's .bin instead of .safetensors
sd = load_file("outputs/finetune/diffusion/partial/partial_final/diffusion_pytorch_model.safetensors")

bad = {k: v for k, v in sd.items() if torch.isnan(v).any() or torch.isinf(v).any()}
print(f"{len(bad)} / {len(sd)} tensors contain NaN or Inf")
if bad:
    print(list(bad.keys())[:10])