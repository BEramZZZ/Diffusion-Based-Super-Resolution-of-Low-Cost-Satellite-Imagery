"""
finetune/load_checkpoint.py
=============================
 
Loads a fine-tuned checkpoint from `run_finetune.py` back into a ready-to-run
model, for each (model, strategy) combination. This is deliberately separate
from the training code because loading needs to reconstruct the exact same
architecture the checkpoint was saved from before weights can be applied --
get that wrong and you silently evaluate something other than what you
trained (see the `lora` diffusion case below, which had exactly this bug
until it was caught before any real evaluation ran on it).
 
Real-ESRGAN: all 4 strategies save a plain `state_dict()` (`.pth`). `full` /
`head_only` / `partial` never change the module tree, so loading is trivial.
`lora` wraps some conv layers in `LoRAConv2d` (see finetune/common.py) --
the SAME wrapping must be re-applied via `apply_strategy` BEFORE
`load_state_dict`, or the saved keys (`...lora_down.weight` etc.) won't
match a freshly-loaded base model's keys.
 
Diffusion upscaler: `full` / `head_only` / `partial` save the whole U-Net
via `save_pretrained` (no architecture change, so `from_pretrained` reloads
it directly). `lora` saves ONLY the adapter deltas via `save_lora_adapter`
(the peft-aware method) -- reloading needs a freshly-loaded BASE U-Net with
the same LoRA config re-applied via `add_adapter`, then `load_lora_adapter`
to load just the deltas onto it.
"""
 
from __future__ import annotations
 
import os
 
import torch
 
from . import finetune_esrgan, finetune_diffusion
from ..models import DiffusionUpscaler, load_diffusion_upscaler, load_realesrgan
 
 
def load_finetuned_esrgan(strategy: str, checkpoint_path: str, lora_rank: int = 8):
    """Returns a `RealESRGANer` (same interface as `models.load_realesrgan`)
    with the fine-tuned weights loaded, ready for `models.run_realesrgan`."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path} -- run "
            f"`python run_finetune.py --model esrgan --strategy {strategy}` first."
        )
 
    upsampler = load_realesrgan()
 
    # Force fp32 to match how it was trained -- also disable the RealESRGANer
    # wrapper's own half-precision casting (`.enhance()` casts the input to
    # fp16 internally if this flag is left True, which would immediately hit
    # the same dtype mismatch fixed earlier during training).
    upsampler.model = upsampler.model.float()
    upsampler.half = False
 
    # Rebuild the exact architecture the checkpoint was saved from (a no-op
    # for full/head_only/partial; wraps the relevant convs in LoRAConv2d for
    # `lora`, matching the wrapping present when the state_dict was saved).
    finetune_esrgan.apply_strategy(upsampler.model, strategy, lora_rank=lora_rank)
 
    device = next(upsampler.model.parameters()).device
    state_dict = torch.load(checkpoint_path, map_location=device)
    # strict=True is deliberate: a silent key mismatch here means evaluating
    # a model that never actually loaded the fine-tuned weights.
    upsampler.model.load_state_dict(state_dict, strict=True)
    upsampler.model.eval()
    return upsampler
 
 
def load_finetuned_diffusion(strategy: str, checkpoint_dir: str, lora_rank: int = 8) -> DiffusionUpscaler:
    """Returns a `DiffusionUpscaler` (same interface as
    `models.load_diffusion_upscaler`) with the fine-tuned weights loaded,
    ready for `models.run_diffusion_upscaler`."""
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_dir} -- run "
            f"`python run_finetune.py --model diffusion --strategy {strategy}` first."
        )
 
    upscaler = load_diffusion_upscaler()
    pipe = upscaler.pipe
 
    # Same fp32 forcing as training, for the same reason (raw fp16 training/
    # inference on adapted weights without a scaler risks silent precision
    # issues, and mixing an fp32-saved checkpoint into an fp16 module would
    # hit the original dtype-mismatch bug again).
    pipe.vae = pipe.vae.float()
    pipe.text_encoder = pipe.text_encoder.float()
 
    if strategy == "lora":
        adapter_file_present = any(
            f.startswith("adapter_") or f.endswith(".safetensors")
            for f in os.listdir(checkpoint_dir)
        )
        if not adapter_file_present:
            raise RuntimeError(
                f"{checkpoint_dir} doesn't look like a `save_lora_adapter` checkpoint "
                f"(no adapter_config/*.safetensors found). If this was trained before "
                f"the save-call fix, re-run "
                f"`python run_finetune.py --model diffusion --strategy lora` "
                f"with the current finetune_diffusion.py before evaluating it."
            )
        pipe.unet = pipe.unet.float()
        finetune_diffusion.apply_strategy(pipe.unet, "lora", lora_rank=lora_rank)
        pipe.unet.load_lora_adapter(checkpoint_dir)
    else:
        from diffusers import UNet2DConditionModel
 
        finetuned_unet = UNet2DConditionModel.from_pretrained(checkpoint_dir)
        pipe.unet = finetuned_unet.to(device=pipe.vae.device, dtype=torch.float32)
 
    pipe.unet.eval()
    return upscaler