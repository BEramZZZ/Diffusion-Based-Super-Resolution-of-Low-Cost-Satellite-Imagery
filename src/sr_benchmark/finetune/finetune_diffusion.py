"""
finetune/finetune_diffusion.py
================================

Applies the 4 fine-tuning strategies to the SD x4 Upscaler's UNet2DConditionModel,
and trains with the pipeline's OWN noise-prediction objective -- reconstructed
here by tracing `StableDiffusionUpscalePipeline.__call__` directly (see
`diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_upscale.py`),
because this pipeline is genuinely non-standard and a generic SD training
loop will silently train the wrong thing. This is the exact failure mode the
project README already diagnosed for the earlier LoRA attempt.

What's actually non-standard here (traced from the pipeline source)
----------------------------------------------------------------------
At inference, the pipeline does NOT encode the low-res image through the VAE
at all. Instead:

    1. The LR image is preprocessed to [-1, 1] pixel space, shape (B,3,h,w)
       where (h, w) is the LR image's OWN pixel resolution (e.g. 128x128 --
       NOT divided by 8, NOT upscaled).
    2. Gaussian noise is added to that LR image itself, at a `noise_level`
       timestep (sampled via a SEPARATE scheduler, `low_res_scheduler`) --
       this is a noise-augmentation trick, not the main denoising process.
    3. The main denoising latents start as pure noise at that SAME (h, w)
       spatial size (see `prepare_latents`, called with `height, width =
       image.shape[2:]` -- i.e. the LR pixel size, not divided by 8 the way
       ordinary SD latents are).
    4. Each denoising step feeds the UNet `cat([noisy_latents, noised_lr_image],
       dim=1)` (4 + 3 = 7 input channels), plus `class_labels=noise_level`
       (the UNet has a `class_embed_type="timestep"` conditioning slot for
       exactly this).
    5. Only at the very end is `vae.decode(final_latents)` called -- and
       THIS checkpoint's VAE decoder is the component that does the actual
       4x pixel upscale (a non-standard decoder for this checkpoint, per the
       project README).

For TRAINING, there is no LR-only forward pass to imitate -- we need a
"clean latent" derived from the HR ground truth to noise and supervise
against, and the pipeline's `__call__` never encodes anything (inference has
no ground truth to encode). The only self-consistent assumption, given step
3 above (latents are spatially sized to match the LR image, not the HR
image), is that this checkpoint's `vae.encode` downsamples by the SR scale
factor (4x) rather than the usual SD 8x -- so `vae.encode(hr_image)` on a
512x512 image yields a (4, 128, 128) latent, exactly matching the LR image's
own spatial size. This is asserted at runtime below (`_assert_shapes`) so a
wrong assumption fails loudly with a clear message instead of silently
training garbage -- verify against the actual downloaded checkpoint's VAE
config before trusting the assumption.
"""

from __future__ import annotations

import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .common import freeze_all, summarize_trainable, unfreeze_all, unfreeze_by_name
from .data import SRPairDataset

try:
    from peft import LoraConfig
except ImportError:  # only needed for the `lora` strategy
    LoraConfig = None


UNET_LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]  # standard SD attention LoRA targets


def apply_strategy(unet: nn.Module, strategy: str, lora_rank: int = 8) -> nn.Module:
    """Configure `unet` (a UNet2DConditionModel) in place for the given
    strategy. Call BEFORE building the optimizer."""

    if strategy == "full":
        unfreeze_all(unet)

    elif strategy == "head_only":
        freeze_all(unet)
        # conv_out is the literal final projection back to the 4 latent
        # channels; conv_norm_out is its preceding normalization -- both are
        # "the output layer" in any meaningful sense.
        unfreeze_by_name(unet, ["conv_out", "conv_norm_out"])

    elif strategy == "partial":
        freeze_all(unet)
        # "Later, closer-to-output" for a U-Net = the up_blocks (decoder
        # path) + the final output conv. down_blocks/mid_block (encoder +
        # bottleneck, closest to the input) stay frozen.
        unfreeze_by_name(unet, ["up_blocks", "conv_norm_out", "conv_out"])

    elif strategy == "lora":
        if LoraConfig is None:
            raise ImportError("The 'lora' strategy requires `peft` -- pip install peft (see requirements.txt).")
        freeze_all(unet)
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            target_modules=UNET_LORA_TARGET_MODULES,
        )
        unet.add_adapter(config)
        # add_adapter already leaves only the injected LoRA params trainable.

    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Must be one of: full, head_only, partial, lora.")

    print(summarize_trainable(unet, label=f"UNet [{strategy}]"))
    return unet


def _assert_shapes(latents: torch.Tensor, lr_image: torch.Tensor) -> None:
    if latents.shape[-2:] != lr_image.shape[-2:]:
        raise RuntimeError(
            f"vae.encode(hr_image) produced spatial size {tuple(latents.shape[-2:])}, "
            f"expected it to match the LR image's own size {tuple(lr_image.shape[-2:])} "
            f"(this pipeline denoises at LR-pixel spatial scale -- see module docstring). "
            f"This checkpoint's VAE encoder downsamples by a different factor than assumed "
            f"here (4x) -- inspect `pipe.vae.config` and adjust hr_size/scale in the dataset, "
            f"or resize the HR image before encoding, so the two match."
        )


def finetune_diffusion_upscaler(
    upscaler,
    strategy: str,
    train_dataset: SRPairDataset,
    epochs: int = 5,
    batch_size: int = 2,
    lr: float = 1e-5,
    out_dir: str = "outputs/finetune_diffusion",
    lora_rank: int = 8,
    max_noise_level: int = 350,
) -> str:
    """Fine-tune the SD x4 upscaler's UNet using the given strategy, with
    the standard epsilon-prediction diffusion loss, reconstructed to match
    this pipeline's actual (non-standard) conditioning setup -- see module
    docstring. `upscaler` is a `DiffusionUpscaler` (from
    `models.load_diffusion_upscaler`).
    """
    os.makedirs(out_dir, exist_ok=True)
    pipe = upscaler.pipe
    device = pipe.unet.device
    dtype = pipe.unet.dtype

    unet = apply_strategy(pipe.unet, strategy, lora_rank=lora_rank)
    unet.train()
    vae = pipe.vae
    vae.requires_grad_(False)
    vae.eval()

    # Fixed text conditioning -- reuse the same prompt the eval benchmark
    # uses, so fine-tuning optimizes the model under the exact conditioning
    # it will be evaluated with. Encoded once and reused for every batch.
    with torch.no_grad():
        text_inputs = pipe.tokenizer(
            [upscaler.prompt], padding="max_length",
            max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt",
        )
        prompt_embeds = pipe.text_encoder(text_inputs.input_ids.to(device))[0]

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError(f"Strategy '{strategy}' left zero trainable parameters -- nothing to optimize.")
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    num_train_timesteps = pipe.scheduler.config.num_train_timesteps

    final_path = os.path.join(out_dir, f"{strategy}_final")
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for lr_batch, hr_batch in loader:
            lr_batch = lr_batch.to(device=device, dtype=dtype)
            hr_batch = hr_batch.to(device=device, dtype=dtype)
            bsz = lr_batch.shape[0]

            optimizer.zero_grad()

            # 1. Encode the HR ground truth to the "clean" latent target.
            #    (See module docstring for why this encode step is the key
            #    assumption -- verify with `_assert_shapes` below.)
            with torch.no_grad():
                clean_latents = vae.encode(hr_batch).latent_dist.sample()
                clean_latents = clean_latents * vae.config.scaling_factor
            _assert_shapes(clean_latents, lr_batch)

            # 2. Noise-augment the LR conditioning image (same mechanism the
            #    pipeline uses at inference via `low_res_scheduler`).
            noise_level = torch.randint(0, max_noise_level, (bsz,), device=device, dtype=torch.long)
            img_noise = torch.randn_like(lr_batch)
            noised_lr = pipe.low_res_scheduler.add_noise(lr_batch, img_noise, noise_level)

            # 3. Standard denoising-diffusion noise-prediction objective on
            #    the HR latent, conditioned on the noised LR image + prompt.
            noise = torch.randn_like(clean_latents)
            timesteps = torch.randint(0, num_train_timesteps, (bsz,), device=device, dtype=torch.long)
            noisy_latents = pipe.scheduler.add_noise(clean_latents, noise, timesteps)

            unet_input = torch.cat([noisy_latents, noised_lr], dim=1)  # 4 + 3 = 7 channels
            batch_prompt_embeds = prompt_embeds.expand(bsz, -1, -1)

            noise_pred = unet(
                unet_input, timesteps,
                encoder_hidden_states=batch_prompt_embeds,
                class_labels=noise_level,
            ).sample

            loss = F.mse_loss(noise_pred.float(), noise.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / max(len(loader), 1)
        print(f"[finetune_diffusion][{strategy}] epoch {epoch}/{epochs}  MSE loss: {avg_loss:.5f}")
        unet.save_pretrained(os.path.join(out_dir, f"{strategy}_epoch{epoch:02d}"))

    unet.save_pretrained(final_path)
    print(f"[finetune_diffusion][{strategy}] saved final checkpoint to {final_path}")
    return final_path
