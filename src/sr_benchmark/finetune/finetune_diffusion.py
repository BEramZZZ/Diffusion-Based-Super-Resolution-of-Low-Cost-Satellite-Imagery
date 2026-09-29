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


def apply_strategy(unet: nn.Module, strategy: str, lora_rank: int = 8, unfreeze_fraction: float = 0.34) -> nn.Module:
    """Configure `unet` (a UNet2DConditionModel) in place for the given
    strategy. Call BEFORE building the optimizer.

    `unfreeze_fraction` only matters for `partial`: it's the fraction of
    up_blocks (counting from the one closest to the output) left trainable.
    The actual up_block count for this checkpoint is printed at runtime
    (not hardcoded here -- this pipeline's UNet is a modified architecture,
    not the standard SD1.5 one, and the exact block count wasn't confirmed
    against the real downloaded checkpoint from this sandbox).
    """

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
        # Original version unfroze ALL of `up_blocks` -- that's the entire
        # decoder half of the U-Net, not a middle ground (275M+ trainable
        # params on the SD x4 upscaler UNet). Narrowed to match the same
        # spirit as the ESRGAN side: only the LAST fraction of up_blocks
        # (closest to the output), not the whole decoder path.
        num_up_blocks = len(unet.up_blocks)
        num_unfrozen = max(1, round(num_up_blocks * unfreeze_fraction))
        first_unfrozen_idx = num_up_blocks - num_unfrozen
        later_up_blocks = [f"up_blocks.{i}" for i in range(first_unfrozen_idx, num_up_blocks)]
        print(f"[finetune_diffusion] unet has {num_up_blocks} up_blocks total; "
              f"unfreezing the last {num_unfrozen} ({later_up_blocks}) per unfreeze_fraction={unfreeze_fraction}")
        unfreeze_by_name(unet, later_up_blocks + ["conv_norm_out", "conv_out"])

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


def _assert_prediction_type(scheduler) -> str:
    """Read the scheduler's declared prediction_type and fail loudly if it's
    something we don't know how to build a target for, instead of silently
    defaulting to epsilon (the bug this fixes). Returns the validated string.
    """
    prediction_type = getattr(scheduler.config, "prediction_type", None)
    if prediction_type not in ("epsilon", "v_prediction"):
        raise RuntimeError(
            f"pipe.scheduler.config.prediction_type = {prediction_type!r} -- expected "
            f"'epsilon' or 'v_prediction'. Refusing to guess a training target; inspect "
            f"pipe.scheduler.config directly and extend the target-selection logic in "
            f"the training loop if this checkpoint uses a different parameterization "
            f"(e.g. 'sample')."
        )
    print(f"[finetune_diffusion] scheduler.config.prediction_type = {prediction_type!r}")
    return prediction_type


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
    unfreeze_fraction: float = 0.34,
    use_amp: bool = True,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    pipe = upscaler.pipe
    device = pipe.unet.device
    use_amp = use_amp and torch.cuda.is_available()

    prediction_type = _assert_prediction_type(pipe.scheduler)

    pipe.unet = pipe.unet.float()
    pipe.vae = pipe.vae.float()
    dtype = torch.float32

    unet = apply_strategy(pipe.unet, strategy, lora_rank=lora_rank, unfreeze_fraction=unfreeze_fraction)
    unet.train()
    vae = pipe.vae
    vae.requires_grad_(False)
    vae.eval()

    pipe.text_encoder = pipe.text_encoder.float()
    with torch.no_grad():
        text_inputs = pipe.tokenizer(
            [upscaler.prompt], padding="max_length",
            max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt",
        )
        prompt_embeds = pipe.text_encoder(text_inputs.input_ids.to(device))[0].to(dtype)

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError(f"Strategy '{strategy}' left zero trainable parameters -- nothing to optimize.")
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"[finetune_diffusion] mixed precision (autocast+GradScaler): {use_amp}")

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

            # 3. Standard denoising-diffusion objective on the HR latent,
            #    conditioned on the noised LR image + prompt. The UNet
            #    forward pass (the expensive part) runs under autocast for
            #    speed; the loss itself is computed in fp32 for stability.
            #
            #    Target depends on the scheduler's prediction_type -- this
            #    checkpoint is v_prediction, NOT epsilon, so the target must
            #    be built with `get_velocity`, not assumed to be raw noise.
            #    See module docstring ("Prediction-target fix").
            noise = torch.randn_like(clean_latents)
            timesteps = torch.randint(0, num_train_timesteps, (bsz,), device=device, dtype=torch.long)
            noisy_latents = pipe.scheduler.add_noise(clean_latents, noise, timesteps)

            if prediction_type == "v_prediction":
                target = pipe.scheduler.get_velocity(clean_latents, noise, timesteps)
            else:  # "epsilon" -- validated in _assert_prediction_type
                target = noise

            unet_input = torch.cat([noisy_latents, noised_lr], dim=1)  # 4 + 3 = 7 channels
            batch_prompt_embeds = prompt_embeds.expand(bsz, -1, -1)

            with torch.amp.autocast("cuda", enabled=use_amp):
                noise_pred = unet(
                    unet_input, timesteps,
                    encoder_hidden_states=batch_prompt_embeds,
                    class_labels=noise_level,
                ).sample
                loss = F.mse_loss(noise_pred.float(), target.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        avg_loss = running_loss / max(len(loader), 1)
        print(f"[finetune_diffusion][{strategy}] epoch {epoch}/{epochs}  {prediction_type} MSE loss: {avg_loss:.5f}")
        # Per-epoch checkpoints were saved here previously -- removed on
        # purpose. Only the final checkpoint (after all epochs) is kept, to
        # avoid piling up N epoch checkpoints per run that nobody loads.

    _save_unet_checkpoint(unet, strategy, final_path)
    print(f"[finetune_diffusion][{strategy}] saved final checkpoint to {final_path}")
    return final_path


def _save_unet_checkpoint(unet: nn.Module, strategy: str, path: str) -> None:
    if strategy == "lora":
        unet.save_lora_adapter(path)
    else:
        unet.save_pretrained(path)