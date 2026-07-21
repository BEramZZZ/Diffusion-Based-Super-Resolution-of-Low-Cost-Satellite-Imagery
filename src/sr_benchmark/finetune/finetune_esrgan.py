"""
finetune/finetune_esrgan.py
============================

Applies the 4 fine-tuning strategies to Real-ESRGAN's RRDBNet architecture,
and trains with the same L1 pixel loss used in the project's earlier,
documented successful full fine-tune (see project README).

RRDBNet layer layout (basicsr.archs.rrdbnet_arch.RRDBNet), in forward order:

    conv_first  -> body[0..num_block-1] (RRDB blocks) -> conv_body (+residual)
                -> conv_up1 -> conv_up2 (pixel-upsampling convs)
                -> conv_hr -> conv_last

This ordering is what "head" (closest to output) and "early / feature
extraction" (closest to input) mean below.

Strategy -> which layers are trainable
---------------------------------------
    full        all of the above.
    head_only   conv_last ONLY (the literal final output layer).
    partial     freeze conv_first + the FIRST 70% of RRDB blocks in `body`;
                unfreeze the LAST 30% of `body`, plus conv_body, conv_up1,
                conv_up2, conv_hr, conv_last (everything "closer to output").
    lora        freeze everything; inject LoRAConv2d adapters into the same
                "closer to output" layers used by `partial` (conv_body,
                conv_up1, conv_up2, conv_hr, conv_last, and the last few RRDB
                blocks) -- LoRA on literally every conv in all 23 RDB blocks
                would be a huge, non-standard adapter count for a benchmark
                like this, so it targets the layers most responsible for the
                model's final output, same spirit as `partial`.
"""

from __future__ import annotations

import os
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .common import (
    freeze_all,
    inject_conv_lora,
    summarize_trainable,
    unfreeze_all,
    unfreeze_by_name,
)
from .data import SRPairDataset


def _partial_target_names(model: nn.Module, unfreeze_fraction: float = 0.3) -> List[str]:
    """Names of the last `unfreeze_fraction` of RRDB blocks in `body`, used
    by both the `partial` and `lora` strategies to define 'later layers'."""
    num_blocks = len(model.body)
    num_unfrozen = max(1, int(round(num_blocks * unfreeze_fraction)))
    first_unfrozen_idx = num_blocks - num_unfrozen
    return [f"body.{i}" for i in range(first_unfrozen_idx, num_blocks)]


def apply_strategy(model: nn.Module, strategy: str, lora_rank: int = 8) -> nn.Module:
    """Configure `model` (an RRDBNet instance) in place for the given
    strategy and return it. Call this BEFORE building the optimizer, since
    the optimizer must only see `requires_grad=True` parameters."""

    if strategy == "full":
        unfreeze_all(model)

    elif strategy == "head_only":
        freeze_all(model)
        unfreeze_by_name(model, ["conv_last"])

    elif strategy == "partial":
        freeze_all(model)
        later_blocks = _partial_target_names(model, unfreeze_fraction=0.3)
        unfreeze_by_name(model, later_blocks + ["conv_body", "conv_up1", "conv_up2", "conv_hr", "conv_last"])

    elif strategy == "lora":
        # Target the same "closer to output" layers as `partial`, but as
        # LoRA adapters rather than directly-trainable full weights. Only
        # the single-conv layers (not the RDB blocks' internal 5 convs each)
        # are targeted directly here to keep the adapter count reasonable;
        # extending into RDB internals is a straightforward follow-up if you
        # want a bigger-capacity LoRA variant.
        inject_conv_lora(model, ["conv_body", "conv_up1", "conv_up2", "conv_hr", "conv_last"], rank=lora_rank)

    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Must be one of: full, head_only, partial, lora.")

    print(summarize_trainable(model, label=f"RRDBNet [{strategy}]"))
    return model


def finetune_realesrgan(
    upsampler,
    strategy: str,
    train_dataset: SRPairDataset,
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-4,
    out_dir: str = "outputs/finetune_esrgan",
    lora_rank: int = 8,
) -> str:
    """Fine-tune the RRDBNet inside a loaded `RealESRGANer` (from
    `models.load_realesrgan`) using the given strategy. Saves a checkpoint
    per epoch plus a final `{strategy}_final.pth`. Returns the final
    checkpoint path.

    Loss: plain L1 on the 4x-upsampled output vs. ground truth -- matches
    the fine-tuning approach already validated in this project (see README:
    "Real-ESRGAN fine-tuning (plain L1 generator fine-tuning, no adversarial
    retraining) was completed successfully").
    """
    os.makedirs(out_dir, exist_ok=True)
    device = next(upsampler.model.parameters()).device
    model = apply_strategy(upsampler.model, strategy, lora_rank=lora_rank)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError(f"Strategy '{strategy}' left zero trainable parameters -- nothing to optimize.")
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    criterion = nn.L1Loss()

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    final_path = os.path.join(out_dir, f"{strategy}_final.pth")
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for lr_batch, hr_batch in loader:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            optimizer.zero_grad()
            pred = model(lr_batch)
            # RRDBNet's own forward pass already applies the 4x upscale;
            # pred and hr_batch should match spatially given hr_size=512 /
            # scale=4 pairing from build_pairs -- if not, it's a mismatch
            # between the dataset's `scale` and the model's own scale=4.
            if pred.shape[-2:] != hr_batch.shape[-2:]:
                raise RuntimeError(
                    f"Shape mismatch: model output {tuple(pred.shape[-2:])} vs. "
                    f"ground truth {tuple(hr_batch.shape[-2:])}. Check that the "
                    f"dataset's `scale` argument matches this model's fixed 4x scale."
                )
            loss = criterion(pred, hr_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / max(len(loader), 1)
        print(f"[finetune_esrgan][{strategy}] epoch {epoch}/{epochs}  L1 loss: {avg_loss:.5f}")
        torch.save(model.state_dict(), os.path.join(out_dir, f"{strategy}_epoch{epoch:02d}.pth"))

    torch.save(model.state_dict(), final_path)
    print(f"[finetune_esrgan][{strategy}] saved final checkpoint to {final_path}")
    return final_path
