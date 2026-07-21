"""
finetune/common.py
===================

Shared machinery for the four fine-tuning strategies used across BOTH models
(the "standard taxonomy" this project applies to Real-ESRGAN and the SD x4
upscaler alike):

    1. full        -- every weight unfrozen.
    2. head_only    -- everything frozen except the final output layer(s).
    3. partial      -- early/feature layers frozen, later layers (closer to
                        the output) unfrozen. A middle ground between 1 and 2.
    4. lora         -- everything frozen, small trainable low-rank adapters
                        added alongside existing layers.

`freeze_all` / `unfreeze_all` / `unfreeze_by_name` are architecture-agnostic.
The actual "which layers count as head / later-layers" mapping is
architecture-specific and lives in finetune_esrgan.py / finetune_diffusion.py
respectively, since RRDBNet (a feed-forward CNN) and UNet2DConditionModel (a
U-Net with attention) don't share a layer taxonomy.
"""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn

STRATEGIES = ("full", "head_only", "partial", "lora")


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def unfreeze_by_name(model: nn.Module, name_substrings: Iterable[str]) -> None:
    """Unfreeze only parameters whose full name contains one of the given
    substrings (e.g. unfreeze_by_name(model, ["conv_last"]))."""
    substrings = list(name_substrings)
    for name, p in model.named_parameters():
        if any(s in name for s in substrings):
            p.requires_grad = True


def trainable_param_count(model: nn.Module) -> "tuple[int, int]":
    """Returns (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def summarize_trainable(model: nn.Module, label: str = "") -> str:
    trainable, total = trainable_param_count(model)
    pct = 100.0 * trainable / max(total, 1)
    return f"[finetune] {label} trainable params: {trainable:,} / {total:,} ({pct:.3f}%)"


# ---------------------------------------------------------------------------
# Generic Conv2d LoRA (for RRDBNet, which has no attention/linear layers for
# the standard peft LoRA machinery to target -- it's pure convolutions).
# ---------------------------------------------------------------------------

class LoRAConv2d(nn.Module):
    """Wraps a frozen base Conv2d with a small trainable low-rank residual:

        out = base_conv(x) + scale * lora_up(lora_down(x))

    `lora_down` keeps the base conv's kernel size/stride/padding (so spatial
    dims match) and projects in_channels -> rank. `lora_up` is a 1x1 conv
    projecting rank -> out_channels. `lora_up` is zero-initialized so the
    wrapped layer is numerically identical to the base conv at the start of
    training (standard LoRA initialization convention).
    """

    def __init__(self, base_conv: nn.Conv2d, rank: int = 8, alpha: float = 8.0):
        super().__init__()
        self.base_conv = base_conv
        for p in self.base_conv.parameters():
            p.requires_grad = False

        out_ch = base_conv.out_channels
        in_ch = base_conv.in_channels
        self.lora_down = nn.Conv2d(
            in_ch, rank,
            kernel_size=base_conv.kernel_size,
            stride=base_conv.stride,
            padding=base_conv.padding,
            bias=False,
        )
        self.lora_up = nn.Conv2d(rank, out_ch, kernel_size=1, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up.weight)
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_conv(x) + self.scale * self.lora_up(self.lora_down(x))


def inject_conv_lora(model: nn.Module, target_names: Iterable[str], rank: int = 8, alpha: float = 8.0) -> List[str]:
    """Replace named Conv2d submodules (dotted attribute paths, e.g.
    'body.20.rdb3.conv5') with LoRAConv2d wrappers, in place.

    Freezes every OTHER parameter in the model first, so that after this
    call the only trainable parameters are the injected LoRA adapters.
    Returns the list of module paths actually wrapped (for logging / sanity
    checking -- a mismatch here usually means the path names changed
    upstream in basicsr/RRDBNet).
    """
    freeze_all(model)
    wrapped: List[str] = []
    targets = set(target_names)

    for full_name in list(targets):
        parent_path, _, attr = full_name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        submodule = getattr(parent, attr, None)
        if not isinstance(submodule, nn.Conv2d):
            print(f"[finetune][warn] '{full_name}' is not an nn.Conv2d (got {type(submodule)}) -- skipped.")
            continue
        setattr(parent, attr, LoRAConv2d(submodule, rank=rank, alpha=alpha))
        wrapped.append(full_name)

    if not wrapped:
        raise RuntimeError(
            "inject_conv_lora wrapped zero modules -- target_names didn't match any "
            "Conv2d submodule. Print model.named_modules() and fix the target list."
        )
    return wrapped
