"""
visualize.py
============

Builds the visual comparisons used in the project report: full comparison
grids (input / model outputs / ground truth, one row per example) and
zoomed-in crops for the hallucination-risk discussion.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from PIL import Image

from .datasets import ImagePair


def save_comparison_grid(
    pairs: Dict[str, ImagePair],
    esrgan_outputs: Dict[str, Image.Image],
    diffusion_outputs: Dict[str, Image.Image],
    out_path: str,
    display_size: int = 512,
) -> None:
    """Save a grid: one row per example, columns = [LR input, Real-ESRGAN
    output, Diffusion output, ground truth]."""
    names = list(pairs.keys())
    n = len(names)

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 4)

    for row, name in enumerate(names):
        pair = pairs[name]

        axes[row, 0].imshow(pair.lr_image.resize((display_size, display_size), Image.NEAREST))
        axes[row, 0].set_title(f"{name}\nLR input")

        axes[row, 1].imshow(esrgan_outputs[name])
        axes[row, 1].set_title("Real-ESRGAN\n(zero-shot)")

        axes[row, 2].imshow(diffusion_outputs[name])
        axes[row, 2].set_title("Diffusion\n(zero-shot)")

        axes[row, 3].imshow(pair.hr_image.resize((display_size, display_size)))
        axes[row, 3].set_title("Ground truth")

        for col in range(4):
            axes[row, col].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[visualize] Saved comparison grid to {out_path}")


def save_hallucination_closeup(
    pair: ImagePair,
    esrgan_output: Image.Image,
    diffusion_output: Image.Image,
    crop_box: Tuple[int, int, int, int],
    out_path: str,
    display_size: int = 512,
) -> None:
    """Save a zoomed-in three-panel comparison (ground truth / Real-ESRGAN /
    diffusion) for a specific crop region -- intended for picking out a clear
    example of GAN smoothing vs. diffusion detail-hallucination for the report.

    Parameters
    ----------
    crop_box:
        (left, top, right, bottom) pixel coordinates within the
        `display_size`-resized image to crop and zoom into.
    """
    hr_resized = pair.hr_image.resize((display_size, display_size))
    esrgan_resized = esrgan_output.resize((display_size, display_size))
    diffusion_resized = diffusion_output.resize((display_size, display_size))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(hr_resized.crop(crop_box))
    axes[0].set_title("Ground truth (zoomed)")
    axes[1].imshow(esrgan_resized.crop(crop_box))
    axes[1].set_title("Real-ESRGAN (zoomed)")
    axes[2].imshow(diffusion_resized.crop(crop_box))
    axes[2].set_title("Diffusion (zoomed)")
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[visualize] Saved hallucination close-up to {out_path}")
