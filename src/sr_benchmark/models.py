"""
models.py
=========

Loading and running the two pretrained super-resolution models under
comparison, in zero-shot inference mode (no fine-tuning):

    1. Real-ESRGAN (x4plus) -- a GAN-based super-resolution model, trained
       with a combination of pixel (L1), perceptual (VGG feature), and
       adversarial losses. A single deterministic forward pass per image.

    2. Stable Diffusion x4 Upscaler -- a text-conditioned latent diffusion
       model, producing output via ~30 iterative denoising steps, each
       conditioned on the low-resolution input image.

See the project README for a full explanation of *why* these two
architectures tend to produce visually different results (GAN smoothing vs.
diffusion detail-hallucination) -- this module only handles the mechanics of
loading and calling them.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .compat import apply_basicsr_compat_shim

REALESRGAN_WEIGHTS_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)
DIFFUSION_MODEL_ID = "stabilityai/stable-diffusion-x4-upscaler"
DEFAULT_PROMPT = "a high resolution aerial satellite photograph, sharp detail, realistic terrain"


def get_device() -> str:
    """Return 'cuda' if a GPU is available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Real-ESRGAN
# ---------------------------------------------------------------------------

def download_realesrgan_weights(weights_dir: str = "weights") -> str:
    """Download the official pretrained Real-ESRGAN x4plus weights if not
    already present locally. Returns the local file path."""
    os.makedirs(weights_dir, exist_ok=True)
    weight_path = os.path.join(weights_dir, "RealESRGAN_x4plus.pth")
    if not os.path.exists(weight_path):
        print("[models] Downloading Real-ESRGAN weights...")
        urllib.request.urlretrieve(REALESRGAN_WEIGHTS_URL, weight_path)
    return weight_path


def load_realesrgan(weights_dir: str = "weights"):
    """Load the pretrained Real-ESRGAN model, ready for inference.

    Returns a `RealESRGANer` instance (from the `realesrgan` package), which
    exposes `.enhance(image_array, outscale=...)`.
    """
    apply_basicsr_compat_shim()  # must happen before importing basicsr/realesrgan

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    weight_path = download_realesrgan_weights(weights_dir)
    device_available = torch.cuda.is_available()

    architecture = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
    )
    upsampler = RealESRGANer(
        scale=4,
        model_path=weight_path,
        model=architecture,
        tile=200,
        tile_pad=10,
        pre_pad=0,
        half=device_available,
    )
    print("[models] Real-ESRGAN loaded.")
    return upsampler


def run_realesrgan(upsampler, lr_image: Image.Image) -> Image.Image:
    """Run Real-ESRGAN 4x super-resolution on a single low-res PIL image."""
    lr_np = np.array(lr_image)
    output_np, _ = upsampler.enhance(lr_np, outscale=4)
    return Image.fromarray(output_np)


# ---------------------------------------------------------------------------
# Stable Diffusion x4 Upscaler
# ---------------------------------------------------------------------------

@dataclass
class DiffusionUpscaler:
    """Thin wrapper bundling the loaded pipeline with its default prompt and
    inference settings, so callers don't need to remember them."""

    pipe: "object"
    prompt: str = DEFAULT_PROMPT
    num_inference_steps: int = 30


def load_diffusion_upscaler(prompt: str = DEFAULT_PROMPT) -> DiffusionUpscaler:
    """Load the pretrained Stable Diffusion x4 upscaler pipeline."""
    from diffusers import StableDiffusionUpscalePipeline

    device_available = torch.cuda.is_available()
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        DIFFUSION_MODEL_ID,
        torch_dtype=torch.float16 if device_available else torch.float32,
        variant="fp16" if device_available else None,
    )
    pipe = pipe.to("cuda" if device_available else "cpu")
    print("[models] Stable Diffusion x4 upscaler loaded.")
    return DiffusionUpscaler(pipe=pipe, prompt=prompt)


def run_diffusion_upscaler(upscaler: DiffusionUpscaler, lr_image: Image.Image) -> Image.Image:
    """Run the diffusion upscaler on a single low-res PIL image."""
    result = upscaler.pipe(
        prompt=upscaler.prompt,
        image=lr_image,
        num_inference_steps=upscaler.num_inference_steps,
    )
    return result.images[0]
