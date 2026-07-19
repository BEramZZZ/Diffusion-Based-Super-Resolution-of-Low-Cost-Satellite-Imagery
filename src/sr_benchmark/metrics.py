"""
metrics.py
==========

Quantitative image-quality metrics used to compare model outputs against
ground truth:

    - PSNR (Peak Signal-to-Noise Ratio): pixel-level fidelity, higher = closer
      to ground truth pixel values. Favors "safe", averaged predictions.
    - SSIM (Structural Similarity): local luminance/contrast/structure
      similarity, higher = better. Still a "distortion" metric, similar
      failure mode to PSNR.
    - LPIPS (Learned Perceptual Image Patch Similarity): compares internal
      VGG feature representations rather than raw pixels, correlating better
      with human perceptual judgment. Lower = more perceptually similar.

Reporting all three together is deliberate: PSNR/SSIM and LPIPS can
disagree, and that disagreement is itself the key finding this benchmark is
built to surface (the "perception-distortion tradeoff" -- Blau & Michaeli,
2018). A model that hallucinates plausible-but-incorrect detail can score
worse on PSNR/SSIM while scoring better (lower) on LPIPS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim

_lpips_model = None  # lazily initialized, since it loads a pretrained VGG network


@dataclass
class MetricResult:
    psnr: float
    ssim: float
    lpips: float


def _get_lpips_model():
    global _lpips_model
    if _lpips_model is None:
        import lpips as lpips_lib

        _lpips_model = lpips_lib.LPIPS(net="vgg")
        if torch.cuda.is_available():
            _lpips_model = _lpips_model.cuda()
    return _lpips_model


def _to_lpips_tensor(pil_img: Image.Image, size: int = 512) -> torch.Tensor:
    """Convert a PIL image into the tensor format LPIPS expects: shape
    (1, 3, H, W), float32, values rescaled from [0, 255] to [-1, 1]."""
    arr = np.array(pil_img.resize((size, size))).astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.cuda() if torch.cuda.is_available() else tensor


def compute_metrics(ground_truth: Image.Image, prediction: Image.Image, size: int = 512) -> MetricResult:
    """Compute PSNR, SSIM, and LPIPS between a ground-truth image and a
    model's prediction. Both images are resized to a common size first, so
    outputs of any resolution can be compared fairly.
    """
    gt_resized = ground_truth.resize((size, size))
    pred_resized = prediction.resize((size, size))

    gt_np = np.array(gt_resized)
    pred_np = np.array(pred_resized)

    psnr_value = _psnr(gt_np, pred_np, data_range=255)
    ssim_value = _ssim(gt_np, pred_np, channel_axis=-1, data_range=255)

    lpips_model = _get_lpips_model()
    with torch.no_grad():
        lpips_value = lpips_model(
            _to_lpips_tensor(gt_resized, size), _to_lpips_tensor(pred_resized, size)
        ).item()

    return MetricResult(psnr=psnr_value, ssim=ssim_value, lpips=lpips_value)
