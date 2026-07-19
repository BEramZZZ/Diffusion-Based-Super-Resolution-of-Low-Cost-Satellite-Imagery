"""
sr_benchmark
============

A benchmark comparing GAN-based (Real-ESRGAN) and diffusion-based
(Stable Diffusion x4 Upscaler) 4x super-resolution on real satellite/aerial
imagery, in zero-shot inference mode.

Modules:
    compat      -- environment compatibility fixes (basicsr/torchvision)
    datasets    -- loading real satellite/aerial imagery and building
                   low-res/high-res evaluation pairs
    models      -- loading and running both super-resolution models
    metrics     -- PSNR, SSIM, and LPIPS computation
    visualize   -- comparison grids and zoomed hallucination-risk crops
"""

__version__ = "1.0.0"
