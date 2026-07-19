#!/usr/bin/env python3
"""
run_zero_shot_benchmark.py
===========================

End-to-end zero-shot comparison of Real-ESRGAN vs. the Stable Diffusion x4
Upscaler on real satellite/aerial imagery (RESISC45), across scene categories
deliberately chosen to surface the difference between GAN-based smoothing
and diffusion-based detail hallucination.

Usage
-----
    python run_zero_shot_benchmark.py [--num-per-category N] [--out-dir DIR]

Outputs (written to --out-dir, default "outputs/"):
    pairs/                          -- saved HR/LR image pairs (PNG)
    esrgan_outputs/                 -- Real-ESRGAN outputs per example (PNG)
    diffusion_outputs/               -- diffusion outputs per example (PNG)
    metrics.csv                      -- per-example PSNR/SSIM/LPIPS for both models
    summary_by_category.csv          -- metrics averaged per scene category
    summary_by_model.csv             -- metrics averaged per model overall
    comparison_grid.png               -- full visual grid (LR / both outputs / GT)

This script is meant to be run once, top to bottom, either locally (with a
CUDA GPU strongly recommended -- the diffusion model is slow on CPU) or in a
Colab notebook via:

    !python run_zero_shot_benchmark.py
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from sr_benchmark.datasets import (  # noqa: E402
    SHOWCASE_CATEGORIES,
    build_pairs,
    load_resisc45_showcase,
    save_pairs,
)
from sr_benchmark.metrics import compute_metrics  # noqa: E402
from sr_benchmark.models import (  # noqa: E402
    load_diffusion_upscaler,
    load_realesrgan,
    run_diffusion_upscaler,
    run_realesrgan,
)
from sr_benchmark.visualize import save_comparison_grid  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-per-category",
        type=int,
        default=2,
        help="How many images per scene category to evaluate (default: 2).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs",
        help="Directory to write all results into (default: outputs/).",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=None,
        help="Override the default showcase categories (space-separated list).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    categories = args.categories or SHOWCASE_CATEGORIES

    pairs_dir = os.path.join(args.out_dir, "pairs")
    esrgan_dir = os.path.join(args.out_dir, "esrgan_outputs")
    diffusion_dir = os.path.join(args.out_dir, "diffusion_outputs")
    for d in (pairs_dir, esrgan_dir, diffusion_dir):
        os.makedirs(d, exist_ok=True)

    # ---------------------------------------------------------------
    # 1. Load data and build evaluation pairs
    # ---------------------------------------------------------------
    hr_images = load_resisc45_showcase(categories=categories, num_per_category=args.num_per_category)
    pairs = build_pairs(hr_images, hr_size=512, scale=4)
    save_pairs(pairs, pairs_dir)

    # ---------------------------------------------------------------
    # 2. Load both models (zero-shot -- no fine-tuning)
    # ---------------------------------------------------------------
    esrgan = load_realesrgan()
    diffusion = load_diffusion_upscaler()

    # ---------------------------------------------------------------
    # 3. Run inference + compute metrics for every example
    # ---------------------------------------------------------------
    esrgan_outputs = {}
    diffusion_outputs = {}
    rows = []

    for name, pair in pairs.items():
        print(f"[benchmark] Processing {name} ({pair.category}) ...")

        esrgan_out = run_realesrgan(esrgan, pair.lr_image)
        esrgan_outputs[name] = esrgan_out
        esrgan_out.save(os.path.join(esrgan_dir, f"{name}.png"))

        diffusion_out = run_diffusion_upscaler(diffusion, pair.lr_image)
        diffusion_outputs[name] = diffusion_out
        diffusion_out.save(os.path.join(diffusion_dir, f"{name}.png"))

        for model_name, output in [("Real-ESRGAN", esrgan_out), ("Diffusion", diffusion_out)]:
            m = compute_metrics(pair.hr_image, output)
            rows.append(
                {
                    "name": name,
                    "category": pair.category,
                    "model": model_name,
                    "psnr": m.psnr,
                    "ssim": m.ssim,
                    "lpips": m.lpips,
                }
            )

    # ---------------------------------------------------------------
    # 4. Save results tables
    # ---------------------------------------------------------------
    df = pd.DataFrame(rows)
    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    df.to_csv(metrics_path, index=False)
    print(f"\n[benchmark] Saved per-example metrics to {metrics_path}")

    by_category = df.groupby(["category", "model"])[["psnr", "ssim", "lpips"]].mean()
    by_category_path = os.path.join(args.out_dir, "summary_by_category.csv")
    by_category.to_csv(by_category_path)
    print(f"[benchmark] Saved per-category summary to {by_category_path}")

    by_model = df.groupby("model")[["psnr", "ssim", "lpips"]].mean()
    by_model_path = os.path.join(args.out_dir, "summary_by_model.csv")
    by_model.to_csv(by_model_path)
    print(f"[benchmark] Saved overall model summary to {by_model_path}")

    print("\n=== Overall averages by model ===")
    print(by_model)
    print("\n=== Averages by category ===")
    print(by_category)

    # ---------------------------------------------------------------
    # 5. Visual comparison grid
    # ---------------------------------------------------------------
    grid_path = os.path.join(args.out_dir, "comparison_grid.png")
    save_comparison_grid(pairs, esrgan_outputs, diffusion_outputs, grid_path)

    print("\n[benchmark] Done. Inspect summary_by_category.csv and comparison_grid.png")
    print("[benchmark] to pick your strongest hallucination-risk example for the report.")


if __name__ == "__main__":
    main()
