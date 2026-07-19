#!/usr/bin/env python3
"""
make_hallucination_closeup.py
===============================

Run this AFTER `run_zero_shot_benchmark.py` has completed. Inspect
`outputs/comparison_grid.png` first, pick the example and crop region that
most clearly shows the difference between Real-ESRGAN's smoothing and the
diffusion model's detail hallucination, then generate a zoomed-in three-panel
figure for your report.

Usage
-----
    python make_hallucination_closeup.py --name dense_residential_00 \\
        --crop 150 150 350 350

`--crop` is (left, top, right, bottom) in pixel coordinates, within the
512x512 display size used by the benchmark.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from PIL import Image

from sr_benchmark.datasets import ImagePair  # noqa: E402
from sr_benchmark.visualize import save_hallucination_closeup  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Example name, e.g. 'dense_residential_00'")
    parser.add_argument(
        "--crop",
        nargs=4,
        type=int,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        required=True,
        help="Crop box in pixel coordinates (512x512 display size).",
    )
    parser.add_argument("--out-dir", default="outputs", help="Directory used by the main benchmark script.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    name = args.name
    category = name.rsplit("_", 1)[0]

    hr_path = os.path.join(args.out_dir, "pairs", f"{name}_hr.png")
    esrgan_path = os.path.join(args.out_dir, "esrgan_outputs", f"{name}.png")
    diffusion_path = os.path.join(args.out_dir, "diffusion_outputs", f"{name}.png")

    for p in (hr_path, esrgan_path, diffusion_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {p} -- run run_zero_shot_benchmark.py first.")

    pair = ImagePair(
        name=name,
        category=category,
        hr_image=Image.open(hr_path).convert("RGB"),
        lr_image=None,  # not needed for the close-up
    )
    esrgan_output = Image.open(esrgan_path).convert("RGB")
    diffusion_output = Image.open(diffusion_path).convert("RGB")

    out_path = os.path.join(args.out_dir, f"hallucination_closeup_{name}.png")
    save_hallucination_closeup(
        pair, esrgan_output, diffusion_output, tuple(args.crop), out_path
    )


if __name__ == "__main__":
    main()
