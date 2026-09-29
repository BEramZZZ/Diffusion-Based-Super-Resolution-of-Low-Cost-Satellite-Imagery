#!/usr/bin/env python3
"""
run_finetune_comparison.py
============================

Runs every model/strategy combination found on disk -- zero-shot baselines
plus all fine-tuned checkpoints -- against the same set of test images, and
saves EACH output as its own independent PNG (not a combined grid), so you
can open, zoom into, or share any single result on its own.

For every selected test image, this produces one folder containing:

    outputs/finetune_eval/<pair_name>/
        lr_input.png
        ground_truth.png
        esrgan_zero_shot.png
        esrgan_full.png
        esrgan_head_only.png
        esrgan_partial.png
        esrgan_lora.png
        diffusion_zero_shot.png
        diffusion_full.png
        diffusion_head_only.png
        diffusion_partial.png
        diffusion_lora.png

(only the model/strategy combinations you ran and that have a checkpoint on
disk are produced -- others are skipped with a log line, same as before.)

Usage
-----
    # default: 1 image per category, every checkpoint found on disk
    python run_finetune_comparison.py

    # only specific strategies/models
    python run_finetune_comparison.py --strategies full lora --models esrgan

    # only specific images by name (skip the "1 per category" default)
    python run_finetune_comparison.py --image-names airplane_00 runway_00

Checkpoint discovery
---------------------
`run_finetune.py` saves each run under a timestamped `run_id` subfolder
(`<finetune_dir>/<model>/<strategy>/<run_id>/<strategy>_final`) so repeated
runs never overwrite each other. `_checkpoint_path` below checks both that
layout AND the older flat layout (`<finetune_dir>/<model>/<strategy>/
<strategy>_final`, from before the run_id change) -- so checkpoints trained
before and after that change are both picked up without moving anything on
disk. If a strategy has more than one timestamped run, the most recently
modified one is used; pass a narrower --finetune-dir if you need an older
run specifically.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from sr_benchmark.datasets import SHOWCASE_CATEGORIES, build_pairs, load_resisc45_showcase  # noqa: E402
from sr_benchmark.finetune.common import STRATEGIES  # noqa: E402
from sr_benchmark.finetune.load_checkpoint import load_finetuned_diffusion, load_finetuned_esrgan  # noqa: E402
from sr_benchmark.models import load_diffusion_upscaler, load_realesrgan, run_diffusion_upscaler, run_realesrgan  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=["esrgan", "diffusion"], choices=["esrgan", "diffusion"])
    parser.add_argument("--strategies", nargs="*", default=list(STRATEGIES), choices=list(STRATEGIES))
    parser.add_argument("--no-zero-shot", action="store_true")
    parser.add_argument("--num-per-category", type=int, default=1,
                         help="How many images per category to include if --image-names isn't given.")
    parser.add_argument("--image-names", nargs="*", default=None,
                         help="Pick specific pair names instead of sampling per-category.")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--finetune-dir", type=str, default="outputs/finetune")
    parser.add_argument("--out-dir", type=str, default="outputs/finetune_eval")
    return parser.parse_args()


def _free_gpu_memory(*objs) -> None:
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _checkpoint_path(finetune_dir: str, model: str, strategy: str) -> str:
    """Locate a strategy's checkpoint, supporting both layouts:

      - old flat layout:        <finetune_dir>/<model>/<strategy>/<name>
      - new run_id layout:      <finetune_dir>/<model>/<strategy>/<run_id>/<name>

    Checks the flat path first (cheap, common case for pre-run_id
    checkpoints), then falls back to scanning immediate subfolders for the
    new layout, returning the most recently modified match if several runs
    exist for the same strategy. Returns the flat path (even if it doesn't
    exist) when nothing is found, so the caller's existing
    os.path.exists()-based skip logic still works unchanged.
    """
    base = os.path.join(finetune_dir, model, strategy)
    name = f"{strategy}_final.pth" if model == "esrgan" else f"{strategy}_final"

    flat = os.path.join(base, name)
    if os.path.exists(flat):
        return flat

    if os.path.isdir(base):
        candidates = []
        for entry in os.listdir(base):
            candidate = os.path.join(base, entry, name)
            if os.path.exists(candidate):
                candidates.append(candidate)
        if candidates:
            candidates.sort(key=os.path.getmtime)
            return candidates[-1]  # most recent run for this strategy

    return flat


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    categories = args.categories or SHOWCASE_CATEGORIES

    print("[compare] loading eval pairs...")
    hr_images = load_resisc45_showcase(categories=categories, num_per_category=args.num_per_category)
    pairs = build_pairs(hr_images, hr_size=512, scale=4)

    if args.image_names:
        pairs = {k: v for k, v in pairs.items() if k in args.image_names}
        missing = set(args.image_names) - set(pairs.keys())
        if missing:
            print(f"[compare][warn] requested image names not found: {missing}")

    if not pairs:
        print("[compare] No pairs selected -- nothing to do.")
        return
    print(f"[compare] Using {len(pairs)} image(s): {list(pairs.keys())}")

    # Save LR input + ground truth once per image, in that image's own folder.
    for pair_name, pair in pairs.items():
        pair_dir = os.path.join(args.out_dir, pair_name)
        os.makedirs(pair_dir, exist_ok=True)
        pair.lr_image.save(os.path.join(pair_dir, "lr_input.png"))
        pair.hr_image.save(os.path.join(pair_dir, "ground_truth.png"))

    def _run_and_save(col_id: str, run_fn) -> None:
        """Run one model/strategy against every pair and save each result
        as its own PNG under that pair's folder -- no grid, no combining."""
        saved = 0
        for pair_name, pair in pairs.items():
            try:
                out_image: Image.Image = run_fn(pair.lr_image)
            except Exception as e:  # noqa: BLE001
                print(f"[compare][warn] {col_id} failed on {pair_name}: {e}")
                continue
            pair_dir = os.path.join(args.out_dir, pair_name)
            out_path = os.path.join(pair_dir, f"{col_id}.png")
            out_image.save(out_path)
            saved += 1
        print(f"[compare] {col_id}: saved {saved}/{len(pairs)} image(s)")

    # --- Zero-shot baselines -------------------------------------------------
    if not args.no_zero_shot:
        if "esrgan" in args.models:
            print("[compare] Real-ESRGAN zero-shot...")
            m = load_realesrgan()
            _run_and_save("esrgan_zero_shot", lambda img, u=m: run_realesrgan(u, img))
            _free_gpu_memory(m)
        if "diffusion" in args.models:
            print("[compare] Diffusion zero-shot...")
            m = load_diffusion_upscaler()
            _run_and_save("diffusion_zero_shot", lambda img, u=m: run_diffusion_upscaler(u, img))
            _free_gpu_memory(m)

    # --- Fine-tuned variants ---------------------------------------------------
    if "esrgan" in args.models:
        for strategy in args.strategies:
            ckpt = _checkpoint_path(args.finetune_dir, "esrgan", strategy)
            if not os.path.exists(ckpt):
                print(f"[compare][skip] esrgan/{strategy}: no checkpoint at {ckpt}")
                continue
            print(f"[compare] Real-ESRGAN [{strategy}] (checkpoint: {ckpt})...")
            try:
                m = load_finetuned_esrgan(strategy, ckpt, lora_rank=args.lora_rank)
            except Exception as e:  # noqa: BLE001
                print(f"[compare][skip] esrgan/{strategy}: failed to load -- {e}")
                continue
            _run_and_save(f"esrgan_{strategy}", lambda img, u=m: run_realesrgan(u, img))
            _free_gpu_memory(m)

    if "diffusion" in args.models:
        for strategy in args.strategies:
            ckpt = _checkpoint_path(args.finetune_dir, "diffusion", strategy)
            if not os.path.exists(ckpt):
                print(f"[compare][skip] diffusion/{strategy}: no checkpoint at {ckpt}")
                continue
            print(f"[compare] Diffusion [{strategy}] (checkpoint: {ckpt})...")
            try:
                m = load_finetuned_diffusion(strategy, ckpt, lora_rank=args.lora_rank)
            except Exception as e:  # noqa: BLE001
                print(f"[compare][skip] diffusion/{strategy}: failed to load -- {e}")
                continue
            _run_and_save(f"diffusion_{strategy}", lambda img, u=m: run_diffusion_upscaler(u, img))
            _free_gpu_memory(m)

    print(f"\n[compare] Done. Independent images saved under {args.out_dir}/<image_name>/")


if __name__ == "__main__":
    main()