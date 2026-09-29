#!/usr/bin/env python3
"""
run_finetune.py
================

Fine-tunes either Real-ESRGAN or the SD x4 diffusion upscaler using one of
the 4 standard fine-tuning strategies (full / head_only / partial / lora).

This is deliberately a single-strategy-per-run script (not "run all 8 in one
go") so each run's compute cost and logs stay separate and resumable --
loop over strategies from the shell if you want all of them back to back.

Usage
-----
    # single combination
    python run_finetune.py --model esrgan --strategy lora
    python run_finetune.py --model diffusion --strategy partial

    # all 4 strategies for one model, one after another
    for s in full head_only partial lora; do
        python run_finetune.py --model esrgan --strategy $s
    done

    # the full 8-combination matrix
    for m in esrgan diffusion; do
      for s in full head_only partial lora; do
        python run_finetune.py --model $m --strategy $s
      done
    done

Outputs land in --out-dir/<model>/<strategy>/<run_id>/ (final checkpoint
only -- see finetune_diffusion.py/finetune_esrgan.py). `run_id` defaults to
a UTC timestamp (YYYYmmdd-HHMMSS) so re-running the same --model/--strategy
combo never overwrites a previous run's output; pass --run-id to control it
yourself (e.g. to intentionally overwrite, or to label a run).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from sr_benchmark.finetune.common import STRATEGIES  # noqa: E402
from sr_benchmark.finetune.data import build_training_dataset  # noqa: E402
from sr_benchmark.finetune.finetune_esrgan import finetune_realesrgan  # noqa: E402
from sr_benchmark.finetune.finetune_diffusion import finetune_diffusion_upscaler  # noqa: E402
from sr_benchmark.models import load_diffusion_upscaler, load_realesrgan  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["esrgan", "diffusion"], required=True)
    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None,
                         help="Defaults: 4 for esrgan, 2 for diffusion (diffusion is far more memory-hungry).")
    parser.add_argument("--lr", type=float, default=None,
                         help="Defaults: 1e-4 for esrgan, 1e-5 for diffusion.")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--unfreeze-fraction", type=float, default=0.34,
                         help="For --strategy partial: fraction of later blocks left trainable "
                              "(counting from the output side). Default 0.34 on the diffusion "
                              "UNet's 3 up_blocks means only the last block is trainable.")
    parser.add_argument("--no-amp", action="store_true",
                         help="Disable mixed-precision (autocast+GradScaler) training for the "
                              "diffusion model and fall back to plain fp32 -- slower, only needed "
                              "if you see NaN losses with AMP on.")
    parser.add_argument("--num-train-per-category", type=int, default=40,
                         help="Training images per RESISC45 category (disjoint from eval images).")
    parser.add_argument("--eval-num-per-category", type=int, default=2,
                         help="Must match --num-per-category used in run_zero_shot_benchmark.py, "
                              "so the training split stays disjoint from what you evaluate on.")
    parser.add_argument("--out-dir", type=str, default="outputs/finetune")
    parser.add_argument("--run-id", type=str, default=None,
                         help="Subfolder name under --out-dir/<model>/<strategy>/ for this run's "
                              "output. Defaults to a UTC timestamp (YYYYmmdd-HHMMSS) so repeated "
                              "runs of the same --model/--strategy never overwrite each other's "
                              "checkpoints. Pass your own to label a run or to intentionally reuse "
                              "(and overwrite) a previous run's folder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_size = args.batch_size or (4 if args.model == "esrgan" else 2)
    learning_rate = args.lr or (1e-4 if args.model == "esrgan" else 1e-5)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.out_dir, args.model, args.strategy, run_id)

    if os.path.exists(out_dir):
        raise FileExistsError(
            f"{out_dir} already exists -- refusing to silently overwrite a previous run's "
            f"checkpoint. Pass a different --run-id, or remove that directory yourself first."
        )

    print(f"[run_finetune] model={args.model} strategy={args.strategy} run_id={run_id} "
          f"epochs={args.epochs} batch_size={batch_size} lr={learning_rate}")
    print(f"[run_finetune] output dir: {out_dir}")

    train_dataset = build_training_dataset(
        num_train_per_category=args.num_train_per_category,
        eval_num_per_category=args.eval_num_per_category,
    )
    print(f"[run_finetune] training set size: {len(train_dataset)} HR/LR pairs")

    if args.model == "esrgan":
        upsampler = load_realesrgan()
        finetune_realesrgan(
            upsampler, args.strategy, train_dataset,
            epochs=args.epochs, batch_size=batch_size, lr=learning_rate,
            out_dir=out_dir, lora_rank=args.lora_rank,
        )
    else:
        upscaler = load_diffusion_upscaler()
        finetune_diffusion_upscaler(
            upscaler, args.strategy, train_dataset,
            epochs=args.epochs, batch_size=batch_size, lr=learning_rate,
            out_dir=out_dir, lora_rank=args.lora_rank,
            unfreeze_fraction=args.unfreeze_fraction, use_amp=not args.no_amp,
        )

    print(f"\n[run_finetune] Done. Final checkpoint in {out_dir}/{args.strategy}_final/")
    print("[run_finetune] Next: point run_zero_shot_benchmark.py at this checkpoint "
          "(or write a small loader swap) to compare fine-tuned vs. zero-shot metrics.")


if __name__ == "__main__":
    main()