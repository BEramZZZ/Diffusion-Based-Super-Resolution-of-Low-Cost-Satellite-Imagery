"""
finetune/data.py
=================

Builds a domain-adaptation TRAINING set from RESISC45, for fine-tuning both
models toward satellite/aerial imagery specifically.

Reuses the same SHOWCASE_CATEGORIES and the same bicubic-degradation pair
construction as the zero-shot benchmark (`sr_benchmark.datasets`), for
consistency -- but pulls DIFFERENT images than the evaluation set, so
fine-tuned models are never evaluated on images they trained on.

How the split is kept disjoint
-------------------------------
`load_resisc45_showcase` (used for eval) always takes the FIRST
`num_per_category` examples per category, in dataset iteration order. This
loader takes the NEXT `num_train_per_category` examples after that offset --
i.e. it skips exactly as many images as the eval script requested, then
starts collecting for training. As long as you fine-tune with the same (or a
smaller) `eval_num_per_category` as you used for `run_zero_shot_benchmark.py`,
there is zero overlap by construction.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from sr_benchmark.datasets import SHOWCASE_CATEGORIES, ImagePair, build_pairs


def load_resisc45_training_split(
    categories: Optional[List[str]] = None,
    num_train_per_category: int = 40,
    eval_num_per_category: int = 2,
) -> Dict[str, Image.Image]:
    """Load a training image set from RESISC45, disjoint from the eval
    showcase set (see module docstring for how disjointness is guaranteed).
    """
    from datasets import load_dataset

    categories = categories or SHOWCASE_CATEGORIES

    print("[finetune/data] Loading RESISC45 training split from Hugging Face...")
    resisc = load_dataset("tanganke/resisc45", split="train")
    class_names = resisc.features["label"].names

    missing = [c for c in categories if c not in class_names]
    if missing:
        raise ValueError(f"Categories not found in RESISC45: {missing}")

    skip_counts = {c: 0 for c in categories}       # images seen so far, per category
    take_counts = {c: 0 for c in categories}       # images collected for training, per category
    images: Dict[str, Image.Image] = {}

    for example in resisc:
        label_name = class_names[example["label"]]
        if label_name not in categories:
            continue
        if skip_counts[label_name] < eval_num_per_category:
            # This example belongs to the eval set -- skip it for training.
            skip_counts[label_name] += 1
            continue
        if take_counts[label_name] >= num_train_per_category:
            continue
        name = f"{label_name}_train_{take_counts[label_name]:03d}"
        images[name] = example["image"].convert("RGB")
        take_counts[label_name] += 1
        if all(c >= num_train_per_category for c in take_counts.values()):
            break

    found_total = sum(take_counts.values())
    expected_total = len(categories) * num_train_per_category
    if found_total < expected_total:
        print(f"[finetune/data][warn] Only found {found_total}/{expected_total} training images "
              f"(some categories may be smaller than requested in this split).")

    print(f"[finetune/data] Loaded {len(images)} training images across {len(categories)} categories.")
    return images


class SRPairDataset(Dataset):
    """Torch Dataset yielding (lr_tensor, hr_tensor) pairs in [-1, 1] range,
    channel-first, ready for either model's training loop.

    Both tensors are float32 in [-1, 1] -- the range both Real-ESRGAN
    training (rescaled internally) and the diffusers VAE/UNet expect.
    """

    def __init__(self, pairs: Dict[str, ImagePair]):
        self.items = list(pairs.values())

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pair = self.items[idx]
        lr = _to_tensor(pair.lr_image)
        hr = _to_tensor(pair.hr_image)
        return lr, hr


def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0   # [0,255] -> [-1,1]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def build_training_dataset(
    categories: Optional[List[str]] = None,
    num_train_per_category: int = 40,
    eval_num_per_category: int = 2,
    hr_size: int = 512,
    scale: int = 4,
) -> SRPairDataset:
    """One-call convenience: load disjoint training images, build HR/LR
    pairs, wrap in a Dataset."""
    hr_images = load_resisc45_training_split(
        categories=categories,
        num_train_per_category=num_train_per_category,
        eval_num_per_category=eval_num_per_category,
    )
    pairs = build_pairs(hr_images, hr_size=hr_size, scale=scale)
    return SRPairDataset(pairs)
