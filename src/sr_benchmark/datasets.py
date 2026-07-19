"""
datasets.py
===========

Loads real satellite/aerial imagery for the super-resolution benchmark, and
builds low-resolution/high-resolution evaluation pairs from it.

Data source: NWPU-RESISC45 (via the Hugging Face `datasets` library)
----------------------------------------------------------------------
RESISC45 is a standard, large (31,500 image) remote-sensing benchmark
covering 45 real-world scene categories, hosted reliably on Hugging Face's
dataset hub. We deliberately avoid live satellite-catalog search APIs
(e.g. Microsoft Planetary Computer's STAC search) here, since that endpoint
has a documented, ongoing high failure rate as of mid-2026
(github.com/microsoft/PlanetaryComputer/issues/476) -- unsuitable for a
reproducible benchmark that others should be able to re-run from a GitHub repo.

Evaluation pairs are built by self-downsampling: each real high-resolution
image is treated as ground truth, and its matching low-resolution input is
created by bicubic downsampling that same image by the scale factor. This is
the standard "bicubic degradation" protocol used throughout the
super-resolution literature (e.g. DIV2K, Set5/Set14 benchmarks). It is a
simplification relative to true cross-sensor pairs (e.g. WorldStrat's real
Sentinel-2-vs-commercial-imagery pairs) -- see the project README for the
full discussion of this tradeoff.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from PIL import Image


# Categories deliberately chosen to surface different failure/success modes:
#   - regular, repetitive man-made structure (where hallucinated detail is
#     most visible and most consequential -- e.g. an invented building or road)
#   - fine linear features (roads, runways -- exactly what the project brief
#     flags as unacceptable to hallucinate)
#   - organic, irregular natural texture (forest, chaparral -- where GAN
#     smoothing vs. diffusion detail-invention looks most different)
#   - open water / coastline (low-texture regions, a good smoothing stress test)
SHOWCASE_CATEGORIES: List[str] = [
    "dense_residential",   # repetitive man-made structure
    "sparse_residential",  # repetitive man-made structure, lower density
    "freeway",              # fine linear features
    "runway",                # fine linear features
    "harbor",                 # mixed structure + water
    "forest",                 # organic, irregular natural texture
    "chaparral",              # organic, irregular natural texture
    "river",                  # low-texture, natural
    "beach",                  # low-texture, natural
    "agricultural",           # regular but non-urban texture
]


@dataclass
class ImagePair:
    """A single evaluation example: a high-res ground truth and its
    corresponding self-downsampled low-res input."""

    name: str
    category: str
    hr_image: Image.Image
    lr_image: Image.Image


def load_resisc45_showcase(
    categories: Optional[List[str]] = None,
    num_per_category: int = 2,
    seed: int = 42,
) -> Dict[str, Image.Image]:
    """Load a small, diverse, reproducible set of real RESISC45 images.

    Parameters
    ----------
    categories:
        Which RESISC45 scene categories to include. Defaults to
        `SHOWCASE_CATEGORIES`, chosen to surface clear differences between
        GAN-based and diffusion-based super-resolution (see module docstring).
    num_per_category:
        How many images to take from each category.
    seed:
        Random seed, for reproducible selection across runs.

    Returns
    -------
    Dict mapping a unique image name (e.g. "forest_00") to a PIL Image.
    """
    from datasets import load_dataset

    categories = categories or SHOWCASE_CATEGORIES

    print("[datasets] Loading RESISC45 from Hugging Face (streams, cached after first run)...")
    resisc = load_dataset("tanganke/resisc45", split="train")
    class_names = resisc.features["label"].names

    missing = [c for c in categories if c not in class_names]
    if missing:
        raise ValueError(
            f"Categories not found in RESISC45: {missing}. "
            f"Available categories: {class_names}"
        )

    counts = {c: 0 for c in categories}
    images: Dict[str, Image.Image] = {}

    # Deterministic selection: iterate the dataset once, in order, taking the
    # first `num_per_category` examples of each requested category. Combined
    # with a fixed `seed` for any downstream random operations (e.g. patch
    # cropping), this keeps the benchmark reproducible run to run.
    for example in resisc:
        label_name = class_names[example["label"]]
        if label_name not in counts or counts[label_name] >= num_per_category:
            continue
        name = f"{label_name}_{counts[label_name]:02d}"
        images[name] = example["image"].convert("RGB")
        counts[label_name] += 1
        if all(c >= num_per_category for c in counts.values()):
            break

    found_total = sum(counts.values())
    expected_total = len(categories) * num_per_category
    if found_total < expected_total:
        print(
            f"[warn] Only found {found_total}/{expected_total} requested images "
            f"(some categories may have fewer available in this split)."
        )

    print(f"[datasets] Loaded {len(images)} images across {len(categories)} categories.")
    return images


def build_pairs(
    hr_images: Dict[str, Image.Image],
    hr_size: int = 512,
    scale: int = 4,
) -> Dict[str, ImagePair]:
    """Build HR/LR evaluation pairs by resizing to a standard size and then
    self-downsampling via bicubic interpolation (the standard SR benchmark
    protocol -- see module docstring for the honest limitation this implies).

    Parameters
    ----------
    hr_images:
        Mapping of image name -> source PIL image (any size).
    hr_size:
        Target size (square) for the "ground truth" high-res image.
    scale:
        Super-resolution factor; the low-res image is `hr_size // scale`.

    Returns
    -------
    Dict mapping image name -> ImagePair.
    """
    lr_size = hr_size // scale
    pairs: Dict[str, ImagePair] = {}

    for name, img in hr_images.items():
        category = name.rsplit("_", 1)[0]
        hr_img = img.resize((hr_size, hr_size), Image.BICUBIC)
        lr_img = hr_img.resize((lr_size, lr_size), Image.BICUBIC)
        pairs[name] = ImagePair(name=name, category=category, hr_image=hr_img, lr_image=lr_img)

    return pairs


def save_pairs(pairs: Dict[str, ImagePair], out_dir: str) -> None:
    """Save every HR/LR pair to disk as PNG files, for later re-use without
    re-downloading or re-processing the source dataset."""
    os.makedirs(out_dir, exist_ok=True)
    for name, pair in pairs.items():
        pair.hr_image.save(os.path.join(out_dir, f"{name}_hr.png"))
        pair.lr_image.save(os.path.join(out_dir, f"{name}_lr.png"))


def load_pairs_from_disk(names: List[str], category_map: Dict[str, str], in_dir: str) -> Dict[str, ImagePair]:
    """Reload previously-saved HR/LR pairs from disk, skipping re-download."""
    pairs: Dict[str, ImagePair] = {}
    for name in names:
        hr_path = os.path.join(in_dir, f"{name}_hr.png")
        lr_path = os.path.join(in_dir, f"{name}_lr.png")
        if not (os.path.exists(hr_path) and os.path.exists(lr_path)):
            raise FileNotFoundError(f"Missing saved pair for '{name}' in {in_dir}")
        pairs[name] = ImagePair(
            name=name,
            category=category_map.get(name, name.rsplit("_", 1)[0]),
            hr_image=Image.open(hr_path).convert("RGB"),
            lr_image=Image.open(lr_path).convert("RGB"),
        )
    return pairs
