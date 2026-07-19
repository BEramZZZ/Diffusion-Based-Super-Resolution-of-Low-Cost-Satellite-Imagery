# Satellite Super-Resolution: GAN vs. Diffusion (Zero-Shot Benchmark)

A benchmark comparing two pretrained 4x image super-resolution approaches —
**Real-ESRGAN** (GAN-based) and the **Stable Diffusion x4 Upscaler**
(diffusion-based) — on real satellite/aerial imagery, in zero-shot inference
mode (no fine-tuning). Built as part of a research internship project on
diffusion-based super-resolution of low-cost satellite imagery.

## What this benchmark actually measures

Beyond just "which model scores higher," this benchmark is built to surface
a specific, well-documented tension in image restoration: the
**perception-distortion tradeoff** (Blau & Michaeli, 2018).

- **Real-ESRGAN**, trained with pixel + perceptual + adversarial losses,
  tends to produce **smoother, safer** output — when genuinely uncertain
  about fine detail, it converges toward an averaged, blurred answer (a
  direct consequence of its pixel-loss term; see `docs/`METHODOLOGY for the
  full explanation).
- **The diffusion model**, sampling stochastically over ~30 denoising steps,
  tends to **commit to specific, sharp-looking detail** — but that detail is
  only ever a statistically *plausible* guess, not a guaranteed match to the
  true ground truth. This is the practical meaning of "hallucination" in
  this context: confident, realistic-looking, but potentially incorrect
  detail (an invented building, road, or texture).

For an application like satellite imagery analysis for a space agency, this
distinction is not cosmetic — a hallucinated road or structure is a real,
consequential error, not just a lower similarity score.

This benchmark reports **PSNR, SSIM, and LPIPS** side by side specifically
because PSNR/SSIM (pixel/structure fidelity) and LPIPS (perceptual
similarity) can disagree, and that disagreement is itself evidence of the
tradeoff above.

## Project structure

```
sr-benchmark/
├── README.md
├── requirements.txt
├── run_zero_shot_benchmark.py     # main script: run this
├── make_hallucination_closeup.py  # secondary script: zoomed comparison figure
└── src/sr_benchmark/
    ├── compat.py       # basicsr/torchvision compatibility fix
    ├── datasets.py     # loading RESISC45, building HR/LR evaluation pairs
    ├── models.py       # loading + running both models
    ├── metrics.py      # PSNR, SSIM, LPIPS
    └── visualize.py    # comparison grids and zoomed crops
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA GPU is strongly recommended. The diffusion model in particular is
slow on CPU (each image takes ~30 iterative denoising steps).

**Note on `torchao`/`peft`:** this benchmark does **not** use LoRA or any
`peft`-based fine-tuning, so those dependency conflicts (documented in the
project's development history) don't apply here — this is inference-only.

## Usage

```bash
# Run the full zero-shot benchmark (default: 2 images x 10 categories = 20 examples)
python run_zero_shot_benchmark.py

# Use more images per category for a more statistically robust result
python run_zero_shot_benchmark.py --num-per-category 5

# After inspecting outputs/comparison_grid.png, generate a zoomed close-up
# for your strongest example (adjust the crop box after looking at the grid)
python make_hallucination_closeup.py --name dense_residential_00 --crop 150 150 350 350
```

Outputs land in `outputs/`:

| File | Contents |
|---|---|
| `pairs/` | Saved HR/LR image pairs used for evaluation |
| `esrgan_outputs/`, `diffusion_outputs/` | Each model's output per example |
| `metrics.csv` | Per-example PSNR/SSIM/LPIPS for both models |
| `summary_by_category.csv` | Metrics averaged per scene category |
| `summary_by_model.csv` | Metrics averaged overall, per model |
| `comparison_grid.png` | Full visual grid: LR input / both outputs / ground truth |

## Data and methodology

**Source dataset: NWPU-RESISC45**, loaded via the Hugging Face `datasets`
library (`tanganke/resisc45`) — 31,500 real satellite/aerial images across 45
scene categories. Chosen deliberately over live satellite-catalog search
APIs (e.g. Microsoft Planetary Computer's STAC search), which have a
documented, ongoing high failure rate as of mid-2026
([microsoft/PlanetaryComputer#476](https://github.com/microsoft/PlanetaryComputer/issues/476))
— unsuitable for a benchmark others should be able to reproduce from this
repository.

**Evaluation categories** (`SHOWCASE_CATEGORIES` in `datasets.py`) were
chosen to surface different model behaviors:
- Dense/sparse residential, freeway, runway, harbor — regular, man-made
  structure, where hallucinated detail is most visible and most
  consequential.
- Forest, chaparral — organic, irregular natural texture, where GAN
  smoothing is most visually apparent.
- River, beach — low-texture regions, a smoothing stress test.
- Agricultural — regular but non-urban repetitive texture.

**HR/LR pair construction**: each real RESISC45 image is resized to 512x512
and treated as ground truth; its low-res input is created by bicubic
downsampling to 128x128 (4x). This is the standard "bicubic degradation"
evaluation protocol used throughout the super-resolution literature.

**Honest limitation**: this is a *self-downsampled* benchmark, not a true
cross-sensor pair (e.g. real low-res Sentinel-2 imagery vs. real independent
high-res commercial imagery, as in the WorldStrat dataset). WorldStrat's full
release (100GB+) was impractical within this project's timeline; this
benchmark substitutes a smaller, standard, reproducible protocol instead.
This is a real, disclosed simplification — see the project report for the
full discussion.

## Fine-tuning (documented, not included here)

An earlier phase of this project attempted LoRA fine-tuning of the diffusion
upscaler on satellite-domain data. This was ultimately not completed
successfully: diagnosis revealed the pipeline uses a non-standard latent
diffusion architecture (diffusing at low-resolution-image spatial scale,
with 4x upscaling performed inside a specialized VAE decoder, and a separate
noise-augmentation scheduler for the conditioning image) that a
standard-latent-diffusion-assumption training loop does not correctly
replicate. Real-ESRGAN fine-tuning (plain L1 generator fine-tuning, no
adversarial retraining) was completed successfully and showed a consistent
PSNR/SSIM improvement over its zero-shot baseline on a held-out NAIP aerial
imagery test set. Full replication of correct diffusion fine-tuning for this
specific pipeline architecture is left as future work.

## References

- X. Wang et al., "ESRGAN: Enhanced Super-Resolution Generative Adversarial
  Networks," ECCV Workshops, 2018.
- C. Saharia et al., "Image Super-Resolution via Iterative Refinement," IEEE
  TPAMI, vol. 45, no. 4, 2022.
- L. Zhang, A. Rao, M. Agrawala, "Adding Conditional Control to Text-to-Image
  Diffusion Models," ICCV, 2023.
- Y. Blau, T. Michaeli, "The Perception-Distortion Tradeoff," CVPR, 2018.
- R. Zhang et al., "The Unreasonable Effectiveness of Deep Features as a
  Perceptual Metric," CVPR, 2018. (LPIPS)
