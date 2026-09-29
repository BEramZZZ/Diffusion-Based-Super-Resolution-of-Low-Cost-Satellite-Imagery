# Satellite Super-Resolution: GAN vs. Diffusion — Zero-Shot Benchmark and Fine-Tuning Comparison

A benchmark and fine-tuning study comparing two 4x image super-resolution approaches —
**Real-ESRGAN** (GAN-based) and the **Stable Diffusion x4 Upscaler** (diffusion-based)
— on satellite/aerial imagery, in both zero-shot inference and across four fine-tuning
strategies (head-only, LoRA, partial, full). Built as part of a research project on
diffusion-based super-resolution of low-cost satellite imagery.

> A live demo platform built on these results is available at [link to platform repo].
> This repository is the research record (methodology, benchmark numbers, training code);
> the platform repository holds the deployed web app and serving code.

## What this project actually measures

Beyond "which model scores higher," this project is built to surface a specific,
well-documented tension in image restoration: the **perception-distortion tradeoff**
(Blau & Michaeli, 2018).

- **Real-ESRGAN**, trained with pixel + perceptual + adversarial losses, tends toward
  **smoother, safer** output — when genuinely uncertain about fine detail, it converges
  toward an averaged, blurred answer.
- **The diffusion model**, sampling stochastically over ~30 denoising steps, tends to
  **commit to specific, sharp-looking detail** — but that detail is only ever a
  statistically _plausible_ guess, not a guaranteed match to the true ground truth. This
  is the practical meaning of "hallucination" in this context: confident,
  realistic-looking, but potentially incorrect detail (an invented texture, or an object
  with no counterpart in the real scene).

For an application like satellite imagery analysis, this distinction is not cosmetic —
hallucinated detail is a real, consequential error, not just a lower similarity score.
The fine-tuning results below are reported with that framing in mind throughout, not
just as a leaderboard.

## Results: fine-tuning comparison (4 strategies x 2 models)

Full numbers in [`outputs/finetune_eval/summary_by_strategy.csv`](outputs/finetune_eval/summary_by_strategy.csv).
Evaluated on 10 held-out NAIP aerial-imagery scenes per strategy after fine-tuning on a
small (400-image) synthetically-degraded NAIP training set.

| Model           | Strategy  | PSNR      | SSIM      | Trainable params | Train time | Note                                      |
| --------------- | --------- | --------- | --------- | ---------------- | ---------- | ----------------------------------------- |
| Real-ESRGAN     | zero-shot | 24.74     | 0.614     | 0                | —          |                                           |
| Real-ESRGAN     | head-only | 22.82     | 0.597     | 1.7K (0.01%)     | 45m        | below zero-shot                           |
| Real-ESRGAN     | lora      | 24.12     | 0.610     | 25K (0.15%)      | 50m        | near no-op                                |
| Real-ESRGAN     | partial   | 28.71     | 0.804     | 5.19M (31%)      | 1h10m      |                                           |
| **Real-ESRGAN** | **full**  | **30.02** | **0.833** | 16.7M (100%)     | 2h         | best overall                              |
| Diffusion       | zero-shot | 22.96     | 0.529     | 0                | —          |                                           |
| Diffusion       | head-only | 23.67     | 0.557     | 9.7K (0.002%)    | 25m        | color-drift failure mode                  |
| Diffusion       | lora      | —         | —         | 1.6M (0.34%)     | 50m        | 0/10 valid outputs (first pass)           |
| Diffusion       | partial   | 26.13     | 0.663     | 7.16M (1.51%)    | 1h         |                                           |
| Diffusion       | full      | 27.18     | 0.699     | unverified       | not logged | checkpoint size anomaly, see Known issues |

Diffusion numbers come from the v-prediction-corrected fine-tuning runs
(`outputs/finetune_eval/`). The earlier epsilon-prediction pass is kept separately in
`outputs/finetune_eval_Old/` for comparison and is **not** used in this table.

**Bottom line:** Real-ESRGAN + full fine-tuning is the strongest result on both metrics,
by a clear margin over every diffusion variant including diffusion's own best strategy.
It also fails in the safer direction — its worst-case behavior is mild under-texturing
(smoothing), not invented structure. See "Recommendation" below.

### Key findings

- **Real-ESRGAN + head-only actually scores below its own zero-shot baseline** (22.82 vs
  24.74 PSNR). Adapting only the final output-projection layer, with nothing else
  unfrozen, made this model _worse_ than doing no fine-tuning at all. Don't ship this
  strategy as if "fine-tuned" implies "improved" — it doesn't, here.
- **Real-ESRGAN + lora is a near no-op** (24.12 vs 24.74 PSNR) — not harmful, but not
  worth the training time either at this rank/target-layer configuration.
- **Diffusion + lora produced zero valid outputs (0/10) in the first pass**, which used
  the epsilon-prediction target described in Known issues. It is excluded from the live
  platform. <!-- TODO: if LoRA was re-run after the v-prediction fix, replace this bullet with the re-run result. -->
- **Diffusion + head-only shows a specific, reproducible color-drift failure mode**:
  structure stays largely intact, but a rust/red cast appears on man-made,
  regular-geometry scenes (e.g. runway markings) and a pink tint appears scattered
  through natural high-entropy texture (e.g. chaparral).
- **Real-ESRGAN + head-only has its own separate failure mode**, found during
  qualitative review: on chaparral specifically, roughly half of the output tile
  collapses into a near-featureless smoothed patch while the other half keeps texture —
  a partial wash-out this project originally expected only from diffusion strategies.
  Averaged over all 10 scenes this strategy still nets _below_ zero-shot (see above);
  this is presumably a contributing cause, not the whole story.
- **Diffusion hallucination examples** observed during qualitative review (not
  captured by PSNR/SSIM): on a beach scene, diffusion output an ornate sand-ripple
  texture with no counterpart in the ground truth; on a dense-residential scene,
  diffusion output a blue/teal blob with no ESRGAN counterpart and no clear
  correspondence to anything in the LR input. Both are consistent with the model's
  fixed, scene-agnostic conditioning prompt rather than being random noise.

## Known issues / open caveats

Documenting these here rather than silently excluding them, since a research repo
claiming completed fine-tuning should be honest about what's still unresolved:

1. **Diffusion + full checkpoint size is inconsistent with a genuine full fine-tune
   (open).** The saved `diffusion_pytorch_model.safetensors` for this strategy is
   ~1.8MB — several orders of magnitude smaller than a full UNet fine-tune should be
   (hundreds of MB at minimum, even in fp16). Because the loading code uses
   `strict=False`, a truncated state dict would load silently, with the model quietly
   falling back to its pretrained weights for whatever wasn't in the checkpoint.
   **The 27.18/0.699 numbers above may not reflect what "full fine-tuning" is supposed
   to measure until this is confirmed.** Since `save_pretrained` writes the complete
   state dict regardless of strategy, the check is to compare the file sizes of the
   `head_only`, `partial`, and `full` checkpoints: if all are similarly small, the
   save path is at fault for every non-LoRA run; if only `full` is small, that single
   run is the culprit.
2. **Diffusion + partial: earlier collapse traced to an objective mismatch
   (resolved).** The Stable Diffusion x4 Upscaler is trained with v-prediction, but the
   first fine-tuning pass used an epsilon-prediction target (target = noise). Every
   gradient step pushed the UNet toward predicting a different quantity than the
   sampler assumes it outputs, which produced a degenerate, near-uniform wash. That run
   also had the lowest training loss of any strategy while its outputs were visibly
   wrong. `finetune_diffusion.py` now trains against the v-prediction target, and all
   diffusion checkpoints reported in the table above come from the corrected version.
   The broken pass is preserved in `outputs/finetune_eval_Old/`.
3. **Research loading code and platform serving code have diverged.**
   `load_checkpoint.py` in this repo handles LoRA checkpoints (rebuilding the adapter
   config and calling `load_lora_adapter`), while the platform's diffusion loader has no
   LoRA branch. This does not matter today because diffusion + LoRA is not deployed,
   but it would need reconciling if LoRA is ever revisited.
4. **No true low-res/high-res satellite pair was used.** Fine-tuning pairs are
   synthetically degraded NAIP aerial imagery, not real Sentinel-2 (or other satellite)
   low-res imagery paired with independent high-res commercial imagery. WorldStrat (the
   dataset actually named in the original project brief) would close this gap but its
   full release (100GB+) was impractical within this project's timeline. This applies
   with equal force to every number in this repository: real-world Sentinel-2
   performance is not proven by these results, only suggested by them.

## Recommendation

Real-ESRGAN + full fine-tuning is the production default: highest PSNR/SSIM by a clear
margin, deterministic (no seed/sampler variance to account for), and its failure mode
(mild smoothing) is bounded and safe compared to diffusion's failure mode (invented,
unverified detail). Diffusion output is retained as a labeled, secondary "generative
candidate" view rather than a default — useful on genuinely ambiguous high-entropy
texture where ESRGAN under-textures relative to ground truth, but not a source of truth
on its own. See the platform repo for how this is implemented as a gated UI choice.

## Project structure

```
research/
├── README.md
├── requirements.txt
├── run_zero_shot_benchmark.py       # zero-shot RESISC45 benchmark
├── run_finetune.py                  # fine-tuning entry point (all strategies)
├── run_finetune_eval.py             # evaluates fine-tuned checkpoints per scene
├── make_hallucination_closeup.py    # zoomed comparison figure generator
├── src/sr_benchmark/
│   ├── compat.py                    # basicsr/torchvision compatibility fix
│   ├── datasets.py                  # loading RESISC45, building HR/LR evaluation pairs
│   ├── models.py                    # loading + running both base models
│   ├── metrics.py                   # PSNR, SSIM, LPIPS
│   ├── visualize.py                 # comparison grids and zoomed crops
│   └── finetune/
│       ├── common.py                # freeze/unfreeze helpers, trainable-param summary
│       ├── data.py                  # SRPairDataset (LR/HR pair construction)
│       ├── finetune_esrgan.py       # Real-ESRGAN strategy implementations
│       ├── finetune_diffusion.py    # diffusion strategies (v-prediction objective)
│       └── load_checkpoint.py       # rebuilds a model from a saved checkpoint
└── outputs/
    ├── pairs/                       # zero-shot benchmark HR/LR pairs
    ├── esrgan_outputs/, diffusion_outputs/
    ├── metrics.csv                  # per-example, zero-shot benchmark
    ├── summary_by_category.csv      # zero-shot, averaged per scene category
    ├── summary_by_model.csv         # zero-shot, averaged per model
    ├── comparison_grid.png
    ├── finetune_eval/               # current results
    │   ├── summary_by_strategy.csv  # the 4-strategy x 2-model table above
    │   └── <scene>/                 # lr_input, ground_truth, every strategy's output
    └── finetune_eval_Old/           # comparison grid from the superseded
                                     # epsilon-prediction diffusion runs
```

Fine-tuned checkpoint weights themselves are **not** committed to this repository
(binary, large, regenerable) — see `.gitignore`. Run `run_finetune.py` to reproduce
them, or see the platform repo for a deployed copy.

## Setup

```
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA GPU is strongly recommended for both benchmarking and fine-tuning. The diffusion
model in particular is slow on CPU (each image takes ~30 iterative denoising steps at
inference; fine-tuning is far slower still).

## Usage

```
# Zero-shot benchmark (default: 2 images x 10 categories = 20 examples)
python run_zero_shot_benchmark.py
python run_zero_shot_benchmark.py --num-per-category 5

# Fine-tuning: pick a model and strategy
python run_finetune.py --model esrgan --strategy full
python run_finetune.py --model esrgan --strategy partial
python run_finetune.py --model esrgan --strategy lora
python run_finetune.py --model esrgan --strategy head_only
python run_finetune.py --model diffusion --strategy full
python run_finetune.py --model diffusion --strategy partial
python run_finetune.py --model diffusion --strategy head_only
# diffusion --strategy lora produced 0/10 valid outputs in the first pass -- see Known issues

# Evaluate fine-tuned checkpoints on the held-out scenes
python run_finetune_eval.py

# After inspecting outputs/comparison_grid.png, generate a zoomed close-up
python make_hallucination_closeup.py --name dense_residential_00 --crop 150 150 350 350
```

## Data and methodology

**Source dataset (zero-shot benchmark): NWPU-RESISC45**, loaded via the Hugging Face
`datasets` library (`tanganke/resisc45`) — 31,500 real satellite/aerial images across 45
scene categories. Chosen deliberately over live satellite-catalog search APIs (e.g.
Microsoft Planetary Computer's STAC search), which have a documented, ongoing high
failure rate as of mid-2026
([microsoft/PlanetaryComputer#476](https://github.com/microsoft/PlanetaryComputer/issues/476))
— unsuitable for a benchmark others should be able to reproduce from this repository.

**Source dataset (fine-tuning): NAIP aerial imagery**, held-out test set of 10 scenes
across 10 categories: sparse/dense residential, forest, freeway, runway, harbor,
rectangular farmland, river, beach, chaparral.

**Evaluation categories** were chosen to surface different model behaviors:

- Dense/sparse residential, freeway, runway, harbor — regular, man-made structure,
  where hallucinated detail is most visible and most consequential.
- Forest, chaparral — organic, irregular natural texture, where GAN smoothing is most
  visually apparent.
- River, beach — low-texture regions, a smoothing stress test.
- Rectangular farmland — regular but non-urban repetitive texture.

**HR/LR pair construction**: each real image is resized to 512x512 and treated as
ground truth; its low-res input is created by bicubic downsampling to 128x128 (4x). This
is the standard "bicubic degradation" evaluation protocol used throughout the
super-resolution literature.

**Diffusion training objective**: the Stable Diffusion x4 Upscaler uses v-prediction, so
fine-tuning targets the velocity (`get_velocity`) rather than the raw noise. Training
against epsilon (noise) targets silently miscalibrates the UNet relative to the sampler;
see Known issues #2.

**Honest limitation**: this is a _self-downsampled_ benchmark, not a true cross-sensor
pair (e.g. real low-res Sentinel-2 imagery vs. real independent high-res commercial
imagery, as in the WorldStrat dataset). WorldStrat's full release (100GB+) was
impractical within this project's timeline; this benchmark substitutes a smaller,
standard, reproducible protocol instead. This applies to both the zero-shot benchmark
and every fine-tuning result in this repository.

## References

- X. Wang et al., "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks,"
  ECCV Workshops, 2018.
- X. Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure
  Synthetic Data," ICCV Workshops, 2021.
- C. Saharia et al., "Image Super-Resolution via Iterative Refinement," IEEE TPAMI, vol.
  45, no. 4, 2022.
- L. Zhang, A. Rao, M. Agrawala, "Adding Conditional Control to Text-to-Image Diffusion
  Models," ICCV, 2023.
- Y. Blau, T. Michaeli, "The Perception-Distortion Tradeoff," CVPR, 2018.
- R. Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual
  Metric," CVPR, 2018. (LPIPS)
- J. Ho et al., "Cascaded Diffusion Models for High Fidelity Image Generation," JMLR, 2022. (conditioning augmentation, used by the Stable Diffusion x4 Upscaler)
