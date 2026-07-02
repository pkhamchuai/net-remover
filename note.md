# Net Removal (Image De-fencing) — Implementation Plan

Plan for benchmarking state-of-the-art net/fence removal on photos. Target machine:
**vermeer** — Ryzen 9 5950X (16C/32T), 125 GiB RAM, RTX 3090 (24 GB VRAM),
Ubuntu 24.04, root NVMe 83 % full (~154 GB free), 2× 7.3 TB HDDs currently unmounted.

## TL;DR — methods to implement / compare

**Stage 1 — mask generation (fixed per image, shared by all inpainters):**
1. **SAM 2** (large) — zero-shot net segmentation via point/box prompts. Primary.
2. **Classical** — adaptive threshold + morphology. Cheap fallback/sanity check.
3. *(later, optional)* **YOLOv8-seg** trained on a custom net dataset — only worth it for recurring net types.

**Stage 2 — inpainting (the actual comparison):**
1. **OpenCV Telea** — classical baseline, no GPU.
2. **LaMa (big-lama)** — expected winner; FFC architecture excels at thin repeating structures.
3. **MI-GAN** — lightweight GAN middle ground (bundled in IOPaint).
4. **SDXL-inpainting** — diffusion contender, viable on 24 GB.
5. *(optional)* **FLUX.1-Fill** — strongest open diffusion inpainter, if disk allows.

**Deliverable:** per-image matplotlib comparison figures, `figsize=(16, 9)` (16:9 aspect),
subplots: **Original → Telea → LaMa → MI-GAN → SDXL (→ FLUX)**, method name + runtime
in titles, axes off, saved to `results/`. Plus one overview grid (rows = images,
cols = methods) on the same 16:9 canvas.

---

## Architecture

Two-stage pipeline: **segment the net → inpaint the masked pixels**.
The comparison controls for Stage 1 by freezing one mask per image, so Stage 2
methods compete fairly.

Working hypothesis (to be demonstrated by the benchmark): **LaMa beats
diffusion models for structure/grid removal** because Fast Fourier Convolutions
capture global context and continue occluded background patterns, whereas SD-family
models tend to hallucinate new objects into the hole. Keep SDXL in the comparison
to show this; add zoomed crop insets on a net region as a second figure row to make
hallucination artifacts visible.

## Setup

Two paths — start with the fast one:

**Fast path — IOPaint** (bundles LaMa, SD, MI-GAN, and SAM behind one API/Web UI):

```bash
pip install iopaint
iopaint start --model=lama --device=cuda --port=8080
```

Drive it headlessly from the benchmark script via its HTTP API.

**Custom path** (only if IOPaint lacks needed knobs): direct wrappers around
`advimman/lama` and `facebookresearch/segment-anything` (or SAM 2), each exposing a
common `remove(image, mask) -> image` interface.

Environment: venv, `torch` (CUDA 12.x), `opencv-python`, `matplotlib`, `iopaint`,
`diffusers`, `segment-anything-2`, `ultralytics` (only if YOLOv8-seg is pursued).

## Hardware notes (vermeer)

- 24 GB VRAM: SAM 2-large, SDXL-inpaint, and FLUX.1-Fill all fit; LaMa can stay
  resident alongside. Full-resolution inference, no tiling needed for typical photos.
- 32 threads / 125 GiB RAM: parallelize mask post-processing, metrics, and figure
  rendering across the test set.
- **Disk caution:** root NVMe has only ~154 GB free; model weights (SAM 2 + LaMa +
  SDXL + FLUX ≈ 40–50 GB) default to `~/.cache`. **Mount one of the 7.3 TB HDDs
  first** and point `HF_HOME`, `TORCH_HOME`, and the data/results dirs there.

## Repo layout

```
data/       test photos (nets/fences)
masks/      frozen binary masks, one per image
methods/    one wrapper per inpainting method, common interface
results/    comparison figures + metrics table
run_benchmark.py
```

## Test data

- 3–5 real photos stressing different cases: chain-link fence, sports net
  (badminton/soccer goal), fine mesh close-up. User photos go in `data/`;
  otherwise pull public de-fencing dataset samples.
- 1–2 **synthetic** cases: overlay a net pattern on clean photos → enables
  PSNR/SSIM/LPIPS against ground truth.

## Mask generation details

- SAM 2 with point/box prompts on the net; classical pipeline as fallback for
  very regular meshes.
- Dilate final masks ~3–5 px so no net residue leaks into inpainting.
- Freeze one mask per image before running any inpainter.

## Evaluation

- Qualitative: the 16:9 comparison figures (+ zoom insets).
- Quantitative: runtime per method; BRISQUE/NIQE on real photos (no reference);
  PSNR/SSIM/LPIPS on the synthetic cases.
- Summary table saved alongside figures in `results/`.

## Order of work

1. Mount an HDD for caches/data (needs sudo); create venv, install IOPaint.
2. End-to-end on one photo: SAM 2 mask → Telea + LaMa → first 16:9 figure.
3. Add MI-GAN and SDXL-inpaint (FLUX if disk allows).
4. Synthetic cases + metrics + overview grid.
5. Revisit YOLOv8-seg only if a recurring net type emerges.
