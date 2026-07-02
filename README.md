# Net Remover

Benchmarking inpainting methods for removing bird/insect protection netting
from photos. The test photos are all shot through a fine mesh net that spans
the **entire frame** (rooftop/balcony shots looking out through the net), not
a localized fence or sports net — that shapes the whole approach below. See
[note.md](note.md) for the original, broader implementation plan.

## Pipeline

**Stage 1 — mask generation** ([generate_masks.py](generate_masks.py)):
classical CLAHE + black-hat morphological thresholding, run per image. SAM 2
(point/box-prompt object segmentation) doesn't fit this dataset — there's no
discrete object to prompt, just a uniform grid texture over the whole photo —
so a background-adaptive classical detector is used instead. Masks are saved
to `masks/`.

**Stage 2 — inpainting** ([methods/iopaint_methods.py](methods/iopaint_methods.py)):
runs the frozen mask through three methods via IOPaint's `ModelManager`:

- **Telea** (OpenCV, classical, no GPU)
- **LaMa** (FFC architecture, GPU)
- **MI-GAN** (lightweight GAN, GPU, run at its native 512×512)

Each method's output gets a light median-filter + unsharp-mask cleanup before
compositing back with the original (unmasked pixels always come from the
untouched original image, never from resized/inpainted output).

SDXL-inpaint was also tried and dropped: ~24s/image versus sub-second for the
other three, for no clear quality win on this kind of thin repeating
structure.

**Orchestration** ([run_benchmark.py](run_benchmark.py)): loops over every
`data/*.JPG`, runs all three methods, and saves a per-image 2×3 comparison
figure (`results/<name>_comparison.png`) plus one overview grid
(`results/overview_grid.png`).

## Running it

```
make setup                    # venv + dependencies
.venv/bin/python3 generate_masks.py
.venv/bin/python3 run_benchmark.py
```

## Results

![Overview grid: Original, Telea, LaMa, MI-GAN across all test photos](results/overview_grid.png)

Rows are Original / Telea / LaMa / MI-GAN; columns are the individual test
photos.

**Telea is the best performer** — by visual inspection (repo owner's
judgment), the plain OpenCV classical method produces the cleanest result of
the three on this dataset. LaMa and MI-GAN don't clearly outperform it here,
despite being the more sophisticated, GPU-based methods. Telea also happens
to be the cheapest option (CPU-only, no model weights).

This somewhat inverts the usual expectation (learned methods beating
classical baselines) and is likely specific to this dataset's structure: a
thin, high-frequency, repeating pattern over mostly smooth/low-detail
backgrounds (sky, clouds, flat concrete) is close to the textbook case
Telea's fast marching method was designed for, whereas LaMa and MI-GAN are
tuned more for larger, irregular holes with structured content around them.

## Known limitations

- The classical mask (bottom-left panel in each comparison figure) is clean
  over sky/cloud but noisier over foliage and can miss the net entirely in
  the darkest regions — room to improve before drawing firm conclusions.
- Inpainting runs at a capped working resolution (1536px long side, or 512x512
  for MI-GAN) rather than native resolution, for speed.
- No quantitative metrics yet (BRISQUE/NIQE/PSNR/SSIM/LPIPS) — this has been
  a qualitative comparison so far.

## Future work

- **FLUX.1-Fill** could be tried as a stronger diffusion contender than
  SDXL-inpaint (see [note.md](note.md)) — it was left out of this pass to
  keep things fast and disk-light, not ruled out on quality grounds.
