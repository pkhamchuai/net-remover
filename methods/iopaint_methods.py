"""Stage 2: inpainting method wrappers, all backed by IOPaint's ModelManager.

Calling ModelManager directly (instead of the `iopaint run` CLI) lets us time
pure inference per method, and reuse one Python process across methods.

ModelManager's contract (confirmed by reading iopaint/batch_processing.py,
which passes a plain RGB array in and does `cv2.cvtColor(result, BGR2RGB)` on
the way out): input is RGB, output is BGR. Get this backwards -- e.g. feeding
it a pre-flipped BGR array -- and colors in the *inpainted* region come out
with red/blue swapped relative to the untouched region pasted back from the
real RGB original (this bit us: navy net lines turned orange after "removal").

We also resize to a working resolution ourselves rather than relying on
IOPaint's built-in `hd_strategy`:
- The default HDStrategy.CROP splits the mask into one bounding box per
  connected component (cv2.findContours) and runs a separate forward pass per
  box. Our net masks are thousands of disjoint grid-line fragments, so CROP
  means thousands of tiny inference calls per image (minutes per image, and
  it OOMs SDXL).
- MI-GAN's model class overrides `__call__` entirely and ignores
  `config.hd_strategy` outright -- it always falls back to its own
  per-contour box-cropping unless the input is exactly 512x512. Box count
  depends on mask fragment count (thousands, for our net grids) rather than
  image resolution, so even downsizing to 1536px left it at ~60s/image.
  MI-GAN was trained at a fixed 512x512 anyway, so we run it at that native
  resolution (pad-to-square + resize) instead of fighting its box logic --
  correct for the model, and its single-pass path is sub-second.
Doing the resize/paste-back ourselves, uniformly, sidesteps both issues and
keeps methods directly comparable at their intended working resolution.
"""

import gc
import time

import cv2
import numpy as np
import torch
from iopaint.model_manager import ModelManager
from iopaint.schema import CV2Flag, HDStrategy, InpaintRequest

_WORK_SIZE = 1536
_MIGAN_SIZE = 512

METHODS = {
    "Telea": (
        "cv2",
        InpaintRequest(cv2_flag=CV2Flag.INPAINT_TELEA, hd_strategy=HDStrategy.ORIGINAL),
    ),
    "LaMa": ("lama", InpaintRequest(hd_strategy=HDStrategy.ORIGINAL)),
    "MI-GAN": ("migan", InpaintRequest(hd_strategy=HDStrategy.ORIGINAL)),
    # SDXL-inpaint dropped: ~24s/image vs sub-second for the other three, and
    # per note.md's own hypothesis, diffusion tends to hallucinate rather than
    # continue background through thin repeating net structure anyway.
}


def _ensure_downloaded(model_id: str):
    """IOPaint's ModelManager only recognizes a model as available once its
    weights are already cached locally, so pull them first (no-op if present).
    """
    if model_id == "cv2":
        return
    from iopaint.download import cli_download_model

    cli_download_model(model_id)


def _resize_max(image: np.ndarray, mask: np.ndarray, limit: int):
    h, w = image.shape[:2]
    scale = min(1.0, limit / max(h, w))
    if scale == 1.0:
        return image, mask
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    small_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return small_image, small_mask


def _median_then_sharpen(image: np.ndarray, median_ksize: int = 5, sharpen_amount: float = 1.0):
    """Clean up residual grid speckle from the (imperfect, low-res) inpaint
    with a median filter, then unsharp-mask to recover perceived detail the
    median filter and upsampling softened."""
    denoised = cv2.medianBlur(image, median_ksize)
    blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(denoised, 1 + sharpen_amount, blurred, -sharpen_amount, 0)
    return sharpened


def _resize_pad_square(image: np.ndarray, mask: np.ndarray, size: int):
    h, w = image.shape[:2]
    side = max(h, w)
    image_sq = cv2.copyMakeBorder(image, 0, side - h, 0, side - w, cv2.BORDER_REFLECT)
    mask_sq = cv2.copyMakeBorder(mask, 0, side - h, 0, side - w, cv2.BORDER_CONSTANT, value=0)
    small_image = cv2.resize(image_sq, (size, size), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask_sq, (size, size), interpolation=cv2.INTER_NEAREST)
    return small_image, small_mask, side


def run_method(method_name: str, image_rgb: np.ndarray, mask_gray: np.ndarray, device: str = "cuda"):
    """Run one inpainting method. Returns (result_rgb, seconds)."""
    model_id, config = METHODS[method_name]

    _ensure_downloaded(model_id)
    # sd_cpu_textencoder is read via kwargs["sd_cpu_textencoder"] (not .get())
    # inside iopaint's SD/SDXL model classes, so it's required even though it
    # only matters for those two methods; harmless no-op for lama/migan/cv2.
    manager = ModelManager(
        name=model_id, device=torch.device(device), sd_cpu_textencoder=False
    )
    try:
        image_rgb = np.ascontiguousarray(image_rgb)
        orig_h, orig_w = image_rgb.shape[:2]

        if method_name == "MI-GAN":
            small_rgb, small_mask, side = _resize_pad_square(
                image_rgb, mask_gray, _MIGAN_SIZE
            )
            start = time.perf_counter()
            small_result_bgr = manager(small_rgb, small_mask, config)
            elapsed = time.perf_counter() - start

            small_result_rgb = small_result_bgr[:, :, ::-1]
            square_result_rgb = cv2.resize(
                small_result_rgb, (side, side), interpolation=cv2.INTER_CUBIC
            )
            result_rgb = square_result_rgb[:orig_h, :orig_w]
        else:
            small_rgb, small_mask = _resize_max(image_rgb, mask_gray, _WORK_SIZE)
            start = time.perf_counter()
            small_result_bgr = manager(small_rgb, small_mask, config)
            elapsed = time.perf_counter() - start

            small_result_rgb = small_result_bgr[:, :, ::-1]
            if small_rgb.shape[:2] != (orig_h, orig_w):
                result_rgb = cv2.resize(
                    small_result_rgb, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC
                )
            else:
                result_rgb = small_result_rgb

        result_rgb = _median_then_sharpen(np.ascontiguousarray(result_rgb))
        keep = mask_gray < 127
        result_rgb[keep] = image_rgb[keep]
    finally:
        del manager
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    return result_rgb, elapsed
