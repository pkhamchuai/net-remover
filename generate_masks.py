"""Stage 1: classical net-mask generation.

The photos in data/ are all shot through a fine bird-net mesh that covers the
entire frame (not a localized fence/goal net), against widely varying
backgrounds (sky, clouds, buildings, trees, rooftop concrete). A black-hat
morphological transform pulls out thin dark structures relative to their
*local* surroundings, which is what makes it work across such varied
backgrounds, unlike a global intensity threshold.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
MASKS_DIR = Path(__file__).parent / "masks"


def generate_net_mask(
    image_bgr: np.ndarray,
    blackhat_ksize: int = 15,
    clahe_clip_limit: float = 3.0,
    clahe_tile_size: int = 32,
    close_ksize: int = 5,
    dilate_px: int = 4,
    min_component_area: int = 40,
) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Equalize local contrast first: the net's contrast against sky is much
    # stronger than against a dark rooftop, so a plain blackhat + global (Otsu)
    # threshold misses the net over dark backgrounds. CLAHE normalizes local
    # contrast so one global threshold works across both regions, without the
    # per-pixel noise blowup a fully adaptive threshold causes in foliage.
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit, tileGridSize=(clahe_tile_size, clahe_tile_size)
    )
    gray = clahe.apply(gray)

    blackhat_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (blackhat_ksize, blackhat_ksize)
    )
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, blackhat_kernel)

    _, mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_ksize, close_ksize)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    # Drop tiny isolated specks (noise) that survive thresholding but aren't
    # part of the connected net grid.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_component_area:
            cleaned[labels == label] = 255
    mask = cleaned

    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
    )
    mask = cv2.dilate(mask, dilate_kernel)

    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blackhat-ksize", type=int, default=15)
    parser.add_argument("--close-ksize", type=int, default=5)
    parser.add_argument("--dilate-px", type=int, default=4)
    parser.add_argument("--min-component-area", type=int, default=40)
    parser.add_argument("--pattern", default="*.JPG")
    args = parser.parse_args()

    MASKS_DIR.mkdir(exist_ok=True)
    image_paths = sorted(DATA_DIR.glob(args.pattern))
    if not image_paths:
        raise SystemExit(f"No images found in {DATA_DIR} matching {args.pattern}")

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"skip (unreadable): {image_path.name}")
            continue
        mask = generate_net_mask(
            image_bgr,
            blackhat_ksize=args.blackhat_ksize,
            close_ksize=args.close_ksize,
            dilate_px=args.dilate_px,
            min_component_area=args.min_component_area,
        )
        net_frac = (mask > 0).mean()
        out_path = MASKS_DIR / f"{image_path.stem}.png"
        cv2.imwrite(str(out_path), mask)
        print(f"{image_path.name}: net pixels = {net_frac:.1%} -> {out_path.name}")


if __name__ == "__main__":
    main()
