"""Run Telea, LaMa and MI-GAN against the frozen net masks in masks/, for
every photo in data/*.JPG, and save 16:9 comparison figures plus one overview
grid to results/.
"""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from methods.iopaint_methods import METHODS, run_method

DATA_DIR = Path(__file__).parent / "data"
MASKS_DIR = Path(__file__).parent / "masks"
RESULTS_DIR = Path(__file__).parent / "results"

METHOD_NAMES = list(METHODS.keys())
COLUMN_TITLES = ["Original"] + METHOD_NAMES


def load_image_rgb(path: Path):
    bgr = cv2.imread(str(path))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    image_paths = sorted(DATA_DIR.glob("*.JPG"))
    if not image_paths:
        raise SystemExit(f"No JPGs found in {DATA_DIR}")

    overview_rows = []  # list of (image_stem, [ (title, image) ... ])

    for image_path in image_paths:
        stem = image_path.stem
        mask_path = MASKS_DIR / f"{stem}.png"
        if not mask_path.exists():
            print(f"skip {stem}: no mask found (run generate_masks.py first)")
            continue

        print(f"=== {stem} ===")
        image_rgb = load_image_rgb(image_path)
        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        method_panels = []
        for method_name in METHOD_NAMES:
            print(f"  running {method_name}...")
            result_rgb, elapsed = run_method(method_name, image_rgb, mask_gray)
            print(f"  {method_name}: {elapsed:.2f}s")
            method_panels.append((method_name, result_rgb, elapsed))

        # per-image comparison figure: 2x3 grid --
        # row 1: original, method 1, method 2
        # row 2: net mask,  method 3, (blank if fewer than 4 methods)
        top_left = [("Original", image_rgb, None)] + method_panels[0:2]
        bottom_left = [("Net mask", mask_gray, None)] + method_panels[2:4]
        grid = [top_left, bottom_left]
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for row_idx, row in enumerate(grid):
            for col_idx in range(3):
                ax = axes[row_idx, col_idx]
                if col_idx >= len(row):
                    ax.axis("off")
                    continue
                title, img, elapsed = row[col_idx]
                ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
                ax.axis("off")
                ax.set_title(title if elapsed is None else f"{title} ({elapsed:.2f}s)")
        fig.suptitle(stem)
        fig.tight_layout()
        out_path = RESULTS_DIR / f"{stem}_comparison.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  saved {out_path}")

        overview_rows.append((stem, [("Original", image_rgb, None)] + method_panels))

    # overview grid: rows = Original + methods, cols = images (transposed)
    n_rows = len(COLUMN_TITLES)
    n_cols = len(overview_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 13.5))
    for col_idx, (stem, panels) in enumerate(overview_rows):
        for row_idx, (title, img, elapsed) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            ax.axis("off")
            if col_idx == 0:
                ax.text(
                    -0.05, 0.5, title, fontsize=8, ha="right", va="center",
                    transform=ax.transAxes,
                )
            if row_idx == 0:
                ax.set_title(stem, fontsize=8)
    fig.tight_layout()
    overview_path = RESULTS_DIR / "overview_grid.png"
    fig.savefig(overview_path, dpi=120)
    plt.close(fig)
    print(f"saved {overview_path}")


if __name__ == "__main__":
    main()
