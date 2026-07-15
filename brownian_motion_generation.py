
"""
brownian_motion_generation_1.py

Improved MCBM generator for HazeFlow.

Features
--------
- Configurable image size
- Configurable number of maps
- Optional reproducible seed
- Saves PNG and/or NPY
- Importable functions for inference
"""

import argparse
import logging
from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


def setup_logger(out_dir: Path):
    logger = logging.getLogger("MCBM")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(out_dir / "generation.log")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    return logger


def generate_mcbm_map(height=518,
                      width=518,
                      multiple_iter=None,
                      sigma=None,
                      seed=None):
    """
    Returns a normalized float32 beta map in [0,1].
    """

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if multiple_iter is None:
        multiple_iter = random.choice([5, 6, 7])

    if sigma is None:
        sigma = random.choice([15, 25, 35])

    Z = np.zeros((height, width), dtype=np.float32)

    current_row = np.random.randint(0, height)
    current_col = np.random.randint(0, width)

    Z[current_row, current_col] = 1.0

    num_points = height * width * multiple_iter

    increments = np.random.normal(
        loc=0.0,
        scale=1.0,
        size=(num_points, 2)
    )

    for i in range(num_points):

        r = np.random.rand()

        if r < 0.25 and current_row > 0:
            current_row -= 1
        elif r < 0.50 and current_row < height - 1:
            current_row += 1
        elif r < 0.75 and current_col > 0:
            current_col -= 1
        elif current_col < width - 1:
            current_col += 1

        Z[current_row, current_col] += 1

        current_row = (current_row + int(increments[i, 0])) % height
        current_col = (current_col + int(increments[i, 1])) % width

        Z[current_row, current_col] += 1

    Z_smoothed = gaussian_filter(Z, sigma=sigma)

    Z_combined = sigma * Z_smoothed + Z

    beta = (Z_combined - Z_combined.min()) / (
        Z_combined.max() - Z_combined.min() + 1e-8
    )

    return beta.astype(np.float32)


def save_beta_map(beta, png_path=None, npy_path=None):

    if png_path is not None:
        img = Image.fromarray((beta * 255).astype(np.uint8))
        img.save(png_path)

    if npy_path is not None:
        np.save(npy_path, beta)


def generate_dataset(args):

    out_dir = Path(args.output)

    png_dir = out_dir / "png"
    npy_dir = out_dir / "npy"

    if args.save_png:
        png_dir.mkdir(parents=True, exist_ok=True)

    if args.save_npy:
        npy_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir)

    logger.info("Starting MCBM generation")
    logger.info("Maps: %d", args.num_maps)

    for idx in tqdm(range(args.num_maps), desc="Generating MCBM"):

        seed = None if args.random_seed else idx

        beta = generate_mcbm_map(
            height=args.height,
            width=args.width,
            multiple_iter=None,
            sigma=None,
            seed=seed
        )

        png_file = png_dir / f"{idx:06d}.png" if args.save_png else None
        npy_file = npy_dir / f"{idx:06d}.npy" if args.save_npy else None

        save_beta_map(beta, png_file, npy_file)

    logger.info("Finished successfully.")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_maps", type=int, default=1000)
    parser.add_argument("--height", type=int, default=518)
    parser.add_argument("--width", type=int, default=518)

    parser.add_argument(
        "--output",
        type=str,
        default="../datasets/MCBM"
    )

    parser.add_argument(
        "--random_seed",
        action="store_true",
        help="Use random seed instead of deterministic seeds."
    )

    parser.add_argument(
        "--save_png",
        action="store_true",
        default=True
    )

    parser.add_argument(
        "--save_npy",
        action="store_true",
        default=True
    )

    args = parser.parse_args()

    generate_dataset(args)


if __name__ == "__main__":
    main()
