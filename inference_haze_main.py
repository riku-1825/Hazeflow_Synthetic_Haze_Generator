import os
import sys
import cv2
import csv
import time
import math
import hashlib
import argparse
import logging
from pathlib import Path

import glob
import random

import numpy as np

from tqdm import tqdm

import torch

from skimage.metrics import (
    structural_similarity,
    peak_signal_noise_ratio,
    mean_squared_error
)

# -------------------------------------------------------------
# Add Depth Anything V2 to python path
# -------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DEPTH_ANYTHING_ROOT = PROJECT_ROOT / "Depth-Anything-V2"

sys.path.append(str(DEPTH_ANYTHING_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2


# -------------------------------------------------------------
# Supported image formats
# -------------------------------------------------------------

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
)

# -------------------------------------------------------------
# Depth Anything configurations
# -------------------------------------------------------------

MODEL_CONFIGS = {

    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384]
    },

    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768]
    },

    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024]
    },

    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536]
    }

}

# -------------------------------------------------------------
# Utility
# -------------------------------------------------------------

def create_directory(path):

    """
    Creates directory if it does not exist.
    """

    os.makedirs(path, exist_ok=True)


def get_image_list(folder):

    """
    Returns sorted list of images (deduplicated so that
    case-insensitive filesystems don't match the same file
    twice via both the lowercase and uppercase extension globs).
    """

    image_set = set()

    for ext in IMAGE_EXTENSIONS:
        image_set.update(Path(folder).glob(f"*{ext}"))
        image_set.update(Path(folder).glob(f"*{ext.upper()}"))

    image_list = sorted(image_set)

    return image_list


# -------------------------------------------------------------
# Logger
# -------------------------------------------------------------

def setup_logger(log_file):

    logger = logging.getLogger("HazeFlow")

    logger.handlers.clear()

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_file)

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


# -------------------------------------------------------------
# Device
# -------------------------------------------------------------

def get_device(device_name):

    if device_name is not None:

        return torch.device(device_name)

    if torch.cuda.is_available():

        return torch.device("cuda:0")

    if torch.backends.mps.is_available():

        return torch.device("mps")

    return torch.device("cpu")


# -------------------------------------------------------------
# Load Depth Anything V2
# -------------------------------------------------------------

def load_depth_model(checkpoint_path,
                     encoder,
                     device):

    print("\nLoading Depth Anything V2...")

    model = DepthAnythingV2(
        **MODEL_CONFIGS[encoder]
    )

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    model.load_state_dict(state_dict)

    model = model.to(device)

    model.eval()

    print("Depth model loaded successfully.\n")

    return model


# -------------------------------------------------------------
# CSV Header
# -------------------------------------------------------------

CSV_HEADER = [

    "Image",

    "PSNR",

    "SSIM",

    "MSE",

    "MAE",

    "Inference_Time(sec)",

    "FPS"

]

# -------------------------------------------------------------
# Summary dictionary
# -------------------------------------------------------------

summary_metrics = {

    "PSNR": [],
    "SSIM": [],
    "MSE": [],
    "MAE": [],
    "TIME": [],
    "FPS": []

}

# -------------------------------------------------------------
# Beta Map Loader
# -------------------------------------------------------------

class BetaMapLoader:

    """
    Load all MCBM beta maps once and
    randomly sample them during inference.
    """

    def __init__(self, beta_folder):

        self.beta_paths = sorted(
            glob.glob(os.path.join(beta_folder, "*.npy"))
        )

        if len(self.beta_paths) == 0:
            raise RuntimeError(
                f"No beta maps found in {beta_folder}"
            )

        print(f"Loaded {len(self.beta_paths)} MCBM maps.")

    def sample(self, height, width):

        beta_path = random.choice(self.beta_paths)

        return self._load_and_resize(beta_path, height, width)

    def sample_fixed(self, height, width, seed):

        """
        Deterministically pick ONE MCBM map based on `seed` and
        return it resized to (height, width). Same seed -> same
        map every time, so this is safe to call independently from
        multiple shard processes (they all derive the same seed
        from the video's input_folder) and safe to call once per
        video instead of once per frame.

        Without this, sample() above picks a brand-new random
        haze pattern on every single call, i.e. every frame -- the
        spatial haze structure jumps around randomly frame-to-frame
        even though the camera barely moves, which is what causes
        visible flicker/discontinuity in the assembled video.
        """

        rng = random.Random(seed)

        beta_path = rng.choice(self.beta_paths)

        return self._load_and_resize(beta_path, height, width)

    def _load_and_resize(self, beta_path, height, width):

        beta = np.load(beta_path)

        beta = cv2.resize(
            beta,
            (width, height),
            interpolation=cv2.INTER_CUBIC
        )

        beta = beta.astype(np.float32)

        beta = np.clip(beta, 0.0, 1.0)

        return beta
    
    
# -------------------------------------------------------------
# Depth Estimation
# -------------------------------------------------------------

@torch.no_grad()
def estimate_depth(
        model,
        image_bgr,
        input_size=518,
        depth_power=1.0,
        depth_scale=1.0):

    """
    Estimate normalized depth.

    Guards against degenerate (near-constant) raw depth output.
    This happens for texture-less frames -- a close, flat patch
    of uniform grass, a blown-out sky region, a motion-blurred or
    corrupted frame, etc. In that case depth.max() == depth.min(),
    so the usual min-max normalization collapses the whole map to
    zero, which downstream means "zero optical thickness" ->
    transmission = 1.0 -> no haze is applied at all (and, for
    --uniform_haze, the background-percentile reference also comes
    out 0, with the same result).

    A texture-less frame is far more likely to *be* a flat/hazy
    region than something optically empty right against the lens,
    so on detecting near-zero dynamic range we fall back to a
    high, "far/background" depth value instead of zero, and log
    it so degenerate frames are visible/auditable.
    """

    depth = model.infer_image(
        image_bgr,
        input_size
    )

    depth = depth.astype(np.float32)

    depth_min = depth.min()
    depth_max = depth.max()
    depth_range = depth_max - depth_min

    DEGENERATE_RANGE_EPS = 1e-6

    if depth_range < DEGENERATE_RANGE_EPS:

        logging.getLogger("HazeFlow").warning(
            "Degenerate depth map detected (range=%.3e). "
            "Falling back to far/background depth for this frame "
            "instead of zero, to avoid a zero-haze output.",
            depth_range
        )

        depth = np.ones_like(depth)

    else:

        depth -= depth_min

        depth /= (depth_max - depth_min + 1e-8)

        # Depth Anything V2's relative model outputs disparity-like
        # values: CLOSER objects get HIGHER raw values, farther
        # objects get LOWER values. The haze formula below needs
        # the opposite -- optical thickness should INCREASE with
        # true distance -- so invert here. Without this, near
        # objects (e.g. foreground pavement) end up hazier than
        # the background (e.g. the mountain), which is physically
        # backwards.
        depth = 1.0 - depth

    # Reshape the depth curve. depth_power=1.0 leaves it untouched;
    # values <1.0 push midrange depths toward "farther" (thicker
    # haze sooner), values >1.0 push midrange depths toward
    # "nearer" (clearer for longer, thickening only close to the
    # true background).
    depth = np.power(depth, depth_power)

    # Scale relative (0-1) depth into the optical-thickness units
    # consumed by transmission = exp(-beta * depth). The old
    # hardcoded value of 3.5 here, combined with beta in [3, 7],
    # pushed beta*depth past 10 for most of the frame -- i.e. every
    # pixel beyond the closest ~5-10% of the scene was already
    # fully saturated to the min_transmission floor, which is why
    # the whole image looked like a flat wash instead of a real
    # near-clear/far-hazy gradient. Keep this small (paper-scale
    # depth, no artificial inflation) and control overall density
    # through beta_min/beta_max instead.
    depth = depth * depth_scale

    depth = cv2.GaussianBlur(depth, (31, 31), 0)
    

    return depth


# -------------------------------------------------------------
# Beta Map
# -------------------------------------------------------------

def generate_beta_map(
        beta_random,
        beta_min,
        beta_max):

    """
    Convert normalized MCBM map
    into atmospheric scattering coefficient.
    """

    beta = beta_min + (
        beta_random *
        (beta_max - beta_min)
    )

    return beta


# -------------------------------------------------------------
# Transmission Map
# -------------------------------------------------------------

def compute_transmission(
        depth,
        beta,
        min_transmission,
        uniform_haze=False,
        background_percentile=90,
        min_background_depth_frac=0.3):

    """
    If uniform_haze is False (default/original behaviour):
        Per-pixel transmission from the depth map -> haze density
        varies with estimated distance (can look patchy/inverted
        if the depth map is noisy near object edges).

    If uniform_haze is True:
        A single representative depth value is taken from the
        `background_percentile`-th percentile of the depth map
        (i.e. the "farthest"/background region such as the
        treeline or sky). One scalar transmission value is
        computed from that depth and beta, then broadcast to
        every pixel -> the whole frame gets exactly the same
        haze density that the background already has, with no
        depth-driven gradient and no edge-halo artifacts.

        Monocular depth models routinely collapse sky to the
        lowest value in the map (no parallax cues). If sky (or
        any other flat, near-zero region) fills more of the
        frame than `100 - background_percentile` percent, the
        chosen percentile can land inside that flat plateau and
        come out as ~0, which would silently disable haze for
        the whole frame. `min_background_depth_frac` guards
        against this: depth_ref is never allowed to fall below
        this fraction of the frame's own max depth, so a
        sky-heavy frame still gets a sensible haze reference
        instead of collapsing to zero.
    """

    if uniform_haze:

        depth_ref = np.percentile(depth, background_percentile)

        depth_floor = min_background_depth_frac * float(depth.max())

        depth_ref = max(depth_ref, depth_floor)

        beta_ref = np.mean(beta)

        transmission_value = np.exp(-beta_ref * depth_ref)

        transmission_value = float(
            np.clip(transmission_value, min_transmission, 1.0)
        )

        transmission = np.full_like(
            depth, transmission_value, dtype=np.float32
        )

    else:

        transmission = np.exp(-beta * depth)

        transmission = np.clip(
            transmission,
            min_transmission,
            1.0
        )

    return transmission
# -------------------------------------------------------------
# Prepare haze components
# -------------------------------------------------------------

def prepare_haze_components(
        model,
        image,
        beta_loader,
        beta_min,
        beta_max,
        min_transmission,
        input_size=518,
        uniform_haze=False,
        background_percentile=90,
        min_background_depth_frac=0.3,
        depth_power=1.0,
        depth_scale=1.0,
        fixed_beta_random=None):

    h, w = image.shape[:2]

    depth = estimate_depth(
        model,
        image,
        input_size,
        depth_power=depth_power,
        depth_scale=depth_scale
    )

    if fixed_beta_random is not None:

        # Reuse the SAME per-video MCBM pattern for every frame
        # instead of drawing a new random one each time (see
        # BetaMapLoader.sample_fixed for why).
        beta_random = fixed_beta_random

    else:

        beta_random = beta_loader.sample(
            h,
            w
        )

    beta = generate_beta_map(
        beta_random,
        beta_min,
        beta_max
    )

    transmission = compute_transmission(
        depth,
        beta,
        min_transmission,
        uniform_haze=uniform_haze,
        background_percentile=background_percentile,
        min_background_depth_frac=min_background_depth_frac
    )

    return depth, beta, transmission


# -------------------------------------------------------------
# Atmospheric Scattering Model
# -------------------------------------------------------------

def apply_atmospheric_scattering(
        clean_image,
        transmission,
        atmospheric_light=1.0):
    """
    Generate hazy image using

        I = J*T + A*(1-T)

    Parameters
    ----------
    clean_image : uint8 BGR image

    transmission : float32
        H x W

    atmospheric_light : float
        Scalar in [0,1]

    Returns
    -------
    hazy_image : uint8
    """

    image = clean_image.astype(np.float32) / 255.0

    transmission = transmission[..., np.newaxis]

    hazy = (
        image * transmission +
        atmospheric_light * (1.0 - transmission)
    )

    hazy = np.clip(hazy, 0.0, 1.0)

    hazy = (hazy * 255).astype(np.uint8)

    return hazy

# -------------------------------------------------------------
# Generate Hazy Image
# -------------------------------------------------------------

def generate_hazy_image(
        model,
        image,
        beta_loader,
        beta_min,
        beta_max,
        min_transmission,
        atmospheric_light,
        input_size=518,
        uniform_haze=False,
        background_percentile=90,
        min_background_depth_frac=0.3,
        depth_power=1.0,
        depth_scale=1.0,
        fixed_beta_random=None):
    """
    Complete haze generation pipeline.

    Returns
    -------
    hazy_image
    depth
    beta
    transmission
    """

    depth, beta, transmission = prepare_haze_components(
        model=model,
        image=image,
        beta_loader=beta_loader,
        beta_min=beta_min,
        beta_max=beta_max,
        min_transmission=min_transmission,
        input_size=input_size,
        uniform_haze=uniform_haze,
        background_percentile=background_percentile,
        min_background_depth_frac=min_background_depth_frac,
        depth_power=depth_power,
        depth_scale=depth_scale,
        fixed_beta_random=fixed_beta_random
    )

    hazy_image = apply_atmospheric_scattering(
        clean_image=image,
        transmission=transmission,
        atmospheric_light=atmospheric_light
    )

    return (
        hazy_image,
        depth,
        beta,
        transmission
    )
    

# -------------------------------------------------------------
# Evaluation Metrics
# -------------------------------------------------------------

def compute_metrics(clean_image, hazy_image):
    """
    Compute image quality metrics.

    Returns
    -------
    dict
    """

    clean = clean_image.astype(np.float32)
    hazy = hazy_image.astype(np.float32)

    mse = mean_squared_error(clean, hazy)

    mae = np.mean(np.abs(clean - hazy))

    psnr = peak_signal_noise_ratio(
        clean,
        hazy,
        data_range=255
    )

    ssim = structural_similarity(
        clean,
        hazy,
        channel_axis=-1,
        data_range=255
    )

    return {
        "PSNR": psnr,
        "SSIM": ssim,
        "MSE": mse,
        "MAE": mae
    }
    
# -------------------------------------------------------------
# CSV Writer
# -------------------------------------------------------------

def write_csv_row(csv_writer,
                  image_name,
                  metrics,
                  inference_time):

    fps = 1.0 / max(inference_time, 1e-8)

    csv_writer.writerow([

        image_name,

        round(metrics["PSNR"], 4),

        round(metrics["SSIM"], 6),

        round(metrics["MSE"], 4),

        round(metrics["MAE"], 4),

        round(inference_time, 4),

        round(fps, 4)

    ])

    return fps

# -------------------------------------------------------------
# Summary Statistics
# -------------------------------------------------------------

def update_summary(metrics,
                   inference_time,
                   fps):

    summary_metrics["PSNR"].append(metrics["PSNR"])

    summary_metrics["SSIM"].append(metrics["SSIM"])

    summary_metrics["MSE"].append(metrics["MSE"])

    summary_metrics["MAE"].append(metrics["MAE"])

    summary_metrics["TIME"].append(inference_time)

    summary_metrics["FPS"].append(fps)
    
# -------------------------------------------------------------
# Save Summary
# -------------------------------------------------------------

def save_summary(summary_path,
                 total_images):

    with open(summary_path, "w") as f:

        f.write("========== SUMMARY ==========\n\n")

        f.write(f"Total Images : {total_images}\n\n")
        
        if len(summary_metrics["PSNR"]) == 0:
            f.write("No valid images processed.\n")
            return

        f.write(
            f"Average PSNR : {np.mean(summary_metrics['PSNR']):.4f}\n"
        )

        f.write(
            f"Average SSIM : {np.mean(summary_metrics['SSIM']):.4f}\n"
        )

        f.write(
            f"Average MSE : {np.mean(summary_metrics['MSE']):.4f}\n"
        )

        f.write(
            f"Average MAE : {np.mean(summary_metrics['MAE']):.4f}\n\n"
        )

        f.write(
            f"Average Inference Time : {np.mean(summary_metrics['TIME']):.4f} sec\n"
        )

        f.write(
            f"Average FPS : {np.mean(summary_metrics['FPS']):.4f}\n"
        )
        
# -------------------------------------------------------------
# CSV Summary Row
# -------------------------------------------------------------

def write_csv_summary(csv_writer):

    csv_writer.writerow([])

    csv_writer.writerow([

        "AVERAGE",

        np.mean(summary_metrics["PSNR"]),

        np.mean(summary_metrics["SSIM"]),

        np.mean(summary_metrics["MSE"]),

        np.mean(summary_metrics["MAE"]),

        np.mean(summary_metrics["TIME"]),

        np.mean(summary_metrics["FPS"])

    ])
    
# -------------------------------------------------------------
# GPU Cleanup
# -------------------------------------------------------------

import gc


def cleanup_gpu():

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        torch.cuda.ipc_collect()
        
# -------------------------------------------------------------
# Dataset Processing
# -------------------------------------------------------------

def process_dataset(
        image_list,
        output_folder,
        depth_model,
        beta_loader,
        csv_writer,
        logger,
        args):

    total_time = 0

    # Pick ONE MCBM haze pattern for this entire video, deterministically
    # derived from --input_folder so that every shard process (which all
    # receive the same --input_folder) independently arrives at the exact
    # same choice, and reuse it for every frame. Without this, sample()
    # would draw a brand-new random pattern on every single frame, which
    # is what caused the haze to visibly jump/flicker between consecutive
    # frames in the assembled video even though the scene barely changes.
    video_seed = int(
        hashlib.sha256(
            str(args.input_folder).encode()
        ).hexdigest(),
        16
    ) % (2 ** 31)

    fixed_beta_random = None

    for image_path in tqdm(image_list):

        start = time.perf_counter()

        image = cv2.imread(str(image_path))

        if image is None:

            logger.error(f"Cannot read {image_path}")

            continue

        if fixed_beta_random is None:

            h, w = image.shape[:2]

            fixed_beta_random = beta_loader.sample_fixed(
                h,
                w,
                video_seed
            )

            logger.info(
                f"Using fixed MCBM haze pattern for this video "
                f"(seed={video_seed}), reused across all frames."
            )

        try:

            hazy_image, depth, beta, transmission = \
                generate_hazy_image(
                    model=depth_model,
                    image=image,
                    beta_loader=beta_loader,
                    beta_min=args.beta_min,
                    beta_max=args.beta_max,
                    min_transmission=args.min_transmission,
                    atmospheric_light=args.atmospheric_light,
                    input_size=args.input_size,
                    uniform_haze=args.uniform_haze,
                    background_percentile=args.background_percentile,
                    min_background_depth_frac=args.min_background_depth_frac,
                    depth_power=args.depth_power,
                    depth_scale=args.depth_scale,
                    fixed_beta_random=fixed_beta_random
                )

            inference_time = (
                time.perf_counter() - start
            )

            total_time += inference_time

            metrics = compute_metrics(
                image,
                hazy_image
            )

            fps = write_csv_row(
                csv_writer,
                image_path.name,
                metrics,
                inference_time
            )

            update_summary(
                metrics,
                inference_time,
                fps
            )
            
            success = cv2.imwrite(
                os.path.join(
                    output_folder,
                    image_path.name
                ),hazy_image
            )
            if not success:
                logger.error(f"Failed to save {image_path.name}")


            logger.info(
                f"{image_path.name} | "
                f"PSNR={metrics['PSNR']:.2f} "
                f"SSIM={metrics['SSIM']:.3f}"
            )

        except Exception as e:

            logger.exception(e)

        finally:

            del image

            if "hazy_image" in locals():
                del hazy_image

            if "depth" in locals():
                del depth

            if "beta" in locals():
                del beta

            if "transmission" in locals():
                del transmission

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            cleanup_gpu()
            

    return total_time


# -------------------------------------------------------------
# Argument Parser
# -------------------------------------------------------------

def build_argparser():

    parser = argparse.ArgumentParser(
        description="Synthetic haze generation using Depth Anything V2 "
                    "and MCBM beta maps, with image-quality evaluation."
    )

    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Folder containing clean input images"
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Folder where hazy images will be saved"
    )

    parser.add_argument(
        "--beta_folder",
        type=str,
        required=True,
        help="Folder containing MCBM beta map .npy files"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Depth Anything V2 checkpoint (.pth)"
    )

    parser.add_argument(
        "--encoder",
        type=str,
        default="vitb",
        choices=list(MODEL_CONFIGS.keys()),
        help="Depth Anything V2 encoder variant"
    )

    parser.add_argument(
        "--input_size",
        type=int,
        default=518,
        help="Input size for depth model inference"
    )

    parser.add_argument(
        "--beta_min",
        type=float,
        default=0.2,
        help="Minimum atmospheric scattering coefficient. The "
             "HazeFlow paper (Sec. 5.1) samples beta uniformly "
             "from [0.2, 2.8]; the previous default of 3.0 was "
             "well above that range and, combined with the old "
             "depth scaling, saturated almost the entire frame to "
             "the min_transmission floor (flat whiteout instead of "
             "a near-clear/far-hazy gradient)."
    )

    parser.add_argument(
        "--beta_max",
        type=float,
        default=2.8,
        help="Maximum atmospheric scattering coefficient. See "
             "--beta_min; matches the HazeFlow paper's [0.2, 2.8] "
             "range (previous default was 7.0)."
    )

    parser.add_argument(
        "--depth_power",
        type=float,
        default=1.0,
        help="Exponent applied to the normalized (0-1) depth map "
             "before scaling. 1.0 leaves it linear. Values <1.0 "
             "push midrange depths toward 'farther' (haze thickens "
             "sooner); values >1.0 push midrange depths toward "
             "'nearer' (stays clearer longer, only thickening near "
             "the true background). The previous hardcoded value "
             "was 0.8, which pushed haze onto the midground."
    )

    parser.add_argument(
        "--depth_scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the normalized (0-1) depth map "
             "to convert it into the optical-thickness units used "
             "by transmission = exp(-beta * depth). The previous "
             "hardcoded value was 3.5, which combined with the old "
             "beta range (3-7) caused beta*depth to exceed 10 for "
             "most of the frame, saturating almost everything to "
             "the min_transmission floor. Keep this near 1.0 and "
             "control overall haze density via --beta_min/--beta_max."
    )

    parser.add_argument(
        "--atmospheric_light",
        type=float,
        default=1.0,
        help="Atmospheric light A in the range [0,1]"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on, e.g. 'cuda:0', 'mps', 'cpu'. "
             "Defaults to auto-detection."
    )

    parser.add_argument(
        "--csv_file",
        type=str,
        default="metrics.csv",
        help="Filename (within output_folder) for the per-image metrics CSV"
    )

    parser.add_argument(
        "--summary_file",
        type=str,
        default="summary.txt",
        help="Filename (within output_folder) for the summary text report"
    )

    parser.add_argument(
        "--log_file",
        type=str,
        default="haze_flow.log",
        help="Filename (within output_folder) for the run log"
    )
    
    parser.add_argument(
    "--min_transmission",
    type=float,
    default=0.25,
    help="Minimum transmission value"
)

    parser.add_argument(
        "--uniform_haze",
        action="store_true",
        help="If set, ignore per-pixel depth variation and apply one "
             "flat transmission value to the whole image, taken from "
             "the background_percentile of the depth map. Produces "
             "spatially uniform haze density instead of a depth-driven "
             "gradient (and avoids edge-halo artifacts at object "
             "boundaries such as the treeline)."
    )

    parser.add_argument(
        "--background_percentile",
        type=float,
        default=90.0,
        help="Percentile of the per-pixel depth map used as the "
             "'background' reference depth when --uniform_haze is set. "
             "Higher values (closer to 100) pick out the farthest "
             "region (e.g. sky/treeline); lower values pick a nearer "
             "region. Only used with --uniform_haze."
    )

    parser.add_argument(
        "--min_background_depth_frac",
        type=float,
        default=0.3,
        help="Floor on the uniform-haze background depth reference, "
             "as a fraction of the frame's own max depth. Prevents "
             "sky-heavy frames (where a flat, near-zero-depth sky "
             "region can dominate the percentile) from collapsing "
             "the reference depth to ~0, which would silently apply "
             "no haze. Only used with --uniform_haze. Set to 0 to "
             "disable the floor and match the previous behaviour."
    )

    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Split the image list into this many shards, so multiple "
             "GPU processes can each handle a slice of the dataset. "
             "Run one process per shard with a different --shard_id "
             "and --device to use multiple GPUs concurrently."
    )

    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="Which shard (0-indexed, < num_shards) this process handles."
    )

    return parser


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main():

    parser = build_argparser()
    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        raise FileNotFoundError(args.input_folder)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    if not os.path.isdir(args.beta_folder):
        raise FileNotFoundError(args.beta_folder)

    create_directory(args.output_folder)

    if args.num_shards > 1:
        stem, ext = os.path.splitext(args.log_file)
        log_filename = f"{stem}_shard{args.shard_id}{ext}"
    else:
        log_filename = args.log_file

    log_path = os.path.join(args.output_folder, log_filename)
    logger = setup_logger(log_path)

    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    depth_model = load_depth_model(
        checkpoint_path=args.checkpoint,
        encoder=args.encoder,
        device=device
    )

    torch.set_grad_enabled(False)

    beta_loader = BetaMapLoader(args.beta_folder)

    image_list = get_image_list(args.input_folder)

    if len(image_list) == 0:
        logger.error(f"No images found in {args.input_folder}")
        sys.exit(1)

    logger.info(f"Found {len(image_list)} images to process.")

    if args.num_shards > 1:

        if not (0 <= args.shard_id < args.num_shards):
            raise ValueError(
                f"--shard_id must be in [0, {args.num_shards - 1}]"
            )

        image_list = image_list[args.shard_id::args.num_shards]

        logger.info(
            f"Shard {args.shard_id}/{args.num_shards}: "
            f"processing {len(image_list)} images on {device}."
        )

    def shard_suffix(filename):

        if args.num_shards <= 1:
            return filename

        stem, ext = os.path.splitext(filename)
        return f"{stem}_shard{args.shard_id}{ext}"

    csv_path = os.path.join(
        args.output_folder, shard_suffix(args.csv_file)
    )
    summary_path = os.path.join(
        args.output_folder, shard_suffix(args.summary_file)
    )

    with open(csv_path, "w", newline="") as csv_file:

        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(CSV_HEADER)

        overall_start = time.perf_counter()

        total_time = process_dataset(
            image_list=image_list,
            output_folder=args.output_folder,
            depth_model=depth_model,
            beta_loader=beta_loader,
            csv_writer=csv_writer,
            logger=logger,
            args=args
        )

        overall_time = time.perf_counter() - overall_start

        write_csv_summary(csv_writer)

    save_summary(summary_path, len(image_list))

    logger.info(
        f"Processed {len(image_list)} images."
    )
    logger.info(f"Inference time only : {total_time:.2f} sec")
    logger.info(f"Total execution time: {overall_time:.2f} sec")

    print(
        f"\nAll done in {overall_time:.2f} sec. "
        f"Results saved to: {args.output_folder}"
    )


if __name__ == "__main__":
    main()