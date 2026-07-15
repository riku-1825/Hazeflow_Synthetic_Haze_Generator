![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

# Hazeflow Synthetic Haze Generator

A synthetic haze generation framework built upon **HazeFlow**, integrating **Depth Anything V2** for monocular depth estimation and **Markov Chain Brownian Motion (MCBM)** based beta map generation. The framework generates realistic hazy images from clean RGB images using the Atmospheric Scattering Model and provides automatic evaluation metrics for large-scale dataset generation.

---

## Features

- Depth Anything V2 based monocular depth estimation
- Markov Chain Brownian Motion (MCBM) beta map generation
- Physics-based Atmospheric Scattering Model (ASM)
- Uniform and configurable haze generation
- Multi-GPU inference support
- Automatic GPU memory management
- Image quality evaluation
- CSV and log generation
- Large-scale dataset processing

---

# Installation

Clone the repository.

```bash
git clone https://github.com/riku-1825/Hazeflow_Synthetic_Haze_Generator.git
cd Hazeflow_Synthetic_Haze_Generator
```

Create a conda environment.

```bash
conda env create -f environment.yaml
conda activate hazeflow
```
---

# Brownian Motion (MCBM) Beta Map Generation

This repository uses **Markov Chain Brownian Motion (MCBM)** to generate spatially varying atmospheric scattering coefficient maps.

Generate beta maps using

```bash
python haze_generation/brownian_motion_generation.py
```

The generated beta maps will be stored inside

```
datasets/
    └── MCBM/
        └── npy/
```

These beta maps are randomly sampled during inference to create diverse haze patterns.

---

# Depth Anything V2

Depth estimation is performed using **Depth Anything V2**.

Official Repository:

https://github.com/DepthAnything/Depth-Anything-V2

Please download the required checkpoint before running inference.

Recommended model

```
depth_anything_v2_vitb.pth
```

Place the checkpoint inside the project directory.

Example:

```
Hazeflow_Synthetic_Haze_Generator/
│
├── depth_anything_v2_vitb.pth
```

---
## System Configuration

All experiments were conducted on the following hardware and software configuration.

| Component | Specification |
|:----------|:--------------|
| **Operating System** | Ubuntu Linux |
| **CPU** | Intel® Xeon® CPU E5-2640 v4 @ 2.40 GHz |
| **CPU Architecture** | x86_64 |
| **GPUs** | 3 × NVIDIA GeForce RTX 2080 Ti |
| **GPU Memory** | 11 GB GDDR6 per GPU (33 GB total) |
| **CUDA Version** | CUDA 13.0 |
| **NVIDIA Driver** | 580.159.03 |
| **Deep Learning Framework** | PyTorch |
| **Depth Estimation Model** | Depth Anything V2 (ViT-B) |
| **Haze Generation Model** | Atmospheric Scattering Model (ASM) with MCBM β maps |
| **Parallel Processing** | Multi-GPU Dataset Sharding |

### GPU Configuration

| GPU | Model | Memory |
|:---:|:------|:------:|
| GPU 0 | NVIDIA GeForce RTX 2080 Ti | 11 GB |
| GPU 1 | NVIDIA GeForce RTX 2080 Ti | 11 GB |
| GPU 2 | NVIDIA GeForce RTX 2080 Ti | 11 GB |

The framework supports **parallel multi-GPU inference** by partitioning the input dataset into independent shards. Each shard is processed concurrently on a separate GPU, significantly reducing the overall processing time for large-scale synthetic haze generation.

## Dataset

This project uses the **DUT Anti-UAV** dataset for generating synthetic hazy images. The dataset provides drone-related visual data captured in outdoor environments with complex backgrounds, making it suitable for evaluating haze synthesis under realistic conditions.

Repository: https://github.com/wangdongdut/DUT-Anti-UAV

Example

```
Dataset/
    ├── frame_000001.png
    ├── frame_000002.png
    ├── frame_000003.png
    └── ...
```

Supported formats

- png
- jpg
- jpeg
- bmp
- tif
- tiff

No depth maps or transmission maps are required.

Depth maps are estimated automatically during inference using Depth Anything V2.

---

# Haze Generation

The main inference script is

```
inference_haze_main.py
```

Example

```bash
python inference_haze_main.py \
    --input_folder path/to/input \
    --output_folder path/to/output \
    --beta_folder datasets/MCBM/npy \
    --checkpoint depth_anything_v2_vitb.pth
```

For multi-GPU processing, use

```
hazing_main.sh
```

Example

```bash
chmod +x hazing_main.sh
./hazing_main.sh
```

The shell script automatically

- splits the dataset into multiple shards
- assigns each shard to a different GPU
- runs inference in parallel
- merges the generated outputs

This significantly reduces the processing time for large datasets.

---

# Evaluation Metrics

For every generated image the framework computes

- PSNR
- SSIM
- MSE
- MAE
- Inference Time
- FPS

Outputs include

```
output/
│
├── generated_images/
├── metrics.csv
├── summary.txt
└── haze_flow.log
```

The CSV file stores per-image metrics, while the summary file reports the average values across the entire dataset.

---

# Results

| Clean Image | Generated Hazy Image |
|-------------|----------------------|
| <img src="https://github.com/user-attachments/assets/da2a2064-59f6-4aa9-8b85-896cc0aae94d" width="500"> | <img src="https://github.com/user-attachments/assets/645be9f6-3e53-4d79-8d78-694465dfae72" width="500">|

The generated haze is produced using

- Depth Anything V2
- MCBM beta maps
- Atmospheric Scattering Model

The framework supports configurable

- haze density
- atmospheric light
- transmission threshold
- background-aware haze generation
  
---

## Quantitative Results

The framework was evaluated on the generated synthetic haze dataset using standard full-reference image quality metrics. The average results are reported below.

| Metric | Average Value |
|:-------|--------------:|
| **PSNR (dB)** | **13.0790** |
| **SSIM** | **0.8396** |
| **MSE** | **3209.8389** |
| **MAE** | **45.3691** |
| **Inference Time (sec/image)** | **0.2628** |
| **FPS** | **4.3408** |

# Repository Structure

```
Hazeflow_Synthetic_Haze_Generator/
│
├── datasets/
├── haze_generation/
├── Depth-Anything-V2/
├── inference_haze_main.py
├── hazing_main.sh
├── requirements.txt
├── environment.yaml
└── README.md
```

---

# Acknowledgements

This repository builds upon the excellent work of the following projects.

## HazeFlow

> HazeFlow: Realistic Synthetic Haze Generation and Dehazing via Rectified Flow.

Official Repository

https://github.com/cloor/HazeFlow

---

## Depth Anything V2

> Depth Anything V2: Monocular Depth Estimation Foundation Model.

Official Repository

https://github.com/DepthAnything/Depth-Anything-V2

---

### DUT Anti-UAV Dataset

The clean image sequences used in this work were obtained from the DUT Anti-UAV dataset.

Repository: https://github.com/wangdongdut/DUT-Anti-UAV

---

If you use this repository in your research, please also cite the original HazeFlow and Depth Anything V2 papers.

---

# License

This project follows the license terms of the original HazeFlow repository. Please refer to the LICENSE file for details.
