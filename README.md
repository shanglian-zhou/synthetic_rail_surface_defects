# Adaptive MCMC-Guided Synthetic Data Generation for Rail Surface Defect Segmentation

This repository contains the **source code and released data** accompanying the manuscript:

**“Markov Chain Monte Carlo-driven Exploration and Refinement for CGAN-based Synthetic Image Generation for Rail Surface Defect Segmentation.”**

The repository is organized to support **method verification, benchmarking, and reproducibility**, with a focus on reviewer access and experimental transparency.

---

## Repository Contents

### 1. Adaptive MCMC Sampling (MATLAB)
**`scripts/adaptive_mcmc_sampling_with_CGAN/`**

- MATLAB implementation of the proposed **adaptive MCMC sampling framework**
- Refines CGAN-generated image–label pairs through:
  - Discriminator-informed Metropolis–Hastings acceptance
  - IoU-based geometric regularization
  - Adaptive label-space perturbation
  - Skewness correction for stabilized sampling
- GPU-compatible via MATLAB Deep Learning Toolbox

This module corresponds to the **core proposed method** in the manuscript.

---

### 2. Generative Model Benchmarks
**`scripts/Stable_diffusion+ControlNet/`**  
**`scripts/`**

Training and inference scripts for baseline synthetic data generation methods:
- Stable Diffusion v1.5 + ControlNet  
- Stable Diffusion v2.1 + ControlNet  
- Vanilla CVAE  
- U-Net-based CVAE  

These methods serve as benchmarks for comparison with the proposed adaptive MCMC–CGAN framework.

---

### 3. Segmentation Model Benchmarks
**`scripts/`**

Training and evaluation scripts for six segmentation models:
- **CNN-based**: U-Net, FPN, LinkNet  
- **ViT-based**: SegFormer, MaskFormer, Mask2Former  

All segmentation models are trained and evaluated under identical data volume and augmentation settings to ensure fair comparison.

---

### 4. Released Synthetic Dataset
**`releases/`**

- **12,000 synthetic image–label pairs**
- Generated and refined using the proposed adaptive MCMC + CGAN framework
- Provided to support reproducibility and future research

Large data files are hosted via GitHub Releases for reviewer access.

---

## Requirements (Summary)

- MATLAB (R2023b or later recommended)  
  - Deep Learning Toolbox  
  - Image Processing Toolbox  
- Python environments for benchmark generative and segmentation models  
  (see subfolder documentation in **`env_settings/`**)

---
For detailed implementation and experimental settings, please refer to the manuscript sections cited above.

## Training
[Environment configuration](https://github.com/shanglian-zhou/synthetic_rail_surface_defects/tree/main/env_settings/) and [training scripts](https://github.com/shanglian-zhou/synthetic_rail_surface_defects/tree/main/scripts/) can facilitate reproducibility of the results.

For detailed experimental settings and evaluation protocols, please refer to the manuscript.
