# Synthetic Data Lab - ADAS Use Cases

This project uses diffusion models to generate synthetic images for Advanced Driver Assistance Systems (ADAS) testing and development.

## Overview

The project generates realistic road scene images using Stable Diffusion v1.5. These synthetic images can be used to create training data or test scenarios for ADAS systems without the need for expensive real-world data collection.

## What It Does

The notebook demonstrates how to generate photorealistic images of road scenarios, such as obstacles on highways, using text prompts. The generated images are saved for use in ADAS development and testing.

## Requirements

- Python 3.12
- CUDA-capable GPU (recommended)
- Required packages:
  - diffusers
  - transformers
  - accelerate
  - torch

## Usage

1. Install dependencies:
```bash
pip install diffusers transformers accelerate torch
```

2. Open the Jupyter notebook `Diffusion_Models_for_ADAS_Use_Cases.ipynb`

3. Run the cells to generate synthetic road scene images

4. Generated images are saved in the `result/` directory

## Model

The project uses the `runwayml/stable-diffusion-v1-5` model from Hugging Face, running in float16 precision on CUDA for faster generation.

## Output

Generated images are saved as PNG files and can be used for ADAS system training, testing, or validation purposes.
