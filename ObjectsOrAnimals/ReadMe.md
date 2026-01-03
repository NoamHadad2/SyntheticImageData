# ObjectsOrAnimals - SyntheticImageData

## Motivation
Real-world driving datasets rarely cover enough animal/object appearances, poses, and scene conditions. Manual collection and labeling is expensive. This project generates synthetic, label-ready samples to expand coverage and stress-test object detection models.

## Problem Definition
Object detection models (YOLO) can overfit to limited real distributions and fail under domain shift (new backgrounds, rare poses, scale changes, occlusions). We aim to generate realistic insertions with consistent geometry and clean placement regions, then compare baseline vs fine-tuned detectors.

## Models
- Segmentation (mask generation): **SAM 3** (Ultralytics `SAM3SemanticPredictor`) for road-area masks via text prompt (default: "road").
- Depth estimation: **MiDaS / DPT** depth-estimation pipeline to compute normalized depth and guide placement scale.
- Image generation: **SDXL Inpainting** (Diffusers `AutoPipelineForInpainting`) to synthesize animals/objects in selected regions.
- Detection: **YOLOv8** (Ultralytics) for baseline evaluation and fine-tuning, reporting Precision/Recall/F1 and mAP metrics.

## Process
1. **Road mask creation**
   - Run SAM3 per image with prompt "road"
   - Select the largest mask and save as a binary PNG (0/255)
2. **Depth-aware placement**
   - Estimate a depth map for each image
   - Use depth at candidate points to choose a realistic insertion size (farther = smaller)
3. **SDXL inpainting**
   - Create an elliptical spot mask inside valid road regions
   - Inpaint on a cropped area around the spot for stability and speed
4. **Dataset and evaluation**
   - Run a YOLOv8 pretrained baseline
   - Fine-tune YOLOv8 on the prepared data and evaluate on test split

## Results (what is produced)
- Example **road masks** and **generated synthetic images** are saved under `result/`
- The YOLO comparison notebook outputs quantitative metrics (Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95) and plots

## Repository Contents

### Notebooks (`notebooks/`)
- `make_dynamic_road_masks.ipynb` - Generate road masks using SAM3.
- `animals_synthetic_generation.ipynb` - Synthetic generation (base version).
- `animals_synthetic_generation-stage2.ipynb` - Stage 2 generation with masks + depth-aware spot sizing + SDXL inpainting.
- `yolo_baseline.ipynb` - YOLOv8 pretrained baseline evaluation on the dataset.
- `Baseline vs Finetune-stage2.ipynb` - Fine-tune YOLOv8 and compare to baseline with metrics and plots.

### Scripts (`src/`)
- `make_dynamic_road_masks.py` - Script version of SAM3 road mask generation (uses `HF_TOKEN`).
- `prepare_dataset.py` - Converts KITTI-style annotations to YOLO format (subset preparation utility).

### Outputs (`result/`)
- `road_masks/` - Example generated masks (`*_mask.png`).
- `animals_synthetic_images/` - Example synthetic outputs (named by image id + prompt).
- `stage2/` - Example stage2 outputs (include class/view and depth factor in filename).

