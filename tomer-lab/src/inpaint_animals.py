from pathlib import Path
from shutil import copy2
import random

import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline


# Project root (בהנחה שהקובץ נמצא ב-src/)
ROOT = Path(__file__).resolve().parents[1]

# Inputs
IN_IMG  = ROOT / "data" / "base" / "images"
IN_MASK = ROOT / "data" / "base" / "hazard_masks"
IN_LBL  = ROOT / "data" / "base" / "hazard_labels"

# Outputs (aug1_animals)
OUT_DIR = ROOT / "data" / "aug1_animals"
OUT_IMG = OUT_DIR / "images"
OUT_LBL = OUT_DIR / "labels"
OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

# Basic config
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LIMIT = 5  # להרצה ראשונה לדיבאג
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"

ANIMALS = ["deer", "dog", "cat", "fox", "boar"]
PROMPT_TPL = "realistic dashcam photo, {animal} on the road, daylight, sharp focus"
NEG_PROMPT = "cartoon, illustration, lowres, blurry, text, watermark"

LIMIT = 5  # תתחיל 1-5, ואז תעלה ל-100

# ----------------------------
# Helpers
# ----------------------------
def list_images(folder: Path, exts: set[str]) -> list[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])

def get_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"

def load_pipe(device: str) -> StableDiffusionInpaintPipeline:
    # על Mac לפעמים float16 עושה בעיות, אז מתחילים בטוח עם float32
    dtype = torch.float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    pipe = pipe.to(device)

    # חיסכון בזיכרון
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    return pipe

# ----------------------------
# Main
# ----------------------------
def main():
    print("ROOT:", ROOT)
    print("exists images:", IN_IMG.exists())
    print("exists masks:", IN_MASK.exists())
    print("exists labels:", IN_LBL.exists())

    imgs = list_images(IN_IMG, EXTS)
    print("found images:", len(imgs))
    for p in imgs[:3]:
        print(" -", p.name)

    device = get_device()
    print("device:", device)

    pipe = load_pipe(device)

    n = min(LIMIT, len(imgs))
    done = 0

    for img_path in imgs[:n]:
        stem = img_path.stem  # 000004

        mask_path = IN_MASK / f"{stem}_mask.png"
        label_path = IN_LBL / f"{stem}.txt"

        if not mask_path.exists():
            print("SKIP missing mask:", mask_path.name)
            continue
        if not label_path.exists():
            print("SKIP missing label:", label_path.name)
            continue

        init_image = Image.open(img_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("L")

        if mask_image.size != init_image.size:
            mask_image = mask_image.resize(init_image.size)

        animal = random.choice(ANIMALS)
        prompt = PROMPT_TPL.format(animal=animal)

        result = pipe(
            prompt=prompt,
            negative_prompt=NEG_PROMPT,
            image=init_image,
            mask_image=mask_image,
            guidance_scale=7.5,
            num_inference_steps=30,
        ).images[0]

        out_img_path = OUT_IMG / img_path.name
        result.save(out_img_path)

        copy2(label_path, OUT_LBL / label_path.name)

        done += 1
        print(f"[{done}] saved:", out_img_path.name)

    print("done:", done, "images")

if __name__ == "__main__":
    main()