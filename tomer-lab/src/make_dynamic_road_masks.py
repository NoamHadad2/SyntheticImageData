from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
from huggingface_hub import login, hf_hub_download
from ultralytics.models.sam import SAM3SemanticPredictor


# ----------------------------
# Settings
# ----------------------------
SEED = 42
PROMPTS = ["road"]  # שנה ל-["dog"] / ["deer"] וכו'
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Project paths (script-friendly, notebook-safe)
try:
    ROOT = Path(__file__).resolve().parents[1]   # .../Autonomous_Project
except NameError:
    ROOT = Path.cwd()

IMAGES_DIR = ROOT / "data" / "subset_100" / "images"
MASKS_DIR  = ROOT / "data" / "subset_100" / "masks_sam3_road"
MASKS_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Reproducibility
# ----------------------------
random.seed(SEED)
np.random.seed(SEED)

print("HF_TOKEN exists?", bool(os.environ.get("HF_TOKEN")))

# ----------------------------
# Hugging Face login (token from env)
# ----------------------------
def hf_login() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set. Run: export HF_TOKEN='...'\n")
    # no interactive prompt, no git-credential saving
    login(token=token, add_to_git_credential=False)


# ----------------------------
# Download SAM3 weights (uses HF cache)
# ----------------------------
def get_sam3_weights() -> str:
    token = os.environ.get("HF_TOKEN")
    return hf_hub_download(
        repo_id="facebook/sam3",
        filename="sam3.pt",
        token=token,  # explicit token keeps it simple
    )


# ----------------------------
# Mask selection: largest area
# ----------------------------
def pick_largest_mask(masks_bool: np.ndarray) -> np.ndarray:
    # masks_bool: (N,H,W) bool
    areas = masks_bool.reshape(masks_bool.shape[0], -1).sum(axis=1)
    return masks_bool[int(areas.argmax())]


def main() -> None:
    assert IMAGES_DIR.exists(), f"Images folder not found: {IMAGES_DIR}"

    hf_login()
    sam3_path = get_sam3_weights()
    print("SAM3 weights:", sam3_path)
    print("Images dir:", IMAGES_DIR)
    print("Masks dir:", MASKS_DIR)

    sam = SAM3SemanticPredictor(
        overrides=dict(
            model=sam3_path,
            task="segment",
            mode="predict",
            verbose=False,
        )
    )

    img_paths = [p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in EXTS]
    print(f"Found {len(img_paths)} images")

    for img_path in img_paths:
        out_path = MASKS_DIR / f"{img_path.stem}_mask.png"
        if out_path.exists():
            continue

        sam.set_image(str(img_path))
        results = sam(text=PROMPTS)

        if not results or results[0].masks is None:
            print(f"[SKIP] no masks: {img_path.name}")
            continue

        masks = results[0].masks.data.detach().cpu().numpy().astype(bool)  # (N,H,W)
        if masks.shape[0] == 0:
            print(f"[SKIP] empty masks: {img_path.name}")
            continue

        best_mask = pick_largest_mask(masks)          # bool HxW
        mask_255 = best_mask.astype(np.uint8) * 255   # uint8 0/255

        Image.fromarray(mask_255).save(out_path)
        print(f"[OK] {img_path.name} -> {out_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()
