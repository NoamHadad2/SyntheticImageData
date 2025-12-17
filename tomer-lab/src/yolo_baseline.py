from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMG_DIR = (
    ROOT / "data" / "subset_100" / "aug" / "aug1_animals" / "comb_for_detaction"
)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Images folder not found: {folder}")
    imgs = sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if not imgs:
        raise RuntimeError(f"No images found in {folder}")
    return imgs


def main() -> None:
    model = YOLO(ROOT / "yolov8s.pt")

    images = list_images(DEFAULT_IMG_DIR)
    print(f"Found {len(images)} images in {DEFAULT_IMG_DIR}")

    animal_classes = {
        "dog",
        "horse",
        "deer",
        "wolf",
        "cat"
    }

    results = model(images, conf=0.25)
    rows: list[dict] = []

    images_with_animals = 0
    all_confidences: list[float] = []
    detections_per_image: list[int] = []

    for img_path, r in zip(images, results):
        animal_confs: list[float] = []
        for box in r.boxes:
            cls_name = model.names[int(box.cls[0])]
            if cls_name in animal_classes:
                conf = float(box.conf[0])
                animal_confs.append(conf)
                all_confidences.append(conf)

        has_animal = len(animal_confs) > 0
        if has_animal:
            images_with_animals += 1

        detections_per_image.append(len(animal_confs))

        rows.append({
            "image_name": img_path.name,
            "has_animal": has_animal,
            "num_animals": len(animal_confs),
            "avg_animal_conf": float(np.mean(animal_confs)) if animal_confs else np.nan,
        })

    detection_rate = images_with_animals / len(images)
    avg_confidence = float(np.mean(all_confidences)) if all_confidences else 0.0
    avg_detections = float(np.mean(detections_per_image)) if detections_per_image else 0.0

    df = pd.DataFrame(rows)

    print("\n=== DataFrame Summary ===")
    print(df.head())

    print("\nImages with animals:", df["has_animal"].sum())
    print("Images without animals:", (~df["has_animal"]).sum())

    print("\n=== Baseline Metrics ===")
    print(f"Detection rate: {detection_rate:.3f}")
    print(f"Average confidence: {avg_confidence:.3f}")
    print(f"Avg detections per image: {avg_detections:.2f}")


if __name__ == "__main__":
    main()
