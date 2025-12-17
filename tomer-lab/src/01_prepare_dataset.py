from __future__ import annotations

from pathlib import Path
from shutil import copy2
from PIL import Image


# -----------------------------
# CONFIG
# -----------------------------
CLASS_MAP = {
    "Car": 0,
    "Pedestrian": 1,
    "Cyclist": 2,
}

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]


# -----------------------------
# PATHS (robust project root)
# -----------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # Autonomous_Project/
DATA_DIR = PROJECT_ROOT / "data"

SUBSET_DIR = DATA_DIR / "subset_100"
IN_IMAGES = SUBSET_DIR / "images"
IN_ANNS = SUBSET_DIR / "annotations"

OUT_DIR = DATA_DIR / "prepared" / "subset_100_yolo"
OUT_IMAGES = OUT_DIR / "images"
OUT_LABELS = OUT_DIR / "labels"


def find_image_for_base(base: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = IN_IMAGES / f"{base}{ext}"
        if p.exists():
            return p
    return None


def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def kitti_line_to_yolo(line: str, img_w: int, img_h: int):
    parts = line.strip().split()
    if not parts:
        return None

    cls = parts[0]
    if cls == "DontCare":
        return None
    if cls not in CLASS_MAP:
        return None

    # KITTI: cls trunc occl alpha left top right bottom ...
    left, top, right, bottom = map(float, parts[4:8])

    # clip to image bounds
    left = clip(left, 0.0, img_w - 1.0)
    right = clip(right, 0.0, img_w - 1.0)
    top = clip(top, 0.0, img_h - 1.0)
    bottom = clip(bottom, 0.0, img_h - 1.0)

    # validate
    if right <= left or bottom <= top:
        return None

    # YOLO normalized
    x_c = ((left + right) / 2.0) / img_w
    y_c = ((top + bottom) / 2.0) / img_h
    w = (right - left) / img_w
    h = (bottom - top) / img_h

    # final safety clamp (rare numeric edge)
    x_c = clip(x_c, 0.0, 1.0)
    y_c = clip(y_c, 0.0, 1.0)
    w = clip(w, 0.0, 1.0)
    h = clip(h, 0.0, 1.0)

    return CLASS_MAP[cls], x_c, y_c, w, h


def convert_one(base: str) -> bool:
    ann_path = IN_ANNS / f"{base}.txt"
    if not ann_path.exists():
        return False

    img_path = find_image_for_base(base)
    if img_path is None:
        return False

    img = Image.open(img_path)
    img_w, img_h = img.size

    yolo_lines = []
    for line in ann_path.read_text().splitlines():
        out = kitti_line_to_yolo(line, img_w, img_h)
        if out is None:
            continue
        cid, x_c, y_c, w, h = out
        yolo_lines.append(f"{cid} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    # write outputs
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.mkdir(parents=True, exist_ok=True)

    copy2(img_path, OUT_IMAGES / img_path.name)
    (OUT_LABELS / f"{base}.txt").write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

    return True


def main():
    if not IN_IMAGES.exists() or not IN_ANNS.exists():
        raise FileNotFoundError(
            "subset_100 לא בנוי כמו שצריך.\n"
            f"צריך להיות:\n- {IN_IMAGES}\n- {IN_ANNS}"
        )

    ann_files = sorted(IN_ANNS.glob("*.txt"))
    bases = [p.stem for p in ann_files]

    ok = 0
    missing_img = []
    for base in bases:
        img_path = find_image_for_base(base)
        if img_path is None:
            missing_img.append(base)
            continue
        if convert_one(base):
            ok += 1

    print("=== KITTI -> YOLO conversion ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input images: {IN_IMAGES}")
    print(f"Input anns:   {IN_ANNS}")
    print(f"Output images: {OUT_IMAGES}")
    print(f"Output labels: {OUT_LABELS}")
    print()
    print(f"Converted pairs: {ok} / {len(bases)}")

    if missing_img:
        print(f"Missing images for {len(missing_img)} annotations. Examples:")
        for x in missing_img[:20]:
            print(f"- {x}")

    print()
    print("Classes:")
    for k, v in CLASS_MAP.items():
        print(f"- {k} -> {v}")


if __name__ == "__main__":
    main()
