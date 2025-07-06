#!/usr/bin/env python3
"""
auto_train_yolov11n.py
-------------------------------------------------
Run this file and it will:

  • Locate its own directory → use that as the dataset root
  • Read class names from classes.txt
  • Auto-split images/ & labels/ into train/ and val/ (80 / 20) if not already split
  • Write dataset.yaml for Ultralytics
  • Launch training on the YOLOv11n base weights

Dependencies
------------
pip install ultralytics pyyaml
"""

from pathlib import Path
import random
import shutil
import yaml
from ultralytics import YOLO


# --------------------------- helpers --------------------------------------
def build_dataset_yaml(data_dir: Path) -> Path:
    """Create dataset.yaml (and train/val split if needed) and return its path."""
    class_file = data_dir / "classes.txt"
    if not class_file.exists():
        raise FileNotFoundError("classes.txt not found next to this script")

    names = [ln.strip() for ln in class_file.read_text().splitlines() if ln.strip()]
    nc = len(names)

    img_dir = data_dir / "images"
    lab_dir = data_dir / "labels"

    # Split if we only have flat images/labels folders
    if not (img_dir / "train").exists():
        all_imgs = sorted(
            list(img_dir.glob("*.jpg"))
            + list(img_dir.glob("*.jpeg"))
            + list(img_dir.glob("*.png"))
        )
        if len(all_imgs) < 10:
            raise RuntimeError("Need ≥10 images to auto-split—create splits manually.")
        random.shuffle(all_imgs)
        split = int(0.8 * len(all_imgs))
        train_imgs, val_imgs = all_imgs[:split], all_imgs[split:]

        for sub in ("train", "val"):
            (img_dir / sub).mkdir(parents=True, exist_ok=True)
            (lab_dir / sub).mkdir(parents=True, exist_ok=True)

        for p in train_imgs:
            shutil.move(str(p), img_dir / "train" / p.name)
            label = lab_dir / f"{p.stem}.txt"
            if label.exists():
                shutil.move(str(label), lab_dir / "train" / label.name)

        for p in val_imgs:
            shutil.move(str(p), img_dir / "val" / p.name)
            label = lab_dir / f"{p.stem}.txt"
            if label.exists():
                shutil.move(str(label), lab_dir / "val" / label.name)

    yaml_path = data_dir / "dataset.yaml"
    yaml_path.write_text(
        yaml.dump(
            {
                "path": str(data_dir),
                "train": "images/train",
                "val": "images/val",
                "nc": nc,
                "names": names,
            },
            sort_keys=False,
        )
    )
    return yaml_path


def main():
    data_dir = Path(__file__).resolve().parent            # the folder we’re in
    yaml_path = build_dataset_yaml(data_dir)

    # Default hyper-parameters (edit if you like)
    EPOCHS = 100
    IMG_SIZE = 640
    BATCH = 16
    DEVICE = "0"        # set to "cpu" or another CUDA index if needed

    model = YOLO("yolov10n.pt")
    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
    )


if __name__ == "__main__":
    main()
