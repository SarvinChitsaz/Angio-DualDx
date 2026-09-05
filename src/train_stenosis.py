"""
Convert ARCADE stenosis annotations to YOLO-seg format and fine-tune
YOLOv8n-seg on the resulting dataset.

Usage:
    python -m src.train_stenosis
"""

import glob
import os

from configs import config as cfg
from data.coco_utils import group_annotations_by_image, load_coco
from models.stenosis_yolo import convert_coco_to_yolo_seg, train_stenosis_yolo, write_yolo_data_yaml


def main():
    coco_stenosis = load_coco(
        sorted(glob.glob(os.path.join(cfg.STENOSIS_DIR, "train", "annotations", "*.json")))[0]
    )
    coco_stenosis_val = load_coco(
        sorted(glob.glob(os.path.join(cfg.STENOSIS_DIR, "val", "annotations", "*.json")))[0]
    )

    convert_coco_to_yolo_seg(
        coco_stenosis, group_annotations_by_image,
        os.path.join(cfg.STENOSIS_DIR, "train", "images"),
        os.path.join(cfg.YOLO_DATA_DIR, "images", "train"),
        os.path.join(cfg.YOLO_DATA_DIR, "labels", "train"),
    )
    convert_coco_to_yolo_seg(
        coco_stenosis_val, group_annotations_by_image,
        os.path.join(cfg.STENOSIS_DIR, "val", "images"),
        os.path.join(cfg.YOLO_DATA_DIR, "images", "val"),
        os.path.join(cfg.YOLO_DATA_DIR, "labels", "val"),
    )

    yaml_path = write_yolo_data_yaml(cfg.YOLO_DATA_DIR, os.path.join(cfg.YOLO_DATA_DIR, "data.yaml"))

    train_stenosis_yolo(
        data_yaml_path=yaml_path,
        base_weights=cfg.YOLO_BASE_WEIGHTS,
        epochs=cfg.YOLO_EPOCHS,
        imgsz=cfg.YOLO_IMG_SIZE,
        batch=cfg.YOLO_BATCH_SIZE,
        patience=cfg.YOLO_PATIENCE,
        seed=cfg.SEED,
        project=cfg.YOLO_RUNS_DIR,
        name="stenosis_seg",
    )
    print("training complete. best weights at:",
          os.path.join(cfg.YOLO_RUNS_DIR, "stenosis_seg", "weights", "best.pt"))


if __name__ == "__main__":
    main()
