"""
Stenosis detection model: YOLOv8-segmentation (nano backbone), fine-tuned
on the ARCADE `stenosis` task, framed as a single-class instance
segmentation problem.

A dedicated detection-style model was chosen over extending the vessel
U-Net to a second head because stenosis pixels are ~0.4% of the image on
average -- dense per-pixel classification collapsed to a trivial
all-background predictor under every loss-balancing scheme tried
(class weighting, focal loss, pos_weight tuning). See the main README's
"Architecture decisions" section for the full comparison and evidence from
the official ARCADE challenge leaderboard, where detection-style methods
(YOLO, Mask R-CNN, DETR variants) also dominate the top ranks.
"""

import os
import shutil
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def convert_coco_to_yolo_seg(coco, group_annotations_by_image, img_dir, out_img_dir, out_label_dir):
    """Convert ARCADE COCO stenosis polygons into YOLO-seg label format."""
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    img_info = {int(im["id"]): im for im in coco["images"]}
    anns_by_img = group_annotations_by_image(coco)

    n_written = 0
    for img_id, im in img_info.items():
        h, w = im["height"], im["width"]
        fname = im["file_name"]
        stem = Path(fname).stem

        src_path = os.path.join(img_dir, fname)
        dst_img_path = os.path.join(out_img_dir, fname)
        if not os.path.exists(dst_img_path):
            shutil.copy(src_path, dst_img_path)

        anns = anns_by_img.get(img_id, [])
        lines = []
        for ann in anns:
            seg = ann.get("segmentation", None)
            if not seg:
                continue
            for poly in seg:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                pts[:, 0] = pts[:, 0] / w
                pts[:, 1] = pts[:, 1] / h
                pts = np.clip(pts, 0.0, 1.0)
                coords_str = " ".join(f"{px:.6f} {py:.6f}" for px, py in pts)
                lines.append(f"0 {coords_str}")

        label_path = os.path.join(out_label_dir, stem + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(lines))
        n_written += 1

    return n_written


def write_yolo_data_yaml(yolo_root, yaml_path):
    yaml_content = f"""
path: {yolo_root}
train: images/train
val: images/val

names:
  0: stenosis
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    return yaml_path


def train_stenosis_yolo(data_yaml_path, base_weights, epochs, imgsz, batch, patience, seed, project, name):
    model = YOLO(base_weights)
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        seed=seed,
        project=project,
        name=name,
        verbose=False,
    )
    return model


def load_stenosis_model(weights_path):
    return YOLO(weights_path)
