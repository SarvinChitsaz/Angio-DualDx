"""
Utilities for reading ARCADE's COCO-format annotations and turning them
into pixel masks / class-mapping dictionaries.

Key finding this module encodes a fix for: in the ARCADE `syntax` task,
COCO `category_id` does not always match the true SYNTAX segment number
stored in `category["name"]` (e.g. category_id=20 -> name="16"). Any
class-name lookup MUST go through `build_category_maps`, never assume
`id == name`.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from configs.config import SEGMENT_DESCRIPTION


def load_coco(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def group_annotations_by_image(coco):
    """Return {image_id: [annotation, ...]}."""
    by_img = defaultdict(list)
    for ann in coco["annotations"]:
        by_img[int(ann["image_id"])].append(ann)
    return by_img


def build_category_maps(coco, segment_description=None):
    """
    Build the two mappings needed everywhere else in the codebase:

    id_to_name        : COCO category_id -> raw SYNTAX name string ("9a", "16", ...)
    id_to_description : COCO category_id -> human-readable description

    Always derive these from the actual `categories` list -- never assume
    a fixed ordering or that id == segment number.
    """
    if segment_description is None:
        segment_description = SEGMENT_DESCRIPTION

    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    id_to_description = {
        cid: segment_description.get(name, name) for cid, name in id_to_name.items()
    }
    return id_to_name, id_to_description


def polygons_to_mask(h, w, anns, ignore_ids=frozenset()):
    """Rasterize a list of COCO annotations into a single-channel class mask."""
    mask = np.zeros((h, w), dtype=np.uint8)

    for ann in anns:
        cid = int(ann["category_id"])
        if cid in ignore_ids:
            continue

        seg = ann.get("segmentation", None)
        if seg is None:
            continue

        for poly in seg:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts = np.round(pts).astype(np.int32)
            cv2.fillPoly(mask, [pts], color=cid)

    return mask


def make_masks(coco, split_dir, ignore_ids=frozenset()):
    """
    Precompute and cache one PNG mask per image under `<split_dir>/masks/`.

    `ignore_ids` should contain {26} for the `syntax` task, since category
    26 ("stenosis") lives in a separate, independently-annotated file and
    must not be rasterized into the vessel mask.
    """
    split_dir = Path(split_dir)
    out_dir = split_dir / "masks"
    out_dir.mkdir(exist_ok=True)

    img_info = {int(im["id"]): im for im in coco["images"]}
    anns_by_img = group_annotations_by_image(coco)

    for img_id, im in img_info.items():
        h, w = int(im["height"]), int(im["width"])
        anns = anns_by_img.get(img_id, [])
        mask = polygons_to_mask(h, w, anns, ignore_ids=ignore_ids)

        fname = Path(im["file_name"]).stem
        Image.fromarray(mask).save(out_dir / f"{fname}_mask.png")

    return out_dir
