import glob
import os
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler


class VesselSegDataset(Dataset):
    def __init__(self, split_dir, augment=False):
        self.img_dir  = os.path.join(split_dir, "images")
        self.mask_dir = os.path.join(split_dir, "masks")

        self.images = sorted(glob.glob(self.img_dir + "/*.png"))
        self.masks  = [
            os.path.join(self.mask_dir, Path(p).stem + "_mask.png") for p in self.images
        ]

        self.tf = A.Compose(
            (
                [
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.03, scale_limit=0.1, rotate_limit=10,
                        border_mode=0, p=0.5,
                    ),
                    A.ElasticTransform(alpha=40, sigma=6, p=0.3),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.4
                    ),
                    A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
                ]
                if augment
                else []
            )
            + [A.Normalize(mean=(0.5,), std=(0.5,)), ToTensorV2()]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = np.array(Image.open(self.images[i]).convert("L"), dtype=np.uint8)
        mask = np.array(Image.open(self.masks[i]), dtype=np.uint8)

        img = img[..., None]
        out = self.tf(image=img, mask=mask)

        x = out["image"].float()
        y = out["mask"].long()
        return x, y


def build_rare_class_sampler(train_images_dir, syntax_by_img, cnt_syntax, rare_threshold, boost=5.0):
    """
    Build a WeightedRandomSampler that gives `boost`x sampling weight to any
    training image containing at least one rare vessel class.
    """
    rare_classes = {c for c in range(1, 26) if cnt_syntax.get(c, 0) < rare_threshold}

    train_images = sorted(glob.glob(os.path.join(train_images_dir, "*.png")))
    weights = []
    for img_path in train_images:
        img_id = int(Path(img_path).stem)
        anns = syntax_by_img.get(img_id, [])
        classes_in_img = {int(a["category_id"]) for a in anns if a["category_id"] != 26}
        weights.append(boost if classes_in_img & rare_classes else 1.0)

    weights = np.array(weights, dtype=np.float32)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    return sampler, rare_classes
