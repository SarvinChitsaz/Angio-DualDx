"""
Vessel segmentation model: a standard U-Net (resnet34 encoder, ImageNet
pretrained) over 26 classes (background + 25 SYNTAX segments), with a
class-weighted CrossEntropy + Dice loss to counter the severe annotation
imbalance across segment classes (see data/README.md).
"""

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from configs.config import ENCODER_NAME, ENCODER_WEIGHTS, IN_CHANNELS, NUM_VESSEL_CLASSES


def build_vessel_model(device):
    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=IN_CHANNELS,
        classes=NUM_VESSEL_CLASSES,
    ).to(device)
    return model


def build_class_weights(cnt_syntax, num_classes, device):
    """
    Inverse-log-frequency class weights for CrossEntropyLoss, derived from
    per-class training annotation counts (data/coco_utils output).
    """
    class_pixel_counts = np.ones(num_classes)
    for c in range(1, num_classes):
        if c in cnt_syntax:
            class_pixel_counts[c] = cnt_syntax[c]

    class_weights = 1.0 / np.log(1.02 + class_pixel_counts / class_pixel_counts.sum())
    class_weights = class_weights / class_weights.mean()
    class_weights[0] = 1.0  # background gets normal weight

    return torch.tensor(class_weights, dtype=torch.float32).to(device)


def build_vessel_loss(class_weights_tensor):
    ce_loss = nn.CrossEntropyLoss(weight=class_weights_tensor)
    dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

    def vessel_loss(logits, y):
        return ce_loss(logits, y) + 0.5 * dice_loss(logits, y)

    return vessel_loss


def load_vessel_checkpoint(model, checkpoint_path, device, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.eval()
    return checkpoint
