"""
Full validation-set evaluation for the vessel segmentation model:
per-class Dice / Precision / Recall / F1 / IoU, plus a confusion matrix.
Also includes the YOLO stenosis evaluation wrapper.

Usage:
    python -m src.eval
"""

from collections import defaultdict

import numpy as np
import torch


def multiclass_dice_per_class(y_true, y_pred, num_classes=26):
    dices = {}
    for c in range(1, num_classes):
        t = y_true == c
        p = y_pred == c
        denom = t.sum() + p.sum()
        if denom == 0:
            continue
        dices[c] = (2.0 * (t & p).sum()) / denom
    return dices


def evaluate_vessel_model(model, val_loader, device, num_classes=26):
    """
    Single pass over the validation set computing:
      - pixel accuracy (incl. background)
      - foreground-only accuracy
      - per-class Dice
      - per-class precision / recall / F1 / IoU
      - a full confusion matrix (row-normalized in the caller if desired)
    """
    model.eval()

    vessel_dice_per_class = defaultdict(list)
    per_class_tp = defaultdict(int)
    per_class_fp = defaultdict(int)
    per_class_fn = defaultdict(int)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    correct_pixels = total_pixels = 0
    correct_fg = total_fg = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            pred = torch.argmax(model(x), dim=1).cpu().numpy()
            y_np = y.numpy()

            correct_pixels += (pred == y_np).sum()
            total_pixels += y_np.size

            fg_mask = y_np != 0
            correct_fg += ((pred == y_np) & fg_mask).sum()
            total_fg += fg_mask.sum()

            for b in range(x.shape[0]):
                for c, d in multiclass_dice_per_class(y_np[b], pred[b], num_classes).items():
                    vessel_dice_per_class[c].append(d)

                gt_flat = y_np[b].flatten()
                pred_flat = pred[b].flatten()
                mask = gt_flat != 0
                for gt_c, pred_c in zip(gt_flat[mask], pred_flat[mask]):
                    confusion[gt_c, pred_c] += 1

            for c in range(1, num_classes):
                t, p = y_np == c, pred == c
                per_class_tp[c] += (t & p).sum()
                per_class_fp[c] += (~t & p).sum()
                per_class_fn[c] += (t & ~p).sum()

    f1_scores, iou_scores = [], []
    for c in range(1, num_classes):
        tp, fp, fn = per_class_tp[c], per_class_fp[c], per_class_fn[c]
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn)
        f1_scores.append(f1)
        iou_scores.append(iou)

    return {
        "pixel_accuracy": correct_pixels / total_pixels,
        "foreground_accuracy": correct_fg / total_fg,
        "mean_dice": np.mean([d for v in vessel_dice_per_class.values() for d in v]),
        "mean_f1": np.mean(f1_scores),
        "mean_iou": np.mean(iou_scores),
        "dice_per_class": vessel_dice_per_class,
        "confusion_matrix": confusion,
    }


def evaluate_stenosis_model(yolo_model, data_yaml_path, split="val"):
    """Wraps ultralytics' built-in val() and derives F1 from precision/recall."""
    metrics = yolo_model.val(data=data_yaml_path, split=split)
    mask_p, mask_r = metrics.seg.mp, metrics.seg.mr
    mask_f1 = 2 * mask_p * mask_r / (mask_p + mask_r) if (mask_p + mask_r) > 0 else 0.0
    return {
        "box_precision": metrics.box.mp, "box_recall": metrics.box.mr, "box_map50": metrics.box.map50,
        "mask_precision": mask_p, "mask_recall": mask_r, "mask_map50": metrics.seg.map50,
        "mask_f1": mask_f1,
    }
