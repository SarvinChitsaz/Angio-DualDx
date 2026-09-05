"""
End-to-end inference: run both models (vessel U-Net + stenosis YOLOv8-seg)
on a single image and return aligned prediction masks, ready for
visualization (src/visualize.py) or the severity index (src/severity_index.py).
"""

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image


def run_full_pipeline(img_path, vessel_model, yolo_model, device, conf=0.25):
    img_gray = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)

    tf = A.Compose([A.Normalize(mean=(0.5,), std=(0.5,)), ToTensorV2()])
    x = tf(image=img_gray[..., None])["image"].unsqueeze(0).float().to(device)

    vessel_model.eval()
    with torch.no_grad():
        vessel_logits = vessel_model(x)
        vessel_pred = torch.argmax(vessel_logits, dim=1)[0].cpu().numpy().astype(np.uint8)

    yolo_result = yolo_model.predict(img_path, conf=conf, imgsz=512, verbose=False)[0]

    stenosis_pred_mask = np.zeros_like(vessel_pred, dtype=np.uint8)
    if yolo_result.masks is not None:
        for m in yolo_result.masks.data.cpu().numpy():
            m_resized = cv2.resize(m, (vessel_pred.shape[1], vessel_pred.shape[0]))
            stenosis_pred_mask[m_resized > 0.5] = 1

    return img_gray, vessel_pred, stenosis_pred_mask
