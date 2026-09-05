"""
Training loop for the vessel segmentation U-Net.

Usage:
    python -m src.train_vessel
"""

import os
import time

import torch
from torch.utils.data import DataLoader

from configs import config as cfg
from data.coco_utils import build_category_maps, group_annotations_by_image, load_coco, make_masks
from data.dataset import VesselSegDataset, build_rare_class_sampler
from models.vessel_unet import build_class_weights, build_vessel_loss, build_vessel_model


class EarlyStopping:
    def __init__(self, patience=8, min_delta=0.0, path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.best_loss = None
        self.counter = 0
        self.stop = False

    def step(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    coco_syntax = load_coco(sorted(
        __import__("glob").glob(os.path.join(cfg.SYNTAX_DIR, "train", "annotations", "*.json"))
    )[0])
    coco_syntax_val = load_coco(sorted(
        __import__("glob").glob(os.path.join(cfg.SYNTAX_DIR, "val", "annotations", "*.json"))
    )[0])

    from collections import Counter
    cnt_syntax = Counter(a["category_id"] for a in coco_syntax["annotations"])
    syntax_by_img = group_annotations_by_image(coco_syntax)
    _, id_to_description = build_category_maps(coco_syntax)

    make_masks(coco_syntax, os.path.join(cfg.SYNTAX_DIR, "train"), ignore_ids={26})
    make_masks(coco_syntax_val, os.path.join(cfg.SYNTAX_DIR, "val"), ignore_ids={26})

    train_ds = VesselSegDataset(os.path.join(cfg.SYNTAX_DIR, "train"), augment=True)
    val_ds = VesselSegDataset(os.path.join(cfg.SYNTAX_DIR, "val"), augment=False)

    sampler, _ = build_rare_class_sampler(
        os.path.join(cfg.SYNTAX_DIR, "train", "images"),
        syntax_by_img, cnt_syntax, cfg.RARE_CLASS_ANNOTATION_THRESHOLD,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.TRAIN_BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg.TRAIN_BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_vessel_model(device)
    class_weights_tensor = build_class_weights(cnt_syntax, cfg.NUM_VESSEL_CLASSES, device)
    vessel_loss = build_vessel_loss(class_weights_tensor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    early_stopping = EarlyStopping(patience=cfg.EARLY_STOP_PATIENCE, path=cfg.VESSEL_CHECKPOINT_PATH)

    train_losses, val_losses = [], []

    for epoch in range(cfg.MAX_EPOCHS):
        start = time.time()

        model.train()
        running_train = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = vessel_loss(model(x), y)
            loss.backward()
            optimizer.step()
            running_train += loss.item() * x.size(0)
        train_loss = running_train / len(train_loader.dataset)

        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                running_val += vessel_loss(model(x), y).item() * x.size(0)
        val_loss = running_val / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{cfg.MAX_EPOCHS} | Train {train_loss:.4f} | Val {val_loss:.4f} | Time: {time.time()-start:.1f}s")

        early_stopping.step(val_loss, model)
        if early_stopping.stop:
            print("Early stopping triggered.")
            break

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": early_stopping.best_loss,
        "epochs_trained": len(train_losses),
    }
    torch.save(checkpoint, cfg.VESSEL_CHECKPOINT_PATH)
    print("saved checkpoint:", cfg.VESSEL_CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
