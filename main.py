"""
Entry point for the ARCADE coronary artery segmentation + stenosis
detection pipeline.

Examples:
    python main.py --stage train-vessel
    python main.py --stage train-stenosis
    python main.py --stage eval
    python main.py --stage demo --image data/raw/arcade/arcade/stenosis/val/images/101.png
"""

import argparse
import os

import torch

from configs import config as cfg


def stage_train_vessel():
    from src.train_vessel import main as train_vessel_main
    train_vessel_main()


def stage_train_stenosis():
    from src.train_stenosis import main as train_stenosis_main
    train_stenosis_main()


def stage_eval():
    import glob
    from collections import Counter

    from data.coco_utils import build_category_maps, group_annotations_by_image, load_coco, make_masks
    from data.dataset import VesselSegDataset
    from models.vessel_unet import build_vessel_model, load_vessel_checkpoint
    from src.eval import evaluate_stenosis_model, evaluate_vessel_model
    from models.stenosis_yolo import load_stenosis_model
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    coco_syntax_val = load_coco(sorted(
        glob.glob(os.path.join(cfg.SYNTAX_DIR, "val", "annotations", "*.json"))
    )[0])
    make_masks(coco_syntax_val, os.path.join(cfg.SYNTAX_DIR, "val"), ignore_ids={26})

    val_ds = VesselSegDataset(os.path.join(cfg.SYNTAX_DIR, "val"), augment=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.TRAIN_BATCH_SIZE, shuffle=False)

    model = build_vessel_model(device)
    load_vessel_checkpoint(model, cfg.VESSEL_CHECKPOINT_PATH, device)

    vessel_metrics = evaluate_vessel_model(model, val_loader, device, cfg.NUM_VESSEL_CLASSES)
    print("Vessel  -> Mean F1:", round(vessel_metrics["mean_f1"], 4),
          " Mean Dice:", round(vessel_metrics["mean_dice"], 4),
          " Mean IoU:", round(vessel_metrics["mean_iou"], 4))

    yolo_best = os.path.join(cfg.YOLO_RUNS_DIR, "stenosis_seg", "weights", "best.pt")
    if os.path.exists(yolo_best):
        yolo_model = load_stenosis_model(yolo_best)
        stenosis_metrics = evaluate_stenosis_model(
            yolo_model, os.path.join(cfg.YOLO_DATA_DIR, "data.yaml")
        )
        print("Stenosis -> Mask F1:", round(stenosis_metrics["mask_f1"], 4))


def stage_demo(image_path):
    from data.coco_utils import build_category_maps, load_coco
    from models.vessel_unet import build_vessel_model, load_vessel_checkpoint
    from models.stenosis_yolo import load_stenosis_model
    from src.pipeline import run_full_pipeline
    from src.visualize import build_fixed_color_map, show_combined

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    model = build_vessel_model(device)
    load_vessel_checkpoint(model, cfg.VESSEL_CHECKPOINT_PATH, device)

    yolo_best = os.path.join(cfg.YOLO_RUNS_DIR, "stenosis_seg", "weights", "best.pt")
    yolo_model = load_stenosis_model(yolo_best)

    coco_syntax_val = load_coco(sorted(
        __import__("glob").glob(os.path.join(cfg.SYNTAX_DIR, "val", "annotations", "*.json"))
    )[0])
    _, id_to_description = build_category_maps(coco_syntax_val)
    fixed_colors = build_fixed_color_map(cfg.NUM_VESSEL_CLASSES)

    img_gray, vessel_pred, stenosis_pred = run_full_pipeline(image_path, model, yolo_model, device)

    show_combined(
        fname=os.path.basename(image_path),
        img_gray=img_gray,
        gt_vessel_mask=vessel_pred,   # no GT available for an arbitrary image; showing pred twice as placeholder
        gt_stenosis_mask=stenosis_pred,
        vessel_pred=vessel_pred,
        stenosis_pred=stenosis_pred,
        fixed_colors=fixed_colors,
        id_to_description=id_to_description,
        save_path=os.path.join(cfg.RESULTS_DIR, "visualizations", "demo_output.png"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                         choices=["train-vessel", "train-stenosis", "eval", "demo"])
    parser.add_argument("--image", default=None, help="image path, required for --stage demo")
    args = parser.parse_args()

    if args.stage == "train-vessel":
        stage_train_vessel()
    elif args.stage == "train-stenosis":
        stage_train_stenosis()
    elif args.stage == "eval":
        stage_eval()
    elif args.stage == "demo":
        if not args.image:
            raise SystemExit("--image is required for --stage demo")
        stage_demo(args.image)
