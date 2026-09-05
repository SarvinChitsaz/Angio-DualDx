# Checkpoints

Trained weights are not committed to this repository (see `.gitignore`).

To reproduce:

```bash
python main.py --stage train-vessel     # -> vessel_unet_final.pth
python main.py --stage train-stenosis   # -> yolo_runs/stenosis_seg/weights/best.pt
```

Both scripts write here automatically.
