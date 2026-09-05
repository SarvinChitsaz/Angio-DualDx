# Dataset — ARCADE

This project uses the **ARCADE** dataset (Automatic Region-based Coronary
Artery Disease diagnostics using x-ray angiography imagEs), released for the
MICCAI 2023 challenge of the same name.

## Download

```bash
mkdir -p data/raw/arcade
wget -O data/raw/arcade.zip "https://zenodo.org/api/records/10390295/files/arcade.zip/content"
unzip -q -o data/raw/arcade.zip -d data/raw/arcade
```

After extraction, `ARCADE_BASE_DIR` (see `configs/config.py`) should point at
`data/raw/arcade/arcade`, which contains two independent COCO-format tasks:

```
arcade/
├── syntax/     # 26-class vessel segmentation (25 anatomical segments + stenosis)
│   ├── train/  # 1000 images
│   ├── val/    # 200 images
│   └── test/
└── stenosis/   # binary stenosis instance segmentation
    ├── train/
    ├── val/
    └── test/
```

## Key findings from dataset exploration

1. **`category_id` ≠ segment number.** In `syntax` annotations, the COCO
   `category_id` field does not reliably match the real SYNTAX segment
   number stored in `category["name"]` (e.g. `category_id=20` has
   `name="16"`). Any class-name lookup must go through
   `data/coco_utils.build_category_maps`, never assume a fixed index-based
   mapping.

2. **`syntax` and `stenosis` share the same 1000 training images**, aligned
   exactly by `file_name` / `image_id`. This makes it possible to run both
   models on the same input and combine their outputs (see
   `src/pipeline.py`).

3. **Stenosis annotations are independent of vessel annotations.** Roughly
   half of all stenosis instances fall outside the bounding box of any
   annotated vessel segment in the same image — i.e. some visibly diseased
   vessels are simply not labeled in the `syntax` task. This makes
   crop-around-vessel localization strategies unreliable and was the reason
   the project moved from a shared-encoder multi-task network to two
   independent, specialized models (U-Net for vessels, YOLOv8-seg for
   stenosis) — see the main README's "Architecture decisions" section.

4. **Extreme class imbalance** on two axes: stenosis pixels are ~0.4% of an
   image on average, and 11 of 25 vessel segment classes have fewer than 100
   training annotations. Both were mitigated (class-weighted loss + rare-class
   oversampling for vessels, `pos_weight`/focal-loss experiments and eventual
   detection-style reframing for stenosis) — full results in the main README.

5. **Annotation is manual (CVAT) and has known inter-rater variability**,
   per the official challenge documentation. Some prediction errors likely
   reflect ground-truth ambiguity rather than pure model failure.
