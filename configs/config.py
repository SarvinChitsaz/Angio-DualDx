import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR      = os.environ.get("ARCADE_BASE_DIR", "./data/raw/arcade")
SYNTAX_DIR    = os.path.join(BASE_DIR, "syntax")
STENOSIS_DIR  = os.path.join(BASE_DIR, "stenosis")

PROCESSED_DIR = "./data/processed"
YOLO_DATA_DIR = os.path.join(PROCESSED_DIR, "yolo_stenosis")

CHECKPOINT_DIR = "./models/checkpoints"
VESSEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "vessel_unet_final.pth")
YOLO_RUNS_DIR = os.path.join(CHECKPOINT_DIR, "yolo_runs")

RESULTS_DIR = "./assets/results"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Vessel segmentation model (U-Net, resnet34 encoder)
# ---------------------------------------------------------------------------
NUM_VESSEL_CLASSES = 26  # 0 = background, 1-25 = SYNTAX segments
ENCODER_NAME       = "resnet34"
ENCODER_WEIGHTS    = "imagenet"
IN_CHANNELS        = 1

TRAIN_BATCH_SIZE   = 8
LEARNING_RATE      = 2e-4
WEIGHT_DECAY       = 1e-4
MAX_EPOCHS         = 45
EARLY_STOP_PATIENCE = 8

RARE_CLASS_ANNOTATION_THRESHOLD = 100   # classes with fewer train annotations
                                         # than this are oversampled

# ---------------------------------------------------------------------------
# Stenosis detection model (YOLOv8-seg)
# ---------------------------------------------------------------------------
YOLO_BASE_WEIGHTS = "yolov8n-seg.pt"
YOLO_IMG_SIZE     = 512
YOLO_BATCH_SIZE   = 8
YOLO_EPOCHS       = 100
YOLO_PATIENCE     = 20
YOLO_CONF_THRESHOLD = 0.25

# ---------------------------------------------------------------------------
# Category ID -> real SYNTAX segment name
# ---------------------------------------------------------------------------
# IMPORTANT: The `category_id` field in the ARCADE COCO annotations does NOT
# always match the real SYNTAX segment number encoded in the `name` field
# (e.g. category_id=20 has name="16"). This mapping must always be built at
# runtime from the actual `categories` list in the annotation JSON
# (see data/coco_utils.py: `build_category_maps`), never hardcoded by index.
SEGMENT_DESCRIPTION = {
    "1": "RCA proximal", "2": "RCA mid", "3": "RCA distal",
    "4": "Posterior descending artery (PDA)", "5": "Left main",
    "6": "LAD proximal", "7": "LAD mid", "8": "LAD apical",
    "9": "First diagonal", "9a": "First diagonal a",
    "10": "Second diagonal", "10a": "Second diagonal a",
    "11": "Proximal circumflex", "12": "Intermediate / anterolateral artery",
    "12a": "Obtuse marginal 1", "12b": "Obtuse marginal 2",
    "13": "Distal circumflex", "14": "Left posterolateral",
    "14a": "Left posterolateral a", "14b": "Left posterolateral b",
    "15": "Posterior descending (from LCX in left dominance)",
    "16": "Posterolateral branch from RCA",
    "16a": "Posterolateral branch from RCA (1st)",
    "16b": "Posterolateral branch from RCA (2nd)",
    "16c": "Posterolateral branch from RCA (3rd)",
    "stenosis": "Stenosis",
}

# ---------------------------------------------------------------------------
# SYNTAX Score segment weighting factors (right-dominant system).
#
# Source: Sianos et al., "The SYNTAX Score: an angiographic tool grading
# the complexity of coronary artery disease", EuroIntervention 2005, and
# subsequent official SYNTAX scoring documentation.
#
# NOTE: this is a simplified, self-contained approximation used only for the
# "Severity Index" bonus module (see src/severity_index.py). It is NOT a
# clinical-grade SYNTAX score: percent diameter narrowing, lesion length,
# calcification, tortuosity, and bifurcation involvement are not available
# in the ARCADE annotations and are therefore not modeled.
# ---------------------------------------------------------------------------
SYNTAX_SEGMENT_WEIGHT = {
    "1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0,
    "5": 5.0,
    "6": 3.5, "7": 2.5, "8": 1.0,
    "9": 1.0, "9a": 1.0,
    "10": 0.5, "10a": 0.5,
    "11": 1.5,
    "12": 1.0, "12a": 1.0, "12b": 1.0,
    "13": 0.5,
    "14": 0.5, "14a": 0.5, "14b": 0.5,
    "15": 1.0,
    "16": 0.5, "16a": 0.5, "16b": 0.5, "16c": 0.5,
}

OCCLUSION_MULTIPLIER = 2.0  # non-occlusive assumption (see README limitations)

# ---------------------------------------------------------------------------
# Official ARCADE (MICCAI 2023) challenge baselines, used for comparison
# ---------------------------------------------------------------------------
OFFICIAL_BASELINES = {
    "vessel_segmentation_f1": {"rank_5_ensemble": 0.3769, "rank_3_yolo_angio": 0.422},
    "stenosis_detection_f1":  {"rank_5_ensemble": 0.3941, "rank_3_stenunet": 0.5348},
}
