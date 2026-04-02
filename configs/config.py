"""
configs/config.py
-----------------
Central configuration for the Dog Emotion Classification project.
Edit BASE_DIR and CSV_PATH to point to your local dataset before running any script.
"""

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path('C:\\Users\\ACER\\OneDrive\\Desktop\\Bull\\Dog-Emotion-Classification-1\\data')
CSV_PATH  = BASE_DIR / 'labels.csv'
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Dataset ──────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
NUM_CLASSES = 4
CLASS_NAMES = ['angry', 'happy', 'relaxed', 'sad']
CLASS_COLORS = {
    'angry'  : '#E74C3C',
    'happy'  : '#2ECC71',
    'relaxed': '#3498DB',
    'sad'    : '#9B59B6',
}
CLASS_EMOJIS = {
    'angry'  : '😠',
    'happy'  : '😊',
    'relaxed': '😌',
    'sad'    : '😢',
}

# ─── Training ─────────────────────────────────────────────────────────────────
SEED          = 42
EPOCHS_PHASE1 = 10       # Head-only training (backbone frozen)
EPOCHS_PHASE2 = 20       # Fine-tuning (last 30 backbone layers unfrozen)
LR_PHASE1     = 1e-3
LR_PHASE2     = 1e-5
DROPOUT_RATE  = 0.4
PATIENCE      = 5        # Early stopping patience for phase 1
PATIENCE_FT   = 6        # Early stopping patience for phase 2
UNFREEZE_LAST = 30       # Number of backbone layers to unfreeze in phase 2

# ─── Train / Val / Test splits ────────────────────────────────────────────────
VAL_RATIO  = 0.15        # 15 % of total dataset
TEST_RATIO = 0.15        # 15 % of total dataset

# ─── ImageNet normalisation (used by both MobileNetV2 and EfficientNetB0) ─────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─── Output file names ────────────────────────────────────────────────────────
MOBILENET_BEST_CKPT  = OUTPUT_DIR / 'best_mobilenetv2.pth'
EFFICIENTNET_BEST_CKPT = OUTPUT_DIR / 'best_efficientnetb0.pth'
MOBILENET_FINAL_CKPT = OUTPUT_DIR / 'mobilenetv2_dog_emotion_final.pth'
EFFICIENTNET_FINAL_CKPT = OUTPUT_DIR / 'efficientnetb0_dog_emotion_final.pth'
LABEL_MAP_PATH       = OUTPUT_DIR / 'label_map.json'
