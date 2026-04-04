"""
scripts/train_efficientnet.py
------------------------------
End-to-end training script for EfficientNetB0.

Run from the project root:
    python scripts/train_efficientnet.py

Outputs saved to outputs/:
    best_efficientnetb0.pth
    efficientnetb0_dog_emotion_final.pth
    training_curves_efficientnet.png
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from configs.config import (
    EFFICIENTNET_BEST_CKPT,
    EFFICIENTNET_FINAL_CKPT,
    LABEL_MAP_PATH,
    OUTPUT_DIR,
    SEED,
    CLASS_NAMES,
    CLASS_EMOJIS,
)
from src.dataset import set_seed, load_dataframes, build_dataloaders
from src.models import build_efficientnetb0
from src.train import train_model
from src.evaluate import evaluate_model, print_classification_report
from utils.visualization import plot_training_curves


def main() -> None:
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Data ──────────────────────────────────────────────────────────────────
    train_df, val_df, test_df, label2idx, idx2label = load_dataframes()
    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df
    )

    # ── Save label map (idempotent) ───────────────────────────────────────────
    if not LABEL_MAP_PATH.exists():
        with open(LABEL_MAP_PATH, 'w') as f:
            json.dump({str(k): v for k, v in idx2label.items()}, f, indent=2)
        print(f'Label map saved → {LABEL_MAP_PATH}')

    # ── Build & train model ───────────────────────────────────────────────────
    model = build_efficientnetb0(freeze_backbone=True, device=device)
    history = train_model(
        model, train_loader, val_loader, device,
        checkpoint_path=EFFICIENTNET_BEST_CKPT,
    )

    # ── Save final weights ────────────────────────────────────────────────────
    torch.save(model.state_dict(), EFFICIENTNET_FINAL_CKPT)
    print(f'Final weights saved → {EFFICIENTNET_FINAL_CKPT}')

    # ── Plot training curves ──────────────────────────────────────────────────
    dummy_hist = {k: [] for k in ['loss', 'accuracy', 'val_loss', 'val_accuracy']}
    dummy_hist['phase1_end'] = 0
    plot_training_curves(
        dummy_hist, history,
        save_path=OUTPUT_DIR / 'training_curves_efficientnet.png',
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(EFFICIENTNET_BEST_CKPT))
    metrics, y_true, y_pred = evaluate_model(
        model, test_loader, device, model_name='EfficientNetB0'
    )
    print_classification_report(
        y_true, y_pred, CLASS_NAMES, CLASS_EMOJIS, model_name='EfficientNetB0'
    )

    print('\n✅ EfficientNetB0 training complete.')


if __name__ == '__main__':
    main()
