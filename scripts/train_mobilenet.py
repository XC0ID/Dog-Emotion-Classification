"""
scripts/train_mobilenet.py
--------------------------
End-to-end training script for MobileNetV2.

Run from the project root:
    python scripts/train_mobilenet.py

Outputs saved to outputs/:
    best_mobilenetv2.pth
    mobilenetv2_dog_emotion_final.pth
    training_curves_mobilenet.png
"""

import sys
import json
from pathlib import Path

# Ensure project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from configs.config import (
    MOBILENET_BEST_CKPT,
    MOBILENET_FINAL_CKPT,
    LABEL_MAP_PATH,
    OUTPUT_DIR,
    SEED,
)
from src.dataset import set_seed, load_dataframes, build_dataloaders
from src.models import build_mobilenetv2
from src.train import train_model
from src.evaluate import evaluate_model, print_classification_report
from utils.visualization import (
    plot_distribution,
    plot_sample_grid,
    plot_training_curves,
    plot_confusion_matrices,
)
from configs.config import CLASS_NAMES, CLASS_EMOJIS


def main() -> None:
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Data ──────────────────────────────────────────────────────────────────
    train_df, val_df, test_df, label2idx, idx2label = load_dataframes()
    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df
    )

    # ── Visualise data ────────────────────────────────────────────────────────
    plot_distribution(train_df.append(val_df).append(test_df),
                      save_path=OUTPUT_DIR / 'distribution_charts.png')
    plot_sample_grid(train_df,
                     save_path=OUTPUT_DIR / 'sample_grid.png')

    # ── Save label map ────────────────────────────────────────────────────────
    with open(LABEL_MAP_PATH, 'w') as f:
        json.dump({str(k): v for k, v in idx2label.items()}, f, indent=2)
    print(f'Label map saved → {LABEL_MAP_PATH}')

    # ── Build & train model ───────────────────────────────────────────────────
    model = build_mobilenetv2(freeze_backbone=True, device=device)
    history = train_model(
        model, train_loader, val_loader, device,
        checkpoint_path=MOBILENET_BEST_CKPT,
    )

    # ── Save final weights ────────────────────────────────────────────────────
    torch.save(model.state_dict(), MOBILENET_FINAL_CKPT)
    print(f'Final weights saved → {MOBILENET_FINAL_CKPT}')

    # ── Plot training curves ──────────────────────────────────────────────────
    # Wrap single history into the dual format expected by visualization
    dummy_hist = {k: [] for k in ['loss', 'accuracy', 'val_loss', 'val_accuracy']}
    dummy_hist['phase1_end'] = 0
    plot_training_curves(
        history, dummy_hist,
        save_path=OUTPUT_DIR / 'training_curves_mobilenet.png',
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(MOBILENET_BEST_CKPT))
    metrics, y_true, y_pred = evaluate_model(
        model, test_loader, device, model_name='MobileNetV2'
    )
    print_classification_report(
        y_true, y_pred, CLASS_NAMES, CLASS_EMOJIS, model_name='MobileNetV2'
    )

    print('\n✅ MobileNetV2 training complete.')


if __name__ == '__main__':
    main()
