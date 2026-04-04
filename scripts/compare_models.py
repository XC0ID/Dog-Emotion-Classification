"""
scripts/compare_models.py
--------------------------
Load both trained models and produce a side-by-side comparison:
  - Metric table (accuracy, AUC, macro F1, loss, parameters)
  - Grouped bar + radar chart
  - Confusion matrices
  - Per-class F1 bar chart
  - Qualitative prediction grid
  - Final recommendation printed to console

Requires both model checkpoints to exist in outputs/ (run train_mobilenet.py
and train_efficientnet.py first).

Run from the project root:
    python scripts/compare_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from configs.config import (
    MOBILENET_BEST_CKPT,
    EFFICIENTNET_BEST_CKPT,
    OUTPUT_DIR,
    SEED,
    CLASS_NAMES,
    CLASS_EMOJIS,
)
from src.dataset import set_seed, load_dataframes, build_dataloaders, get_val_transform
from src.models import build_mobilenetv2, build_efficientnetb0
from src.evaluate import evaluate_model, print_classification_report
from utils.visualization import (
    plot_model_comparison,
    plot_confusion_matrices,
    plot_per_class_f1,
    plot_sample_predictions,
)


def main() -> None:
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    # ── Data ──────────────────────────────────────────────────────────────────
    train_df, val_df, test_df, label2idx, idx2label = load_dataframes()
    _, _, test_loader = build_dataloaders(train_df, val_df, test_df)

    # ── Load models ───────────────────────────────────────────────────────────
    if not MOBILENET_BEST_CKPT.exists():
        raise FileNotFoundError(
            f'{MOBILENET_BEST_CKPT} not found. Run train_mobilenet.py first.'
        )
    if not EFFICIENTNET_BEST_CKPT.exists():
        raise FileNotFoundError(
            f'{EFFICIENTNET_BEST_CKPT} not found. Run train_efficientnet.py first.'
        )

    model_mob = build_mobilenetv2(freeze_backbone=False, device=device)
    model_mob.load_state_dict(torch.load(MOBILENET_BEST_CKPT, map_location=device))

    model_eff = build_efficientnetb0(freeze_backbone=False, device=device)
    model_eff.load_state_dict(torch.load(EFFICIENTNET_BEST_CKPT, map_location=device))

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics_mob, y_true_mob, y_pred_mob = evaluate_model(
        model_mob, test_loader, device, 'MobileNetV2'
    )
    metrics_eff, y_true_eff, y_pred_eff = evaluate_model(
        model_eff, test_loader, device, 'EfficientNetB0'
    )

    # ── Classification reports ────────────────────────────────────────────────
    print_classification_report(y_true_mob, y_pred_mob, CLASS_NAMES, CLASS_EMOJIS, 'MobileNetV2')
    print_classification_report(y_true_eff, y_pred_eff, CLASS_NAMES, CLASS_EMOJIS, 'EfficientNetB0')

    # ── Comparison table ──────────────────────────────────────────────────────
    params_mob = sum(p.numel() for p in model_mob.parameters()) / 1e6
    params_eff = sum(p.numel() for p in model_eff.parameters()) / 1e6

    comparison_df = pd.DataFrame([metrics_mob, metrics_eff]).set_index('model')

    final_table = pd.DataFrame({
        'Model'        : ['MobileNetV2', 'EfficientNetB0'],
        'Parameters'   : [f'{params_mob:.2f}M', f'{params_eff:.2f}M'],
        'Test Accuracy': [f"{metrics_mob['accuracy']*100:.2f}%",
                          f"{metrics_eff['accuracy']*100:.2f}%"],
        'Macro F1'     : [f"{metrics_mob['f1']:.4f}", f"{metrics_eff['f1']:.4f}"],
        'AUC'          : [f"{metrics_mob['auc']:.4f}", f"{metrics_eff['auc']:.4f}"],
        'Speed'        : ['⚡⚡⚡', '⚡⚡'],
    })

    print('=' * 70)
    print('         🏆  FINAL MODEL COMPARISON SUMMARY  🏆')
    print('=' * 70)
    print(final_table.to_string(index=False))
    print('=' * 70)

    winner = (
        'EfficientNetB0'
        if metrics_eff['accuracy'] >= metrics_mob['accuracy']
        else 'MobileNetV2'
    )
    print(f"""
📌 RECOMMENDATION
─────────────────────────────────────────────────────
🏆 Best Overall Accuracy : {winner}
⚡ Fastest Inference     : MobileNetV2
🎯 Balanced Choice       : EfficientNetB0

→ For real-time / edge deployment → MobileNetV2
→ For maximum accuracy            → EfficientNetB0
─────────────────────────────────────────────────────
""")

    # ── Charts ────────────────────────────────────────────────────────────────
    plot_model_comparison(comparison_df,
                          save_path=OUTPUT_DIR / 'model_comparison.png')
    plot_confusion_matrices(
        y_true_mob, y_pred_mob, y_true_eff, y_pred_eff,
        CLASS_NAMES,
        save_path=OUTPUT_DIR / 'confusion_matrices.png',
    )
    plot_per_class_f1(
        y_true_mob, y_pred_mob, y_true_eff, y_pred_eff,
        CLASS_NAMES,
        save_path=OUTPUT_DIR / 'per_class_f1.png',
    )
    plot_sample_predictions(
        test_df, model_mob, model_eff,
        get_val_transform(), idx2label, device,
        n_display=12, seed=SEED,
        save_path=OUTPUT_DIR / 'sample_predictions.png',
    )

    print('\n✅ Comparison complete. All charts saved to outputs/.')


if __name__ == '__main__':
    main()
