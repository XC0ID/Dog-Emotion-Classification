"""
utils/visualization.py
-----------------------
All plotting helpers for the Dog Emotion Classification project.

Functions
---------
plot_distribution(df, save_path)
    Bar chart + donut chart + lollipop chart of class distribution.

plot_sample_grid(df, n_samples, save_path)
    Grid of n_samples random images per emotion class.

plot_augmentation_preview(sample_path, n_aug, save_path)
    Original image vs N augmented versions side by side.

plot_training_curves(hist_mob, hist_eff, save_path)
    Accuracy and loss curves for both models with fine-tune boundary marker.

plot_model_comparison(comparison_df, save_path)
    Grouped bar chart + radar chart comparing accuracy, AUC, macro F1.

plot_confusion_matrices(y_true_mob, y_pred_mob, y_true_eff, y_pred_eff,
                         class_names, save_path)
    Side-by-side normalised confusion matrices.

plot_per_class_f1(y_true_mob, y_pred_mob, y_true_eff, y_pred_eff,
                   class_names, save_path)
    Grouped bar chart of per-class F1 scores.

plot_sample_predictions(test_df, model_mob, model_eff, val_transform,
                         idx2label, device, n_display, save_path)
    Grid of test images annotated with true and predicted labels.
"""

import random
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score

from configs.config import CLASS_COLORS, CLASS_EMOJIS, IMG_SIZE

# ─── Global style ─────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def _save(fig: plt.Figure, path: Path | None, tight: bool = True) -> None:
    if tight:
        plt.tight_layout()
    if path:
        fig.savefig(path, bbox_inches='tight', dpi=150)
        print(f'  💾 Saved → {path}')
    plt.show()


# ─── Distribution ─────────────────────────────────────────────────────────────
def plot_distribution(df, save_path: Path | None = None) -> None:
    """Bar + donut + lollipop charts of class distribution."""
    counts = df['label'].value_counts().sort_index()
    colors = [CLASS_COLORS[c] for c in counts.index]

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle('Dog Emotion Dataset — Label Distribution',
                 fontsize=16, fontweight='bold', y=1.02)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Bar chart
    ax1 = fig.add_subplot(gs[0])
    bars = ax1.bar(counts.index, counts.values, color=colors,
                   edgecolor='white', linewidth=1.5, width=0.6)
    for bar, val in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 10, f'{val:,}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_title('Image Count per Class', fontweight='bold')
    ax1.set_ylabel('Number of Images')
    ax1.set_xlabel('Emotion')
    ax1.set_ylim(0, counts.max() * 1.15)
    ax1.tick_params(axis='x', rotation=15)

    # Donut chart
    ax2 = fig.add_subplot(gs[1])
    wedges, texts, autotexts = ax2.pie(
        counts.values,
        labels=[f"{CLASS_EMOJIS[c]} {c}" for c in counts.index],
        colors=colors, autopct='%1.1f%%', startangle=140,
        pctdistance=0.78,
        wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight('bold')
    ax2.set_title('Class Distribution (Donut)', fontweight='bold')
    ax2.text(0, 0, f'{len(df):,}\nimages', ha='center', va='center',
             fontsize=11, fontweight='bold', color='#2C3E50')

    # Lollipop
    ax3 = fig.add_subplot(gs[2])
    y_pos = range(len(counts))
    ax3.hlines(y_pos, 0, counts.values, colors=colors, linewidth=3, alpha=0.7)
    ax3.scatter(counts.values, y_pos, color=colors, s=120, zorder=5)
    ax3.set_yticks(list(y_pos))
    ax3.set_yticklabels(
        [f"{CLASS_EMOJIS[c]} {c.capitalize()}" for c in counts.index],
        fontsize=10,
    )
    for i, val in enumerate(counts.values):
        ax3.text(val + 5, i, str(val), va='center', fontsize=9, fontweight='bold')
    ax3.set_title('Lollipop — Balance Check', fontweight='bold')
    ax3.set_xlabel('Count')
    ax3.set_xlim(0, counts.max() * 1.12)

    _save(fig, save_path)


# ─── Sample grid ──────────────────────────────────────────────────────────────
def plot_sample_grid(
    df,
    n_samples: int = 4,
    seed: int = 42,
    save_path: Path | None = None,
) -> None:
    """Display n_samples random images per emotion class."""
    class_names = sorted(df['label'].unique())
    n_classes   = len(class_names)
    fig, axes   = plt.subplots(n_classes, n_samples,
                                figsize=(n_samples * 3.2, n_classes * 3.2))
    fig.suptitle('🐶 Sample Images per Emotion Class',
                 fontsize=16, fontweight='bold', y=1.01)

    for row_idx, emotion in enumerate(class_names):
        subset = df[df['label'] == emotion].sample(n=n_samples, random_state=seed)
        color  = CLASS_COLORS[emotion]
        for col_idx, (_, row) in enumerate(subset.iterrows()):
            ax = axes[row_idx][col_idx]
            try:
                img = Image.open(str(row['filepath'])).convert('RGB').resize(IMG_SIZE)
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes)
            ax.axis('off')
            if col_idx == 0:
                ax.set_ylabel(
                    f"{CLASS_EMOJIS[emotion]} {emotion.upper()}",
                    fontsize=11, fontweight='bold', color=color,
                    rotation=0, labelpad=60, va='center',
                )
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(2.5)

    _save(fig, save_path)


# ─── Augmentation preview ─────────────────────────────────────────────────────
def plot_augmentation_preview(
    sample_path: str,
    n_aug: int = 7,
    save_path: Path | None = None,
) -> None:
    """Show original + N augmented variants of a single image."""
    aug_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])

    def augment_once(path: str) -> np.ndarray:
        img = Image.open(path).convert('RGB')
        t   = aug_transform(img)
        return (t.permute(1, 2, 0) * 255).numpy().astype('uint8')

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle('Data Augmentation Preview', fontsize=14, fontweight='bold')
    orig = Image.open(sample_path).convert('RGB').resize(IMG_SIZE)
    axes[0][0].imshow(orig)
    axes[0][0].set_title('Original', fontweight='bold', color='#2ECC71')
    axes[0][0].axis('off')
    for i, ax in enumerate(axes.flat[1:n_aug + 1]):
        ax.imshow(augment_once(sample_path))
        ax.set_title(f'Aug #{i + 1}', fontsize=9)
        ax.axis('off')
    _save(fig, save_path)


# ─── Training curves ──────────────────────────────────────────────────────────
def plot_training_curves(
    hist_mob: dict,
    hist_eff: dict,
    save_path: Path | None = None,
) -> None:
    """Accuracy and loss curves for both models."""
    phase1_end = hist_mob.get('phase1_end', len(hist_mob['accuracy']) // 2)

    def _plot(hist, title, color, ax_acc, ax_loss):
        ep = range(1, len(hist['accuracy']) + 1)
        ax_acc.plot(ep, hist['accuracy'],     color=color, lw=2, label='Train Acc')
        ax_acc.plot(ep, hist['val_accuracy'], color=color, lw=2, ls='--', label='Val Acc')
        ax_acc.axvline(phase1_end, ls=':', color='gray', lw=1.5, label='Fine-tune ↓')
        ax_acc.set_title(f'{title} — Accuracy', fontweight='bold')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.set_xlabel('Epoch')
        ax_acc.legend(fontsize=8)
        ax_acc.set_ylim(0, 1.05)

        ax_loss.plot(ep, hist['loss'],     color=color, lw=2, label='Train Loss')
        ax_loss.plot(ep, hist['val_loss'], color=color, lw=2, ls='--', label='Val Loss')
        ax_loss.axvline(phase1_end, ls=':', color='gray', lw=1.5)
        ax_loss.set_title(f'{title} — Loss', fontweight='bold')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.legend(fontsize=8)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('📉 Training & Validation Curves', fontsize=16, fontweight='bold')
    _plot(hist_mob, 'MobileNetV2',   '#E74C3C', axes[0][0], axes[1][0])
    _plot(hist_eff, 'EfficientNetB0', '#3498DB', axes[0][1], axes[1][1])
    _save(fig, save_path)


# ─── Model comparison ─────────────────────────────────────────────────────────
def plot_model_comparison(comparison_df, save_path: Path | None = None) -> None:
    """Grouped bar chart + radar chart of key metrics."""
    import pandas as pd
    metrics_to_plot = ['accuracy', 'auc', 'f1']
    metric_labels   = ['Accuracy', 'AUC', 'Macro F1']
    model_colors    = ['#E74C3C', '#3498DB']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('🏆 Model Comparison — MobileNetV2 vs EfficientNetB0',
                 fontsize=15, fontweight='bold')

    # Grouped bar
    x, width = np.arange(len(metric_labels)), 0.35
    ax = axes[0]
    for i, (model_name, color) in enumerate(zip(comparison_df.index, model_colors)):
        vals = [comparison_df.loc[model_name, m] for m in metrics_to_plot]
        bars = ax.bar(x + i * width, vals, width, label=model_name,
                      color=color, alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score')
    ax.set_title('Metric Comparison', fontweight='bold')
    ax.legend()

    # Radar chart
    ax2 = fig.add_subplot(1, 2, 2, projection='polar')
    N      = len(metric_labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)
    ax2.set_thetagrids(np.degrees(angles[:-1]), metric_labels, fontsize=10)
    for model_name, color in zip(comparison_df.index, model_colors):
        vals = [comparison_df.loc[model_name, m] for m in metrics_to_plot] + \
               [comparison_df.loc[model_name, metrics_to_plot[0]]]
        ax2.plot(angles, vals, 'o-', lw=2, color=color, label=model_name)
        ax2.fill(angles, vals, alpha=0.15, color=color)
    ax2.set_ylim(0, 1)
    ax2.set_title('Radar Chart', fontweight='bold', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15))

    _save(fig, save_path)


# ─── Confusion matrices ───────────────────────────────────────────────────────
def plot_confusion_matrices(
    y_true_mob, y_pred_mob,
    y_true_eff, y_pred_eff,
    class_names: list[str],
    save_path: Path | None = None,
) -> None:
    """Side-by-side normalised confusion matrices."""
    def _heatmap(y_true, y_pred, title, ax, cmap):
        cm      = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        labels  = [f"{CLASS_EMOJIS[c]}\n{c.capitalize()}" for c in sorted(class_names)]
        sns.heatmap(
            cm_norm, annot=cm, fmt='d', cmap=cmap,
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Proportion'},
            ax=ax, annot_kws={'size': 10},
        )
        ax.set_title(title, fontweight='bold', fontsize=12, pad=12)
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)
        ax.tick_params(axis='both', labelsize=9)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Confusion Matrices (Test Set)', fontsize=15, fontweight='bold')
    _heatmap(y_true_mob, y_pred_mob, 'MobileNetV2',   ax1, 'Reds')
    _heatmap(y_true_eff, y_pred_eff, 'EfficientNetB0', ax2, 'Blues')
    _save(fig, save_path)


# ─── Per-class F1 ─────────────────────────────────────────────────────────────
def plot_per_class_f1(
    y_true_mob, y_pred_mob,
    y_true_eff, y_pred_eff,
    class_names: list[str],
    save_path: Path | None = None,
) -> None:
    """Grouped bar chart of per-class F1 scores."""
    sorted_classes = sorted(class_names)
    f1_mob = f1_score(y_true_mob, y_pred_mob, average=None)
    f1_eff = f1_score(y_true_eff, y_pred_eff, average=None)
    labels = [f"{CLASS_EMOJIS[c]} {c.capitalize()}" for c in sorted_classes]
    x, width = np.arange(len(sorted_classes)), 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - width / 2, f1_mob, width, label='MobileNetV2',    color='#E74C3C', alpha=0.85)
    b2 = ax.bar(x + width / 2, f1_eff, width, label='EfficientNetB0', color='#3498DB', alpha=0.85)
    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f'{bar.get_height():.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Score — Both Models', fontsize=14, fontweight='bold')
    ax.legend()
    ax.axhline(0.80, ls='--', color='gray', lw=1)
    _save(fig, save_path)


# ─── Sample predictions ───────────────────────────────────────────────────────
def plot_sample_predictions(
    test_df,
    model_mob: nn.Module,
    model_eff: nn.Module,
    val_transform: transforms.Compose,
    idx2label: dict,
    device: torch.device,
    n_display: int = 12,
    seed: int = 42,
    save_path: Path | None = None,
) -> None:
    """Grid of test images annotated with predictions from both models."""
    sample = test_df.sample(n=n_display, random_state=seed).reset_index(drop=True)
    cols   = 4
    rows   = (n_display + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3.8))
    fig.suptitle('🔍 Model Predictions on Sample Test Images',
                 fontsize=15, fontweight='bold', y=1.01)

    for idx, (_, row) in enumerate(sample.iterrows()):
        ax = axes[idx // cols][idx % cols]
        try:
            orig_img   = Image.open(str(row['filepath'])).convert('RGB')
            tensor_img = val_transform(orig_img).unsqueeze(0).to(device)

            with torch.no_grad():
                proba_mob = torch.softmax(model_mob(tensor_img), dim=1)[0]
                cls_mob   = idx2label[proba_mob.argmax().item()]
                conf_mob  = proba_mob.max().item()

                proba_eff = torch.softmax(model_eff(tensor_img), dim=1)[0]
                cls_eff   = idx2label[proba_eff.argmax().item()]
                conf_eff  = proba_eff.max().item()

            true_label   = row['label']
            border_color = '#2ECC71' if (cls_mob == true_label and cls_eff == true_label) else '#E74C3C'

            ax.imshow(orig_img)
            ax.set_title(
                f"True: {CLASS_EMOJIS.get(true_label, '')} {true_label}\n"
                f"MobV2: {cls_mob} ({conf_mob:.0%}) | "
                f"EffB0: {cls_eff} ({conf_eff:.0%})",
                fontsize=7.5, pad=4,
            )
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(border_color)
                spine.set_linewidth(3)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error\n{e}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=7)
        ax.axis('off')

    _save(fig, save_path)
