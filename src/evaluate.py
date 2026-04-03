"""
src/evaluate.py
---------------
Model evaluation on the held-out test set.

Computes:
  - Overall accuracy
  - Macro F1-score
  - AUC (one-vs-rest)
  - Per-class classification report
  - Confusion matrix (raw counts)

Usage
-----
    from src.evaluate import evaluate_model

    metrics, y_true, y_pred = evaluate_model(
        model, test_loader, device, model_name='MobileNetV2'
    )
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    model_name: str = 'Model',
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Evaluate `model` on `test_loader` and print a summary.

    Parameters
    ----------
    model : nn.Module
        Trained model (weights already loaded).
    test_loader : DataLoader
        Test set DataLoader (shuffle=False).
    device : torch.device
    model_name : str
        Display name used in print output.

    Returns
    -------
    metrics : dict
        Keys: model, loss, accuracy, auc, f1
    y_true : np.ndarray   (int labels)
    y_pred : np.ndarray   (int predictions)
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()

    y_true_list, y_pred_list, y_proba_list = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            total_loss += criterion(outputs, labels).item()
            proba = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            y_true_list.extend(labels.cpu().numpy())
            y_pred_list.extend(predicted.cpu().numpy())
            y_proba_list.extend(proba.cpu().numpy())

    y_true  = np.array(y_true_list)
    y_pred  = np.array(y_pred_list)
    y_proba = np.array(y_proba_list)

    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(y_true, y_pred)
    f1       = f1_score(y_true, y_pred, average='macro')
    auc      = roc_auc_score(y_true, y_proba, multi_class='ovr')

    sep = '─' * 45
    print(f'\n{sep}')
    print(f'  {model_name}')
    print(f'{sep}')
    print(f'  Loss     : {avg_loss:.4f}')
    print(f'  Accuracy : {accuracy:.4f}  ({accuracy * 100:.2f}%)')
    print(f'  AUC      : {auc:.4f}')
    print(f'  Macro F1 : {f1:.4f}')
    print(f'{sep}')

    metrics = dict(
        model=model_name,
        loss=avg_loss,
        accuracy=accuracy,
        auc=auc,
        f1=f1,
    )
    return metrics, y_true, y_pred


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    class_emojis: dict[str, str],
    model_name: str = 'Model',
) -> None:
    """Print a per-class classification report with emojis."""
    sorted_classes = sorted(class_names)
    target_names = [
        f"{class_emojis.get(c, '')} {c.capitalize()}" for c in sorted_classes
    ]
    print('=' * 60)
    print(f'  📋 {model_name} — Classification Report')
    print('=' * 60)
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Return the raw confusion matrix."""
    return confusion_matrix(y_true, y_pred)
