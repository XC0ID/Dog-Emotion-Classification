"""
src/train.py
------------
Training utilities for the two-phase transfer learning strategy.

Phase 1 — Head Training
    Backbone weights are frozen; only the custom EmotionHead is trained.
    Runs for EPOCHS_PHASE1 epochs with LR_PHASE1.

Phase 2 — Fine-Tuning
    Last UNFREEZE_LAST backbone layers are unfrozen and trained together with
    the head at a much lower learning rate (LR_PHASE2).

Usage
-----
    from src.train import train_model
    from src.models import build_mobilenetv2

    model = build_mobilenetv2(freeze_backbone=True, device=device)
    history = train_model(model, train_loader, val_loader, device,
                          checkpoint_path='outputs/best_mob.pth')
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from configs.config import (
    EPOCHS_PHASE1, EPOCHS_PHASE2,
    LR_PHASE1, LR_PHASE2,
    PATIENCE, PATIENCE_FT,
    UNFREEZE_LAST,
)
from src.models import _unfreeze_last_n


# ─── Single epoch helpers ─────────────────────────────────────────────────────
def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    return total_loss / len(loader), correct / total


def _validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run validation. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

    return total_loss / len(loader), correct / total


# ─── Phase runner ─────────────────────────────────────────────────────────────
def _run_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    n_epochs: int,
    patience: int,
    checkpoint_path: Path,
    best_val_acc: float,
    phase_label: str,
) -> tuple[dict, float]:
    """
    Generic training loop with early stopping and checkpointing.

    Returns
    -------
    history : dict with keys loss, accuracy, val_loss, val_accuracy
    best_val_acc : float — best validation accuracy seen so far
    """
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    patience_counter = 0

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = _train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = _validate(model, val_loader, criterion, device)

        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        # Log every epoch in phase 1, every 2 in phase 2
        log_every = 1 if 'Phase 1' in phase_label else 2
        if epoch % log_every == 0 or epoch == n_epochs:
            print(
                f'[{phase_label}] Epoch {epoch:>3}/{n_epochs} | '
                f'Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | '
                f'Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}'
            )

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'  ⏹  Early stopping at epoch {epoch}')
                break

    print(f'  ✅ {phase_label} complete — best val_acc: {best_val_acc:.4f}')
    return history, best_val_acc


# ─── Public API ───────────────────────────────────────────────────────────────
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    checkpoint_path: str | Path = 'outputs/best_model.pth',
) -> dict:
    """
    Run Phase 1 (head training) followed by Phase 2 (fine-tuning).

    The best checkpoint by validation accuracy is saved to `checkpoint_path`.

    Parameters
    ----------
    model : nn.Module
        A MobileNetV2Classifier or EfficientNetB0Classifier with frozen backbone.
    train_loader, val_loader : DataLoader
    device : torch.device
    checkpoint_path : str or Path
        Where to save the best model weights.

    Returns
    -------
    history : dict
        Keys: loss, accuracy, val_loss, val_accuracy (combined phases)
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()

    # ── Phase 1: train head only ──────────────────────────────────────────────
    print('\n🏋️  Phase 1 — Training classification head (backbone frozen)...')
    optimizer1 = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR_PHASE1
    )
    hist1, best_val_acc = _run_phase(
        model, train_loader, val_loader, criterion, optimizer1, device,
        n_epochs=EPOCHS_PHASE1,
        patience=PATIENCE,
        checkpoint_path=checkpoint_path,
        best_val_acc=0.0,
        phase_label='Phase 1',
    )

    # ── Phase 2: unfreeze last N backbone layers and fine-tune ────────────────
    print('\n🔬  Phase 2 — Fine-tuning last {} backbone layers...'.format(
        UNFREEZE_LAST
    ))
    _unfreeze_last_n(model, n=UNFREEZE_LAST)
    optimizer2 = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR_PHASE2
    )
    hist2, _ = _run_phase(
        model, train_loader, val_loader, criterion, optimizer2, device,
        n_epochs=EPOCHS_PHASE2,
        patience=PATIENCE_FT,
        checkpoint_path=checkpoint_path,
        best_val_acc=best_val_acc,
        phase_label='Phase 2',
    )

    # Merge histories
    full_history = {k: hist1[k] + hist2[k] for k in hist1}
    full_history['phase1_end'] = len(hist1['accuracy'])
    return full_history
