"""
src/models.py
-------------
Model factory functions for MobileNetV2 and EfficientNetB0 with a custom
classification head suited for dog emotion classification.

Both models follow the same architecture pattern:
    Pre-trained backbone (ImageNet)
        → Identity (original classifier replaced)
        → BatchNorm1d
        → Linear(256) → ReLU → Dropout(0.4)
        → Linear(128) → ReLU → Dropout(0.25)
        → Linear(num_classes)

Usage
-----
    from src.models import build_mobilenetv2, build_efficientnetb0

    model_mob = build_mobilenetv2(num_classes=4, freeze_backbone=True)
    model_eff = build_efficientnetb0(num_classes=4, freeze_backbone=True)
"""

import torch
import torch.nn as nn
import torchvision.models as models

from configs.config import NUM_CLASSES, DROPOUT_RATE


# ─── Shared classification head ───────────────────────────────────────────────
class EmotionHead(nn.Module):
    """
    Shared classification head used by both architectures.

    Parameters
    ----------
    in_features : int
        Number of output features from the backbone.
    num_classes : int
        Number of target emotion classes (default: 4).
    dropout_rate : float
        Dropout probability for the first dropout layer (default: 0.4).
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = DROPOUT_RATE,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── MobileNetV2 ──────────────────────────────────────────────────────────────
class MobileNetV2Classifier(nn.Module):
    """
    MobileNetV2 backbone + custom EmotionHead.

    Attributes
    ----------
    base : torchvision MobileNetV2 with classifier replaced by Identity
    head : EmotionHead
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = DROPOUT_RATE,
    ) -> None:
        super().__init__()
        base = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V2
        )
        in_features = base.classifier[-1].in_features
        base.classifier = nn.Identity()
        self.base = base
        self.head = EmotionHead(in_features, num_classes, dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base(x)
        return self.head(x)


# ─── EfficientNetB0 ───────────────────────────────────────────────────────────
class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNetB0 backbone + custom EmotionHead.

    Attributes
    ----------
    base : torchvision EfficientNet-B0 with classifier replaced by Identity
    head : EmotionHead
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = DROPOUT_RATE,
    ) -> None:
        super().__init__()
        base = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        in_features = base.classifier[1].in_features
        base.classifier = nn.Identity()
        self.base = base
        self.head = EmotionHead(in_features, num_classes, dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base(x)
        return self.head(x)


# ─── Factory helpers ──────────────────────────────────────────────────────────
def _freeze_backbone(model: nn.Module) -> None:
    """Freeze all backbone (base) parameters."""
    for param in model.base.parameters():
        param.requires_grad = False


def _unfreeze_last_n(model: nn.Module, n: int = 30) -> None:
    """
    Freeze all backbone layers except the last `n`.
    Used for phase-2 fine-tuning.
    """
    params = list(model.base.parameters())
    cutoff = len(params) - n
    for i, param in enumerate(params):
        param.requires_grad = i >= cutoff


def build_mobilenetv2(
    num_classes: int = NUM_CLASSES,
    freeze_backbone: bool = True,
    device: torch.device | None = None,
) -> MobileNetV2Classifier:
    """
    Build and return a MobileNetV2Classifier.

    Parameters
    ----------
    freeze_backbone : bool
        If True, freeze backbone weights (Phase 1 mode).
        Set to False or call _unfreeze_last_n() for Phase 2.
    device : torch.device, optional
        Move model to this device.
    """
    model = MobileNetV2Classifier(num_classes=num_classes)
    if freeze_backbone:
        _freeze_backbone(model)
    if device is not None:
        model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'MobileNetV2  : {n_params:.2f}M total params')
    return model


def build_efficientnetb0(
    num_classes: int = NUM_CLASSES,
    freeze_backbone: bool = True,
    device: torch.device | None = None,
) -> EfficientNetB0Classifier:
    """
    Build and return an EfficientNetB0Classifier.

    Parameters
    ----------
    freeze_backbone : bool
        If True, freeze backbone weights (Phase 1 mode).
    device : torch.device, optional
        Move model to this device.
    """
    model = EfficientNetB0Classifier(num_classes=num_classes)
    if freeze_backbone:
        _freeze_backbone(model)
    if device is not None:
        model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'EfficientNetB0: {n_params:.2f}M total params')
    return model
