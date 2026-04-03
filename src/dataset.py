"""
src/dataset.py
--------------
PyTorch Dataset class and DataLoader factory for the Dog Emotion dataset.

The dataset is loaded from a CSV file that maps image filenames to emotion
labels.  Images are stored in per-class subdirectories under BASE_DIR.

Usage
-----
    from src.dataset import load_dataframes, build_dataloaders

    train_df, val_df, test_df, label2idx, idx2label = load_dataframes()
    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df
    )
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from configs.config import (
    BASE_DIR, CSV_PATH,
    IMG_SIZE, BATCH_SIZE, SEED,
    VAL_RATIO, TEST_RATIO,
    IMAGENET_MEAN, IMAGENET_STD,
)


# ─── Reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_dataframes(
    base_dir: Path = BASE_DIR,
    csv_path: Path = CSV_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    """
    Load the dataset CSV, verify file existence, encode labels, and split into
    train / val / test DataFrames.

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
        Each contains columns: filename, label, filepath, label_idx
    label2idx : dict[str, int]
    idx2label : dict[int, str]
    """
    df = pd.read_csv(csv_path, index_col=0)
    df.columns = ['filename', 'label']
    df['label'] = df['label'].str.strip().str.lower()

    # Build full file paths
    df['filepath'] = df.apply(
        lambda r: base_dir / r['label'] / r['filename'], axis=1
    )

    # Drop missing images
    df['exists'] = df['filepath'].apply(lambda p: p.exists())
    n_missing = (~df['exists']).sum()
    if n_missing:
        print(f'⚠️  Skipping {n_missing} missing file(s).')
    df = df[df['exists']].reset_index(drop=True)

    # Encode labels
    label2idx = {c: i for i, c in enumerate(sorted(df['label'].unique()))}
    idx2label = {v: k for k, v in label2idx.items()}
    df['label_idx'] = df['label'].map(label2idx)

    # Stratified split: 70 / 15 / 15
    temp_size = VAL_RATIO + TEST_RATIO          # 0.30
    half      = TEST_RATIO / temp_size           # 0.50 of the 30 %

    train_df, temp_df = train_test_split(
        df, test_size=temp_size, stratify=df['label'], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=half, stratify=temp_df['label'], random_state=SEED
    )

    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    print(f'Train : {len(train_df):,}  ({len(train_df)/len(df)*100:.1f}%)')
    print(f'Val   : {len(val_df):,}   ({len(val_df)/len(df)*100:.1f}%)')
    print(f'Test  : {len(test_df):,}   ({len(test_df)/len(df)*100:.1f}%)')

    return train_df, val_df, test_df, label2idx, idx2label


# ─── Transform factories ──────────────────────────────────────────────────────
def get_train_transform() -> transforms.Compose:
    """Return augmentation pipeline for the training set."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform() -> transforms.Compose:
    """Return deterministic pipeline for val/test sets."""
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─── Dataset class ────────────────────────────────────────────────────────────
class DogEmotionDataset(Dataset):
    """
    Maps rows in a DataFrame to (image_tensor, label_index) pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'filepath' and 'label_idx' columns.
    transform : torchvision.transforms.Compose, optional
        Image transforms to apply.  Defaults to val transform (no augmentation).
    """

    def __init__(self, df: pd.DataFrame, transform=None) -> None:
        self.df        = df.reset_index(drop=True)
        self.transform = transform or get_val_transform()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row   = self.df.iloc[idx]
        img   = Image.open(str(row['filepath'])).convert('RGB')
        label = int(row['label_idx'])
        img   = self.transform(img)
        return img, label


# ─── DataLoader factory ───────────────────────────────────────────────────────
def build_dataloaders(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wrap DataFrames in Dataset objects and return DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    train_ds = DogEmotionDataset(train_df, transform=get_train_transform())
    val_ds   = DogEmotionDataset(val_df,   transform=get_val_transform())
    test_ds  = DogEmotionDataset(test_df,  transform=get_val_transform())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
