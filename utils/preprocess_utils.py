"""
Preprocessing utilities for Stage 1 animation cleanup.
Handles RGBA image loading, resizing, normalization, and dataset creation.
"""

import os
import cv2
import numpy as np
import torch

try:
    import cv2
except ImportError:
    raise ImportError("OpenCV (cv2) is required. Install with: pip install opencv-python")

# Optional imports for dataset functionality (only needed for training)
# Catch all exceptions to handle NumPy compatibility issues
try:
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader
    import albumentations as A
    HAS_DATASET_DEPS = True
except (ImportError, AttributeError, Exception):
    # Silently fail - these are only needed for training, not inference
    HAS_DATASET_DEPS = False
    pd = None
    Dataset = None
    DataLoader = None
    A = None

# Create a dummy Dataset class if not available (for class definition)
if Dataset is None:
    class Dataset:
        pass


def load_rgba(path):
    """Loads PNG and ensures 4 channels (RGBA)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    if img.shape[-1] == 3:  # no alpha → add opaque
        alpha = np.ones((img.shape[0], img.shape[1], 1), dtype=np.uint8) * 255
        img = np.concatenate([img, alpha], axis=-1)
    return img


def resize_and_pad_rgba(img, target_size=512):
    """Resizes with aspect ratio preserved, pads with transparent alpha."""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size, target_size, 4), dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
    return canvas


def normalize_rgba(img):
    """Normalize RGB to [-1,1], alpha to [0,1]."""
    img = img.astype(np.float32)
    rgb = (img[..., :3] / 127.5) - 1.0
    alpha = img[..., 3:] / 255.0
    return np.concatenate([rgb, alpha], axis=-1)


def get_train_augs():
    """Get augmentation pipeline for training."""
    if not HAS_DATASET_DEPS:
        raise ImportError("albumentations is required for augmentation. Install with: pip install albumentations")
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=7, border_mode=cv2.BORDER_CONSTANT, fill_value=0, p=0.3),
        A.RandomScale(scale_limit=0.1, p=0.3),
        A.RandomBrightnessContrast(0.05, 0.05, p=0.3),
        A.GaussianBlur(blur_limit=(3, 3), p=0.2),
    ], additional_targets={'target': 'image'})


# Only define Dataset class if dependencies are available
if HAS_DATASET_DEPS:
    class AnimationPhaseDataset(Dataset):
        """Dataset for animation phase pairs."""
        
        def __init__(self, manifest_path, augment=False, size=512):
            if not os.path.exists(manifest_path):
                raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
            self.df = pd.read_csv(manifest_path)
            self.size = size
            self.augment = augment
            self.augs = get_train_augs() if augment else None

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            input_img = load_rgba(row.input_path)
            target_img = load_rgba(row.target_path)

            # Preprocess
            input_img = resize_and_pad_rgba(input_img, self.size)
            target_img = resize_and_pad_rgba(target_img, self.size)

            # Albumentations expects dicts
            if self.augs:
                transformed = self.augs(image=input_img, target=target_img)
                input_img = transformed["image"]
                target_img = transformed["target"]

            # Normalize
            input_img = normalize_rgba(input_img)
            target_img = normalize_rgba(target_img)

            # Convert to Tensor
            input_tensor = torch.from_numpy(input_img.transpose(2, 0, 1)).float()
            target_tensor = torch.from_numpy(target_img.transpose(2, 0, 1)).float()

            return {
                "input": input_tensor,
                "target": target_tensor,
                "input_phase": row.input_phase,
                "target_phase": row.target_phase
            }
else:
    # Dummy class if dependencies not available
    class AnimationPhaseDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError("Dataset dependencies not available. Install with: pip install pandas torch albumentations")


def get_dataloaders(train_csv, val_csv, test_csv, batch_size=4, num_workers=2, augment=True, size=512):
    """Create dataloaders for train, validation, and test sets."""
    if not HAS_DATASET_DEPS:
        raise ImportError("Dataset dependencies required. Install with: pip install pandas torch albumentations")
    train_ds = AnimationPhaseDataset(train_csv, augment=augment, size=size)
    val_ds = AnimationPhaseDataset(val_csv, augment=False, size=size)
    test_ds = AnimationPhaseDataset(test_csv, augment=False, size=size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, test_loader

