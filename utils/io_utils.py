"""
I/O utilities for Stage 1 animation cleanup.
Handles file operations, directory creation, and tensor-to-image conversion.
"""

import os
import torch
import numpy as np
import cv2


def ensure_dir(path):
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(path, exist_ok=True)
    return path


def save_tensor_as_png(tensor, output_path, denormalize=True):
    """
    Save a tensor as PNG image.
    
    Args:
        tensor: PyTorch tensor of shape (4, H, W) or (C, H, W) where C=4
                RGB channels in [-1, 1], Alpha in [0, 1]
        output_path: Path to save the image
        denormalize: If True, denormalize RGB from [-1,1] to [0,255] and alpha from [0,1] to [0,255]
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Convert to numpy
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
    
    # Handle different tensor shapes
    if tensor.dim() == 4:  # (B, C, H, W) - take first batch
        tensor = tensor[0]
    elif tensor.dim() == 3:  # (C, H, W) - good
        pass
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}. Expected (C,H,W) or (B,C,H,W)")
    
    # Convert to numpy
    img = tensor.numpy()
    
    # Transpose from (C, H, W) to (H, W, C)
    if img.shape[0] == 4 or img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    
    if denormalize:
        # Denormalize RGB from [-1, 1] to [0, 255]
        if img.shape[2] >= 3:
            img[..., :3] = (img[..., :3] + 1.0) * 127.5
            img[..., :3] = np.clip(img[..., :3], 0, 255)
        
        # Denormalize Alpha from [0, 1] to [0, 255]
        if img.shape[2] == 4:
            img[..., 3] = img[..., 3] * 255.0
            img[..., 3] = np.clip(img[..., 3], 0, 255)
    
    # Convert to uint8
    img = img.astype(np.uint8)
    
    # Convert BGR to RGB if needed (OpenCV uses BGR)
    if img.shape[2] >= 3:
        img[..., :3] = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
    
    # Save as PNG
    if img.shape[2] == 4:
        # RGBA
        cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    else:
        # RGB
        cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

