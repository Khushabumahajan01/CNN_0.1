"""Shared image preprocessing helpers for training, augmentation, and prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def preprocess_image(image: Image.Image) -> Image.Image:
    """Convert an image to grayscale, crop the digit, center it, and resize to 28x28."""
    gray = ImageOps.grayscale(image)
    arr = np.array(gray)

    # Make dark ink bright and white background dark, matching MNIST-like data.
    if arr.mean() > 127:
        arr = 255 - arr

    # Crop to the visible digit so resizing preserves the stroke area.
    mask = arr > 25
    if mask.any():
        y_indices, x_indices = np.where(mask)
        arr = arr[y_indices.min() : y_indices.max() + 1, x_indices.min() : x_indices.max() + 1]

    digit_img = Image.fromarray(arr.astype(np.uint8))
    digit_img.thumbnail((20, 20), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    x_offset = (28 - digit_img.width) // 2
    y_offset = (28 - digit_img.height) // 2
    canvas.paste(digit_img, (x_offset, y_offset))
    return canvas


def image_to_model_array(image: Image.Image) -> np.ndarray:
    """Preprocess an image and return a normalized CNN input array."""
    processed = preprocess_image(image)
    image_array = np.array(processed, dtype=np.float32) / 255.0
    return image_array.reshape((1, 28, 28, 1))


def image_path_to_model_array(image_path: Path) -> np.ndarray:
    """Load an image path and return a normalized CNN input array."""
    return image_to_model_array(Image.open(image_path))
