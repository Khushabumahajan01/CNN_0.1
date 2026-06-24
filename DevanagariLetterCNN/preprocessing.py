"""Image preprocessing helpers for Devanagari letter training and prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


IMAGE_SIZE = 28
INK_SIZE = 22


def preprocess_image(image: Image.Image) -> Image.Image:
    """Convert a letter image into a centered 28x28 grayscale model input."""
    gray = ImageOps.grayscale(image)
    arr = np.array(gray)

    # Training images are black ink on a white background. The CNN works better
    # with bright ink on a dark background, like MNIST.
    if arr.mean() > 127:
        arr = 255 - arr

    mask = arr > 25
    if mask.any():
        y_indices, x_indices = np.where(mask)
        arr = arr[y_indices.min() : y_indices.max() + 1, x_indices.min() : x_indices.max() + 1]

    letter = Image.fromarray(arr.astype(np.uint8))
    letter.thumbnail((INK_SIZE, INK_SIZE), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    x_offset = (IMAGE_SIZE - letter.width) // 2
    y_offset = (IMAGE_SIZE - letter.height) // 2
    canvas.paste(letter, (x_offset, y_offset))
    return canvas


def image_to_model_array(image: Image.Image) -> np.ndarray:
    """Preprocess one image and return a normalized CNN input array."""
    processed = preprocess_image(image)
    image_array = np.array(processed, dtype=np.float32) / 255.0
    return image_array.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 1))


def image_path_to_model_array(image_path: Path) -> np.ndarray:
    """Load an image path and return a normalized CNN input array."""
    with Image.open(image_path) as image:
        return image_to_model_array(image)
