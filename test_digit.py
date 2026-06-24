"""
Predict a handwritten digit image with the trained CNN model.

Usage:
    python test_digit.py path/to/new_digit.png
    python test_digit.py
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
from preprocessing import image_path_to_model_array


# Reduce TensorFlow startup noise before TensorFlow is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf


tf.get_logger().setLevel(logging.ERROR)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "digit_model.h5"
DATASET_DIR = PROJECT_DIR / "dataset"


def default_test_image() -> Path:
    """Return a sample processed digit image when no image path is provided."""
    for digit in range(10):
        digit_dir = DATASET_DIR / str(digit)
        source_images = sorted(digit_dir.glob("*_source.png"))
        if source_images:
            return source_images[0]

        any_images = sorted(digit_dir.glob("*.png"))
        if any_images:
            return any_images[0]

    raise FileNotFoundError("No test image was provided and no image was found in dataset/0 through dataset/9.")


def parse_args() -> argparse.Namespace:
    """Parse the path to the image that should be predicted."""
    parser = argparse.ArgumentParser(description="Predict a single handwritten digit.")
    parser.add_argument("image", nargs="?", type=Path, help="Path to the digit image.")
    return parser.parse_args()


def main() -> None:
    """Load the saved model, predict the image, and print the predicted digit."""
    args = parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError("digit_model.h5 was not found. Run train.py first.")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    image_path = args.image if args.image is not None else default_test_image()
    if not image_path.exists():
        raise FileNotFoundError(f"Image was not found: {image_path}")

    image = image_path_to_model_array(image_path)
    probabilities = model.predict(image, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))

    print(f"Image: {image_path}")
    print(f"Predicted Digit: {predicted_digit}")


if __name__ == "__main__":
    main()
