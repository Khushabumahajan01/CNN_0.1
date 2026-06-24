"""
Predict one Devanagari letter image with the trained CNN model.

Usage:
    python test_letter.py path/to/letter.png
    python test_letter.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from preprocessing import image_path_to_model_array

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf


tf.get_logger().setLevel(logging.ERROR)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "letter_model.h5"
CLASS_FILE = PROJECT_DIR / "class_names.json"
DATASET_DIR = PROJECT_DIR / "dataset"


def configure_console() -> None:
    """Print Devanagari text correctly in Windows terminals when possible."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def load_class_data() -> tuple[list[str], dict[str, str]]:
    """Load class order and readable Devanagari labels."""
    if not CLASS_FILE.exists():
        raise FileNotFoundError("class_names.json was not found. Run train.py first.")

    data = json.loads(CLASS_FILE.read_text(encoding="utf-8"))
    return data["class_names"], data["letters"]


def default_test_image(class_names: list[str]) -> Path:
    """Return one sample letter image when no image path is provided."""
    for class_name in class_names:
        class_dir = DATASET_DIR / class_name
        images = sorted(class_dir.glob("*.png"))
        if images:
            return images[0]

    raise FileNotFoundError("No image was provided and no image was found in dataset/.")


def parse_args() -> argparse.Namespace:
    """Parse the image path to predict."""
    parser = argparse.ArgumentParser(description="Predict a single Devanagari letter.")
    parser.add_argument("image", nargs="?", type=Path, help="Path to the letter image.")
    return parser.parse_args()


def main() -> None:
    """Load the model, predict one image, and print the best result."""
    configure_console()
    args = parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError("letter_model.h5 was not found. Run train.py first.")

    class_names, letters = load_class_data()
    image_path = args.image if args.image is not None else default_test_image(class_names)
    if not image_path.exists():
        raise FileNotFoundError(f"Image was not found: {image_path}")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    image = image_path_to_model_array(image_path)
    probabilities = model.predict(image, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    predicted_letter = letters[predicted_class]
    confidence = float(probabilities[predicted_index])

    print(f"Image: {image_path}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Predicted Letter: {predicted_letter}")
    print(f"Confidence: {confidence:.2%}")

    print("\nAll probabilities:")
    for class_name, probability in zip(class_names, probabilities):
        print(f"{letters[class_name]} ({class_name}): {float(probability):.2%}")


if __name__ == "__main__":
    main()
