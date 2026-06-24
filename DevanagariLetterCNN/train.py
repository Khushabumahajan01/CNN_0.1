"""
Train a CNN to predict selected Devanagari letters.

The project expects this source layout:

devnagri_dataset/
    devanagari_dataset/
        ai/
        o/
        au/
        am/
        ah/
        ka/
        kha/
        ga/

It creates:
- dataset/
- augmented_dataset/
- letter_model.h5
- class_names.json
- accuracy_plot.png
- loss_plot.png
- confusion_matrix.png
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
from PIL import Image
from preprocessing import preprocess_image
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator, save_img


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_DIR / "devnagri_dataset" / "devanagari_dataset"
DATASET_DIR = PROJECT_DIR / "dataset"
AUGMENTED_DIR = PROJECT_DIR / "augmented_dataset"
MODEL_PATH = PROJECT_DIR / "letter_model.h5"
CLASS_FILE = PROJECT_DIR / "class_names.json"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

CLASS_NAMES = ["ai", "o", "au", "am", "ah", "ka", "kha", "ga"]
LETTER_LABELS = {
    "ai": "ऐ",
    "o": "ओ",
    "au": "औ",
    "am": "अं",
    "ah": "अः",
    "ka": "क",
    "kha": "ख",
    "ga": "ग",
}


def configure_plot_font() -> None:
    """Use a Devanagari-capable font for plots when one is installed."""
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("Nirmala UI", "Mangal", "Aparajita", "Kokila"):
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            return


def image_paths_in_dir(directory: Path) -> list[Path]:
    """Return supported image files from a directory."""
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def create_class_folders(base_dir: Path) -> None:
    """Create one folder for each Devanagari letter class."""
    for class_name in CLASS_NAMES:
        (base_dir / class_name).mkdir(parents=True, exist_ok=True)


def save_class_names() -> None:
    """Save class order and readable labels used by training and prediction."""
    data = {
        "class_names": CLASS_NAMES,
        "letters": LETTER_LABELS,
    }
    CLASS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_base_dataset(force_rebuild: bool = False) -> None:
    """Preprocess source letter images into dataset/<class_name>/ folders."""
    if force_rebuild and DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    create_class_folders(DATASET_DIR)
    save_class_names()

    ready = all(image_paths_in_dir(DATASET_DIR / class_name) for class_name in CLASS_NAMES)
    if ready and not force_rebuild:
        print("Base dataset already exists.")
        return

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source dataset was not found: {SOURCE_DIR}")

    for class_name in CLASS_NAMES:
        source_files = image_paths_in_dir(SOURCE_DIR / class_name)
        if not source_files:
            raise ValueError(f"No source images found for class: {class_name}")

        target_dir = DATASET_DIR / class_name
        for old_file in target_dir.glob("*.png"):
            old_file.unlink()

        for index, source_file in enumerate(source_files):
            with Image.open(source_file) as image:
                processed = preprocess_image(image)
            processed.save(target_dir / f"{class_name}_source_{index:03d}.png")

    print("Saved preprocessed letter images in dataset/.")


def augment_dataset(images_per_class: int = 250, force_rebuild: bool = False) -> None:
    """Create augmented data for every Devanagari letter class."""
    if force_rebuild and AUGMENTED_DIR.exists():
        shutil.rmtree(AUGMENTED_DIR)

    create_class_folders(AUGMENTED_DIR)

    datagen = ImageDataGenerator(
        rotation_range=12,
        zoom_range=0.15,
        width_shift_range=0.12,
        height_shift_range=0.12,
        shear_range=0.10,
        fill_mode="constant",
        cval=0.0,
    )

    for class_name in CLASS_NAMES:
        output_dir = AUGMENTED_DIR / class_name
        existing_count = len(list(output_dir.glob("*.png")))
        if existing_count >= images_per_class:
            continue

        source_files = image_paths_in_dir(DATASET_DIR / class_name)
        if not source_files:
            raise ValueError(f"No processed source images found for class: {class_name}")

        source_index = 0
        for index in range(existing_count, images_per_class):
            source_file = source_files[source_index % len(source_files)]
            source_index += 1

            with Image.open(source_file) as image:
                processed = preprocess_image(image)
            image_array = np.array(processed, dtype=np.float32) / 255.0
            image_array = image_array.reshape((28, 28, 1))
            augmented = datagen.random_transform(image_array)
            save_img(output_dir / f"{class_name}_aug_{index:04d}.png", augmented, scale=True)

    print(f"Generated at least {images_per_class} augmented images for every class.")


def load_augmented_data() -> tuple[np.ndarray, np.ndarray]:
    """Load augmented images and numeric labels."""
    images: list[np.ndarray] = []
    labels: list[int] = []

    for label, class_name in enumerate(CLASS_NAMES):
        for image_path in sorted((AUGMENTED_DIR / class_name).glob("*.png")):
            with Image.open(image_path) as image:
                gray = image.convert("L").resize((28, 28))
            images.append(np.array(gray, dtype=np.float32) / 255.0)
            labels.append(label)

    x = np.array(images, dtype=np.float32).reshape((-1, 28, 28, 1))
    y = np.array(labels, dtype=np.int64)
    return x, y


def train_test_split(x: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> tuple[np.ndarray, ...]:
    """Split data into training and testing sets."""
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(x))
    test_count = int(len(x) * test_size)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    return x[train_indices], x[test_indices], y[train_indices], y[test_indices]


def build_model() -> Sequential:
    """Build a small readable CNN using RMSprop optimizer."""
    model = Sequential(
        [
            Input(shape=(28, 28, 1)),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(96, (3, 3), activation="relu"),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.35),
            Dense(len(CLASS_NAMES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer=RMSprop(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_history(history: tf.keras.callbacks.History) -> None:
    """Save accuracy and loss curves."""
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Devanagari Letter Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PROJECT_DIR / "accuracy_plot.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Devanagari Letter Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PROJECT_DIR / "loss_plot.png")
    plt.close()


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print and save a confusion matrix."""
    class_count = len(CLASS_NAMES)
    matrix = np.zeros((class_count, class_count), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1

    print("Confusion Matrix:")
    print(matrix)

    labels = [LETTER_LABELS[name] for name in CLASS_NAMES]
    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Letter")
    plt.ylabel("True Letter")
    plt.xticks(range(class_count), labels)
    plt.yticks(range(class_count), labels)
    plt.colorbar()

    for row in range(class_count):
        for col in range(class_count):
            plt.text(col, row, matrix[row, col], ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(PROJECT_DIR / "confusion_matrix.png")
    plt.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line settings."""
    parser = argparse.ArgumentParser(description="Train a Devanagari letter CNN.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--augment-count", type=int, default=250, help="Augmented images per class.")
    parser.add_argument("--force-rebuild", action="store_true", help="Recreate dataset and augmented images.")
    return parser.parse_args()


def main() -> None:
    """Run the complete training pipeline."""
    args = parse_args()
    configure_plot_font()
    tf.random.set_seed(42)
    np.random.seed(42)

    build_base_dataset(force_rebuild=args.force_rebuild)
    augment_dataset(images_per_class=args.augment_count, force_rebuild=args.force_rebuild)

    x, y = load_augmented_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = build_model()
    model.summary()

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    plot_training_history(history)

    probabilities = model.predict(x_test, verbose=0)
    predicted_labels = np.argmax(probabilities, axis=1)
    save_confusion_matrix(y_test, predicted_labels)


if __name__ == "__main__":
    main()
