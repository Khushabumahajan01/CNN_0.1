"""
Train a CNN for handwritten digit recognition from a custom digit source.

The script can read either:
1. One combined image containing digits 0-9 from left to right, or
2. A folder containing digit images that sort into digit order.

It creates:
- dataset/0 ... dataset/9
- augmented_dataset/0 ... augmented_dataset/9
- digit_model.h5
- accuracy_plot.png
- loss_plot.png
- confusion_matrix.png
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from preprocessing import preprocess_image
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, save_img


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
AUGMENTED_DIR = PROJECT_DIR / "augmented_dataset"
MODEL_PATH = PROJECT_DIR / "digit_model.h5"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def create_digit_folders(base_dir: Path) -> None:
    """Create one folder per digit class."""
    for digit in range(10):
        (base_dir / str(digit)).mkdir(parents=True, exist_ok=True)


def image_paths_in_dir(directory: Path) -> list[Path]:
    """Return supported image files in a directory."""
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def has_digit_subfolders(directory: Path) -> bool:
    """Check whether a folder is organized as 0/, 1/, ... 9/."""
    return all((directory / str(digit)).is_dir() for digit in range(10))


def open_images(paths: list[Path]) -> list[Image.Image]:
    """Open image paths and detach them from file handles."""
    images: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.copy())
    return images


def find_connected_components(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Find bounding boxes for connected foreground components in a binary mask."""
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []

    for start_y in range(height):
        for start_x in range(width):
            if not mask[start_y, start_x] or visited[start_y, start_x]:
                continue

            stack = [(start_y, start_x)]
            visited[start_y, start_x] = True
            min_x = max_x = start_x
            min_y = max_y = start_y

            while stack:
                y, x = stack.pop()
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)

                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            if (max_x - min_x + 1) * (max_y - min_y + 1) > 30:
                boxes.append((min_x, min_y, max_x + 1, max_y + 1))

    return boxes


def split_combined_digit_image(source_image: Path) -> list[Image.Image]:
    """Automatically split one custom image into ten digit images."""
    image = Image.open(source_image).convert("L")
    arr = np.array(image)

    # Threshold the image so foreground handwriting can be separated from background.
    if arr.mean() > 127:
        foreground = arr < 220
    else:
        foreground = arr > 35

    boxes = find_connected_components(foreground)
    boxes = sorted(boxes, key=lambda box: (box[0], box[1]))

    # If connected-component detection does not produce ten boxes, fall back to equal slices.
    if len(boxes) != 10:
        width, height = image.size
        slice_width = width / 10
        boxes = []
        for index in range(10):
            left = int(round(index * slice_width))
            right = int(round((index + 1) * slice_width))
            boxes.append((left, 0, right, height))

    return [image.crop(box) for box in boxes[:10]]


def source_images_from_folder(source_dir: Path) -> dict[int, list[Image.Image]]:
    """Load digit images from either labeled subfolders or a flat sorted folder."""
    if has_digit_subfolders(source_dir):
        return source_images_from_labeled_dir(source_dir)

    paths = image_paths_in_dir(source_dir)
    if len(paths) < 10:
        raise ValueError(f"Expected at least 10 image files in {source_dir}, found {len(paths)}.")

    return {digit: open_images([paths[digit]]) for digit in range(10)}


def source_images_from_labeled_dir(labeled_dir: Path) -> dict[int, list[Image.Image]]:
    """Load source images from labeled <root>/<digit> folders."""
    digit_images: dict[int, list[Image.Image]] = {}
    for digit in range(10):
        digit_dir = labeled_dir / str(digit)
        paths = image_paths_in_dir(digit_dir)
        if not paths:
            raise ValueError(f"No source image found in {digit_dir}.")
        digit_images[digit] = open_images(paths)
    return digit_images


def base_dataset_is_ready() -> bool:
    """Check that every digit folder contains at least one already processed 28x28 image."""
    for digit in range(10):
        image_files = image_paths_in_dir(DATASET_DIR / str(digit))
        if not image_files:
            return False

        try:
            if not any(Image.open(image_file).size == (28, 28) for image_file in image_files):
                return False
        except OSError:
            return False

    return True


def build_base_dataset(source: Path, force_rebuild: bool = False) -> None:
    """Split/copy the custom source into dataset folders labeled 0 through 9."""
    if force_rebuild and DATASET_DIR.exists() and source.exists():
        shutil.rmtree(DATASET_DIR)
    elif force_rebuild and DATASET_DIR.exists() and not source.exists():
        print("Source image/folder was not found. Keeping dataset/ so existing labeled images are not deleted.")

    create_digit_folders(DATASET_DIR)

    if base_dataset_is_ready() and not force_rebuild:
        print("Base dataset already contains 28x28 processed images.")
        return

    if not force_rebuild:
        print("Base dataset is missing processed 28x28 images. Rebuilding processed source images.")

    if source.is_dir():
        digit_images = source_images_from_folder(source)
    elif source.is_file():
        digit_images = {digit: [image] for digit, image in enumerate(split_combined_digit_image(source))}
    elif DATASET_DIR.exists():
        print("Source image/folder was not found. Using existing labeled dataset images as the source.")
        digit_images = source_images_from_labeled_dir(DATASET_DIR)
    else:
        raise FileNotFoundError(f"Source path does not exist: {source}")

    if set(digit_images) != set(range(10)):
        raise ValueError("The source must provide digits 0-9.")

    for digit in range(10):
        for index, image in enumerate(digit_images[digit]):
            processed = preprocess_image(image)
            suffix = "source" if index == 0 else f"source_{index + 1}"
            processed.save(DATASET_DIR / str(digit) / f"{digit}_{suffix}.png")

    print("Saved split and preprocessed digit images in dataset/0 through dataset/9.")


def augment_dataset(images_per_digit: int = 200, force_rebuild: bool = False) -> None:
    """Create augmented digit images using Keras ImageDataGenerator."""
    if force_rebuild and AUGMENTED_DIR.exists():
        shutil.rmtree(AUGMENTED_DIR)

    create_digit_folders(AUGMENTED_DIR)

    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        fill_mode="constant",
        cval=0.0,
    )

    for digit in range(10):
        output_dir = AUGMENTED_DIR / str(digit)
        existing_count = len(list(output_dir.glob("*.png")))
        if existing_count >= images_per_digit:
            continue

        source_files = image_paths_in_dir(DATASET_DIR / str(digit))
        if not source_files:
            raise ValueError(f"No source images found for digit {digit}.")

        source_index = 0
        for index in range(existing_count, images_per_digit):
            # Use every source image for this digit, not only the first one.
            # This lets files such as 7_source2.png and 7_source3.png improve diversity.
            source_file = source_files[source_index % len(source_files)]
            source_index += 1
            image = preprocess_image(Image.open(source_file))
            image_array = np.array(image, dtype=np.float32) / 255.0
            image_array = image_array.reshape((1, 28, 28, 1))
            augmented = datagen.random_transform(image_array[0])
            save_img(output_dir / f"{digit}_aug_{index:04d}.png", augmented, scale=True)

    print(f"Generated at least {images_per_digit} augmented images for every digit.")


def load_augmented_data() -> tuple[np.ndarray, np.ndarray]:
    """Load augmented images and normalize pixel values between 0 and 1."""
    images: list[np.ndarray] = []
    labels: list[int] = []

    for digit in range(10):
        for image_path in sorted((AUGMENTED_DIR / str(digit)).glob("*.png")):
            image = Image.open(image_path).convert("L").resize((28, 28))
            images.append(np.array(image, dtype=np.float32) / 255.0)
            labels.append(digit)

    x = np.array(images, dtype=np.float32).reshape((-1, 28, 28, 1))
    y = np.array(labels, dtype=np.int64)
    return x, y


def train_test_split(x: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> tuple[np.ndarray, ...]:
    """Split the dataset into 80% training and 20% testing."""
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(x))
    test_count = int(len(x) * test_size)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    return x[train_indices], x[test_indices], y[train_indices], y[test_indices]


def build_model() -> Sequential:
    """Build the required TensorFlow/Keras CNN architecture."""
    model = Sequential(
        [
            Input(shape=(28, 28, 1)),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def plot_training_history(history: tf.keras.callbacks.History) -> None:
    """Save accuracy and loss plots for training and validation."""
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PROJECT_DIR / "accuracy_plot.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PROJECT_DIR / "loss_plot.png")
    plt.close()


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print and save a confusion matrix after testing."""
    matrix = np.zeros((10, 10), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1

    print("Confusion Matrix:")
    print(matrix)

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Digit")
    plt.ylabel("True Digit")
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.colorbar()

    for row in range(10):
        for col in range(10):
            plt.text(col, row, matrix[row, col], ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(PROJECT_DIR / "confusion_matrix.png")
    plt.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    default_source_folder = PROJECT_DIR.parent / "digits.png"
    parser = argparse.ArgumentParser(description="Train a CNN digit classifier from a custom digit image.")
    parser.add_argument("--source", type=Path, default=default_source_folder, help="Combined digit image or folder.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--augment-count", type=int, default=200, help="Augmented images per digit.")
    parser.add_argument("--force-rebuild", action="store_true", help="Recreate dataset and augmented images.")
    return parser.parse_args()


def main() -> None:
    """Run the complete digit recognition training pipeline."""
    args = parse_args()
    tf.random.set_seed(42)
    np.random.seed(42)

    build_base_dataset(args.source, force_rebuild=args.force_rebuild)
    augment_dataset(images_per_digit=args.augment_count, force_rebuild=args.force_rebuild)

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

    predictions = model.predict(x_test, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    save_confusion_matrix(y_test, predicted_labels)


if __name__ == "__main__":
    main()
