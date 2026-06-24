"""Create augmented Devanagari letter images without training the model."""

from __future__ import annotations

from train import augment_dataset, build_base_dataset


if __name__ == "__main__":
    build_base_dataset(force_rebuild=False)
    augment_dataset(images_per_class=250, force_rebuild=False)
    print("Augmentation complete.")
