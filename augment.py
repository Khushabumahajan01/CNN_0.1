import os

import numpy as np
from PIL import Image
from preprocessing import preprocess_image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_dir = "dataset"
output_dir = "augmented_dataset"

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2
)

for digit in range(10):

    source_folder = os.path.join(dataset_dir, str(digit))
    target_folder = os.path.join(output_dir, str(digit))

    os.makedirs(target_folder, exist_ok=True)

    for filename in os.listdir(source_folder):

        img_path = os.path.join(source_folder, filename)

        # Use the same crop, center, grayscale, resize, and normalize steps as training/testing.
        img = preprocess_image(Image.open(img_path))
        x = np.array(img, dtype=np.float32) / 255.0
        x = x.reshape((1, 28, 28, 1))

        count = 0

        for batch in datagen.flow(
            x,
            batch_size=1,
            save_to_dir=target_folder,
            save_prefix=str(digit),
            save_format='png'
        ):
            count += 1

            if count >= 200:
                break

print("Augmentation Complete")
