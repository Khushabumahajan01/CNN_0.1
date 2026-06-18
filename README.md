# DigitCNN

DigitCNN is a TensorFlow/Keras convolutional neural network project for handwritten digit recognition. It trains a digit classifier for classes `0` through `9` using your own handwritten digit images, generates augmented training samples, saves the trained model, and creates training evaluation plots.

## Features

- Trains a CNN for handwritten digit classification.
- Supports common image formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, and `.webp`.
- Accepts multiple dataset styles:
  - one combined image containing digits `0` to `9` from left to right
  - a flat folder with at least ten digit images sorted by filename
  - labeled folders such as `dataset/0`, `dataset/1`, ... `dataset/9`
  - a custom source folder with `0` through `9` subfolders
- Preprocesses images into MNIST-like `28x28` grayscale inputs.
- Creates augmented images with rotation, zoom, shift, and shear.
- Saves the trained model as `digit_model.h5`.
- Saves `accuracy_plot.png`, `loss_plot.png`, and `confusion_matrix.png`.

## Project Structure

```text
DigitCNN/
    train.py              # Train the CNN model
    test_digit.py         # Predict one digit image
    preprocessing.py      # Shared preprocessing helpers
    augment.py            # Extra augmentation script
    requirements.txt      # Python dependencies
    README.md
    dataset/              # Local labeled source images, ignored by Git
    augmented_dataset/    # Generated augmented data, ignored by Git
```

Generated files such as the trained model, plots, `dataset/`, and `augmented_dataset/` are ignored by Git.

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source venv/bin/activate
```

## Dataset Setup

The easiest layout is:

```text
dataset/
    0/
        0.jpeg
    1/
        1.png
    2/
        2.png
    ...
    9/
        9.png
```

You can add more than one image per digit folder. The training script will use all supported image files it finds.

You can also pass a custom source:

```bash
python train.py --source path/to/digits.png
```

or:

```bash
python train.py --source path/to/source_folder
```

## Training

Run:

```bash
python train.py
```

Useful options:

```bash
python train.py --epochs 20
python train.py --batch-size 32
python train.py --augment-count 200
python train.py --force-rebuild
python train.py --source path/to/source
```

`--force-rebuild` recreates processed dataset images and augmented images. If the source path is missing, the script keeps the existing labeled `dataset/` folder so your local images are not deleted.

## Prediction

After training, predict a new digit image with:

```bash
python test_digit.py path/to/new_digit.png
```

The script preprocesses the image the same way as the training data before sending it to the model.

## Outputs

After training, the project creates:

```text
digit_model.h5
accuracy_plot.png
loss_plot.png
confusion_matrix.png
```

These files are generated locally and are not committed to the repository.

## Notes

TensorFlow may print oneDNN or CPU/GPU warnings when training starts. Those messages are informational and do not usually indicate a failure. If training completes and saves `digit_model.h5`, the pipeline ran successfully.
