# DigitCNN

TensorFlow/Keras CNN project for handwritten digit recognition from a custom digit source.

## Project Structure

```text
DigitCNN/
    dataset/
    augmented_dataset/
    train.py
    test_digit.py
    digit_model.h5
    accuracy_plot.png
    loss_plot.png
```

`digit_model.h5`, `accuracy_plot.png`, `loss_plot.png`, and `confusion_matrix.png` are created after training.

## Train

The default source is `../digits.png`. It may be either a single combined image with digits 0-9 from left to right, or a folder with at least ten digit images sorted in digit order.

```bash
pip install -r requirements.txt
python train.py --force-rebuild
```

The training script:

- Splits the custom source into digit folders.
- Converts images to grayscale.
- Resizes images to 28x28.
- Normalizes pixel values between 0 and 1.
- Generates at least 200 augmented images per digit with `ImageDataGenerator`.
- Splits the data into 80% training and 20% testing.
- Trains the CNN for 20 epochs with batch size 32.
- Prints the model summary.
- Saves the model and plots.
- Prints and saves a confusion matrix.

## Predict

```bash
python test_digit.py path/to/new_digit.png
```
