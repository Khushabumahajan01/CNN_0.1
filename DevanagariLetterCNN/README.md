# DevanagariLetterCNN

Simple TensorFlow/Keras CNN project for predicting these Devanagari letters:

`ऐ`, `ओ`, `औ`, `अं`, `अः`, `क`, `ख`, `ग`

The model uses **RMSprop (Root Mean Square Propagation)** as the optimizer.

## Project Structure

```text
DevanagariLetterCNN/
    train.py
    test_letter.py
    preprocessing.py
    augment.py
    requirements.txt
    README.md
    devnagri_dataset/       # Original letter dataset
    dataset/                # Preprocessed 28x28 images
    augmented_dataset/      # Generated augmented images
```

## Install

Use Python 3.11 for TensorFlow on Windows. Some newer Python versions do not
have TensorFlow wheels yet.

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

Useful options:

```bash
python train.py --epochs 20
python train.py --augment-count 250
python train.py --force-rebuild
```

## Predict

```bash
python test_letter.py path/to/letter.png
```

If no image path is given, `test_letter.py` predicts one sample from `dataset/`.

## Outputs

Training creates:

```text
letter_model.h5
class_names.json
accuracy_plot.png
loss_plot.png
confusion_matrix.png
dataset/
augmented_dataset/
```
