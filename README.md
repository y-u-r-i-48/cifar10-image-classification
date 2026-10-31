# Image Classification with MLP and CNN on CIFAR-10

A deep learning project that builds and compares multiple neural network architectures — MLP, CNN, and Transfer Learning — for image classification on the CIFAR-10 dataset using TensorFlow and Keras.

## Project Overview

This project explores how different neural network architectures perform on a 10-class image classification task. It progresses from a simple MLP baseline to a CNN with regularization techniques, and finally to transfer learning with a pretrained ResNet50.

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — 60,000 color images (32×32 pixels) across 10 classes, loaded directly via `keras.datasets.cifar10`.

| Label | Class |
|:---:|---|
| 0 | Airplane |
| 1 | Automobile |
| 2 | Bird |
| 3 | Cat |
| 4 | Deer |
| 5 | Dog |
| 6 | Frog |
| 7 | Horse |
| 8 | Ship |
| 9 | Truck |

- 50,000 training images / 10,000 test images
- No separate download needed — Keras loads it automatically

## Models & Results

| Model | Techniques | Test Accuracy |
|---|---|---|
| MLP | Flatten → Dense(512) → Dropout → Dense(256) → Dropout | ~43% |
| CNN | Conv layers + MaxPooling + Dropout + Data Augmentation | **82.32%** |
| Transfer Learning | Frozen ResNet50 + Dense head | 36.85% |

The CNN with data augmentation and dropout was the best performer, achieving **82%+ test accuracy**. The transfer learning result was lower because ResNet50 was pretrained on large 224×224 images, and CIFAR-10's 32×32 resolution limits how well those features transfer.

## Techniques Used

- **Data Augmentation** — random flips, rotations, and shifts to reduce overfitting
- **Dropout** — added after dense layers to improve generalization
- **Early Stopping** — monitors validation accuracy with patience=3 to stop training at the right time
- **Transfer Learning** — pretrained ResNet50 with frozen base layers and a custom classification head

## File Structure

```
├── Son_Yuri_hw4.ipynb   # Main notebook with all models
├── .gitignore
└── README.md
```

## Requirements

```
tensorflow
keras
numpy
```

## Usage

1. Open `Son_Yuri_hw4.ipynb` in Google Colab (recommended, GPU runtime) or Jupyter
2. Run all cells in order — the CIFAR-10 dataset downloads automatically
3. No additional setup needed
