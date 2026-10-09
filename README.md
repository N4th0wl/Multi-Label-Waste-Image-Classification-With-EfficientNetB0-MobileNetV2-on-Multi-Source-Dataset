# Waste Image Classification with EfficientNetB0 and MobileNetV2

**Six-class waste image classification using transfer learning (EfficientNetB0 and MobileNetV2) on a multi-source dataset.**

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow)
![Task](https://img.shields.io/badge/Task-Multi--class%20Classification-green)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Pipeline](#pipeline)
4. [Model Architecture](#model-architecture)
5. [Training Strategy](#training-strategy)
6. [Experimental Results](#experimental-results)
7. [Error Analysis](#error-analysis)
8. [Explainable AI (Grad-CAM)](#explainable-ai-grad-cam)
9. [Inference on New Images](#inference-on-new-images)
10. [Getting Started](#getting-started)
11. [Hyperparameter Configuration](#hyperparameter-configuration)
12. [Future Work](#future-work)
13. [Technologies](#technologies)

---

## Project Overview

This project implements a system that recognizes the type of waste in a photograph and assigns it to one of six categories:

| No. | Class | Description |
|-----|-------|-------------|
| 1 | `cardboard` | Cardboard |
| 2 | `glass` | Glass bottles and fragments |
| 3 | `metal` | Cans and metal objects |
| 4 | `paper` | Paper |
| 5 | `plastic` | Bottles, packaging, and plastic bags |
| 6 | `trash` | General and residual waste |

Two CNN architectures pre-trained on ImageNet, **MobileNetV2** (lightweight) and **EfficientNetB0** (more accurate), are compared under identical conditions, using the same data split, pipeline, and hyperparameters. The project additionally evaluates **Test-Time Augmentation (TTA)**, a probability-averaging **ensemble** of both models, and **Grad-CAM** visualizations that show which image regions drive each prediction.

**Main result:** EfficientNetB0 is the best-performing model, achieving an **accuracy of 93.44%** and a **macro F1-score of 0.9338** on the held-out test set (1,036 images).

---

## Dataset

The dataset is obtained directly from the research GitHub repository (the `dataset/` folder), which allows the experiment to be reproduced without manually downloading additional sources.

### Dataset Statistics

| Stage | Number of Images |
|-------|-----------------:|
| Images found in the repository | 7,042 |
| Corrupted images | 0 |
| Byte-identical duplicates (SHA-256) | 48 |
| Perceptual-hash duplicates (pHash) | 92 |
| **Total after cleaning** | **6,902** |

### Class Distribution

| Class | Before Cleaning | After Cleaning | Train | Validation | Test |
|-------|----------------:|---------------:|------:|-----------:|-----:|
| cardboard | 858 | 858 | 600 | 129 | 129 |
| glass | 857 | 836 | 585 | 126 | 125 |
| metal | 1,037 | 980 | 686 | 147 | 147 |
| paper | 853 | 823 | 576 | 123 | 124 |
| plastic | 1,842 | 1,830 | 1,280 | 275 | 275 |
| trash | 1,595 | 1,575 | 1,103 | 236 | 236 |
| **Total** | **7,042** | **6,902** | **4,830** | **1,036** | **1,036** |

The dataset is **imbalanced** (`plastic` and `trash` are considerably larger than the other classes); therefore, class weighting is applied during training.

### Labeling Rules

- Only folders whose names match one of the six target classes are used.
- Ambiguous source labels such as `recyclable`, `non-recyclable`, `organic`, `food waste`, `textile`, `vegetation`, and `miscellaneous` are **deliberately excluded** rather than arbitrarily mapped to another class (for example, `recyclable` → `plastic`), in order to preserve semantic validity.
- Folder names with numeric prefixes (for example, `0_cardboard`) are recognized automatically.

---

## Pipeline

```
Clone GitHub repository
      │
      ▼
Automatically discover labeled images (6 classes)
      │
      ▼
Integrity validation ─► Duplicate removal (SHA-256 + pHash)
      │
      ▼
Stratified 70 / 15 / 15 split (BEFORE augmentation)
      │
      ▼
Materialize train/ val/ test/ folders + save CSV manifest
      │
      ▼
tf.data pipeline (cache → shuffle → batch → augmentation)
      │
      ▼
Baseline training (frozen backbone)
      │
      ▼
Fine-tuning (unfreeze upper backbone layers)
      │
      ▼
Evaluation: Accuracy, Precision, Recall, F1, Confusion Matrix, TTA, Ensemble
      │
      ▼
Error analysis + Grad-CAM + Inference on new images
```

### Stage Details

**1. Data cleaning**
- **Integrity check:** every image is opened and fully decoded with PIL; corrupted images are discarded.
- **Exact duplicates:** detected using the SHA-256 hash of the file contents.
- **Perceptual duplicates:** detected using a 64-bit DCT-based pHash (resize to 32×32, take the 8×8 low-frequency coefficients, and threshold at the median). Matching is performed as an **exact match** and only **within the same class**. This is intentionally conservative so that visually similar but genuinely different waste images are not removed.

**2. Data splitting**
- A stratified 70% / 15% / 15% split (train / validation / test) with `random_state=42` preserves class proportions across all subsets.
- The split is performed **before** augmentation to prevent data leakage.
- A manifest file (`waste_dataset_manifest.csv`) records the split assignment of every image for traceability.

**3. Input pipeline (`tf.data`)**
- Images are loaded as `uint8` and cached in memory, so each JPEG is decoded only once.
- Shuffling is performed **per image, before batching**, and is reshuffled every epoch. An automatic sanity check verifies that each training batch contains at least four different classes.
- Augmentation is applied **after caching**, so it differs from epoch to epoch.
- Validation and test sets are neither shuffled nor augmented, which keeps their ordering deterministic for error analysis.

**4. Data augmentation (training data only)**

| Transformation | Parameter |
|----------------|-----------|
| Random flip | Horizontal |
| Random rotation | ±15% (0.15) |
| Random zoom | 20% |
| Random translation | 10% (horizontal and vertical) |
| Random contrast | 20% |

Vertical flipping is intentionally omitted because it produces unrealistic object orientations.

---

## Model Architecture

Both models share the same structure and differ only in their backbone:

```
Input (288×288×3)
   │
Preprocessing
   │  • MobileNetV2    : Rescaling to the range [-1, 1]
   │  • EfficientNetB0 : pass-through (preprocessing is built into the model)
   ▼
Pre-trained Backbone (ImageNet, include_top=False)
   ▼
GlobalAveragePooling2D
   ▼
Dropout (0.30)
   ▼
Dense (256, ReLU, L2 = 1e-4)
   ▼
Dropout (0.50)
   ▼
Dense (6, Softmax)
```

| Model | Total Parameters | `.keras` File Size |
|-------|-----------------:|-------------------:|
| MobileNetV2 | 2,587,462 | 27.71 MB |
| EfficientNetB0 | 4,379,049 | 45.51 MB |

> **Note:** Keras displays a warning that the pre-trained MobileNetV2 weights for non-224 inputs are loaded from the default 224×224 weights. This is expected behavior; because the architecture is fully convolutional, the weights remain applicable to 288×288 inputs.

---

## Training Strategy

Each model is trained in **two phases**.

### Phase 1: Baseline (Frozen Backbone)
- The entire backbone is frozen; only the classification head is trained.
- Learning rate of `1e-3`, with a maximum of 12 epochs.
- Early stopping with a patience of 4.

### Phase 2: Fine-Tuning
- The upper layers of the backbone are unfrozen: the last 60 layers for **MobileNetV2** (39 of 154 layers become trainable) and the last 90 layers for **EfficientNetB0** (71 of 238 layers become trainable).
- **BatchNormalization layers remain frozen** to preserve stable statistics.
- Learning rate of `1e-4`, with a maximum of 30 epochs and early stopping with a patience of 6.

### Regularization and Optimization

| Technique | Value |
|-----------|-------|
| Optimizer | AdamW (weight decay `1e-4`) |
| Loss | Categorical Crossentropy with label smoothing `0.10` |
| Class weights | `balanced` (scikit-learn) |
| Dropout | 0.30 and 0.50 |
| L2 regularization | `1e-4` on the 256-unit Dense layer |
| ModelCheckpoint | Saves the best model according to `val_accuracy` |
| EarlyStopping | Monitors `val_accuracy`, `restore_best_weights=True` |
| ReduceLROnPlateau | Monitors `val_loss`, factor 0.5, patience 2 |

**Class weights:**

| cardboard | glass | metal | paper | plastic | trash |
|----------:|------:|------:|------:|--------:|------:|
| 1.342 | 1.376 | 1.173 | 1.398 | 0.629 | 0.730 |

Training was conducted on Google Colab with a T4 GPU (approximately 100–140 seconds per epoch) using TensorFlow 2.21.

---

## Experimental Results

All figures below are computed on the **test set (1,036 images)**, which was never used for training or model selection.

### Model Comparison

| Model | TTA | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|-------|:---:|---------:|----------------:|-------------:|---------:|------------:|
| MobileNetV2 | No | 0.8842 | 0.8771 | 0.8824 | 0.8793 | 0.8843 |
| **EfficientNetB0** | No | **0.9344** | **0.9319** | **0.9361** | **0.9338** | **0.9345** |
| MobileNetV2 | Yes | 0.8871 | 0.8811 | 0.8844 | 0.8822 | 0.8873 |
| EfficientNetB0 | Yes | 0.9353 | 0.9304 | 0.9356 | 0.9328 | 0.9354 |
| Ensemble (MobileNetV2 + EfficientNetB0) | No | 0.9199 | 0.9176 | 0.9207 | 0.9191 | 0.9199 |
| Ensemble (MobileNetV2 + EfficientNetB0) | Yes | 0.9228 | 0.9202 | 0.9230 | 0.9214 | 0.9227 |

**TTA** (Test-Time Augmentation) here means averaging the predictions of the original image and its horizontally flipped version.

### Per-Class Metrics: EfficientNetB0 (without TTA)

| Class | Precision | Recall | F1-score | Support |
|-------|----------:|-------:|---------:|--------:|
| cardboard | 0.9375 | 0.9302 | 0.9339 | 129 |
| glass | 0.9070 | 0.9360 | 0.9213 | 125 |
| metal | 0.9216 | 0.9592 | 0.9400 | 147 |
| paper | 0.9280 | 0.9355 | 0.9317 | 124 |
| plastic | 0.9236 | 0.9236 | 0.9236 | 275 |
| trash | 0.9735 | 0.9322 | 0.9524 | 236 |

### Per-Class Metrics: MobileNetV2 (without TTA)

| Class | Precision | Recall | F1-score | Support |
|-------|----------:|-------:|---------:|--------:|
| cardboard | 0.8852 | 0.8372 | 0.8606 | 129 |
| glass | 0.8571 | 0.9120 | 0.8837 | 125 |
| metal | 0.8742 | 0.8980 | 0.8859 | 147 |
| paper | 0.8244 | 0.8710 | 0.8471 | 124 |
| plastic | 0.9007 | 0.8909 | 0.8958 | 275 |
| trash | 0.9207 | 0.8856 | 0.9028 | 236 |

### Key Findings

- **EfficientNetB0 outperforms MobileNetV2 by approximately five percentage points** in both accuracy and macro F1, at the cost of a larger model (45.5 MB versus 27.7 MB).
- **TTA yields only a marginal improvement** (+0.09 to +0.29 percentage points in accuracy). For EfficientNetB0, macro F1 decreases slightly (0.9338 → 0.9328), so the benefit of TTA in this setting is negligible.
- **The ensemble does not outperform EfficientNetB0 alone.** Averaging probabilities with the weaker MobileNetV2 lowers overall performance (0.9228 versus 0.9353 with TTA). In this experiment, EfficientNetB0 on its own is the best choice.
- The final model is selected by **macro F1**, followed by accuracy, because the dataset is imbalanced and macro F1 better reflects balanced performance across classes.

---

## Error Analysis

EfficientNetB0 makes **68 errors** out of 1,036 test images. The notebook displays the 12 most confident misclassifications, together with image galleries for the most frequently confused class pairs.

Most frequent error pairs for the Ensemble with TTA (true label → predicted label):

| True → Predicted | Count |
|------------------|------:|
| trash → plastic | 10 |
| plastic → trash | 7 |
| plastic → glass | 7 |
| cardboard → paper | 5 |
| glass → plastic | 5 |
| plastic → metal | 5 |
| glass → metal | 4 |
| paper → cardboard | 4 |

**Observed patterns:**
- **`trash` and `plastic`** are the most frequently confused classes. The `trash` class is heterogeneous (a mixture of many object types), so its boundary with `plastic` is blurred.
- **Plastic ↔ glass ↔ metal** confusions arise from transparent or reflective objects (bottles, packaging) that look visually similar.
- **Cardboard ↔ paper** confusions arise from similar texture and color.

---

## Explainable AI (Grad-CAM)

Grad-CAM is used to visualize the image regions that most influence a model's prediction. The implementation:

- **Automatically locates** the last 4D convolutional feature map in the backbone (`out_relu` for MobileNetV2 and `top_activation` for EfficientNetB0), rather than relying on hard-coded layer indices.
- Computes gradients with respect to the **pre-softmax logits** to avoid saturation.
- Normalizes the heatmap and overlays it (JET colormap) on the original image.
- Displays a side-by-side **MobileNetV2 versus EfficientNetB0** comparison grid for one test example per class.

---

## Inference on New Images

The final part of the notebook provides an `upload_and_predict()` function that predicts the waste class of new images:

1. The user uploads one or more images (upload button in Colab, or file path input outside Colab).
2. Preprocessing is identical to training: EXIF orientation correction, RGB conversion, and resizing to 288×288.
3. Predictions are produced by **both models with horizontal-flip TTA**, and the probabilities are averaged into an *ensemble* result.
4. The output includes the image, predicted class, confidence, a per-class probability chart, and a per-model breakdown.
5. If confidence is **below 50%**, the result is flagged as *"model is uncertain"*.

> The models recognize only the six classes listed above. Images outside these categories are still assigned to one of them, so clear photographs with a single centered object are recommended.

---

## Getting Started

### Option 1: Google Colab (Recommended)

1. Open `Multi-Label-Waste-Image-Classification-With-EfficientNetB0-MobileNetV2-on-Multi-Source-Dataset.ipynb` in Google Colab.
2. Enable the GPU: **Runtime → Change runtime type → T4 GPU**.
3. Run all cells in order (**Runtime → Run all**). The notebook will:
   - clone the repository containing the dataset,
   - clean and split the data,
   - train both models,
   - evaluate and save the results.

> Without a GPU, training runs on the CPU and is considerably slower (approximately 6–10 minutes per epoch).

### Option 2: Local Environment

```bash
git clone https://github.com/N4th0wl/Multi-Label-Waste-Image-Classification-With-EfficientNetB0-MobileNetV2-on-Multi-Source-Dataset.git
cd Multi-Label-Waste-Image-Classification-With-EfficientNetB0-MobileNetV2-on-Multi-Source-Dataset

pip install tensorflow numpy pandas matplotlib seaborn opencv-python pillow scikit-learn
jupyter notebook
```

Because the notebook uses Colab-specific paths (`/content/...`), update the variables `REPO_DIR`, `RAW_EXTRACT_DIR`, `COMBINED_DIR`, and `DATASET_DIR` in the configuration cell, as well as the model output paths, to match your local machine.

### Generated Outputs

| File | Description |
|------|-------------|
| `final_mobilenet_v2.keras` | Fine-tuned MobileNetV2 model |
| `final_efficientnet_b0.keras` | Fine-tuned EfficientNetB0 model |
| `best_<model>_<phase>.keras` | Best checkpoint of each training phase |
| `waste_dataset_manifest.csv` | List of files with their label, source, and split |

### Loading a Model for Prediction

```python
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

model = tf.keras.models.load_model("final_efficientnet_b0.keras")

img = ImageOps.exif_transpose(Image.open("example.jpg")).convert("RGB")
x = tf.image.resize(tf.keras.utils.img_to_array(img), (288, 288)).numpy()
x = np.expand_dims(x, axis=0)          # pixel scale 0-255, identical to training

probs = model.predict(x)[0]
print(CLASSES[int(np.argmax(probs))], f"{probs.max():.1%}")
```

---

## Hyperparameter Configuration

| Parameter | Value |
|-----------|-------|
| Random seed | 42 |
| Image size | 288 × 288 |
| Batch size | 32 |
| Split ratio | 70 / 15 / 15 |
| Baseline epochs | 12 |
| Fine-tuning epochs | 30 |
| Baseline learning rate | 1e-3 |
| Fine-tuning learning rate | 1e-4 |
| Dropout (head) | 0.30 and 0.50 |
| Label smoothing | 0.10 |
| Weight decay (AdamW) | 1e-4 |
| Pre-trained weights | ImageNet |
| Fine-tuned layers | MobileNetV2: 60, EfficientNetB0: 90 |

---

## Future Work

- Extend the label set (organic waste, textiles, e-waste, batteries) and add real-world field photographs.
- Store per-image *source* metadata to enable cross-dataset evaluation.
- Evaluate additional backbones (EfficientNetV2, ConvNeXt, Vision Transformer).
- Address class imbalance with focal loss or oversampling.
- Combine the classifier with **object detection** (e.g., YOLO) for images containing multiple waste items.
- Convert the models to **TensorFlow Lite** for deployment on mobile and edge devices.
- Build a demonstration application (Streamlit, Gradio, or a mobile app).

---

## Technologies

- **Framework:** TensorFlow / Keras 2.21
- **Models:** MobileNetV2, EfficientNetB0 (ImageNet pre-trained)
- **Data and evaluation:** NumPy, Pandas, scikit-learn
- **Image processing:** Pillow, OpenCV
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Google Colab (T4 GPU)
