# 🌿 PlantGuard

**PlantGuard** is a deep learning-based plant disease detection system that classifies plant leaf images into 38 disease and healthy categories. It uses a fine-tuned EfficientNetB0 model trained on the PlantVillage dataset and is deployed as a Flask web application.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Supported Plants & Diseases](#supported-plants--diseases)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Team](#team)

---

## Overview

PlantGuard helps farmers and agricultural professionals quickly identify plant diseases from a simple leaf photo. Upload an image of a plant leaf and the model will predict the disease (or confirm the plant is healthy) along with a confidence score — and suggest possible treatments.

---

## Demo

> Upload a leaf image through the web interface and get an instant prediction with confidence score and recommended cures.

---

## Dataset

The model was trained on the **PlantVillage** dataset, split into:

| Split      | Images  | Classes |
|------------|---------|---------|
| Training   | 43,429  | 38      |
| Validation | 5,417   | 38      |
| Test       | 5,459   | 38      |

**Data Augmentation** applied during training: horizontal flip, vertical flip.

---

## Model Architecture

PlantGuard uses **transfer learning** with **EfficientNetB0** as the backbone, pre-trained on ImageNet.

- **Input size:** 224 × 224 × 3 (RGB)
- **Base model:** EfficientNetB0 (frozen initially)
- **Custom head:** GlobalAveragePooling → Dense(256, ReLU) → BatchNorm → Dropout → Dense(38, Softmax)
- **Total parameters:** ~4.7M (18 MB)
- **Trainable parameters (fine-tuning phase):** ~1.4M

### Training Strategy

Training was done in **two phases**:

**Phase 1 — Feature Extraction (10 epochs)**
- EfficientNetB0 base fully frozen
- Optimizer: Adam (lr = 0.001)
- Best val accuracy: **96.58%**

**Phase 2 — Fine-Tuning (30 epochs)**
- Only `block7` layers unfrozen; BatchNorm layers remain frozen
- Optimizer: Adam (lr = 0.00001)
- Best val accuracy: **98.60%**

**Callbacks used:** ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

---

## Results

| Metric             | Value    |
|--------------------|----------|
| Best Val Accuracy  | ~98.6%   |
| Final Val Loss     | ~0.043   |
| Classes            | 38       |

The model outputs a predicted class name and a confidence percentage for each image.

---

## Supported Plants & Diseases

The model detects **38 conditions** across **14 plant species**:

| Plant       | Conditions |
|-------------|-----------|
| Apple       | Scab, Black rot, Cedar apple rust, Healthy |
| Blueberry   | Healthy |
| Cherry      | Powdery mildew, Healthy |
| Corn (Maize)| Cercospora leaf spot, Common rust, Northern Leaf Blight, Healthy |
| Grape       | Black rot, Esca (Black Measles), Leaf blight, Healthy |
| Orange      | Huanglongbing (Citrus greening) |
| Peach       | Bacterial spot, Healthy |
| Pepper bell | Bacterial spot, Healthy |
| Potato      | Early blight, Late blight, Healthy |
| Raspberry   | Healthy |
| Soybean     | Healthy |
| Squash      | Powdery mildew |
| Strawberry  | Leaf scorch, Healthy |
| Tomato      | Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites, Target spot, Yellow Leaf Curl Virus, Mosaic virus, Healthy |

---

## Project Structure

```
PlantGuard/
├── templates/              # HTML templates for the Flask app
├── app.py                  # Flask web application
├── cures.py                # Disease treatment recommendations
├── best_model.keras        # Trained Keras model
├── PlantGuard.ipynb        # Training notebook (Kaggle)
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python runtime version
└── Developers              # Team information
```

---

## Installation

### Prerequisites

- Python 3.11+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yazan-03/PlantGuard.git
cd PlantGuard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Flask app
python app.py
```

Then open your browser and go to `http://localhost:5000`.

---

## Usage

1. Open the web app in your browser.
2. Upload a clear photo of a plant leaf.
3. Click **Predict**.
4. View the predicted disease name, confidence score, and recommended treatment.

---

## Technologies Used

- **Python 3.11**
- **TensorFlow / Keras 2.18** — model training and inference
- **EfficientNetB0** — pre-trained backbone (ImageNet)
- **Flask** — web application framework
- **OpenCV** — image preprocessing
- **Scikit-learn** — evaluation metrics
- **Matplotlib / Seaborn** — training visualization
- **Kaggle** — training environment (Tesla T4 GPU)

---

## Team

See the [`Developers`](./Developers) file for the full list of contributors.

---

## License

This project is for educational and research purposes.
