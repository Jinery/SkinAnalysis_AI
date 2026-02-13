<div align="center">

# SkinAnalysis AI 🔬🧠

**Deep Learning model for skin condition classification**  
MobileNetV2-based transfer learning model for detecting skin conditions from facial images

[![Python](https://img.shields.io/badge/Python-3.11.9-blue?style=for-the-badge&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-3.12.0-red?style=for-the-badge&logo=keras)](https://keras.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 📋 Overview

SkinAnalysis AI is a deep learning model designed to classify skin conditions from facial images into three categories:
- **Healthy** - Normal skin
- **Nevus** - Moles and benign growths  
- **Problem** - Potentially problematic lesions (BCC, ACK, SEK, SCC)

The model uses **transfer learning** with **MobileNetV2** architecture, making it lightweight and efficient for mobile deployment.

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.11.9
- pip package manager
- (Optional) CUDA-compatible GPU for faster training

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Jinery/SkinAnalysis_AI.git
cd SkinAnalysis_AI
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

## 🏋️ Training the Model
### Basic training
```bash
python train.py
```

### The training process:
1. **Creates data pipeline with 80/20 train/validation split**
2. **Builds MobileNetV2 with custom classification head**
3. **Trains for 30 epochs (configurable)**
4. **Fine-tunes for additional 7 epochs**
5. **Saves model to model/SkinAnalysis_AI.keras**

### Configuration
```python
IMG_SIZE = (224, 224)  # Input image size
BATCH_SIZE = 64        # Batch size
EPOCH = 30             # Training epochs
```

### Training features
* **Data augmentation: Random flips, rotations, zoom, noise, brightness, contrast**
* **Class weighting: Handles imbalanced datasets automatically**
* **Callbacks: Early stopping, model checkpoint, learning rate reduction**
* **Fine-tuning: Unfreezes base model for final optimization**

## 🔮 Making Predictions
### Test with multiple images

**Place test images in the check/ folder and run:**
``` bash
python predict.py
```
**The script will:**
* Load the trained model
* Process all images from check/ folder
* Display predictions with confidence scores

### Programmatic usage
```python
from predict import predict_single_image

# Local image
result = predict_single_image("path/to/image.jpg")

# URL image
result = predict_single_image("https://example.com/image.jpg")

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}%")
print("Top predictions:", result['top_predictions'])
```

### Output format
```json
{
    'success': True,
    'image': 'filename.jpg',
    'prediction': 'nevus',
    'confidence': 87.34,
    'top_predictions': [
        ['nevus', 87.34],
        ['healthy', 10.21],
        ['problem', 2.45]
    ]
}
```

## 📈 Visualization
**Training history and confusion matrices are automatically generated:**

### Training curves
```python
from paint import plot_training_history

plot_training_history(history)  # Saves to visualizations/training_history.png
```

### Confusion matrix
```python
from paint import plot_prediction_matrix

plot_prediction_matrix(model, validation_dataset, class_names, history)
```
**Outputs include:**
* Accuracy and loss curves
* Normalized confusion matrix
* Raw confusion matrix with counts
* Classification report (precision, recall, F1-score)

## 🧪 Model Architecture

```text
Input (224x224x3)
    ↓
Data Augmentation
    ↓
MobileNetV2 (pretrained, frozen initially)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(512, ReLU) + Dropout(0.4)
    ↓
Dense(256, ReLU) + Dropout(0.3)
    ↓
Dense(3, Softmax)
```

## ⚙️ Configuration
### Path management (utils.py)

**Toggle between local and Google Colab environments:**
```python
is_colab: bool = False  # Set to True for Colab
```

**Paths automatically adjust:**
* Local: ../dataset/, ../model/, etc.
* Colab: /content/dataset/, /content/model/, etc.

### Model naming
**Model files are defined in utils.py:**
```python
def get_model_and_weights_name():
    return ("SkinAnalysis_AI.keras", "SkinAnalysis_AI_best_weights.keras")
```

<div align="center"> Made with ❤️ and TensorFlow </div>