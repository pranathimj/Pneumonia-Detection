# 🧠 Automated Pneumonia Detection via Convolutional Neural Networks

> A deep learning-based solution to detect pneumonia from chest X-rays with radiologist-level accuracy.

![Pneumonia Detection](https://img.shields.io/badge/AI-Healthcare-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Accuracy](https://img.shields.io/badge/Accuracy-98.46%25-brightgreen)

---

## 📌 Project Overview

Pneumonia remains one of the leading causes of death globally, particularly affecting children and the elderly. Early and accurate diagnosis is crucial. This project leverages **Convolutional Neural Networks (CNNs)** to detect pneumonia in chest X-ray images — providing fast, reliable, and scalable diagnostic support.

---

## 🚀 Highlights

- 🏥 **Clinically Inspired Design**  
- 📈 **Achieved 98.46% Accuracy**  
- 🧪 **Rigorous Testing & 5-Fold Cross-Validation**  
- 🧰 **Custom CNN Architecture**  
- 🎯 **Saliency Maps (Grad-CAM) for Interpretability**

---

## 🧬 Dataset

- 📦 Source: [Kaggle Pediatric Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- 🔢 Total Images: 5,856  
  - 👶 Normal: 1,583  
  - 🤒 Pneumonia: 4,273  
- ⚖️ Split:
  - 70% Training  
  - 10% Validation  
  - 20% Testing

---

## 🛠️ Methodology

### 🔍 Preprocessing
- Resize to 160×160
- Normalize pixel values
- Convert grayscale to RGB
- Real-time data augmentation using `ImageDataGenerator`

### 🧠 CNN Architecture
- 5 convolutional blocks: Conv → BatchNorm → ReLU → Pool
- Fully connected layer: 128 neurons
- Dropout: 0.5
- Output: Sigmoid for binary classification

### ⚙️ Training Setup
- Optimizer: Adam (`lr=0.001`)
- Loss: Binary Cross-Entropy
- Epochs: 30 (early stopping)
- Validation Strategy: 5-fold cross-validation

---

## 📊 Performance Metrics

| Metric      | Value        |
|-------------|--------------|
| Accuracy    | **98.46%**   |
| Precision   | 98.06%       |
| Recall      | 98.89%       |
| F1-Score    | 98.47%       |
| AUC         | 0.996        |

> 📌 CNN outperformed all classical ML baselines and even some pre-trained transfer learning models like DenseNet-121.

---

## 📈 Visual Explanations

Implemented **Grad-CAM** to visualize which regions influenced the model's decision the most:

![GradCAM Example](https://raw.githubusercontent.com/pranathimj/Pneumonia-Detection/main/assets/gradcam.png)

---

## 📚 Key Comparisons

| Model             | Accuracy  |
|------------------|-----------|
| Random Forest     | 97.6%     |
| DenseNet-121      | 98.1%     |
| **Our CNN (Custom)** | **98.46%** |
| ResNet-50         | 97.8%     |

---

## 📦 Requirements

- Python 3.8+
- TensorFlow 2.4
- Keras 2.4
- NumPy, Matplotlib, scikit-learn

Install all dependencies:


pip install -r requirements.txt

# Step 1: Activate your virtual environment
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Step 2: Train the model
python train.py

# Step 3: Predict from new images
python predict.py --image path/to/chest-xray.jpg

