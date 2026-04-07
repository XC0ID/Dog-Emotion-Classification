# 🐶 Dog Emotion Classification

A deep learning-based computer vision project designed to classify dog emotions from images using transfer learning techniques. This project leverages pretrained CNN architectures like MobileNetV2 and EfficientNetB0 to achieve high performance with limited data.

---

## 📌 Project Objective

The main objective is to build a robust image classification system that can identify different emotional states of dogs. This can be useful in:
- Animal behavior analysis
- Veterinary applications
- Pet monitoring systems
- Smart surveillance

---

## 🧠 Models Implemented

| Model              | Type              | Key Advantage |
|-------------------|------------------|--------------|
| MobileNetV2       | Lightweight CNN  | Fast & efficient |
| EfficientNetB0    | Scaled CNN       | Higher accuracy |

Both models use a shared classification head called **EmotionHead**.

---

## 🏗️ Project Architecture

The system follows a modular deep learning pipeline:

1. Data Loading & Preprocessing  
2. Model Selection (MobileNet / EfficientNet)  
3. Transfer Learning  
4. Training (2 Phases)  
5. Evaluation  
6. Visualization  
7. Model Comparison  

---
## 📂 Project Structure
```
dog-emotion-classification/
├── README.md
├── requirements.txt
├── configs/
│   └── config.py
├── data/
│   └── README.md
├── src/
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   └── evaluate.py
├── utils/
│   └── visualization.py
├── scripts/
│   ├── train_mobilenet.py
│   ├── train_efficientnet.py
│   └── compare_models.py
├── notebooks/
│   └── dog-emotion-classification.ipynb
└── outputs/
```

---

## ⚙️ Installation

```bash
git clone https://github.com/XC0ID/Dog-Emotion-Classification.git
cd Dog-Emotion-Classification
pip install -r requirements.txt
```

---
## 📊 Dataset

* Download dataset from Kaggle (see data/README.md)
* Organize into class folders

---

## 🚀 Training

* Train MobileNetV2
```python scripts/train_mobilenet.py```
* Train EfficientNetB0
```python scripts/train_efficientnet.py```

---

## 📈 Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

---

## 📊 Visualization Features

Implemented in ```utils/visualization.py```:

* Class distribution plots
* Training vs validation loss curves
* Accuracy curves
* Confusion matrix
* ROC curves
* Model comparison graphs

---
## 🔍 Model Comparison

```python scripts/compare_models.py```

Outputs:

Performance comparison
Best model recommendation

---
## 📁 Outputs

Stored in ```outputs/```:

* Trained models (.pth)
* Graphs & plots
* Evaluation reports
* Label mapping (JSON)

---
## 🧩 Key Features

✔ Modular architecture

✔ Config-driven hyperparameters

✔ Transfer learning (2-phase training)

✔ Multi-model support

✔ Rich evaluation metrics

✔ Visualization utilities

---
## ⚠️ Limitations
Dataset size may be limited
Emotion labeling may be subjective
Performance depends on data quality

---
## 🚀 Future Enhancements

Add more emotion categories
Use Vision Transformers (ViT)
Deploy using Flask/Streamlit
Real-time emotion detection
Mobile app integration
