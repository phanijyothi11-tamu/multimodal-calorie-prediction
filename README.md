# 🍽️ Multimodal Calorie Prediction using Bi-LSTM and CNN

A multimodal deep learning model combining Bi-LSTM and CNN to predict calorie intake using CGM time-series data, neural features, and food images. Achieved **RMSRE 0.33**.

---

## 🚀 Overview

This project presents a deep learning approach for accurate calorie prediction by integrating multiple data modalities:

* Continuous Glucose Monitoring (CGM) time-series data
* Neural / physiological features
* Meal images

The architecture combines **Bi-directional LSTM (Bi-LSTM)** for sequential data modeling and **Convolutional Neural Networks (CNNs)** for image feature extraction, followed by a fusion layer for final prediction.

---

## 🧠 Key Features

* **Bi-LSTM for Time-Series Analysis**

  * Captures temporal patterns in CGM data
  * Uses attention mechanism to focus on important signals

* **CNN for Image Processing**

  * Extracts visual features from meal images
  * Identifies food type and portion-related cues

* **Multimodal Data Fusion**

  * Combines outputs from different subnetworks
  * Improves prediction accuracy over single-modality models

* **Custom Data Preprocessing**

  * Synchronizes heterogeneous data sources
  * Handles time-series, structured, and image inputs

---

## 🏗️ Model Architecture

```
CGM Data ──▶ Bi-LSTM ──┐
                       ├──▶ Feature Fusion ──▶ Dense Layers ──▶ Prediction
Neural Data ───────────┤
Meal Images ─▶ CNN ────┘
```

---

## 📊 Performance

* **Metric Used**: Root Mean Square Relative Error (RMSRE)
* **Best RMSRE Achieved**: **0.33**

Additional Metrics:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)

---

## 📅 Timeline

* **Project Duration**: September 2024 – December 2024

---

## 🛠 Tech Stack

* Python
* PyTorch / TensorFlow
* NumPy, Pandas
* OpenCV / PIL
* Matplotlib / Seaborn

---

## 📁 Dataset

This project leverages **synthetic or anonymized multimodal datasets**, including:

* CGM time-series readings
* Neural / physiological features
* Meal image datasets

⚠️ The dataset is not publicly included due to confidentiality and privacy considerations.
You may plug in your own compatible data or contact for further details.

---

## ⚙️ How to Run

1. Clone the repository:
   git clone https://github.com/yourusername/multimodal-calorie-prediction.git

2. Navigate to the project folder:
   cd multimodal-calorie-prediction

3. Install required dependencies:
   pip install -r requirements.txt

4. Run the main script:
   python caloriePrediction.py

---
## 💡 Highlights

* Multimodal learning combining time-series, structured, and image data
* Use of attention mechanisms for improved prediction
* Real-world application in health monitoring and nutrition tracking
* Demonstrates strong understanding of deep learning and data fusion

---

## 🔗 Future Work

* Integrate real-world datasets
* Deploy as a web application (FastAPI + React)
* Improve explainability using attention visualization

---

## 👩‍💻 Author

**Phani Jyothi Kurada**
M.S. Computer Science, Texas A&M University

