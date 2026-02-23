# 🍽️ Multimodal Calorie Prediction using Bi-LSTM and CNN

A multimodal deep learning system that predicts calorie intake by integrating **Continuous Glucose Monitoring (CGM) time-series data, neural features, and meal images**.

Achieved **Root Mean Square Relative Error (RMSRE) as low as 0.33**, demonstrating strong prediction accuracy.

---

## 🚀 Overview

Traditional calorie estimation methods rely on a single data source and often lack accuracy.  
This project introduces a **multimodal deep learning architecture** that combines:

- 📈 Time-series physiological data (CGM)
- 🧠 Neural / structured features
- 🖼️ Food images  

The model leverages **Bi-LSTM for sequential learning** and **CNN for image feature extraction**, followed by a **fusion layer for prediction**.

---

## 🧠 Key Features

- 🔁 **Bi-LSTM with Attention**
  - Captures temporal dependencies in CGM time-series data
  - Focuses on important time steps using attention mechanisms

- 🖼️ **CNN-based Image Processing**
  - Extracts visual features from meal images
  - Identifies food type and portion-related cues

- 🔗 **Multimodal Fusion**
  - Combines outputs from multiple subnetworks
  - Improves prediction accuracy over single-modality models

- ⚙️ **Custom Data Pipelines**
  - Synchronizes heterogeneous data sources
  - Handles time-series + structured + image inputs

---

## 🏗️ Model Architecture
