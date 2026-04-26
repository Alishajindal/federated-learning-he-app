---
title: Federated Medical AI App
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
---

# Blood Cell Classification Demo

## Demonstration

<p align="center">
  <img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExdGZpMmpuYTc0NzMyMzN3bWpiZTgzZTRkdDVua3ZqbGJnYmN0YThpcSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/Ldfhwxsp3JtHVj6NV8/giphy.gif" width="650"/>
</p>

---

## Overview

This application provides a deployed interface for a privacy-preserving medical image classification system based on federated learning and homomorphic encryption.

Users can upload a blood cell image and obtain predictions from a model trained without sharing raw medical data.

---

## Features

- Real-time image classification  
- Top-3 predicted classes with confidence scores  
- Probability visualization  
- Streamlit-based interactive interface  

---

## Model Summary

- Vision Transformer (ViT)  
- Federated Learning (Non-IID setting)  
- CKKS Homomorphic Encryption  
- Secure parameter aggregation  

---

## How to Use

1. Upload a blood cell image  
2. The model processes the image  
3. Predictions and confidence scores are displayed  

---

## Repository Structure

```
.
├── app.py                 # Streamlit interface
├── predict.py             # Inference pipeline
├── models.py              # Model architecture
├── FeSVBiS.py             # Federated ViT implementation
├── SLViT.py               # Supporting modules
├── utils.py               # Utility functions
├── he_utils.py            # Homomorphic encryption utilities
├── dataset.py             # Dataset helpers (internal use)
├── final_state_dict.pth   # Trained model weights
└── requirements.txt       # Dependencies
```
---

## Disclaimer

This application is intended for research and demonstration purposes only.  
It is not suitable for clinical or medical use.

---

## Main Project

Full research implementation and experimental results:

👉 https://github.com/Alishajindal/federated-learning-he

---

## Author

Alisha Jindal  
B.E. Computer Engineering  
Thapar Institute of Engineering and Technology
