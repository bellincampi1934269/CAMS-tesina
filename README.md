# Multimodal Prediction of Type 2 Diabetes Onset

This repository contains the source code and experimental results for the CAMS course final project. The objective of this project is to implement a multimodal neural network capable of predicting the onset of Type 2 Diabetes using longitudinal Electronic Health Records (EHR).

## Project Overview

Predicting disease onset from clinical data is challenging due to the asynchronous and irregularly sampled nature of medical visits. This project tackles this issue by combining static patient demographics with dynamic, time-series clinical observations (such as HbA1c, fasting blood glucose, BMI, and blood pressure).

### Key Features
* **Multimodal Architecture:** A PyTorch-based neural network that processes both time-invariant data (e.g., biological sex, baseline age) and temporal sequences.
* **Temporal Discretization & Imputation:** Implementation of a Zero-Order Hold (Forward Fill) strategy to manage missing values in irregularly sampled temporal data without altering the clinical chronological sequence.
* **Class Imbalance Handling:** Utilization of a dynamically weighted Binary Cross-Entropy loss function to penalize False Negatives, making the model highly sensitive for early-warning clinical decision support.

## Dataset

The model was trained and evaluated on a cohort of 12,352 synthetic patients generated using [Synthea](https://github.com/synthetichealth/synthea). Synthea simulates the life cycle of virtual patients based on Markov chain transition models derived from real clinical guidelines, providing dense, longitudinal data.

*(Note: The original raw `.csv` files are not included in this repository due to file size limits, but the cohort can be reproduced using the Synthea engine and targeting the SNOMED-CT code `44054006` for Type 2 Diabetes).*

## Repository Structure

* `build_tensors.py`: Script for data engineering. It loads the Synthea CSV files, extracts the specific cohort, applies the Zero-Order Hold imputation, and builds the static and dynamic tensors.
* `train_model.py`: Contains the PyTorch multimodal neural network architecture, the training loop, and the evaluation metrics.
* `plots/`: Directory containing the visual evaluation of the model, including the Learning Curve, ROC-AUC Curve, and Confusion Matrix.

## Results

The proposed architecture was evaluated on a strictly held-out test set (80/20 split), achieving the following results:
* **ROC-AUC Score:** 0.9927
* **False Negatives:** 0 (The model successfully identified all patients at risk within the test set).

## Requirements

To run the scripts, ensure you have the following libraries installed:
```bash
pip install pandas numpy torch scikit-learn matplotlib
