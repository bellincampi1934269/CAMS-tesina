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

* `00_check_dataset.py`: Preliminary script for raw data exploration, sanity checks, and disease prevalence calculation.
* `01_extract_cohort.py`: Script for cohort extraction and diagnosis filtering using specific SNOMED-CT codes.
* `02_extract_features.py`: Script dedicated to extracting static demographics and dynamic clinical variables from the filtered cohort.
* `03_build_tensors.py`: Data engineering script implementing temporal discretization, Zero-Order Hold imputation, and the construction of static and dynamic tensors.
* `04_dataset.py`: Custom PyTorch Dataset class implementation for handling multimodal inputs (static features, longitudinal sequences, and attention masks).
* `05_model.py`: Definition of the PyTorch multimodal neural network architecture (Late Fusion with MLP and LSTM branches).
* `06_train.py`: The main script containing the PyTorch training loop, optimization, and evaluation metrics.
* `07_plots.py`: Script for generating the visual evaluation metrics (Learning curves, ROC-AUC, and Confusion Matrix).
* `plots/`: Directory containing the final generated evaluation images.

## Results

The proposed architecture was evaluated on a strictly held-out test set (80/20 split), achieving the following results:
* **ROC-AUC Score:** 0.9927
* **False Negatives:** 0 (The model successfully identified all patients at risk within the test set).

## Requirements

To run the scripts, ensure you have the following libraries installed:
```bash
pip install pandas numpy torch scikit-learn matplotlib
