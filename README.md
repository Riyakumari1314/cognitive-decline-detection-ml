# Cognitive Decline Detection Using Speech and Typing Patterns
# Multimodal Cognitive Decline Detection via Passive Digital Biomarkers

An end-to-end machine learning pipeline for early detection of cognitive decline using **spontaneous speech/linguistic markers** and **keystroke/typing dynamics**. The framework evaluates unimodal classifiers alongside a participant-matched **Early Fusion Multimodal Classifier**.

---

## Project Architecture

```text
cognitive-decline-detection-ml/
├── data/
│   ├── docs/
│   │   └── methodology.md          # Experimental protocol & zero-leakage split design
│   ├── speech_benchmark.csv        # Participant speech & linguistic benchmark records
│   └── typing_benchmark.csv        # Participant-matched keystroke dynamics records
├── results/
│   ├── speech_model_cm.png         # Speech Unimodal Confusion Matrix
│   ├── typing_model_cm.png         # Typing Unimodal Confusion Matrix
│   └── multimodal_model_cm.png     # Multimodal Fusion Confusion Matrix
├── src/
│   ├── preprocessing/
│   │   ├── speech_preprocessing.py # Median imputation & boundary validation (Speech)
│   │   └── typing_preprocessing.py # Median imputation & boundary validation (Typing)
│   ├── features/
│   │   ├── speech_features.py      # 8 clinical linguistic & acoustic-proxy biomarkers
│   │   └── typing_features.py      # 8 motor-cognitive keystroke biomarkers
│   ├── models/
│   │   ├── speech_model.py         # StandardScaler + RandomForest (5-Fold CV)
│   │   ├── typing_model.py         # StandardScaler + RandomForest (5-Fold CV)
│   │   └── multimodal_model.py     # Feature Fusion + GradientBoosting (5-Fold CV)
│   └── evaluation/
│       ├── metrics.py              # Accuracy, Precision, Recall, F1-Score, ROC-AUC
│       └── confusion_matrix.py     # Confusion matrix plotting & export
├── download_benchmarks.py          # Automated dataset ingestion & formatting script
├── main.py                         # End-to-end training & evaluation entrypoint
└── requirements.txt                # Project dependencies


## Overview

This research project investigates the use of machine learning to detect cognitive decline from behavioural patterns in speech and typing.

The study evaluates three approaches:

1. Speech-only analysis
2. Typing-only analysis
3. Multimodal speech + typing analysis

## Research Objective

The objective is to investigate whether combining speech and typing features can improve the reliability and generalizability of cognitive-decline screening compared with using either modality independently.

## Research Pipeline

Speech Data
→ Preprocessing
→ Speech Feature Extraction
→ Speech Model

Typing Data
→ Preprocessing
→ Typing Feature Extraction
→ Typing Model

Speech Features + Typing Features
→ Feature Fusion
→ Multimodal Machine Learning Model
→ Cognitive Status Prediction

## Key Research Areas

- Speech analysis
- Keystroke dynamics
- Feature engineering
- Machine learning
- Multimodal learning
- Cognitive-decline screening
- Model evaluation
- Data augmentation
- Generalization and data-leakage prevention

## Machine Learning Models

The study will evaluate models such as:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting / XGBoost

## Evaluation Metrics

The models will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

## Dataset

The project will use appropriate research benchmark datasets for speech and typing analysis.

Restricted datasets will not be uploaded to this repository.

Dataset information and access details will be documented separately.

## Data Privacy and Ethics

This project is intended for research purposes.

Private or restricted participant data will not be stored in this repository.

Any future collection of human participant data will follow appropriate informed-consent, privacy, security, and ethical-review requirements.

## Project Status

🚧 Research project currently under development.

Experimental results will be added after the models are trained and evaluated.

## Authors

Riya Kumari  
Jyoti  
Aditya Singh saini, 
Vaibhav BHardwaj

University of Petroleum and Energy Studies (UPES), Dehradun
