# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:23:53 2026

@author: jenny
"""

# End-to-End Climate Risk Analytics Pipeline & Predictive Modeling

An automated, production-grade ELT (Extract, Load, Transform) data pipeline and machine learning workflow built to ingest, clean, and model over **2 million historical storm event records** spanning 70+ years of NOAA weather data. 

This system handles client-side data consolidation, leverages cloud data warehousing for high-performance feature engineering, and deploys an optimized gradient-boosted tree classifier to predict severe financial property damage under severe class imbalance conditions.

---

## Data Architecture & Lifecycle

The project is structured using a modern data stack pattern that separates data ingestion, warehouse processing, and local statistical computing:

1. **Ingestion (Extract & Load):** A Python orchestration engine utilizing the Google Cloud BigQuery API to handle client-side data unification, schema relaxation matching across decades of disjointed legacy CSV tracking files, and bulk batch ingestion.
2. **Transformation (ELT):** Direct warehouse-level feature engineering using Google BigQuery Studio. Employs advanced SQL manipulation, regex string parsing to standardize inconsistent alphanumeric damage flags (e.g., "50K", "2M"), and sparse target variable generation.
3. **Machine Learning Pipeline:** An offline Python modeling ecosystem that pulls pristine warehouse feature layers into local memory, handles categorical one-hot encoding, mitigates extreme class imbalances, and trains a highly sensitive classifier.
4. **Closing the Loop:** The pipeline automatically writes predictive outputs, including calculated risk probabilities, directly back to a dedicated BigQuery core analytics table for downstream business intelligence execution.

---

## Machine Learning & Performance Diagnostics

### The Real-World Obstacle: Severe Class Imbalance
The dataset contains an extreme class imbalance with a **19.69:1 ratio**, where severe damage incidents comprise only **4.8%** of the 2-million-row history. Standard accuracy metrics would reward a naive model that guesses zero every time. 

To overcome this, the modeling pipeline implements a **stratified split** to ensure equal class representation across cross-validation buckets and applies a customized hyperparameter penalty (`scale_pos_weight`) to heavily penalize minority misclassifications.

### Model Metrics (XGBoost Classifier)

* **ROC-AUC Score:** `0.8498` — Demonstrating exceptional baseline discriminative power to isolate high-risk environmental attributes.
* **Precision-Recall AUC (PR-AUC):** `0.2874` — Representing a **6x performance multiplier** over a random baseline classifier (`0.048`) at pinpointing minority risk signatures.
* **Recall (Severe Class):** `0.77` — Successfully capturing **77% of all severe damage events**, prioritizing the minimized financial risk exposure critical for insurance underwriter application frameworks.

---

## Repository Structure

```text
├── .gitignore                 # Shields cloud service credentials and python cache
├── README.md                  # Project documentation and performance diagnostics
├── requirements.txt           # Explicit python library dependencies
├── warehouse/
│   └── ml_features_joined.sql # Native BigQuery SQL script for ELT feature transformations
└── modeling/
    └── train_storm_model.py   # Full training pipeline, feature plotting, and BQ write-back