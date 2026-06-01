# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:37:41 2026

@author: jenny
"""

import os
from google.cloud import bigquery
import pandas as pd
import numpy as np

# Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from xgboost import XGBClassifier

# 1. Authenticate with your downloaded JSON key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\jenny\OneDrive\Desktop\code\NOAA weather data\datatest-498117-a06f231a1a88.json"

# Initialize BigQuery Client
project_id = "datatest-498117"
dataset_id = "clean_storm_data"
client = bigquery.Client(project=project_id)

print("Fetching engineered feature table from BigQuery...")
query = f"SELECT EVENT_ID,STATE, storm_month, simplified_event_type, magnitude_clean, injuries_clean, deaths_clean, is_severe_damage FROM `{project_id}.{dataset_id}.ml_features_joined`"
df = client.query(query).to_dataframe()
print(f"Data successfully loaded into memory. Shape: {df.shape}")

# 2. Separate Features (X) and Target (y)
# Since EVENT_ID is a unique key string (not a mathematical pattern), we don't want XGBoost trying to use it to learn weather traits. We need to tell X to drop it alongside our target column.
X = df.drop(columns=["is_severe_damage", "EVENT_ID"])
y = df["is_severe_damage"]

# Calculate class imbalance to help XGBoost weight the rare severe storms
num_zeros = np.sum(y == 0)
num_ones = np.sum(y == 1)
scale_pos_weight_value = num_zeros / num_ones
print(f"Dataset breakdown: Minor/No Damage={num_zeros}, Severe Damage={num_ones}")
print(f"Calculated class imbalance ratio: {scale_pos_weight_value:.2f}")

# 3. Train/Test Split (80/20)
# Stratify=y ensures both training and testing sets get an equal percentage of severe storms
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Preprocessing Pipeline for Categorical Columns
# One-hot encodes categories like 'STATE' and 'simplified_event_type'
categorical_cols = ["STATE", "simplified_event_type"]
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough" # Keeps numeric columns as they are
)

print("Preprocessing features...")
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# 5. Initialize and Train XGBoost Classifier
print("Training XGBoost Classifier...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight_value, # Tells XGBoost to focus harder on rare severe storms
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train_transformed, y_train)
print("Model training complete!")

# 6. Evaluate Model Performance
y_pred = model.predict(X_test_transformed)
y_proba = model.predict_proba(X_test_transformed)[:, 1]

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# Calculate Precision-Recall Area Under Curve (PR-AUC) - The ultimate metric for imbalanced data
precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
print(f"Precision-Recall AUC (PR-AUC): {pr_auc:.4f}\n")




import matplotlib.pyplot as plt

# 1. Dynamically pull EVERY single feature name exactly as it was fed to XGBoost
all_feature_names = preprocessor.get_feature_names_out()

# Clean up the names (removes prefixes like 'cat__' and 'remainder__')
all_feature_names = [name.split('__')[-1] for name in all_feature_names]

# 2. Extract importance scores from the trained model
importances = model.feature_importances_

print(f"Debug Info: Feature names array length = {len(all_feature_names)}")
print(f"Debug Info: Importance values array length = {len(importances)}")

# 3. Map them together and sort them
feature_imp_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# 4. Plot the Top 15 most influential features
plt.figure(figsize=(10, 6))
plt.barh(feature_imp_df['Feature'].head(15)[::-1], feature_imp_df['Importance'].head(15)[::-1], color='teal')
plt.xlabel('XGBoost Feature Importance Score')
plt.title('Top 15 Most Influential Features for Predicting Severe Storm Damage')
plt.tight_layout()
plt.show()

# =========================================================================
# 5. CLOSING THE LOOP: WRITE PREDICTIONS BACK TO BIGQUERY
# =========================================================================
# Create a clean results DataFrame using the test split indexes to pull IDs
results_df = pd.DataFrame({
    'EVENT_ID': df.loc[X_test.index, 'EVENT_ID'],
    'actual_severity': y_test,
    'predicted_severity': y_pred,
    'risk_probability': y_proba  # The decimal probability calculated by XGBoost
})

print(f"Uploading {len(results_df)} model prediction records back to BigQuery...")

# Configure the destination table path in your warehouse
output_table_ref = f"{project_id}.{dataset_id}.ml_model_predictions"

# Stream it up (WRITE_TRUNCATE ensures it overwrites old runs cleanly)
upload_job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
upload_job = client.load_table_from_dataframe(results_df, output_table_ref, job_config=upload_job_config)
upload_job.result()  # Wait for the cloud upload to finish

print("Successfully synced machine learning predictions back to the warehouse! Table: ml_model_predictions")