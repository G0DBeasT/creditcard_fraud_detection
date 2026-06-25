# Credit Card Fraud Detection

## Overview
This project focuses on predicting fraudulent credit card transactions. The primary challenge is dealing with a highly imbalanced dataset, where frauds account for a tiny fraction of the total transactions. Specialized techniques such as class weights and hyperparameter tuning for Average Precision are utilized to build a robust detection model.

## Dataset
The dataset used is the `creditcard` dataset from OpenML (ID 1597). 
- **Total Transactions:** 284,807
- **Fraudulent Transactions:** 492 (0.173%)
- **Features:** 28 PCA-transformed numerical features (V1-V28), `Amount`, and `Class` (target variable: 0 for non-fraud, 1 for fraud).

## Machine Learning Pipeline

1. **Data Loading:** 
   The dataset is fetched directly from OpenML using `scikit-learn`'s `fetch_openml`.

2. **Exploratory Data Analysis (EDA):**
   Visualization of the class distribution highlights the severe class imbalance, informing the choice of evaluation metrics and model strategies.

3. **Data Preprocessing:**
   The `Amount` feature is scaled using `StandardScaler`. The V1-V28 features are already PCA-transformed and scaled.

4. **Model Training & Fine-Tuning:**
   A `RandomForestClassifier` is trained with `class_weight='balanced'` to handle the imbalance. `RandomizedSearchCV` is used to find the best hyperparameters, optimizing specifically for the **Average Precision Score (AUPRC)** rather than accuracy.

5. **Model Evaluation:**
   Since accuracy is misleading for highly imbalanced data, the model is evaluated using:
   - Precision, Recall, and F1-Score
   - Precision-Recall Curve
   - Confusion Matrix
   - Average Precision Score (AUPRC)

## Libraries Used
- `pandas`, `numpy` for data manipulation
- `matplotlib`, `seaborn` for visualization
- `scikit-learn` for preprocessing, modeling, and evaluation
