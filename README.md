# Credit Card Fraud Detection

## Project Overview

### Ribal Zaiter 8940
### Achraf Hoteit 8830

This project tackles the problem of **credit card fraud detection** using a dataset from **Kaggle** that contains highly **imbalanced classes** (only 0.17% fraudulent transactions).

The goal is to **accurately detect fraud** using machine learning techniques, evaluate their effectiveness with proper metrics, and explain the model’s decisions using **SHAP** for transparency.

---

## Dataset Description

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Rows**: 284,807 transactions
- **Features**: 30 (including `Time`, `Amount`, and 28 PCA components `V1–V28`)
- **Target**: `Class` (0 = Non-Fraud, 1 = Fraud)

---

## Data Exploration

- No missing or negative values in `Time` or `Amount`.
- Heavy class imbalance:
  - **Class 0** (Non-Fraud): 284,315
  - **Class 1** (Fraud): 492

Visualization confirmed the imbalance, which required special handling before training models.

---

## Handling Imbalanced Data

We applied two main techniques:
- **Undersampling**: For SVM with kernel
- **SMOTE (Synthetic Minority Over-sampling Technique)**: For Linear SVM, Random Forest, and XGBoost

---

## Models Trained

| Model                  | Technique Used        |
|------------------------|------------------------|
| **SVM with Kernel**     | Undersampling (5000 non-fraud) |
| **Linear SVM**          | SMOTE                 |
| **Random Forest**       | SMOTE                 |
| **XGBoost**             | SMOTE                 |

---

## Evaluation Metrics

- **AUC-ROC**: Measures model performance, especially for imbalanced data.
- **Recall (Fraud)**: Focus on catching as many fraudulent cases as possible.
- **Confusion Matrix & Classification Report**: For detailed breakdown.

---

## Model Results

| Model          | AUC-ROC | Recall (Fraud Class) |
|----------------|---------|----------------------|
| SVM (Kernel)   | ~0.985   | High (Detects most frauds) |
| Linear SVM     | ~0.973   | Moderate             |
| Random Forest  | ~0.984   | High                 |
| XGBoost        | **~0.99** | **Very High (Best overall)** |

> **XGBoost** outperformed all others in terms of AUC.

---

## Model Explainability with SHAP

We used **SHAP** to explain:
- Which features contribute most to fraud predictions
- Why the model classified a specific sample as fraud or not

**Key Findings**:
- Features like **V11, V13, and V9** had the highest impact.
- SHAP waterfall and force plots helped visualize the reasoning for individual predictions.

---

## Final Comparison Plot

A dual-axis plot was created to compare models by:
- **AUC-ROC score**
- **Recall on Fraud Class**

This helps visualize the trade-offs and strengths of each model.

---
