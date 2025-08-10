# Confusion Matrix Made Simple: Accuracy, Precision, Recall & F1-Score

**Author:** Simanga Mchunu  
**Date:** August 08, 2025  
 

---

## 📌 Overview

When working with classification algorithms in machine learning — such as **Logistic Regression**, **K-Nearest Neighbors**, or **Support Vector Classifiers** — evaluation metrics like **MAE**, **MSE**, or **RMSE** are not suitable.  
Instead, we rely on a **confusion matrix** and a **classification report** to evaluate model performance.

This project demonstrates:
- How to generate and interpret a **confusion matrix**.
- How to calculate **Accuracy**, **Precision**, **Recall**, and **F1-Score** from it.
- When to prioritize specific metrics depending on your data context.

---

## 📂 Dataset

We use the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which contains **569 samples** and **30 numeric features** describing tumor characteristics, along with a binary target:
- **M** = Malignant (Cancerous)
- **B** = Benign (Non-Cancerous)

**Source:**  
Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993).  
Breast Cancer Wisconsin (Diagnostic) [Dataset].  
[UCI Machine Learning Repository](https://doi.org/10.24432/C5DW2B)  
Licensed under **CC BY 4.0**.

---

## ⚙️ Steps

### 1. Load and Preprocess Data
- Drop the **ID** column.
- Encode the target variable:
  - **M** → `1`
  - **B** → `0`
- Train-test split with **30% test set**, stratified by class.
- Standardize features using **StandardScaler**.

### 2. Train Model
- Logistic Regression with `max_iter=10000`.

### 3. Evaluate Model
- Generate **Confusion Matrix** and **Classification Report**.
- Plot the confusion matrix with **Seaborn heatmap**.

---

## 📊 Confusion Matrix Interpretation

|               | Predicted Malignant | Predicted Benign |
|---------------|--------------------|------------------|
| **Actual Malignant** | True Positives (TP) = 60 | False Negatives (FN) = 4 |
| **Actual Benign**    | False Positives (FP) = 1 | True Negatives (TN) = 106 |

---

## 📈 Metrics Definitions

- **Accuracy** = `(TP + TN) / Total`
- **Precision** = `TP / (TP + FP)`  
  _When the model predicts "Malignant", how often is it correct?_
- **Recall** = `TP / (TP + FN)`  
  _Of all actual Malignant cases, how many did the model identify correctly?_
- **F1-Score** = `2 * (Precision * Recall) / (Precision + Recall)`  
  _Balances Precision and Recall._

---

## 📌 Macro vs Weighted Average

- **Macro Average**: Equal weight to all classes, regardless of class size.
- **Weighted Average**: Weights each class’s metric by its sample count.

---

## 🎯 Threshold Tuning

In medical contexts (e.g., cancer detection), **Recall** is often more critical than Accuracy.  
We can adjust the classification threshold (default `0.5`) to increase Recall.

Example: Setting threshold = `0.3` increased Recall for Malignant tumors from **0.94** to **0.97**, without lowering Precision.

---

## 🛠 Example Code Snippet

```python
# Predict probabilities
y_probs = model.predict_proba(X_test)[:, 1]

# Apply custom threshold
threshold = 0.3
y_pred_custom = (y_probs >= threshold).astype(int)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_custom, labels=[1, 0])

# Classification Report
report = classification_report(y_test, y_pred_custom, target_names=["Benign", "Malignant"])
```

---
# 📌 Key Takeaways

Accuracy alone can be misleading in imbalanced datasets.

In healthcare:

- Prioritize Recall (catch as many positive cases as possible).

- Keep Precision reasonably high to avoid unnecessary interventions.

- In spam/fraud detection:

- Prioritize Precision (reduce false alarms).

- F1-score is useful when both Precision and Recall are equally important.

# 📜 License

This project uses the Breast Cancer Wisconsin (Diagnostic) Dataset under the [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).
You are free to use this work for educational or commercial purposes with attribution.