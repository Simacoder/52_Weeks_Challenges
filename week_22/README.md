# FNB DATAFEST CHALLENGE

# Personalized Solutions Recommender System 

Welcome to our solution for the **Personalized Solutions Dataquest Challenge**, where we built a predictive and recommendation system to enhance user engagement on a financial services platform.

---

##  Problem Statement

The challenge was to:
- Predict meaningful customer interactions (`CLICK`, `CHECKOUT`) from raw behavioral data.
- Recommend relevant financial products to users based on their activity and profile.

---

##  Table of Contents

1. [Dataset Description](#dataset-description)
2. [Data Preparation](#data-preparation)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Modeling Approaches](#modeling-approaches)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Recommendation System](#recommendation-system)
7. [Feature Importance](#feature-importance)
8. [Wow Moments](#wow-moments)
9. [Conclusion & Next Steps](#conclusion--next-steps)

---

##  Dataset Description

| Column       | Type         | Description |
|--------------|--------------|-------------|
| `idcol`      | User ID      | Unique customer identifier |
| `interaction`| Categorical  | DISPLAY, CLICK, CHECKOUT |
| `int_date`   | Date         | Date of interaction |
| `item`       | Categorical  | Item code |
| `item_type`  | Category     | TRANSACT, LEND, INVEST, etc. |
| `item_descrip`| Text        | Description of the item |
| `page`, `tod`| Context      | Time of Day, App Page |
| `segment`, `beh_segment` | User Features | Broad and detailed segmentation |
| `active_ind` | Activity     | Cold Start, Semi Active, Active |

---

##  Data Preparation

- Date conversion and missing value handling
- Created `rating`: DISPLAY = 0, CLICK = 1, CHECKOUT = 2
- Train/Test Split using time-based logic
- Categorical encoding and feature scaling

---

##  Exploratory Data Analysis (EDA)

- Weekly interaction trends visualized for DISPLAY, CLICK, CHECKOUT
- Feature correlation heatmap showing influence of `segment`, `item_type`, `active_ind`

---

##  Modeling Approaches

### Classification Models
- **Logistic Regression** (baseline)
- **Random Forest**
- **Bagging Classifier**
- **Gradient Boosting**

### Ensemble Learning
- Improved performance using Bagging and Boosting to reduce variance and bias.

---

##  Evaluation Metrics

We evaluated models using:
- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC
- Log Loss
- Precision@10 (for top-N relevance)

---

##  Recommendation System

### Content-Based Filtering
- TF-IDF + Cosine Similarity on item descriptions
- Recommended similar items per product

### Collaborative Filtering
- SVD-based user-item matrix factorization
- Suggested items based on similar users’ behaviors

---

##  Feature Importance

Key influencing features:
- `segment`
- `active_ind`
- `item_type`

Visualized using:
- Random Forest Feature Importance
- Permutation Importance

---

##  Wow Moments

-  Transformed `interaction` into numerical ratings for modeling
-  Hybrid Recommender System combining content & collaborative filtering
-  Ensemble visualizations with trend lines
-  Segment-based insights revealed behavioral drivers
-  Time-aware test split to simulate real-world predictions
-  Precision@K focused on top-N quality

---

##  Conclusion & Next Steps

### Achievements:
- Accurate interaction prediction with ensemble methods
- Personalized recommendations using hybrid approach

### Next Steps:
- Add time-series aware modeling
- Improve recommendations for cold-start users
- Automate pipeline for retraining and monitoring

---

##  Authors

- Simanga Mchunu
- Contact: simacoder@hotmail.com

---

## 📁 Project Structure

```bash
├── data/
│   └── data.csv
├── notebooks/
│   └── model_training.ipynb
├── src/
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── recommender.py
├── Recommender Presentation.pdf
├── README.md
```
