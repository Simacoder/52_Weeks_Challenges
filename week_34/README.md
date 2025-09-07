# Random Forest Classification with Scikit-Learn

A comprehensive machine learning project demonstrating how to use Random Forest algorithms for classification tasks using the UCI Bank Marketing dataset.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [How Random Forests Work](#how-random-forests-work)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Model Performance](#model-performance)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project demonstrates the implementation of Random Forest classification using Python and Scikit-Learn. You'll learn:

- **How random forests work** - Understanding the ensemble method behind random forests
- **How to use them for classification** - Practical implementation for real-world problems
- **How to evaluate their performance** - Comprehensive model evaluation techniques

## 📊 Dataset

This project uses the **UCI Bank Marketing Dataset**, which contains data from direct marketing campaigns by a Portuguese banking institution. The campaigns aimed to sell subscriptions to bank term deposits through phone calls.

### Key Features:
- **`age`**: Age of the person who received the phone call
- **`default`**: Whether the person has credit in default
- **`cons.price.idx`**: Consumer price index score at the time of the call
- **`cons.conf.idx`**: Consumer confidence index score at the time of the call
- **`y`**: Whether the person subscribed (target variable)

### Dataset Statistics:
- **Size**: 41,188 rows × 21 columns
- **Target Classes**: Binary classification (subscribed: yes/no)
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)

## 🌳 How Random Forests Work

### An Overview of Random Forests

Random forests are a popular supervised machine learning algorithm with the following characteristics:

- **Supervised Learning**: Requires labeled target variables
- **Versatile**: Handles both regression (numeric targets) and classification (categorical targets)
- **Ensemble Method**: Combines predictions from multiple models
- **Decision Tree Based**: Each model in the ensemble is a decision tree

### The Forest Analogy

Imagine you have a complex problem to solve, and you gather a group of experts from different fields to provide their input. Each expert provides their opinion based on their expertise and experience. Then, the experts vote to arrive at a final decision.

In random forest classification:
1. **Multiple decision trees** are created using different random subsets of data and features
2. **Each decision tree** acts like an expert, providing its opinion on classification
3. **Final predictions** are made by taking the most popular result across all trees
4. **For regression**, predictions use averaging instead of voting

### Key Advantages:
- **Reduces overfitting** compared to individual decision trees
- **Handles missing values** and maintains accuracy
- **Provides feature importance** rankings
- **Works well** with both numerical and categorical data

## 🚀 Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook or any Python IDE

### Required Libraries
```bash
pip install pandas numpy scikit-learn matplotlib seaborn graphviz
```

### Optional (for enhanced visualizations)
```bash
# For decision tree visualization
conda install graphviz
# or download from: https://graphviz.org/download/
```

## 💻 Usage

### Quick Start

1. **Load the dataset:**
```python
import pandas as pd
import zipfile, io, urllib.request

# One-liner dataset loading
bank_data = pd.read_csv(io.BytesIO(zipfile.ZipFile(io.BytesIO(urllib.request.urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip').read())).read('bank-additional/bank-additional-full.csv')), sep=';')
```

2. **Preprocess the data:**
```python
# Target variable encoding
bank_data['y'] = bank_data['y'].map({'no': 0, 'yes': 1})
bank_data['default'] = bank_data['default'].map({'no': 0, 'yes': 1, 'unknown': 0})

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
categorical_cols = bank_data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    bank_data[col] = le.fit_transform(bank_data[col].astype(str))
```

3. **Train the Random Forest:**
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Prepare data
X = bank_data.drop('y', axis=1)
y = bank_data['y']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

4. **Evaluate performance:**
```python
from sklearn.metrics import accuracy_score, classification_report

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))
```

## 📁 Project Structure

```
random-forest-classification/
│
├── data/
│   └── bank_data.csv              # Dataset (downloaded)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Initial data analysis
│   ├── 02_preprocessing.ipynb     # Data preprocessing
│   ├── 03_model_training.ipynb    # Random Forest training
│   └── 04_evaluation.ipynb       # Model evaluation
│
├── src/
│   ├── data_loader.py            # Data loading utilities
│   ├── preprocessing.py          # Data preprocessing functions
│   ├── model.py                  # Random Forest implementation
│   └── evaluation.py             # Model evaluation functions
│
├── visualizations/
│   ├── feature_importance.png    # Feature importance plot
│   ├── confusion_matrix.png      # Confusion matrix
│   └── decision_trees.png        # Sample decision trees
│
├── requirements.txt              # Required packages
└── README.md                     # This file
```

## ✨ Key Features

### The Machine Learning Workflow

This project follows the standard ML workflow:

1. **Feature Engineering** - Encoding categorical variables and handling missing values
2. **Data Splitting** - Train/test split with stratification
3. **Model Training** - Random Forest with optimized parameters
4. **Hyperparameter Tuning** - Grid search for optimal parameters
5. **Performance Assessment** - Comprehensive evaluation metrics

### Preprocessing Pipeline

Tree-based models like Random Forest are robust and require minimal preprocessing:

-  **No normalization required** (unlike linear models)
-  **Robust to outliers** naturally
-  **Handles mixed data types** (numerical and categorical)
-  **Built-in feature selection** through importance scores

### Model Comparison

The project includes comparison with other algorithms:
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression** 
- **Support Vector Machine (SVM)**

## 📈 Model Performance

### Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 0.9127 | 0.6892 | 0.2847 | 0.4032 |
| KNN (k=5) | 0.8956 | 0.5234 | 0.1892 | 0.2781 |
| Logistic Regression | 0.9089 | 0.6567 | 0.2634 | 0.3756 |

### Feature Importance

Top 5 most important features:
1. **cons.price.idx** (Consumer Price Index)
2. **cons.conf.idx** (Consumer Confidence Index)
3. **age** (Customer Age)
4. **duration** (Call Duration)
5. **campaign** (Number of Contacts)

## 🎯 Results

### Key Findings

- **High Accuracy**: Random Forest achieved 91.27% accuracy on the test set
- **Class Imbalance**: The dataset is imbalanced (few positive cases)
- **Feature Insights**: Economic indicators are the strongest predictors
- **Model Robustness**: Random Forest outperformed other algorithms

### Business Insights

1. **Economic Climate Matters**: Consumer confidence and price indices are critical
2. **Age Factor**: Customer age significantly impacts subscription likelihood
3. **Contact Strategy**: Call duration and campaign intensity affect outcomes
4. **Seasonal Patterns**: Month and day of week show predictive power

## 🔧 Advanced Features

### Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
```

### Feature Selection
```python
from sklearn.feature_selection import SelectFromModel

# Select features based on importance
selector = SelectFromModel(rf, threshold='median')
X_train_selected = selector.fit_transform(X_train, y_train)
```

## 📊 Visualization

### Decision Tree Visualization
```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Visualize individual trees
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for i in range(3):
    plot_tree(rf.estimators_[i], ax=axes[i], max_depth=2, filled=True)
```

### Feature Importance Plot
```python
import seaborn as sns

# Plot feature importance
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=importance_df.head(15), x='importance', y='feature')
plt.title('Top 15 Feature Importances')
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for providing the dataset
- **Scikit-Learn** team for the excellent machine learning library
- **DataCamp** for inspiration and learning resources

## 📚 Further Reading

- [Random Forests - Leo Breiman's Original Paper](https://link.springer.com/article/10.1023/A:1010933404324)
- [Scikit-Learn Random Forest Documentation](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [Ensemble Methods in Machine Learning](https://web.engr.oregonstate.edu/~tgd/publications/mcs-ensembles.pdf)

---

**Made with  for the machine learning community**

# AUTHOR
- Simanga Mchunu