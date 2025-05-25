import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# --- Load and format data ---
diabetes = load_diabetes(as_frame=True)
df = diabetes.frame.dropna()

# --- Train/test split ---
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
pred_cols = diabetes.feature_names
target_col = 'target'

# --- Base Tree Function ---
def plain_vanilla_tree(df_train, target_col, pred_cols, max_depth=3):
    X = df_train[pred_cols]
    y = df_train[target_col]
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    tree.fit(X, y)
    return tree

# --- Bagging Model ---
def train_bagging_trees(df, target_col, pred_cols, n_trees=100, max_depth=3):
    """
    Trains an ensemble of decision trees on bootstrapped samples (bagging).
    """
    train_trees = []
    for i in range(n_trees):
        temp_boot = resample(df, replace=True, random_state=42+i)  # bootstrap sample
        temp_tree = plain_vanilla_tree(temp_boot, target_col, pred_cols, max_depth)
        train_trees.append(temp_tree)
    return train_trees

# --- Prediction from Bagging Trees ---
def bagging_trees_pred(df, train_trees, pred_cols):
    """
    Averages predictions from multiple decision trees.
    """
    x = df[pred_cols]
    all_preds = np.array([tree.predict(x) for tree in train_trees])
    avg_preds = np.mean(all_preds, axis=0)
    return avg_preds

# --- Train and Evaluate Bagging Model ---
n_trees = 100
max_depth = 5

# Train bagging ensemble
bagged_model = train_bagging_trees(train_df, target_col, pred_cols, n_trees=n_trees, max_depth=max_depth)

# Predict
preds = bagging_trees_pred(test_df, bagged_model, pred_cols)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test_df[target_col], preds))
print(f"Bagging Model RMSE (n_trees={n_trees}, max_depth={max_depth}): {rmse:.2f}")

# --- Visualization ---
plt.figure(figsize=(8, 6))
plt.scatter(test_df[target_col], preds, alpha=0.7, color='forestgreen', label='Predicted vs Actual')
plt.plot([test_df[target_col].min(), test_df[target_col].max()],
         [test_df[target_col].min(), test_df[target_col].max()],
         'r--', label='Perfect Prediction')
plt.xlabel('Actual Target')
plt.ylabel('Predicted Target')
plt.title(f'Bagging Trees Predictions (n={n_trees})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
