import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load and prepare the dataset
diabetes = load_diabetes(as_frame=True)
df = diabetes.frame.dropna()

# Split into training and testing
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
pred_cols = diabetes.feature_names

# --- Base Tree Model ---
def plain_vanilla_tree(df_train, target_col, pred_cols, max_depth=3, weights=None):
    X_train = df_train[pred_cols]
    y_train = df_train[target_col]
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    if weights is not None:
        tree.fit(X_train, y_train, sample_weight=weights)
    else:
        tree.fit(X_train, y_train)
    return tree

# --- Boosting: Residual Correction ---
def boost_resid_correction(df_train, target_col, pred_cols, num_models, learning_rate=1, max_depth=3):
    """
    Creates a simple gradient boosting ensemble using residuals.
    """
    # Initial model
    model1 = plain_vanilla_tree(df_train, target_col, pred_cols, max_depth=max_depth)
    initial_preds = model1.predict(df_train[pred_cols])
    df_train = df_train.copy()
    df_train['resids'] = df_train[target_col] - initial_preds
    
    # Boosting
    models = []
    for i in range(num_models):
        temp_model = plain_vanilla_tree(df_train, 'resids', pred_cols, max_depth=max_depth)
        models.append(temp_model)
        temp_pred_resids = temp_model.predict(df_train[pred_cols])
        df_train['resids'] = df_train['resids'] - (learning_rate * temp_pred_resids)
        
    boosting_model = {
        'initial_model': model1,
        'models': models,
        'learning_rate': learning_rate,
        'pred_cols': pred_cols
    }
    return boosting_model

# --- Prediction with Boosting Model ---
def boost_resid_correction_predict(df, boosting_models, chart=False):
    """
    Predicts using a residual-based boosting model and optionally visualizes.
    """
    pred_cols = boosting_models['pred_cols']
    pred = boosting_models['initial_model'].predict(df[pred_cols])
    
    for model in boosting_models['models']:
        temp_resid_preds = model.predict(df[pred_cols])
        pred += boosting_models['learning_rate'] * temp_resid_preds

    if chart:
        plt.figure(figsize=(8, 6))
        plt.scatter(df['target'], pred, alpha=0.6, color='dodgerblue', label='Predicted vs Actual')
        plt.plot([df['target'].min(), df['target'].max()],
                 [df['target'].min(), df['target'].max()],
                 'r--', label='Perfect Prediction')
        plt.xlabel('Actual Target')
        plt.ylabel('Predicted Target')
        plt.title('Boosting Predictions vs Actual')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    rmse = np.sqrt(mean_squared_error(df['target'], pred))
    return pred, rmse

# --- Grid Search for Hyperparameters ---
n_trees = [5, 10, 30, 50, 100, 125, 150, 200, 250, 300]
learning_rates = [0.001, 0.01, 0.1, 0.25, 0.50, 0.75, 0.95, 1]
max_depths = list(range(1, 16))

perf_dict = {}

for tree in n_trees:
    for learning_rate in learning_rates:
        for max_depth in max_depths:
            temp_model = boost_resid_correction(train_df, 'target', pred_cols, tree,
                                                learning_rate=learning_rate, max_depth=max_depth)
            temp_model['target_col'] = 'target'
            _, rmse = boost_resid_correction_predict(test_df, temp_model)
            key = f"{tree}_{learning_rate}_{max_depth}"
            perf_dict[key] = rmse

# Best result
min_key = min(perf_dict, key=perf_dict.get)
print(f"Best config: {min_key}, RMSE: {perf_dict[min_key]:.4f}")
