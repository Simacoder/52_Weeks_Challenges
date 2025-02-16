from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(data_path, model_path):
    df = pd.read_csv(data_path)

    # Convert date columns to datetime, then extract useful features
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = pd.to_datetime(df[col])
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df.drop(columns=[col], inplace=True)  # Drop original datetime column
        except Exception:
            pass  # Skip if not a date column

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Split data into features and target
    X = df.drop(columns=['Sales'])
    y = df['Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model MAE: {mae:.4f}")

    # Save the model
    joblib.dump(model, model_path)
    print(f"Model saved at: {model_path}")



    # Visualization of actual vs predicted sales
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, edgecolor=None)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', label="Perfect Prediction")
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs Predicted Sales")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Feature Importance Analysis
    feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    # Plot Feature Importance
    plt.figure(figsize=(10, 5))
    sns.barplot(x=feature_importance, y=feature_importance.index, palette="viridis")
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title("Feature Importance in Random Forest Model")
    plt.show()


    # Error Distribution Analysis
    errors = y_test - y_pred

    plt.figure(figsize=(10, 5))
    sns.histplot(errors, bins=30, kde=True, color="blue", alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', label="Zero Error Line")
    plt.xlabel("Prediction Error (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Errors")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    train_model('data/processed/cleaned_sales_data.csv', 'models/model.joblib')
