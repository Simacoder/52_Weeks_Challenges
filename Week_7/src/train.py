from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import pandas as pd

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

if __name__ == "__main__":
    train_model('data/processed/cleaned_sales_data.csv', 'models/model.joblib')
