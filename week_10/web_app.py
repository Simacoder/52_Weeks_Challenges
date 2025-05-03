import pandas as pd
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os

# Title and author information
st.title("🚖 Uber Fare Prediction System")
st.markdown("### 📊 Research by Simanga Mchunu, Machine Learning Engineer")

# Load and preprocess dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/Data.csv")
        st.write(f"✅ Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")

        # Convert datetime columns
        df["ride_start"] = pd.to_datetime(df["ride_start"], errors='coerce')
        df["ride_end"] = pd.to_datetime(df["ride_end"], errors='coerce')

        # Handle missing or invalid datetime values
        df = df.dropna(subset=["ride_start", "ride_end"])

        # Calculate ride duration
        df["ride_duration"] = (df["ride_end"] - df["ride_start"]).dt.total_seconds() / 60
        df["ride_hour"] = df["ride_start"].dt.hour
        df["distance_km"] = pd.to_numeric(df["distance"], errors='coerce') / 1000  # Convert meters to km
        df["ride_price"] = pd.to_numeric(df["ride_price"], errors='coerce')

        # Fill missing numerical values with median
        df.fillna(df.median(numeric_only=True), inplace=True)

        # Identify categorical columns and apply one-hot encoding
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Drop unnecessary columns
        df.drop(columns=["ride_start", "ride_end"], errors='ignore', inplace=True)

        st.write(f"✅ Processed dataset has {df.shape[0]} rows and {df.shape[1]} columns")
        return df

    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return pd.DataFrame()

# Train and evaluate models
def train_model():
    df_transformed = load_data()

    if df_transformed.empty:
        st.error("❌ Error: The processed dataset is empty. Check data loading and preprocessing steps.")
        return

    X = df_transformed.drop(columns=["ride_price"], errors='ignore')
    y = df_transformed["ride_price"]

    # Handle missing values
    X.fillna(X.median(numeric_only=True), inplace=True)  # Replace NaNs in features
    y.dropna(inplace=True)  # Drop NaNs in target variable

    if X.empty or y.empty:
        st.error("❌ Error: Features or target variable is empty after preprocessing.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
    }

    best_model = None
    best_score = float("-inf")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        st.write(f"✅ **{name}** - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {score:.2f}")

        if score > best_score:
            best_score = score
            best_model = model

    # Save the best model and feature names
    joblib.dump((best_model, list(X_train.columns)), "model_uber.pkl")
    st.success(f"✅ Best Model Selected: {best_model}")

# Streamlit UI for prediction
def predict():
    st.sidebar.header("🚕 Enter Ride Details")

    distance_km = st.sidebar.number_input("📏 Distance (km)", min_value=0.1, max_value=100.0, value=5.0)
    ride_hour = st.sidebar.slider("⏰ Ride Start Hour", min_value=0, max_value=23, value=12)
    ride_duration = st.sidebar.number_input("⏳ Ride Duration (mins)", min_value=1, max_value=120, value=15)

    # Check if model exists
    if not os.path.exists("model_uber.pkl"):
        st.error("❌ Model file not found. Please train the model first.")
        return

    # Load model (handling different formats)
    loaded_data = joblib.load("model_uber.pkl")

    if isinstance(loaded_data, tuple):
        model, feature_names = loaded_data
    else:
        model = loaded_data
        feature_names = ["distance_km", "ride_hour", "ride_duration"]  # Default feature names

    # Create input DataFrame
    input_data = pd.DataFrame([[distance_km, ride_hour, ride_duration]], columns=["distance_km", "ride_hour", "ride_duration"])

    # Ensure input data has the same features as training data
    missing_cols = set(feature_names) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0  # Add missing columns with default value 0

    extra_cols = set(input_data.columns) - set(feature_names)
    input_data.drop(columns=extra_cols, errors='ignore', inplace=True)  # Drop extra columns

    input_data = input_data[feature_names]  # Reorder columns

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.subheader("🎯 Predicted Ride Fare")
    st.write(f"💰 **Estimated Fare: ${prediction:.2f}**")

# Streamlit Execution
if __name__ == "__main__":
    train_button = st.sidebar.button("🚀 Train Model")
    if train_button:
        train_model()

    predict()
