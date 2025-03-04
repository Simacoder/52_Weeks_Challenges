import pandas as pd
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Title and author information
st.title("Uber Fare Prediction System")
st.markdown("### Research by Simanga Mchunu, Machine Learning Engineer")

# Load and preprocess dataset
def load_data():
    df = pd.read_csv("data/data.csv")
    
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
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Drop unnecessary columns
    df = df.drop(columns=["ride_start", "ride_end"], errors='ignore')
    
    # Drop rows with invalid numerical values
    df = df.dropna()
    
    return df

# Train and evaluate models
def train_model():
    df_transformed = load_data()
    
    X = df_transformed.drop(columns=["ride_price"], errors='ignore')
    y = df_transformed["ride_price"]
    
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
        
        st.write(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {score:.2f}")
        
        if score > best_score:
            best_score = score
            best_model = model  # Ensure the best model is stored
    
    # Hyperparameter tuning using GridSearchCV (for Random Forest as example)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20]
    }
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)
    best_rf_model = grid_search.best_estimator_
    st.write(f"Best Random Forest Model: {best_rf_model}")
    
    joblib.dump(best_model, "model_uber.pkl")  # Save best model
    st.success(f"Best Model Selected: {best_model}")

# Streamlit UI for prediction
def predict():
    st.sidebar.header("Enter Ride Details")
    distance_km = st.sidebar.number_input("Distance (km)", min_value=0.1, max_value=100.0, value=5.0)
    ride_hour = st.sidebar.slider("Ride Start Hour", min_value=0, max_value=23, value=12)
    ride_duration = st.sidebar.number_input("Ride Duration (mins)", min_value=1, max_value=120, value=15)
    
    model = joblib.load("model_uber.pkl")
    input_data = pd.DataFrame([[distance_km, ride_hour, ride_duration]], columns=["distance_km", "ride_hour", "ride_duration"])
    prediction = model.predict(input_data)[0]
    
    st.subheader("Predicted Ride Fare")
    st.write(f"Estimated Fare: ${prediction:.2f}")

if __name__ == "__main__":
    train_model()
    predict()
