import os
import io
import base64
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

# ----------------------------
# FastAPI initialization
# ----------------------------
app = FastAPI(title="California Housing API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Load dataset
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "california_housing_alternative.csv")

try:
    df = pd.read_csv(CSV_PATH)
    if 'MedHouseVal' in df.columns:
        df.rename(columns={'MedHouseVal': 'median_house_value'}, inplace=True)
except Exception as e:
    print("Error loading CSV:", e)
    df = pd.DataFrame()

# ----------------------------
# Models folder
# ----------------------------
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ----------------------------
# Global variables
# ----------------------------
X_train, X_test, y_train, y_test, feature_columns = None, None, None, None, None

# ----------------------------
# Endpoints
# ----------------------------
@app.get("/data")
def get_data():
    if df.empty:
        return {"error": "Dataset not loaded"}
    return {"rows": len(df), "columns": list(df.columns)}

@app.get("/plot")
def plot_correlation():
    try:
        numeric_df = df.select_dtypes(include='number')
        if 'median_house_value' not in numeric_df.columns:
            return {"error": "'median_house_value' column not found in numeric dataset"}

        plt.figure(figsize=(8,6))
        numeric_df.corr()['median_house_value'].sort_values(ascending=False).plot(kind='bar')
        plt.title("Correlation with median_house_value")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode()
        plt.close()
        return {"plot_base64": img_base64}
    except Exception as e:
        return {"error": str(e)}

@app.post("/train_model")
def train_model(test_size: float = 0.2):
    global X_train, X_test, y_train, y_test, feature_columns
    try:
        if df.empty or 'median_house_value' not in df.columns:
            return {"error": "Dataset not ready"}

        X = df.drop(columns=['median_house_value'])
        y = df['median_house_value']

        # One-hot encode categorical
        X = pd.get_dummies(X, drop_first=True)

        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        feature_columns = X_imputed.columns.tolist()

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=test_size, random_state=42)

        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train LinearRegression
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Versioning
        existing_versions = [
            int(f.split("_v")[1].split(".")[0])
            for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")
        ]
        version = max(existing_versions)+1 if existing_versions else 1

        # Save model
        joblib.dump({
            "model": model,
            "scaler": scaler,
            "imputer": imputer,
            "feature_columns": feature_columns
        }, os.path.join(MODELS_DIR, f"model_v{version}.joblib"))

        return {"message": "Model trained", "version": version, "rmse": rmse}
    except Exception as e:
        return {"error": str(e)}

@app.get("/models")
def list_models():
    try:
        if X_test is None or y_test is None:
            return {"error": "No trained model to evaluate"}
        models_info = []
        for f in os.listdir(MODELS_DIR):
            if f.endswith(".joblib"):
                data = joblib.load(os.path.join(MODELS_DIR, f))
                model = data['model']
                scaler = data['scaler']
                imputer = data['imputer']
                X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
                X_test_scaled = scaler.transform(X_test_imputed)
                y_pred = model.predict(X_test_scaled)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                models_info.append({"model_file": f, "rmse": rmse})
        return models_info
    except Exception as e:
        return {"error": str(e)}

# ----------------------------
# Fixed /predict endpoint
# ----------------------------
@app.get("/predict")
def predict(
    longitude: float = Query(...),
    latitude: float = Query(...),
    housing_median_age: float = Query(...),
    total_rooms: float = Query(...),
    total_bedrooms: float = Query(...),
    population: float = Query(...),
    households: float = Query(...),
    median_income: float = Query(...),
    ocean_proximity: str = Query(...)
):
    try:
        # Load latest model
        model_files = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")])
        if not model_files:
            return {"error": "No models trained"}
        model_file = model_files[-1]
        data_model = joblib.load(os.path.join(MODELS_DIR, model_file))
        model = data_model['model']
        scaler = data_model['scaler']
        imputer = data_model['imputer']
        feature_columns = data_model['feature_columns']

        # Prepare input
        X_new = pd.DataFrame([{
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
            "ocean_proximity": ocean_proximity
        }])
        # One-hot encode categorical
        X_new = pd.get_dummies(X_new)
        for col in feature_columns:
            if col not in X_new.columns:
                X_new[col] = 0
        X_new = X_new[feature_columns]

        # Impute and scale
        X_new_imputed = pd.DataFrame(imputer.transform(X_new), columns=X_new.columns)
        X_new_scaled = scaler.transform(X_new_imputed)

        # Predict
        pred = model.predict(X_new_scaled)[0]
        return {"prediction": pred, "model_version": model_file}

    except Exception as e:
        return {"error": str(e)}

@app.get("/predict_trend")
def predict_trend(variable: str = "median_income", version: int = None):
    try:
        if df.empty:
            return {"error": "Dataset not loaded"}

        model_files = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")])
        if not model_files:
            return {"error": "No models trained"}

        if version is None:
            model_file = model_files[-1]
        else:
            model_file = f"model_v{version}.joblib"
            if model_file not in model_files:
                return {"error": "Version not found"}

        data_model = joblib.load(os.path.join(MODELS_DIR, model_file))
        model = data_model['model']
        scaler = data_model['scaler']
        imputer = data_model['imputer']
        feature_columns = data_model['feature_columns']

        if variable not in df.columns or df[variable].dtype.kind not in 'iufc':
            return {"error": f"Variable '{variable}' must be numeric"}

        trend_values = np.linspace(df[variable].min(), df[variable].max(), 50)
        trend_df = pd.DataFrame({variable: trend_values})

        for col in feature_columns:
            if col == variable:
                continue
            if col in df.columns:
                if df[col].dtype.kind in 'iufc':
                    trend_df[col] = df[col].median()
                else:
                    trend_df[col] = 0
            else:
                trend_df[col] = 0

        for col in feature_columns:
            if col not in trend_df.columns:
                trend_df[col] = 0
        trend_df = trend_df[feature_columns]

        X_trend = pd.DataFrame(imputer.transform(trend_df), columns=trend_df.columns)
        X_trend_scaled = scaler.transform(X_trend)
        y_pred = model.predict(X_trend_scaled)

        plt.figure(figsize=(8,6))
        plt.plot(trend_values, y_pred, marker='o')
        plt.xlabel(variable)
        plt.ylabel("Predicted median_house_value")
        plt.title(f"Prediction Trend over {variable}")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode()
        plt.close()

        return {
            "variable": variable,
            "trend_values": trend_values.tolist(),
            "predictions": y_pred.tolist(),
            "plot_base64": plot_base64,
            "model_version": model_file
        }

    except Exception as e:
        return {"error": str(e)}
