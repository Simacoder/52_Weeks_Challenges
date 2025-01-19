import uvicorn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
#import requests

# Initialize FastAPI
app = FastAPI()

# Define the request body format for prediction
class PredictionFeatures(BaseModel):
    experience_level_encoded: float
    company_size_encoded: float
    employment_type_PT: int
    job_title_Data_Engineer: int
    job_title_Data_Manager: int
    job_title_Data_Scientist: int
    job_title_Machine_Learning_Engineer: int

# Global variable to store the loaded model
model = None
X_columns = None  # To store the columns of the training data

# Download the model and columns
def download_model():
    global model, X_columns
    model = joblib.load('lin_regression.sav')
    print(model.feature_names_in_)
    
    # To get the columns from the training data
    # Load the training data and apply the same preprocessing as done during training
    salary_data = pd.read_csv('ds_salaries.csv')  # Update with your training data path
    X = salary_data.drop(columns=['salary_in_usd'])
    X = pd.get_dummies(X, drop_first=True)
    X_columns = X.columns

# Download the model immediately when the script runs
download_model()

# Function to preprocess the input data
def preprocess_input_data(features):
    # Convert the features into a DataFrame
    input_data = pd.DataFrame([features.dict()])  # Using dict() to convert the Pydantic model to a dictionary
    
    # Apply the same one-hot encoding used during training
    input_data_encoded = pd.get_dummies(input_data, drop_first=True)
    
    # Align the columns of the input data with the trained model's columns
    input_data_encoded = input_data_encoded.reindex(columns=X_columns, fill_value=0)
    
    return input_data_encoded

# API root endpoint
@app.get("/")
async def index():
    return {"message": "Welcome to the Data Science Income API. Use the /predict feature to predict your income"}

# Prediction endpoint
@app.post("/predict")
async def predict(features: PredictionFeatures):
    try:
        # Convert input features to DataFrame
        input_data = pd.DataFrame([{
            "experience_level_encoded": features.experience_level_encoded,
            "company_size_encoded": features.company_size_encoded,
            "employment_type_PT": features.employment_type_PT,
            "job_title_Data_Engineer": features.job_title_Data_Engineer,
            "job_title_Data_Manager": features.job_title_Data_Manager,
            "job_title_Data_Scientist": features.job_title_Data_Scientist,
            "job_title_Machine_Learning_Engineer": features.job_title_Machine_Learning_Engineer
        }])

        # Align input_data columns with the model's expected features
        input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

        # Debugging logs
        print(f"Aligned input data columns: {input_data.columns}")
        print(f"Input data shape: {input_data.shape}")

        # Predict using the model
        prediction = model.predict(input_data)[0]

        return {"Salary (USD)": prediction}
    except ValueError as e:
        print(f"ValueError during prediction: {e}")
        return {"error": "Prediction failed. Ensure input features match training data."}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {"error": "An unexpected error occurred during prediction."}


# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
