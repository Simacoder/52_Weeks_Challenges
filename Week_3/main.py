# import packages
import uvicorn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import requests

# Initialie Fast FastApi
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


# Global variable to tore the loaded model 
model = None

# Download the model 
def download_model():
    global model 
    model = joblib.load('lin_regression.sav')
    
# Download the model immediately when the script runs
download_model()

# API root endpoint
@app.get("/")
async def index():
    return {"message": "Welcome to the Data Science Income API. Use the /predict feature to predict your income"}
    
# prediction endpoint 
@app.post("/predict")
async def predict(features: PredictionFeatures):
    # Create an input dataframe  for Prediction 

    input_data = pd.DataFrame([{
        "experience_level_encoded": features.experience_level_encoded,
        "company_size_encoded": features.company_size_encoded,
        "employment_type_PT": features.employment_type_PT,
        "job_title_Data_Engineer":features.job_title_Data_Engineer,
        "job_title_Data_Manager": features.job_title_Data_Manager,
         "job_title_Data_Scientist": features.job_title_Data_Scientist,
        "job_title_Machine_Learning_Engineer": features.job_title_Machine_Learning_Engineer

    }])

    # Predict use the loaded model
    prediction = model.predict(input_data)[0]
    return {
        "Salary (USD)":prediction
    }

url = 'http://127.0.0.1:8000/predict'
# dummy data to tet API 
data = {
    "experience_level_encoded": 3.0,
    "company_size_encoded": 3.0,
    "employment_type_PT": 0,
    "job_title_Data_Engineer": 0,
    "job_title_Data_Manager": 1,
    "job_title_Data_Scientist": 0,
    "job_title_Machine_Learning_Engineer": 0
}

# amke a Post request  to the API
response = requests.post(url, json=data)

# print response
response.json
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
        