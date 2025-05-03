# import request package
import requests

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
