# Data Science Income Prediction API

Welcome to the **Data Science Income Prediction API**, a project that bridges the gap between data science and software engineering. This API allows users to predict income levels for various data science roles based on job-related features, utilizing a machine learning model deployed with FastAPI.

## 🚀 Project Overview

As the field of data science evolves, the demand for **full-stack data scientists**—professionals capable of managing the entire data science lifecycle—has grown significantly. A critical skill in this domain is the ability to take machine learning models out of Jupyter notebooks and deploy them into production. This project demonstrates how to:

- Develop a machine learning model.
- Deploy it using **FastAPI**, a modern, high-performance web framework for building APIs.
- Prepare for containerization with tools like **Docker** for scalability and ease of deployment.

## 📁 Project Structure

```plaintext
.
├── app.py                # The main application script
├── lin_regress.sav       # Pre-trained machine learning model (Linear Regression)
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── Dockerfile            # Docker configuration for containerization
```

 # 🔧 Features

 1. **Income Prediction API:**

- Accepts job-related features such as experience level, company size, employment type, and job titles.
- Returns predicted income (in USD).
  
1. **FastAPI Integration:**

- Lightweight, fast, and ideal for building scalable APIs.

3. **Model Deployment:**

- A pre-trained machine learning model (lin_regress.sav) is loaded and served.

# 🛠️ Requirements

- Python 3.8+
- Joblib (for loading the trained model)
- FastAPI
- Pandas
- Uvicorn (ASGI server)

Install dependencies using:
```bash
    pip install -r requirements.txt
```

 🏗️ **How to Run the Project**

1. **Clone the Repository:**

```bash
    git clone https://github.com/Simacoder/52_weeks_challenges.git
    cd 52_weeks_challenges/
```
2. **Run the API Locally:**
   
```bash
    py -m uvicorn main:app --reload
```

3. **Access the API:**
   - Root endpoint: http://127.0.0.1:8000/
   - Prediction endpoint: http://127.0.0.1:8000/predict

4. **Test the Prediction Endpoint:** Send a POST request with the following JSON
   
   body:
```bash
    {
    "experience_level_encoded": 3.0,
    "company_size_encoded": 2.0,
    "employment_type_PT": 1,
    "job_title_Data_Engineer": 0,
    "job_title_Data_Manager": 1,
    "job_title_Data_Scientist": 0,
    "job_title_Machine_Learning_Engineer": 0
}
```
The API will respond with the predicted salary in USD.

🐳 **Docker Integration**

1. **Build the Docker Image:**
```bash
    docker build . -t apiserver
```
2. **Run the Docker Container:**
    
```bash
    docker run -p 8000:8000 apiserver
```

3. **Access the API:** Visit http://localhost:8000/ to interact with the API

✨ **Why This Project Matters**
In today's data-driven world, organizations increasingly value data scientists who can:

- Develop models and software solutions.
- Understand and implement ML Ops best practices.
- Deliver models as production-ready services.
By completing this project, you'll enhance your skills as a **full-stack data scientist**, capable of engaging in all stages of the data science lifecycle.

🤝 **Contributing**
- Contributions are welcome! Feel free to open an issue or submit a pull request to improve this project.

📝 **License**
- This project is licensed under the MIT License. See the LICENSE file for more details.


# AUTHOR
- Simanga Mchunu

