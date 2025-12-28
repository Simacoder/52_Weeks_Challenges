# 🏡 California Housing Explorer Dashboard

A comprehensive data science project demonstrating how to build a **personal API** with FastAPI and create an interactive dashboard with Streamlit. This project showcases the power of modular, reusable APIs for data science workflows.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Technologies Used](#technologies-used)
- [Key Concepts](#key-concepts)
- [Contributing](#contributing)

---

## 🎯 Overview

This project demonstrates the practical benefits of building a **personal API** for your data science work:

- **Reusability**: Use the same API endpoints across multiple applications (Streamlit, Jupyter notebooks, scripts)
- **Modularity**: Separate data processing logic from presentation layers
- **Scalability**: Deploy your models and data anywhere without duplicating code
- **Collaboration**: Share clean data and models with teammates through well-defined endpoints
- **Maintainability**: Version your models and encapsulate business logic centrally

The California Housing dataset is used as an example, with a complete ML pipeline and interactive dashboard built on top of the API.

---

## ✨ Features

### Backend (FastAPI)
- 📊 Load and explore the California Housing dataset
- 📈 Generate correlation plots and statistical analysis
- 🤖 Train multiple machine learning models (Linear Regression)
- 💾 Model versioning and persistence with joblib
- 🔮 Make predictions on new data
- 📉 Analyze prediction trends across variables
- ⚡ Automatic API documentation with Swagger UI

### Frontend (Streamlit)
- 🖥️ Interactive, tabbed dashboard interface
- 📊 Dataset exploration and statistics
- 📈 Correlation visualization
- 🛠️ Model training and performance tracking
- 📉 Interactive trend analysis
- 🖊️ Single property prediction interface
- 💰 Formatted currency output and real-time feedback

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                      │
│  (5 Tabs: Dataset, Correlation, Models, Trends, Predict)  │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP Requests
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                        │
│  (/data, /plot, /train_model, /models, /predict, /trend)  │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
    CSV Data        ML Models         Scalers
    california_     (joblib files)    (Preprocessing)
    housing.csv
```

---

## 📦 Prerequisites

- Python 3.8+
- pip or conda
- Basic understanding of:
  - FastAPI and REST APIs
  - Machine Learning with scikit-learn
  - Streamlit for dashboards

---

## 🚀 Installation

### 1. Clone or Download the Project
```bash
git clone 52_Weeks_CHallenges.git
cd 52_Weeks_Challenges
cd week_49
```

### 2. Create a Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n housing python=3.9
conda activate housing
```

### 3. Install Dependencies
```bash
pip install fastapi uvicorn scikit-learn pandas matplotlib plotly streamlit requests joblib
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### 4. Prepare the Data
Ensure `california_housing_alternative.csv` is in the project root directory.

---

## 💻 Usage

### Step 1: Start the FastAPI Backend
```bash
uvicorn main:app --reload
```

The API will be available at:
- **Main**: http://127.0.0.1:8000
- **Swagger Docs**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

### Step 2: Start the Streamlit Dashboard (in a new terminal)
```bash
streamlit run app.py
```

The dashboard will open at:
- http://localhost:8501

### Step 3: Explore!
Navigate through the 5 tabs to explore the dataset, train models, and make predictions.

---

## 📁 Project Structure

```
california-housing-dashboard/
├── main.py                           # FastAPI backend
├── app.py                            # Streamlit frontend
├── california_housing_alternative.csv # Dataset
├── models/                           # Directory for trained models
│   └── model_v1.joblib
│   └── model_v2.joblib
│   └── ...
├── requirements.txt                  # Project dependencies
└── README.md                         # This file
```

---

## 🔌 API Endpoints

### Data Management
- **`GET /data`** - Get dataset info (rows, columns)
- **`GET /plot`** - Get correlation plot as base64 image

### Model Training & Management
- **`POST /train_model`** - Train a new model, returns version and RMSE
- **`GET /models`** - List all trained models with their RMSE scores

### Predictions
- **`GET /predict`** - Make a single prediction
  - Query Parameters: `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `ocean_proximity`
  - Returns: `prediction`, `model_version`

- **`GET /predict_trend`** - Analyze prediction trends
  - Query Parameters: `variable` (median_income, longitude, latitude, housing_median_age), `version` (optional)
  - Returns: Trend plot, values, and predictions

### Documentation
- **`GET /docs`** - Swagger UI (interactive API documentation)
- **`GET /redoc`** - ReDoc (alternative documentation)

---

## 🛠️ Technologies Used

### Backend
- **FastAPI** - Modern, fast web framework for building APIs
- **Uvicorn** - ASGI web server
- **scikit-learn** - Machine learning library
- **Pandas** - Data manipulation and analysis
- **Matplotlib** - Plotting and visualization
- **joblib** - Model persistence and serialization
- **NumPy** - Numerical computing

### Frontend
- **Streamlit** - Rapid web app development for data science
- **Plotly** - Interactive visualizations
- **Requests** - HTTP client library

---

## 💡 Key Concepts

### Personal API Benefits

1. **Reusability**: Write preprocessing logic once in the API, use it everywhere
2. **Scalability**: Deploy the API to the cloud (AWS, Azure, GCP) without changing client code
3. **Collaboration**: Share data and models with teammates through clean API contracts
4. **Testing**: Test new models instantly across all connected applications
5. **Versioning**: Maintain multiple model versions and switch between them seamlessly

### ML Pipeline Flow
```
Raw Data → Preprocessing → Feature Scaling → Model Training
    ↓
API Endpoints ← Model Storage (joblib)
    ↓
Streamlit Dashboard (or other clients)
```

### Model Versioning
Each trained model is saved with a version number (v1, v2, ...) and can be referenced later. The `/predict` endpoint automatically uses the latest version.

---

## 🔄 Workflow Example

1. **Load Data**: API reads CSV, validates columns
2. **Explore**: Streamlit dashboard shows correlation plots
3. **Train**: Click "Train Model" button
4. **Evaluate**: View RMSE across different versions
5. **Predict**: Enter property features and get price predictions
6. **Analyze**: View how predictions change across variables

---

## 📊 Sample API Calls

### Get Dataset Info
```bash
curl http://127.0.0.1:8000/data
```

### Make a Prediction
```bash
curl "http://127.0.0.1:8000/predict?longitude=-122.0&latitude=37.0&housing_median_age=30&total_rooms=1000&total_bedrooms=200&population=500&households=150&median_income=5.0&ocean_proximity=NEAR%20BAY"
```

### Get Trend Analysis
```bash
curl "http://127.0.0.1:8000/predict_trend?variable=median_income"
```

---

## 🚀 Deployment Ideas

Once you've mastered the local setup, consider deploying to:

- **Docker**: Containerize both FastAPI and Streamlit
- **Cloud Platforms**: 
  - Heroku (FastAPI backend)
  - AWS EC2 / Elastic Beanstalk
  - Google Cloud Run
  - Azure App Service
- **Raspberry Pi**: Run locally for IoT applications
- **Kubernetes**: For production-scale deployments

---

## 🐛 Troubleshooting

### API not responding
- Ensure FastAPI is running: `uvicorn main:app --reload`
- Check that the API URL is correct: `http://127.0.0.1:8000`

### Models not training
- Verify `california_housing_alternative.csv` exists in the project root
- Check that the CSV has a `MedHouseVal` or `median_house_value` column

### Streamlit connection errors
- Make sure the FastAPI backend is running before starting Streamlit
- Check firewall settings

### Out of memory
- The dataset is small (~20k rows), so this shouldn't happen
- If using larger datasets, consider pagination or batching in the API

---

## 📚 Learning Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [REST API Best Practices](https://restfulapi.net/)

---

## 🎓 Next Steps

### Enhance the Project
- Add authentication (JWT tokens)
- Implement caching for faster responses
- Add more ML algorithms (Random Forest, Gradient Boosting)
- Create unit and integration tests
- Add logging and monitoring
- Implement pagination for large datasets

### Expand Functionality
- Add feature engineering endpoints
- Create model comparison tools
- Build hyperparameter tuning interface
- Add data upload capabilities
- Implement model explainability (SHAP values)

---

## 👥 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is open source and available under the MIT License.

---

## 🤝 Support

Have questions or found a bug? Please open an issue in the repository!

---

## 🎉 Acknowledgments

Built to demonstrate the power of **Personal APIs** for data science workflows, inspired by modern best practices in ML engineering and API design.

Happy coding! 🚀

---

**Last Updated**: December 2025  
**Maintained By**: Simanga Mchunu  
**Status**: Active Development