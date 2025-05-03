# Bolt Fare Prediction System

## Research by Simanga Mchunu, Machine Learning Engineer

### Problem Statement
Bolt passengers often face uncertainties regarding ride fares due to fluctuating factors such as distance, time of the ride, and traffic conditions. This project aims to build a machine learning model to accurately predict Uber ride fares based on historical ride data, improving cost transparency for passengers and optimizing pricing strategies for ride-hailing services.

### Insights from the Data
- **Ride Duration vs. Price:** Longer rides generally result in higher fares, but peak hours introduce additional pricing fluctuations.
- **Time of Day Impact:** Late-night and early-morning rides tend to have surge pricing.
- **Distance Influence:** Distance is the primary determinant of fare, though external factors like demand surges affect final costs.
- **Feature Importance:** Features like ride duration, ride start hour, and ride distance contribute significantly to fare prediction accuracy.

### Tech Stack
#### Programming Language:
- Python

#### Libraries & Frameworks:
- **Data Processing:** pandas, numpy
- **Machine Learning Models:** scikit-learn, XGBoost
- **Model Evaluation:** mean_absolute_error, mean_squared_error, r2_score
- **Hyperparameter Tuning:** GridSearchCV
- **Deployment:** Streamlit

### Project Workflow
1. **Data Preprocessing:**
   - Load and clean the dataset (handle missing values, convert timestamps, calculate ride duration).
   - Convert categorical features into numerical representations.
2. **Exploratory Data Analysis (EDA):**
   - Understand data distributions and relationships using statistical analysis.
3. **Model Training & Evaluation:**
   - Train models: Linear Regression, Random Forest, and XGBoost.
   - Evaluate using RMSE, MAE, and R² score.
   - Select the best-performing model based on evaluation metrics.
4. **Hyperparameter Tuning:**
   - Optimize model parameters using GridSearchCV for improved performance.
5. **Deployment:**
   - Develop an interactive Streamlit web application for users to input ride details and get fare predictions.

### How to Run the Project
1. Clone this repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run the Streamlit app using `streamlit run app.py`.
4. Input ride details to get fare predictions.

### Future Enhancements
- Integrate real-time traffic data for better accuracy.
- Implement deep learning models (LSTMs) for time-series forecasting of demand trends.
- Deploy as a web service with an API for integration with Uber-like platforms.

### Contact
For any queries or collaboration opportunities, feel free to reach out to Simanga Mchunu.

---

This project demonstrates a professional, data-driven approach to fare prediction in the ride-hailing industry. 🚀

