# 🛒 BigMart Sales Prediction using Machine Learning Pipeline

This project aims to predict product sales in various BigMart outlets using a machine learning model built with a complete Scikit-learn pipeline. It covers data preprocessing, feature engineering, model training, and evaluation — all automated using the `Pipeline` and `ColumnTransformer` tools from Scikit-learn.

---

## 📌 Problem Statement

Given historical sales data for BigMart outlets, including information about items (e.g., weight, category, price) and outlets (e.g., size, location, type), the goal is to forecast **Item Outlet Sales** — a regression problem.

---

## 💡 Project Highlights

- Automated end-to-end **ML pipeline**
- **Preprocessing steps**: Imputation, encoding, and scaling
- Trained a **Random Forest Regressor**
- Evaluated with **Root Mean Squared Error (RMSE)**
- Designed to be easily extendable and production-ready

---

## 🧱 Tech Stack

- **Language**: Python 3.8+
- **Libraries**:  
  - `pandas`, `numpy`, `matplotlib`
  - `scikit-learn` for modeling and pipelines
  - `category_encoders` for encoding (if needed)
  
---

## 🔄 Workflow

1. **Data Loading**: Read training and test CSV files
2. **Exploratory Data Analysis**: View basic info, missing values, data types
3. **Preprocessing**:  
   - Fill missing values in `Item_Weight` and `Outlet_Size`  
   - Encode categorical variables  
   - Scale numerical features
4. **Modeling**:  
   - Train a `RandomForestRegressor`  
   - Evaluate using RMSE
5. **Pipeline Integration**: Automate all preprocessing + model training in one step

---

## 🚀 How to Run

1. **Clone this repo**
   ``` bash
        git clone https://github.com/simacoder/52_Weeks_Challenges.git
        cd 52_Weeks_Challenges
        cd week_24
    ```
---

## **Install dependencies**

```bash
    pip install -r requirements.txt
```
## **Run the script**


```bash
python MLpipeline.py
```

## 📁 **Project Structure**

```bash
📦 week_23/
├── data/
│   ├── train.csv
│   └── test.csv
├── MLpipeline.py
├── README.md
└── requirements.txt
```

## 📊 **Evaluation Metric**

- Root Mean Squared Error (RMSE) on training and test sets

- Optional: R² Score for understanding model fit

## 📌 **Notes**

Make sure the train.csv and test.csv files are placed in the data/ folder.

You can extend this pipeline for hyperparameter tuning using GridSearchCV or RandomizedSearchCV.

## 📜 **License**
- This project is open source under the [MIT License].

## 🙌 **Acknowledgements**

Based on the BigMart Sales Prediction challenge and inspired by educational content by Lakshay Arora.

# AUTHOR
- Simanga Mchunu