import streamlit as st
import requests
import base64
import pandas as pd
import plotly.express as px

API_URL = "http://127.0.0.1:8000"

# ------------------------------
# Page configuration
# ------------------------------
st.set_page_config(
    page_title="California Housing Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏡 California Housing Explorer Dashboard")

# ------------------------------
# Load dataset for categorical values
# ------------------------------
data_resp = requests.get(f"{API_URL}/data").json()
if "error" in data_resp:
    df = pd.DataFrame()
else:
    df = pd.DataFrame(columns=data_resp["columns"])

# For categorical dropdown - fetch from API
try:
    ocean_resp = requests.get(f"{API_URL}/data").json()
    if "error" not in ocean_resp and "columns" in ocean_resp:
        # Try to get unique values from the dataset
        data_full = requests.get(f"{API_URL}/data").json()
        ocean_proximity_values = ["NEAR BAY","<1H OCEAN","INLAND","NEAR OCEAN","ISLAND"]
    else:
        ocean_proximity_values = ["NEAR BAY","<1H OCEAN","INLAND","NEAR OCEAN","ISLAND"]
except:
    # Fallback values if dataset not yet loaded
    ocean_proximity_values = ["NEAR BAY","<1H OCEAN","INLAND","NEAR OCEAN","ISLAND"]

# ------------------------------
# Tabs for better navigation
# ------------------------------
tabs = st.tabs(["Dataset", "Correlation", "Models & RMSE", "Trend Prediction", "Single Prediction"])

# ------------------------------ Dataset Tab ------------------------------
with tabs[0]:
    st.header("📊 Dataset Info")
    if "error" in data_resp:
        st.error(data_resp["error"])
    else:
        st.write(f"Number of rows: {data_resp['rows']}")
        st.write(f"Columns: {', '.join(data_resp['columns'])}")

# ------------------------------ Correlation Tab ------------------------------
with tabs[1]:
    st.header("📈 Correlation Plot")
    plot_resp = requests.get(f"{API_URL}/plot").json()
    if "plot_base64" in plot_resp:
        st.image(base64.b64decode(plot_resp["plot_base64"]), width="stretch")
    else:
        st.error(plot_resp.get("error","Unknown error"))

# ------------------------------ Models & RMSE Tab ------------------------------
with tabs[2]:
    st.header("🛠️ Models and RMSE")
    col1, col2 = st.columns([1,2])
    
    with col1:
        st.subheader("Train a New Model")
        if st.button("Train Model"):
            train_resp = requests.post(f"{API_URL}/train_model").json()
            if "error" in train_resp:
                st.error(train_resp["error"])
            else:
                st.success(f"Model v{train_resp['version']} trained! RMSE: {train_resp['rmse']:.2f}")

    with col2:
        st.subheader("Existing Models")
        models_resp = requests.get(f"{API_URL}/models").json()
        if isinstance(models_resp, list) and len(models_resp) > 0:
            models_df = pd.DataFrame(models_resp)
            st.dataframe(models_df)

            # Interactive RMSE vs Version
            models_df["version"] = models_df["model_file"].str.extract(r'v(\d+)').astype(int)
            fig = px.line(models_df.sort_values("version"), x="version", y="rmse",
                          markers=True, title="RMSE per Model Version",
                          labels={"version":"Model Version","rmse":"RMSE"})
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No trained models yet.")

# ------------------------------ Trend Prediction Tab ------------------------------
with tabs[3]:
    st.header("📉 Prediction Trend")
    variable = st.selectbox("Select variable for trend", ["median_income","longitude","latitude","housing_median_age"])
    trend_resp = requests.get(f"{API_URL}/predict_trend", params={"variable": variable}).json()
    if "error" in trend_resp:
        st.error(trend_resp["error"])
    elif "plot_base64" in trend_resp:
        st.image(base64.b64decode(trend_resp["plot_base64"]), width="stretch")
        with st.expander("Show Trend Data"):
            st.dataframe(pd.DataFrame({
                variable: trend_resp["trend_values"],
                "Predicted median_house_value": trend_resp["predictions"]
            }))
    else:
        st.error("Unexpected response from API")

# ------------------------------ Single Prediction Tab ------------------------------
with tabs[4]:
    st.header("🖊️ Make a Prediction")
    st.write("Enter the feature values:")
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            longitude = st.number_input("Longitude", value=-122.0, step=0.1)
            latitude = st.number_input("Latitude", value=37.0, step=0.1)
            housing_median_age = st.number_input("Housing Median Age", value=30, step=1)
            total_rooms = st.number_input("Total Rooms", value=1000, step=10)
        with col2:
            total_bedrooms = st.number_input("Total Bedrooms", value=200, step=10)
            population = st.number_input("Population", value=500, step=10)
            households = st.number_input("Households", value=150, step=10)
            median_income = st.number_input("Median Income", value=5.0, step=0.1)
        with col3:
            ocean_proximity = st.selectbox("Ocean Proximity", ocean_proximity_values)
        
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            # Prepare inputs as individual query parameters
            inputs = {
                "longitude": float(longitude),
                "latitude": float(latitude),
                "housing_median_age": float(housing_median_age),
                "total_rooms": float(total_rooms),
                "total_bedrooms": float(total_bedrooms),
                "population": float(population),
                "households": float(households),
                "median_income": float(median_income),
                "ocean_proximity": str(ocean_proximity)
            }
            
            try:
                # Use GET request with params (matches your FastAPI endpoint)
                pred_resp = requests.get(f"{API_URL}/predict", params=inputs).json()
                
                if "error" in pred_resp:
                    st.error(f" Error: {pred_resp['error']}")
                elif "prediction" in pred_resp:
                    st.success(f" Predicted median_house_value: R{pred_resp['prediction']:,.2f}")
                    st.info(f" Model used: {pred_resp['model_version']}")
                else:
                    st.error(" Unexpected response from API")
                    st.write(pred_resp)
            except Exception as e:
                st.error(f" Connection error: {str(e)}")