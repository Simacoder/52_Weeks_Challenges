import os
import io
import base64
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

st.set_page_config(page_title="California Housing ML", layout="wide", initial_sidebar_state="collapsed")

# ----------------------------
# Setup directories and load data
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "california_housing_alternative.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(CSV_PATH)
        if 'MedHouseVal' in df.columns:
            df.rename(columns={'MedHouseVal': 'median_house_value'}, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

df = load_data()

# ----------------------------
# Global state variables
# ----------------------------
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_test = None
    st.session_state.feature_columns = None

# ----------------------------
# Custom CSS for better styling
# ----------------------------
st.markdown("""
    <style>
    .main-header {
        font-size: 3em;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 1.2em;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<p class="main-header">🏠 California Housing</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ML Price Prediction System</p>', unsafe_allow_html=True)

if df.empty:
    st.error("❌ Dataset not loaded. Please check the CSV file path.")
    st.stop()

# ----------------------------
# Tab Navigation
# ----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Explorer",
    "🤖 Train Model",
    "🎯 Make Predictions",
    "📋 Model Management",
    "📉 Trend Analysis"
])

# ============================
# TAB 1: DATA EXPLORER
# ============================
with tab1:
    st.header("📊 Dataset Explorer")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📈 Total Rows", len(df))
    with col2:
        st.metric("📋 Total Columns", len(df.columns))
    with col3:
        st.metric("❌ Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("💰 Target Column", "median_house_value")
    
    st.divider()
    
    # Data Preview
    st.subheader("Data Preview")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df.head(10), use_container_width=True)
    with col2:
        st.write("**Data Types:**")
        st.dataframe(df.dtypes.to_frame(name="Type"), use_container_width=True)
    
    st.divider()
    
    # Statistics
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.divider()
    
    # Correlation Analysis
    st.subheader("📈 Correlation with House Value")
    try:
        numeric_df = df.select_dtypes(include='number')
        if 'median_house_value' in numeric_df.columns:
            corr_values = numeric_df.corr()['median_house_value'].sort_values(ascending=False)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(corr_values.index, corr_values.values, color='steelblue')
                ax.set_xlabel("Correlation Coefficient")
                ax.set_title("Feature Importance by Correlation")
                ax.grid(axis='x', alpha=0.3)
                st.pyplot(fig)
            with col1:
                st.write("**Top Correlated Features:**")
                st.dataframe(corr_values.head(10).to_frame(name="Correlation"), use_container_width=True)
        else:
            st.warning("'median_house_value' column not found")
    except Exception as e:
        st.error(f"Error plotting correlation: {e}")

# ============================
# TAB 2: TRAIN MODEL
# ============================
with tab2:
    st.header("🤖 Train Machine Learning Model")
    
    st.info("ℹ️ Configure training parameters and train your Linear Regression model")
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Size Ratio", 0.1, 0.5, 0.2, help="Proportion of data used for testing")
    with col2:
        random_state = st.number_input("Random State (for reproducibility)", 0, 1000, 42)
    
    st.divider()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        train_button = st.button("🚀 Train Model", key="train_btn", use_container_width=True)
    
    if train_button:
        with st.spinner("⏳ Training model... This may take a moment"):
            try:
                if 'median_house_value' not in df.columns:
                    st.error("'median_house_value' column not found")
                else:
                    X = df.drop(columns=['median_house_value'])
                    y = df['median_house_value']

                    # One-hot encode categorical
                    X = pd.get_dummies(X, drop_first=True)

                    # Impute missing values
                    imputer = SimpleImputer(strategy='median')
                    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
                    feature_columns = X_imputed.columns.tolist()

                    # Train/test split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_imputed, y, test_size=test_size, random_state=random_state
                    )

                    # Save to session state
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.feature_columns = feature_columns

                    # Feature scaling
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # Train LinearRegression
                    model = LinearRegression()
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = model.score(X_test_scaled, y_test)

                    # Versioning
                    existing_versions = [
                        int(f.split("_v")[1].split(".")[0])
                        for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")
                    ]
                    version = max(existing_versions) + 1 if existing_versions else 1

                    # Save model
                    joblib.dump({
                        "model": model,
                        "scaler": scaler,
                        "imputer": imputer,
                        "feature_columns": feature_columns
                    }, os.path.join(MODELS_DIR, f"model_v{version}.joblib"))

                    # Display results
                    st.success(f"✅ Model trained successfully! **Version {version}**")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🎯 RMSE", f"${rmse:,.0f}")
                    with col2:
                        st.metric("📊 R² Score", f"{r2:.4f}")
                    with col3:
                        st.metric("📈 Train Samples", len(X_train))
                    with col4:
                        st.metric("🧪 Test Samples", len(X_test))
                    
                    st.divider()
                    st.write("**Model Details:**")
                    details_col1, details_col2 = st.columns(2)
                    with details_col1:
                        st.write(f"- **Features Used:** {len(feature_columns)}")
                        st.write(f"- **Train/Test Split:** {100*(1-test_size):.0f}% / {100*test_size:.0f}%")
                    with details_col2:
                        st.write(f"- **Model Type:** Linear Regression")
                        st.write(f"- **Saved Location:** models/model_v{version}.joblib")

            except Exception as e:
                st.error(f"❌ Error training model: {e}")

# ============================
# TAB 3: MAKE PREDICTIONS
# ============================
with tab3:
    st.header("🎯 Make Price Predictions")
    
    model_files = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")])

    if not model_files:
        st.warning("⚠️ No trained models available. Please train a model first in the 'Train Model' tab!")
    else:
        st.info(f"ℹ️ {len(model_files)} model(s) available")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            selected_model = st.selectbox("Select Model", model_files, key="pred_model")
        with col2:
            st.write("")
            st.write("")
            st.write(f"**Selected:** {selected_model}")
        
        st.divider()
        st.subheader("📝 Enter Property Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Location & Geography:**")
            longitude = st.number_input("Longitude", value=-120.0, format="%.2f")
            latitude = st.number_input("Latitude", value=37.0, format="%.2f")
            ocean_proximity = st.selectbox("Ocean Proximity", 
                                          ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])
        
        with col2:
            st.write("**Building & Occupancy:**")
            housing_median_age = st.number_input("Housing Median Age (years)", value=30.0, format="%.1f")
            total_rooms = st.number_input("Total Rooms", value=3000.0, format="%.0f")
            total_bedrooms = st.number_input("Total Bedrooms", value=600.0, format="%.0f")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.write("**Demographics:**")
            population = st.number_input("Population", value=1500.0, format="%.0f")
            households = st.number_input("Households", value=500.0, format="%.0f")
        
        with col4:
            st.write("**Economics:**")
            median_income = st.number_input("Median Income (in R10,000s)", value=5.0, format="%.2f")
        
        st.divider()
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_button = st.button("🔮 Predict Price", key="predict_btn", use_container_width=True)
        
        if predict_button:
            try:
                data_model = joblib.load(os.path.join(MODELS_DIR, selected_model))
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
                prediction = model.predict(X_new_scaled)[0]
                
                st.divider()
                st.success("✅ Prediction Complete!")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.metric("💰 Predicted House Value", f"R{prediction:,.2f}", delta=None)
                with col2:
                    st.write("")
                    st.write(f"**Model:** {selected_model}")
                
                # Show input summary
                st.write("**Input Summary:**")
                input_summary = pd.DataFrame([{
                    "Longitude": longitude,
                    "Latitude": latitude,
                    "Age": housing_median_age,
                    "Total Rooms": total_rooms,
                    "Bedrooms": total_bedrooms,
                    "Population": population,
                    "Households": households,
                    "Median Income": f"R{median_income*10000:,.0f}",
                    "Ocean Proximity": ocean_proximity
                }]).T
                st.dataframe(input_summary)

            except Exception as e:
                st.error(f"❌ Error making prediction: {e}")

# ============================
# TAB 4: MODEL MANAGEMENT
# ============================
with tab4:
    st.header("📋 Model Management & Evaluation")
    
    model_files = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")])
    
    if not model_files:
        st.info("ℹ️ No trained models yet. Train a model in the 'Train Model' tab to see it here!")
    else:
        st.subheader(f"Available Models ({len(model_files)})")
        
        col1, col2 = st.columns([2, 1])
        with col2:
            if st.button("🔄 Evaluate All Models", use_container_width=True):
                st.session_state.evaluate = True
        
        if st.session_state.get('evaluate', False):
            with st.spinner("📊 Evaluating models..."):
                try:
                    if st.session_state.X_test is None:
                        st.error("❌ No test data available. Train a model first.")
                    else:
                        models_info = []
                        for f in model_files:
                            data = joblib.load(os.path.join(MODELS_DIR, f))
                            model = data['model']
                            scaler = data['scaler']
                            imputer = data['imputer']
                            X_test_imputed = pd.DataFrame(
                                imputer.transform(st.session_state.X_test), 
                                columns=st.session_state.X_test.columns
                            )
                            X_test_scaled = scaler.transform(X_test_imputed)
                            y_pred = model.predict(X_test_scaled)
                            rmse = np.sqrt(mean_squared_error(st.session_state.y_test, y_pred))
                            r2 = model.score(X_test_scaled, st.session_state.y_test)
                            models_info.append({
                                "Model": f, 
                                "RMSE": f"${rmse:,.0f}",
                                "R² Score": f"{r2:.4f}",
                                "Trained": "✅"
                            })
                        
                        st.dataframe(pd.DataFrame(models_info), use_container_width=True)
                        st.success("✅ Evaluation complete!")
                        st.session_state.evaluate = False
                except Exception as e:
                    st.error(f"❌ Error evaluating models: {e}")
        
        st.divider()
        
        st.subheader("Model Files")
        for model_file in model_files:
            file_size = os.path.getsize(os.path.join(MODELS_DIR, model_file)) / 1024
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"📦 **{model_file}**")
            with col2:
                st.write(f"Size: {file_size:.2f} KB")
            with col3:
                if st.button(f"🗑️ Delete", key=f"delete_{model_file}"):
                    os.remove(os.path.join(MODELS_DIR, model_file))
                    st.success(f"Deleted {model_file}")
                    st.rerun()

# ============================
# TAB 5: TREND ANALYSIS
# ============================
with tab5:
    st.header("📉 Trend Analysis")
    
    model_files = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")])
    
    if not model_files:
        st.info("ℹ️ No trained models available. Train a model first to see trends!")
    else:
        st.info("ℹ️ Analyze how predicted prices change across different variables")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_trend_model = st.selectbox("Select Model", model_files, key="trend_model")
        with col2:
            numeric_cols = [col for col in df.columns if df[col].dtype.kind in 'iufc' and col != 'median_house_value']
            trend_variable = st.selectbox("Select Variable", numeric_cols)
        
        st.divider()
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            trend_button = st.button("📊 Generate Trend Plot", use_container_width=True)
        
        if trend_button:
            with st.spinner("⏳ Generating trend analysis..."):
                try:
                    data_model = joblib.load(os.path.join(MODELS_DIR, selected_trend_model))
                    model = data_model['model']
                    scaler = data_model['scaler']
                    imputer = data_model['imputer']
                    feature_columns = data_model['feature_columns']

                    if trend_variable not in df.columns:
                        st.error(f"Variable '{trend_variable}' not found")
                    else:
                        trend_values = np.linspace(df[trend_variable].min(), df[trend_variable].max(), 50)
                        trend_df = pd.DataFrame({trend_variable: trend_values})

                        for col in feature_columns:
                            if col == trend_variable:
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

                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(trend_values, y_pred, marker='o', linewidth=3, markersize=6, color='steelblue')
                        ax.fill_between(trend_values, y_pred, alpha=0.3, color='steelblue')
                        ax.set_xlabel(f"{trend_variable}", fontsize=12)
                        ax.set_ylabel("Predicted median_house_value (R)", fontsize=12)
                        ax.set_title(f"Price Trend Analysis: Impact of {trend_variable}", fontsize=14, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                        st.pyplot(fig)
                        
                        st.divider()
                        st.subheader("📊 Trend Data")
                        trend_data = pd.DataFrame({
                            trend_variable: trend_values,
                            "Predicted Price": y_pred
                        })
                        st.dataframe(trend_data, use_container_width=True)

                except Exception as e:
                    st.error(f" Error generating trend: {e}")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #999;'>Built with Streamlit With Simanga | California Housing Dataset</p>", unsafe_allow_html=True)