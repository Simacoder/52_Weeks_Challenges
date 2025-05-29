# Full Recommender System Pipeline: From Data Cleaning to Model & Insights
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Load data
file_path = "data/dq_recsys_challenge_2025(in).csv"  
df = pd.read_csv(file_path)

# Convert date column
df['int_date'] = pd.to_datetime(df['int_date'], format='%d-%b-%y')

# Drop rows with missing values in critical columns
df = df.dropna(subset=['interaction', 'item', 'page', 'tod', 'item_type', 'segment', 'active_ind'])

# Create a numeric rating column from interactions
df['rating'] = df['interaction'].map({'DISPLAY': 0, 'CLICK': 1, 'CHECKOUT': 2})

# Datetime-based train/test split: last 2 weeks for test
cutoff = df['int_date'].max() - pd.Timedelta(weeks=2)
train_df = df[df['int_date'] <= cutoff]
test_df = df[df['int_date'] > cutoff]

# Time Series Analysis
weekly = df.groupby(['int_date', 'interaction']).size().unstack().fillna(0)
weekly = weekly.resample('W').sum()
plt.figure(figsize=(12, 6))
weekly.plot()
plt.title('Weekly Interaction Trends')
plt.xlabel('Date')
plt.ylabel('Interaction Counts')
plt.grid(True)
plt.tight_layout()
plt.show()

# Encode relevant columns for modeling
encode_cols = ['page', 'tod', 'item_type', 'segment', 'beh_segment', 'active_ind']
for col in encode_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Prepare data for correlation analysis
correlation_df = df[['rating'] + encode_cols].copy()
corr_matrix = correlation_df.corr()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# Feature and label preparation for modeling
features = df[encode_cols]
labels = (df['rating'] > 0).astype(int)  # Binary: 0 (display), 1 (click or checkout)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Evaluation metrics
print("Model Evaluation")
print("----------------")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
