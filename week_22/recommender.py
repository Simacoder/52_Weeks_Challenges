import pandas as pd

# Load the dataset
file_path = "data/dq_recsys_challenge_2025(in).csv"
df = pd.read_csv(file_path)
# Display basic information
df_info = df.info()
df_head = df.head()
df_description = df.describe(include='all')

df_info, df_head, df_description

# Convert `int_date` to datetime format
df['int_date'] = pd.to_datetime(df['int_date'], format='%d-%b-%y')

# Create a label column for implicit feedback
interaction_mapping = {
    'DISPLAY': 0.0,
    'CLICK': 0.5,
    'CHECKOUT': 1.0
}
df['label'] = df['interaction'].map(interaction_mapping)

# Preview changes
df[['interaction', 'label', 'int_date']].head(10)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import numpy as np

# Filter out DISPLAY-only interactions for model training
df_filtered = df[df['interaction'] != 'DISPLAY'].copy()

# Encode categorical features
categorical_cols = ['item', 'page', 'tod', 'item_type', 'segment', 'beh_segment', 'active_ind']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_filtered[col] = le.fit_transform(df_filtered[col])
    label_encoders[col] = le

# Define features and target
features = ['idcol', 'item', 'page', 'tod', 'item_type', 'segment', 'beh_segment', 'active_ind']
target = 'label'

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df_filtered[features], df_filtered[target], test_size=0.2, random_state=42
)

# Prepare LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, categorical_feature=categorical_cols)

# Train LightGBM model
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'verbose': -1
}

from lightgbm import early_stopping

model = lgb.train(
    params,
    train_data,
    valid_sets=[test_data],
    num_boost_round=100,
    callbacks=[early_stopping(stopping_rounds=10)]
)

model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100, early_stopping_rounds=10)

