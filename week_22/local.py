from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

# Load your dataset
df = pd.read_csv("data/dq_recsys_challenge_2025(in).csv")

# Preprocessing
df['int_date'] = pd.to_datetime(df['int_date'], format='%d-%b-%y')
interaction_map = {'DISPLAY': 0.0, 'CLICK': 0.5, 'CHECKOUT': 1.0}
df['label'] = df['interaction'].map(interaction_map)
df_filtered = df[df['interaction'] != 'DISPLAY'].copy()

# Label encode categorical features
categorical_cols = ['item', 'page', 'tod', 'item_type', 'segment', 'beh_segment', 'active_ind']
for col in categorical_cols:
    df_filtered[col] = LabelEncoder().fit_transform(df_filtered[col])

# Features and target
features = ['idcol', 'item', 'page', 'tod', 'item_type', 'segment', 'beh_segment', 'active_ind']
X = df_filtered[features]
y = df_filtered['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE:", rmse)
