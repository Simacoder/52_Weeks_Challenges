import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.inspection import permutation_importance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# --- Load Data ---
file_path = "data/dq_recsys_challenge_2025(in).csv"
df = pd.read_csv(file_path)

# Define the user ID column name (adjust if needed)
user_id_col = 'idcol'  # Change this if your user id column has a different name

# --- Preprocessing ---
df['int_date'] = pd.to_datetime(df['int_date'], format='%d-%b-%y')
df = df.dropna(subset=['interaction', 'item', 'page', 'tod', 'item_type', 'segment', 'active_ind', user_id_col])

# Map interaction to numeric rating
df['rating'] = df['interaction'].map({'DISPLAY': 0, 'CLICK': 1, 'CHECKOUT': 2})

# Split train/test by date
cutoff = df['int_date'].max() - pd.Timedelta(weeks=2)
train_df = df[df['int_date'] <= cutoff]
test_df = df[df['int_date'] > cutoff]

# --- Weekly Interaction Trends ---
weekly = df.groupby(['int_date', 'interaction']).size().unstack().fillna(0)
weekly = weekly.resample('W').sum()
plt.figure(figsize=(12, 6))
weekly.plot(ax=plt.gca())
plt.title('Weekly Interaction Trends')
plt.xlabel('Date')
plt.ylabel('Interaction Counts')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Encode categorical features ---
encode_cols = ['page', 'tod', 'item_type', 'segment', 'beh_segment', 'active_ind']
for col in encode_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Correlation heatmap
correlation_df = df[['rating'] + encode_cols].copy()
corr_matrix = correlation_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# --- Prepare supervised modeling dataset ---
features = df[encode_cols]
labels = (df['rating'] > 0).astype(int)  # 1 if CLICK or CHECKOUT else 0

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Logistic Regression Grid Search ---
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['liblinear'],
    'class_weight': ['balanced'],
    'max_iter': [100, 200, 300]
}
grid_search = GridSearchCV(
    LogisticRegression(),
    param_grid,
    scoring='roc_auc',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)
best_log_model = grid_search.best_estimator_

# --- Random Forest ---
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
rf_model.fit(X_train_scaled, y_train)

# --- Bagging Random Forest ---
bagging_rf = BaggingClassifier(
    estimator=RandomForestClassifier(n_estimators=50, max_depth=8, class_weight='balanced', random_state=42),
    n_estimators=10,
    random_state=42,
    n_jobs=-1
)
bagging_rf.fit(X_train_scaled, y_train)

# --- Gradient Boosting ---
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb_model.fit(X_train_scaled, y_train)

# --- Model Evaluation ---
models = {
    'Logistic Regression': best_log_model,
    'Random Forest': rf_model,
    'Bagging RF': bagging_rf,
    'Gradient Boosting': gb_model,
}

metrics_list = []
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    logloss = log_loss(y_test, y_proba)
    
    k = 10
    top_k_indices = np.argsort(y_proba)[-k:][::-1]
    precision_at_k = y_test.iloc[top_k_indices].sum() / k
    
    print(f"\nModel: {name}")
    print("---------------------------")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print(f"Precision@{k}: {precision_at_k:.4f}")
    
    metrics_list.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'AUC-ROC': auc,
        'Log Loss': logloss,
        'Precision@10': precision_at_k
    })

metrics_df = pd.DataFrame(metrics_list).set_index('Model')

# --- Plot metrics comparison ---
plt.figure(figsize=(12, 7))
for metric in metrics_df.columns:
    plt.plot(metrics_df.index, metrics_df[metric], marker='o', label=metric)
plt.title('Model Performance Comparison')
plt.xlabel('Model')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Feature Importance RF ---
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), np.array(encode_cols)[indices], rotation=45)
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# --- Permutation Importance Logistic Regression ---
result = permutation_importance(best_log_model, X_test_scaled, y_test, n_repeats=10, random_state=42, scoring='roc_auc')
sorted_idx = result.importances_mean.argsort()[::-1]

plt.figure(figsize=(10, 6))
plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=np.array(encode_cols)[sorted_idx])
plt.title("Permutation Importance (Logistic Regression)")
plt.tight_layout()
plt.show()

# --- Precision@K plot for RF ---
y_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]
ks = range(1, 51)
precisions_at_k = []
for k in ks:
    top_k_indices = np.argsort(y_proba_rf)[-k:][::-1]
    precision_at_k = y_test.iloc[top_k_indices].sum() / k
    precisions_at_k.append(precision_at_k)

plt.figure(figsize=(8, 5))
plt.plot(ks, precisions_at_k, marker='o')
plt.title('Precision@K for Random Forest')
plt.xlabel('K')
plt.ylabel('Precision@K')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Content-Based Filtering using item_description ---
item_descriptions = df[['item', 'item_descrip']].drop_duplicates().reset_index(drop=True)
item_descriptions['item_descrip'] = item_descriptions['item_descrip'].fillna('')

tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = tfidf.fit_transform(item_descriptions['item_descrip'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_similar_items(item_id, top_n=10):
    try:
        idx = item_descriptions[item_descriptions['item'] == item_id].index[0]
    except IndexError:
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    similar_indices = [i[0] for i in sim_scores]
    return item_descriptions.iloc[similar_indices]['item'].tolist()

print("\nContent-Based Recommendations for item:", item_descriptions.iloc[0]['item'])
print(get_similar_items(item_descriptions.iloc[0]['item']))

# --- Collaborative Filtering with SVD ---
user_item_df = df.pivot_table(index=user_id_col, columns='item', values='rating', fill_value=0)
user_item_matrix = csr_matrix(user_item_df.values)

svd = TruncatedSVD(n_components=20, random_state=42)
user_factors = svd.fit_transform(user_item_matrix)
item_factors = svd.components_.T

def recommend_items_collaborative(user_id, top_n=10):
    if user_id not in user_item_df.index:
        return []
    user_idx = user_item_df.index.get_loc(user_id)
    user_vector = user_factors[user_idx, :]
    scores = item_factors.dot(user_vector)
    item_indices = np.argsort(scores)[::-1]
    recommended_items = user_item_df.columns[item_indices]
    interacted_items = user_item_df.loc[user_id][user_item_df.loc[user_id] > 0].index
    recommendations = [item for item in recommended_items if item not in interacted_items]
    return recommendations[:top_n]

print("\nCollaborative Filtering Recommendations for user:", user_item_df.index[0])
print(recommend_items_collaborative(user_item_df.index[0]))
