import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor

# ===========================================================
# LOAD CLEANED DATASET
# ===========================================================

df = pd.read_csv("Preprocessed_Student_Performance_Data.csv")


# ===========================================================
# LABEL ENCODING FOR CATEGORICAL COLUMNS
# ===========================================================

df_encoded = df.copy()

categorical_cols = df_encoded.select_dtypes(include=['object']).columns
print("\nCategorical columns to be label encoded:", list(categorical_cols))

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])

print("\n=== Encoded head ===")
print(df_encoded.head())

# ===========================================================
# STAGE 5 — MODEL BUILDING (REGRESSION)
# ===========================================================

target = "Exam_Score"   # <-- NEW TARGET VARIABLE

X = df_encoded.drop(columns=[target])
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===========================================================
# BASELINE MODELS (Decision Tree & Random Forest)
# ===========================================================
# ---- Decision Tree Regressor ----
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Save the Decision Tree model
joblib.dump(dt, "decision_tree_model.pkl")
print("Decision Tree model saved.")


# ---- Random Forest Regressor ----
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Save the Random Forest model
joblib.dump(rf, "random_forest_baseline.pkl")
print("Baseline Random Forest model saved as random_forest_baseline.pkl")


# Evaluation function
def evaluate_regression(model_name, y_true, y_pred):
    print(f"\n===== {model_name} =====")
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R²  :", r2)

# Evaluate both models
evaluate_regression("Decision Tree Regressor", y_test, y_pred_dt)
evaluate_regression("Random Forest Regressor", y_test, y_pred_rf)

# ===========================================================
# STAGE 5 — K-MEANS CLUSTERING
# ===========================================================

# Elbow Method
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    wcss.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method for K-Means")
plt.xlabel("Number of clusters (k)")
plt.ylabel("WCSS")
plt.show()

# Train KMeans with k=3 (example)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)
df_encoded["Cluster"] = clusters

# Compute cluster score averages for labeling
cluster_labels = {}

for c in np.unique(clusters):
    cluster_mean = df_encoded[df_encoded["Cluster"] == c]["Exam_Score"].mean()
    cluster_labels[c] = cluster_mean

# Sort by mean score (low → high)
sorted_labels = sorted(cluster_labels.items(), key=lambda x: x[1])

# Map cluster index to a human-friendly label
label_map = {
    sorted_labels[0][0]: "Düşük Başarı",
    sorted_labels[1][0]: "Orta Başarı",
    sorted_labels[2][0]: "Yüksek Başarı"
}

# Save label map
joblib.dump(label_map, "cluster_labels.pkl")
print("Cluster labels saved!")

# Silhouette score
sil = silhouette_score(X, clusters)
print("\nSilhouette Score:", sil)

print("\n=== Cluster counts ===")
print(df_encoded["Cluster"].value_counts())

# Save the K-Means model
joblib.dump(kmeans, "kmeans_cluster_model.pkl")
print("KMeans model saved.")

# Scatter plot for K-Means
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df_encoded["Hours_Studied"],
    y=df_encoded["Exam_Score"],
    hue=df_encoded["Cluster"],
    palette="viridis"
)
plt.title("K-Means Clusters (Hours_Studied vs Exam_Score)")
plt.show()

# ===========================================================
# ADVANCED MODELS (Train/Val/Test + Tuned Random Forest + XGBoost)
# ===========================================================
# ----------------------------------------------------------
# 1) Train–Validation–Test separation and
# 2) Hyperparameter tuning for Random Forest
# ----------------------------------------------------------
# ----------------------------------------------------------
# TRAIN - VALIDATION - TEST SPLIT
# ----------------------------------------------------------

target = "Exam_Score"

X = df_encoded.drop(columns=[target])
y = df_encoded[target]

# First split into train (70%) and temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Split temp into validation (15%) and test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print("Train size:", X_train.shape)
print("Validation size:", X_val.shape)
print("Test size:", X_test.shape)

# ----------------------------------------------------------
# RANDOM FOREST HYPERPARAMETER TUNING
# ----------------------------------------------------------

param_dist = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False]
}

rf = RandomForestRegressor(random_state=42)

# Randomized Search
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=25,              # Try 25 different combinations
    cv=3,                  # cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("\nBest Parameters:", random_search.best_params_)

# First, get the best model
best_rf = random_search.best_estimator_

# Then, save the model
joblib.dump(best_rf, "random_forest_model.pkl")
print("Random Forest model saved as random_forest_model.pkl")

# ----------------------------------------------------------
# FINAL MODEL WITH BEST PARAMETERS
# ----------------------------------------------------------

best_rf = random_search.best_estimator_

# Validation performance
val_pred = best_rf.predict(X_val)

print("\n===== Validation Set Performance =====")
print("MSE:", mean_squared_error(y_val, val_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_val, val_pred)))
print("R²:", r2_score(y_val, val_pred))

# Test performance
test_pred = best_rf.predict(X_test)

print("\n===== FINAL TEST PERFORMANCE =====")
print("MSE:", mean_squared_error(y_test, test_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, test_pred)))
print("R²:", r2_score(y_test, test_pred))

# ----------------------------------------------------------
# 1) Adding the XGBoost model
# 2) Extracting Feature Importance plots
# ----------------------------------------------------------

# ----------------------------------------------------------
# 1) XGBOOST HYPERPARAMETER TUNING
# ----------------------------------------------------------

xgb = XGBRegressor(objective="reg:squarederror", random_state=42)

param_grid = {
    "n_estimators": [200, 300, 400, 500, 600],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "subsample": [0.6, 0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "gamma": [0, 0.1, 0.2]
}

xgb_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=25,
    scoring="r2",
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

print("Fitting XGBoost…")
xgb_search.fit(X_train, y_train)

print("\nBest XGBoost Parameters:", xgb_search.best_params_)

# Get the best model
best_xgb = xgb_search.best_estimator_

# Save the XGBoost model
joblib.dump(best_xgb, "xgboost_model.pkl")


# ----------------------------------------------------------
# VALIDATION RESULTS
# ----------------------------------------------------------

val_pred_xgb = best_xgb.predict(X_val)

print("\n===== XGBoost Validation Performance =====")
print("MSE:", mean_squared_error(y_val, val_pred_xgb))
print("RMSE:", np.sqrt(mean_squared_error(y_val, val_pred_xgb)))
print("R²:", r2_score(y_val, val_pred_xgb))

# ----------------------------------------------------------
# TEST RESULTS
# ----------------------------------------------------------

test_pred_xgb = best_xgb.predict(X_test)

print("\n===== XGBoost TEST Performance =====")
print("MSE:", mean_squared_error(y_test, test_pred_xgb))
print("RMSE:", np.sqrt(mean_squared_error(y_test, test_pred_xgb)))
print("R²:", r2_score(y_test, test_pred_xgb))


# ----------------------------------------------------------
# 2) FEATURE IMPORTANCE (Random Forest + XGBoost)
# ----------------------------------------------------------

importances = best_rf.feature_importances_
xgb_importances = best_xgb.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "RF_Importance": importances,
    "XGB_Importance": xgb_importances
})

print("\n=== Feature Importance Comparison ===")
print(importance_df.sort_values("XGB_Importance", ascending=False))

# ----------------------------------------------------------
# FEATURE IMPORTANCE PLOT (XGBoost)
# ----------------------------------------------------------

plt.figure(figsize=(10, 8))
sorted_idx = np.argsort(xgb_importances)[::-1]
sns.barplot(x=xgb_importances[sorted_idx], y=X_train.columns[sorted_idx], palette="viridis")
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# FEATURE IMPORTANCE PLOT (Random Forest)
# ----------------------------------------------------------

plt.figure(figsize=(10, 8))
sorted_idx = np.argsort(importances)[::-1]
sns.barplot(x=importances[sorted_idx], y=X_train.columns[sorted_idx], palette="magma")
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

print(df.columns.tolist())