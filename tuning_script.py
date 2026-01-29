import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. Data Loading and Preparation
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Helper function for evaluation
results = {}

def evaluate_model(name, model, X_t, y_t):
    y_pred = model.predict(X_t)
    mse = mean_squared_error(y_t, y_pred)
    results[name] = mse
    print(f"{name:30} : MSE = {mse:.6f}")

# 2. Baseline Model
lr = LinearRegression()
lr.fit(X_train, y_train)
evaluate_model('Linear Regression', lr, X_test, y_test)

# 3. Ridge Tuning
print("\n--- Ridge Tuning ---")
ridge_alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
for a in ridge_alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train, y_train)
    evaluate_model(f'Ridge (alpha={a})', ridge, X_test, y_test)

# 4. Lasso Tuning
print("\n--- Lasso Tuning ---")
lasso_alphas = [0.0001, 0.001, 0.01, 0.05, 0.1]
for a in lasso_alphas:
    lasso = Lasso(alpha=a)
    lasso.fit(X_train, y_train)
    evaluate_model(f'Lasso (alpha={a})', lasso, X_test, y_test)

# 5. ElasticNet Tuning
print("\n--- ElasticNet Tuning ---")
en_params = [(0.1, 0.1), (0.1, 0.5), (0.5, 0.1), (0.01, 0.01)]
for a, l1 in en_params:
    en = make_pipeline(StandardScaler(), ElasticNet(alpha=a, l1_ratio=l1))
    en.fit(X_train, y_train)
    evaluate_model(f'ElasticNet (alpha={a}, l1={l1})', en, X_test, y_test)

# 6. Final Comparison
sorted_results = sorted(results.items(), key=lambda x: x[1])
print("\n--- Summary (Ranked by MSE) ---")
for i, (name, mse) in enumerate(sorted_results, 1):
    print(f"{i}. {name:30}: {mse:.6f}")

best_model_name, best_mse = sorted_results[0]
print(f"\nBest Model: {best_model_name} with MSE: {best_mse:.6f}")
