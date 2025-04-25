"""
model_feature_selection.py
Automatically selects optimal features for each model type based on model-specific importances.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

def select_top_features_rf(X_train, y_train, n=5):
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:n]
    return [X_train.columns[i] for i in top_idx]

def select_top_features_ridge(X_train, y_train, n=5):
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    coefs = np.abs(ridge.coef_)
    top_idx = np.argsort(coefs)[::-1][:n]
    return [X_train.columns[i] for i in top_idx]

def select_top_features_xgb(X_train, y_train, n=5):
    if not HAS_XGB:
        return []
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    importances = xgb_model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:n]
    return [X_train.columns[i] for i in top_idx]

# Main utility to get dict of features per model
def get_model_specific_features(X_train, y_train, n=5):
    features = {}
    features['ridge'] = select_top_features_ridge(X_train, y_train, n)
    features['random_forest'] = select_top_features_rf(X_train, y_train, n)
    if HAS_XGB:
        features['xgboost'] = select_top_features_xgb(X_train, y_train, n)
    else:
        features['xgboost'] = []
    return features
