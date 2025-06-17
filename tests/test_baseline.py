import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import sys
import os

# --- Add src directory to Python path ---
# This assumes your tests directory is parallel to your src directory
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, src_path)

from baseline_trainer import (
    tune_and_train_model, scale_features, define_task_and_split, evaluate_predictions
)

# Mocking the tune_and_train_model function for simplicity in this test
# Alternatively, create sample data and call the actual function if feasible
class MockTrainedModel:
     def predict(self, X):
        # Return predictions with the same number of rows as X
        return np.random.randn(len(X))

@pytest.fixture
def sample_scaled_data():
    """Creates minimal sample scaled data."""
    X_train_scaled = pd.DataFrame(np.random.rand(100, 5), columns=[f'f{i}' for i in range(5)])
    y_train = pd.Series(np.random.rand(100) * 0.01)
    X_test_scaled = pd.DataFrame(np.random.rand(20, 5), columns=[f'f{i}' for i in range(5)])
    y_test = pd.Series(np.random.rand(20) * 0.01) # Added y_test fixture
    return X_train_scaled, y_train, X_test_scaled, y_test

@pytest.fixture
def sample_full_data():
    """Creates a small sample DataFrame mimicking processed features for end-to-end tests."""
    np.random.seed(42)
    n = 120
    df = pd.DataFrame({
        'symbol': np.random.choice(['A', 'B'], size=n),
        'log_return': np.random.randn(n) * 0.01,
        'sma_5': np.random.randn(n),
        'sma_20': np.random.randn(n),
        'sma_60': np.random.randn(n),
        'volatility_20': np.random.rand(n),
        'rsi_14': np.random.rand(n),
        'day_of_week': np.random.randint(0, 5, n),
        'month': np.random.randint(1, 13, n),
        'quarter': np.random.randint(1, 5, n),
        'log_return_lag_1': np.random.randn(n),
        'log_return_lag_2': np.random.randn(n),
        'log_return_lag_3': np.random.randn(n),
        'log_return_lag_5': np.random.randn(n),
        'volume_lag_1': np.random.rand(n) * 1000,
    }, index=pd.date_range('2020-01-01', periods=n, freq='B'))
    return df


def test_define_task_and_split(sample_full_data):
    """Test that define_task_and_split splits and aligns data correctly."""
    feature_cols = [
        'sma_5', 'sma_20', 'sma_60', 'volatility_20', 'rsi_14',
        'day_of_week', 'month', 'quarter', 'log_return_lag_1',
        'log_return_lag_2', 'log_return_lag_3', 'log_return_lag_5',
        'volume_lag_1'
    ]
    train_end = '2020-02-28'
    val_end = '2020-03-31'
    X_train, y_train, X_val, y_val, X_test, y_test, test_actual = define_task_and_split(
        sample_full_data, feature_cols, train_end, val_end
    )
    # Account for dropped last row per symbol due to shift(-1)
    n_symbols = sample_full_data['symbol'].nunique()
    expected = len(sample_full_data.dropna(subset=['log_return'])) - n_symbols
    actual = len(X_train) + len(X_val) + len(X_test)
    assert actual == expected


def test_scale_features(sample_full_data):
    """Test scale_features returns DataFrames of correct shape and type."""
    feature_cols = [
        'sma_5', 'sma_20', 'sma_60', 'volatility_20', 'rsi_14',
        'day_of_week', 'month', 'quarter', 'log_return_lag_1',
        'log_return_lag_2', 'log_return_lag_3', 'log_return_lag_5',
        'volume_lag_1'
    ]
    train_end = '2020-02-28'
    val_end = '2020-03-31'
    X_train, y_train, X_val, y_val, X_test, y_test, _ = define_task_and_split(
        sample_full_data, feature_cols, train_end, val_end
    )
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)
    assert isinstance(X_train_scaled, pd.DataFrame)
    assert X_train_scaled.shape == X_train.shape
    assert np.allclose(X_train_scaled.mean(), 0, atol=1)
    assert np.allclose(X_train_scaled.std(), 1, atol=1)


def test_tune_and_train_model_runs(sample_scaled_data):
    """Test that tune_and_train_model returns a fitted model and best param selection works."""
    X_train_scaled, y_train, X_test_scaled, y_test = sample_scaled_data
    params = {'alpha': [0.1, 1.0, 10.0]}
    model = tune_and_train_model(
        "Ridge", Ridge(), params, X_train_scaled, y_train, cv_splits=3
    )
    assert hasattr(model, 'predict')
    preds = model.predict(X_test_scaled)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_test_scaled.shape[0]


def test_evaluate_predictions_shape_and_metrics(sample_scaled_data):
    """Test evaluate_predictions returns correct shape and logs metrics."""
    X_train_scaled, y_train, X_test_scaled, y_test = sample_scaled_data
    model = Ridge(alpha=1.0).fit(X_train_scaled, y_train)
    pred_series = evaluate_predictions("Ridge", model, X_test_scaled, y_test)
    assert isinstance(pred_series, pd.Series)
    assert pred_series.shape[0] == X_test_scaled.shape[0]
    assert pred_series.name == "Ridge_pred"


def test_baseline_prediction_shape(sample_scaled_data):
    """Tests if the trained baseline model prediction has the correct shape."""
    X_train_scaled, y_train, X_test_scaled, y_test = sample_scaled_data

    # Simulate training a simple model (or use a mock)
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)

    # Make predictions
    predictions = model.predict(X_test_scaled)

    # Assert shape and type
    assert isinstance(predictions, np.ndarray), "Predictions should be a numpy array"
    assert predictions.shape == (len(X_test_scaled),), f"Prediction shape mismatch: expected ({len(X_test_scaled)},), got {predictions.shape}"
    assert predictions.dtype == np.float64 or predictions.dtype == np.float32, "Predictions should be float"

# Optionally, add tests for error handling, edge cases, and portfolio simulation if you want full pipeline coverage.