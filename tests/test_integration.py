import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

# Adjust imports based on actual structure
from src.models.predictive_model import PortfolioPredictiveModel
from src.portfolio_optimizer import optimize_portfolio
from src.federated.client import Client # To test the client method

# --- Fixtures ---
@pytest.fixture
def sample_mu_sigma():
    symbols = ['AAPL', 'MSFT', 'GOOG']
    mu = pd.Series([0.10, 0.12, 0.11], index=symbols)
    data = {'AAPL': [0.01, 0.005, -0.01],
            'MSFT': [0.005, 0.012, -0.008],
            'GOOG': [-0.01, -0.008, 0.015]}
    returns = pd.DataFrame(data)
    # Use pypfopt if available for robust cov calculation example
    try:
        from pypfopt import risk_models
        Sigma = risk_models.sample_cov(returns, returns_data=True, frequency=1) # Frequency 1 for simple test case
    except ImportError:
        Sigma = returns.cov() # Fallback
    return mu, Sigma, symbols

@pytest.fixture
def sample_features_and_returns(sample_mu_sigma):
     mu, Sigma, symbols = sample_mu_sigma
     n_features = 5
     n_samples = 10
     # Sample features (index doesn't matter much here)
     X = pd.DataFrame(np.random.rand(n_samples, n_features), columns=[f'f{i}' for i in range(n_features)])
     # Sample historical returns DF for cov calc
     hist_dates = pd.date_range(end='2023-01-10', periods=60, freq='D') # Enough for lookback
     hist_returns = pd.DataFrame(np.random.randn(60, len(symbols)) * 0.01, index=hist_dates, columns=symbols)
     return X, hist_returns, symbols, n_features

# --- Tests ---
def test_model_predict_to_mu(sample_features_and_returns):
    """Test if predictive model outputs mu proxy in expected format."""
    X, _, symbols, n_features = sample_features_and_returns
    n_outputs = len(symbols)
    model = PortfolioPredictiveModel(n_features=n_features, n_outputs=n_outputs)
    # Need to fit model slightly to initialize internal estimators if using MultiOutputRegressor
    model.fit(X.iloc[:1].to_numpy(), np.random.rand(1, n_outputs))

    # Predict using last row of features
    predictions = model.predict(X.iloc[-1:].to_numpy()) # Shape (1, n_outputs)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (1, n_outputs)

def test_mu_sigma_to_mvo(sample_mu_sigma):
    """Test if optimizer takes mu/Sigma and returns valid weights."""
    mu, Sigma, symbols = sample_mu_sigma
    if optimize_portfolio is None: # Check if imported
        pytest.skip("optimize_portfolio function not available (likely PyPortfolioOpt missing)")

    config = {'risk_free_rate': 0.01, 'weight_bounds': (0, 1)}
    weights_dict = optimize_portfolio(mu, Sigma, config)

    assert weights_dict is not None, "Optimization failed to return weights."
    assert isinstance(weights_dict, dict)
    assert set(weights_dict.keys()) == set(symbols) # Check symbols match
    weights = np.array(list(weights_dict.values()))
    assert np.isclose(weights.sum(), 1.0, atol=1e-4), "Weights do not sum to 1."
    assert np.all(weights >= -1e-6) and np.all(weights <= 1.0 + 1e-6), "Weights outside bounds (0, 1)." # Allow tolerance

# Mock Client for testing the workflow method
class MockPredictiveModel:
    def __init__(self, n_features, n_outputs, model_params=None):
        self.n_outputs = n_outputs
    def get_parameters(self): return [np.random.rand(self.n_outputs, 5), np.random.rand(self.n_outputs)]
    def set_parameters(self, p): pass
    def predict(self, X): return np.random.rand(X.shape[0], self.n_outputs) * 0.001 # Return plausible returns

@patch('src.federated.client.optimize_portfolio') # Mock the optimizer call at Client import location
@patch('src.federated.client.PortfolioPredictiveModel', MockPredictiveModel) # Use mock model
@patch('src.federated.client.pd.read_parquet') # Mock data loading
def test_client_optimization_workflow(mock_read_parquet, mock_optimize, sample_features_and_returns):
    """Test the end-to-end client method call."""
    X, hist_returns, symbols, n_features = sample_features_and_returns
    n_outputs = len(symbols)

    # Setup mock return values
    mock_read_parquet.return_value = pd.DataFrame({'symbol': ['AAPL']*10, 'log_return': [0.0]*10, 'target_log_return':[0.0]*10, **{f'f{i}':[0.0]*10 for i in range(n_features)}}) # Mock loaded data structure
    mock_optimize.return_value = {'AAPL': 0.5, 'MSFT': 0.3, 'GOOG': 0.2} # Expected optimizer output

    # Instantiate client (will use MockPredictiveModel)
    # Provide necessary args, n_features, n_outputs now derived from data
    client = Client(client_id='test_client', n_features=n_features, n_outputs=n_outputs, data_path='dummy/path')
    client.symbols = symbols # Manually set symbols as load_data is mocked/simplified

    # Call the workflow method
    config = {'risk_free_rate': 0.01, 'cov_lookback': 60}
    # Pass only the last row of features for prediction, and historical returns for cov
    optimal_weights = client.run_local_optimization(X.iloc[-1:], hist_returns, config)

    # Assertions
    mock_optimize.assert_called_once() # Check if optimizer was called
    args, kwargs = mock_optimize.call_args
    assert isinstance(args[0], pd.Series) # mu
    assert isinstance(args[1], pd.DataFrame) # Sigma
    assert args[0].shape == (n_outputs,)
    assert args[1].shape == (n_outputs, n_outputs)
    assert optimal_weights == mock_optimize.return_value # Check if output matches mock
