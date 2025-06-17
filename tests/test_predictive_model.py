import numpy as np
import pytest
from src.models.predictive_model import PortfolioPredictiveModel, mse_loss


def test_portfolio_predictive_model_fit_and_predict():
    n_samples = 20
    n_features = 5
    n_outputs = 3
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples, n_outputs)

    model = PortfolioPredictiveModel(n_features=n_features, n_outputs=n_outputs)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (n_samples, n_outputs)
    # Should not be all zeros after fitting
    assert not np.allclose(preds, 0)


def test_portfolio_predictive_model_get_set_parameters():
    n_samples = 10
    n_features = 4
    n_outputs = 2
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples, n_outputs)
    model = PortfolioPredictiveModel(n_features=n_features, n_outputs=n_outputs)
    model.fit(X, y)
    params = model.get_parameters()
    # Set parameters to zeros and check prediction is all zeros
    zero_params = [np.zeros((n_outputs, n_features)), np.zeros(n_outputs)]
    model.set_parameters(zero_params)
    preds = model.predict(X)
    assert np.allclose(preds, 0)
    # Restore parameters and check predictions change
    model.set_parameters(params)
    preds2 = model.predict(X)
    assert not np.allclose(preds2, 0)


def test_portfolio_predictive_model_loss():
    n_samples = 15
    n_features = 3
    n_outputs = 2
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples, n_outputs)
    model = PortfolioPredictiveModel(n_features=n_features, n_outputs=n_outputs)
    # Before fitting, should return MSE to zero prediction
    loss_unfit = model.calculate_loss(X, y)
    assert np.isclose(loss_unfit, np.mean(np.square(y)))
    model.fit(X, y)
    y_pred = model.predict(X)
    loss_fit = model.calculate_loss(X, y)
    # After fitting, loss should be less than or equal to unfit loss (not always strictly less)
    assert loss_fit <= loss_unfit
    # mse_loss function should match model loss calculation
    assert np.isclose(mse_loss(y, y_pred), np.mean(np.square(y - y_pred)))


def test_portfolio_predictive_model_shapes():
    n_samples = 8
    n_features = 6
    n_outputs = 4
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples, n_outputs)
    model = PortfolioPredictiveModel(n_features=n_features, n_outputs=n_outputs)
    model.fit(X, y)
    params = model.get_parameters()
    assert params[0].shape == (n_outputs, n_features)
    assert params[1].shape == (n_outputs,)
