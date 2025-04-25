import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler # Assuming scaling happens outside
import logging
import pickle
from typing import List, Tuple, Optional, Dict, Any

# Placeholder type for model parameters
ModelParams = List[np.ndarray] # For Ridge: [coef_array, intercept_array]

class PortfolioPredictiveModel:
    """
    Wrapper for a multi-output regression model (e.g., Ridge) to predict
    expected returns (mu proxy) for multiple assets.
    Compatible with the FL client structure.
    """
    def __init__(self, n_features: int, n_outputs: int, model_params: Optional[Dict] = None):
        """
        Initializes the predictive model.

        Args:
            n_features (int): Number of input features.
            n_outputs (int): Number of target assets to predict returns for.
            model_params (Optional[Dict]): Parameters for the underlying estimator (e.g., {'alpha': 1.0} for Ridge).
        """
        if model_params is None:
            model_params = {'alpha': 1.0} # Default Ridge alpha

        # Use MultiOutputRegressor wrapping Ridge
        self.base_estimator = Ridge(**model_params)
        self._model = MultiOutputRegressor(self.base_estimator)
        self.n_features = n_features
        self.n_outputs = n_outputs
        self._is_fitted = False

    def get_parameters(self) -> ModelParams:
        """Returns the model parameters (coefficients and intercepts for each output)."""
        if not hasattr(self._model, 'estimators_') or not self._model.estimators_:
            # Return zero arrays with correct shapes if not fitted
            # Coef shape: (n_outputs, n_features), Intercept shape: (n_outputs,)
            return [np.zeros((self.n_outputs, self.n_features)), np.zeros(self.n_outputs)]

        all_coefs = np.array([est.coef_ for est in self._model.estimators_]) # Shape: (n_outputs, n_features)
        all_intercepts = np.array([est.intercept_ for est in self._model.estimators_]) # Shape: (n_outputs,)
        return [all_coefs, all_intercepts]

    def set_parameters(self, parameters: ModelParams) -> None:
        """Sets the model parameters."""
        if len(parameters) != 2:
            raise ValueError("Expected 2 parameter arrays (coefficients, intercepts).")

        coefs = parameters[0] # Shape: (n_outputs, n_features)
        intercepts = parameters[1] # Shape: (n_outputs,)

        if coefs.shape != (self.n_outputs, self.n_features):
            raise ValueError(f"Coefficient shape mismatch: Expected {(self.n_outputs, self.n_features)}, got {coefs.shape}")
        if intercepts.shape != (self.n_outputs,):
             raise ValueError(f"Intercept shape mismatch: Expected {(self.n_outputs,)}, got {intercepts.shape}")

        # Ensure estimators list exists and has the right length
        if not hasattr(self._model, 'estimators_') or len(self._model.estimators_) != self.n_outputs:
             # Need to fit briefly to create estimators if setting params before first fit
             # This is a workaround for sklearn's MultiOutputRegressor structure
             logging.warning("Model not fitted yet. Fitting with dummy data to initialize estimators before setting parameters.")
             dummy_X = np.zeros((1, self.n_features))
             dummy_y = np.zeros((1, self.n_outputs))
             try:
                 self._model.fit(dummy_X, dummy_y)
             except Exception as e:
                 logging.error(f"Dummy fit failed: {e}")
                 raise

        # Set parameters for each internal estimator
        for i, estimator in enumerate(self._model.estimators_):
            estimator.coef_ = coefs[i, :]
            estimator.intercept_ = intercepts[i]

        self._is_fitted = True
        logging.debug("Model parameters set.")


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the model to the entire dataset X, y."""
        self._model.fit(X, y)
        self._is_fitted = True

    # Note: partial_fit is not directly supported by MultiOutputRegressor wrapping Ridge.
    # If SGDRegressor is used inside MultiOutput, partial_fit might work differently.
    # For FL simulation with epochs, the Client.train logic needs adaptation (see Task 2).

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Makes predictions (expected returns for n_outputs assets)."""
        if not self._is_fitted:
            logging.warning("Model predicting based on initial parameters as it hasn't been fitted.")
            # Need initial state or return zeros
            return np.zeros((X.shape[0], self.n_outputs))
        return self._model.predict(X) # Returns shape (n_samples, n_outputs)

    def calculate_loss(self, X: np.ndarray, y: np.ndarray) -> float:
         """Calculates average MSE loss across all outputs."""
         if not self._is_fitted:
             return np.mean(np.square(y)) # Loss if predicting zeros (average across samples and outputs)

         y_pred = self.predict(X)
         loss = np.mean(np.square(y - y_pred)) # Mean across samples and outputs
         return float(loss)

# --- Loss Function Definition (MSE) ---
def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Standard Mean Squared Error calculation."""
    return float(np.mean(np.square(y_true - y_pred)))

# Handling Σ: This model predicts μ (next-day returns). The methodology requires continuing to use empirical rolling covariance for Σ during the portfolio optimization step. This is will be handled in a subsequent task.
