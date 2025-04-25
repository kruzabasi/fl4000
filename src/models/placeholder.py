# FILE: src/models/placeholder.py

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.exceptions import NotFittedError
import logging
from typing import List, Tuple, Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PlaceholderLinearModel:
    """
    A wrapper around SGDRegressor for use as a placeholder model in FL simulation.
    Supports getting/setting parameters and partial fitting for epochs.
    """
    def __init__(self, n_features: int, random_state: Optional[int] = None):
        # Initialize with warm_start=True to allow partial_fit over epochs
        self._model = SGDRegressor(
            loss="squared_error",
            penalty=None, # No regularization handled by model itself yet
            fit_intercept=True,
            max_iter=1, # Only one pass per partial_fit call
            tol=None, # Let FL control convergence
            shuffle=True,
            random_state=random_state,
            warm_start=True
        )
        self.n_features = n_features
        self._is_fitted = False # Track if model has been fitted at all

    def get_parameters(self) -> List[np.ndarray]:
        """Returns the model parameters (coefficients and intercept)."""
        if not self._is_fitted:
             # Return zero arrays with correct shapes if not fitted
             # coef_ shape: (1, n_features) for regression, but SGDRegressor stores as (n_features,)
             # intercept_ shape: (1,)
             return [np.zeros(self.n_features), np.zeros(1)]

        coef = self._model.coef_.copy().flatten() # Ensure 1D array
        intercept = self._model.intercept_.copy()
        return [coef, intercept]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Sets the model parameters."""
        if len(parameters) == 2:
            coef = parameters[0].flatten() # Ensure 1D
            intercept = parameters[1]

            if coef.shape != (self.n_features,):
                 raise ValueError(f"Coefficient shape mismatch: Expected ({self.n_features},), got {coef.shape}")
            if intercept.shape != (1,):
                 # Allow scalar intercept to be set
                 if np.isscalar(intercept) or intercept.shape == ():
                     intercept = np.array([intercept])
                 elif intercept.shape != (1,):
                     raise ValueError(f"Intercept shape mismatch: Expected (1,), got {intercept.shape}")


            self._model.coef_ = coef
            self._model.intercept_ = intercept
            self._is_fitted = True # Mark as 'fitted' once parameters are set
        else:
            raise ValueError(f"Expected 2 parameter arrays (coef, intercept), got {len(parameters)}")


    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Performs one epoch (pass) of training using partial_fit."""
        # Need to initialize coef_/intercept_ if not already done
        if not hasattr(self._model, 'coef_') or not hasattr(self._model, 'intercept_'):
            # Initialize with zeros or small random values if desired
            self._model.coef_ = np.zeros(self.n_features)
            self._model.intercept_ = np.zeros(1)

        self._model.partial_fit(X, y)
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Makes predictions."""
        if not self._is_fitted:
             logging.warning("Model predicting based on initial (zero) parameters as it hasn't been fitted or parameters set.")
             # Return zeros or handle as appropriate
             return np.zeros(X.shape[0])
        return self._model.predict(X)

    def calculate_loss(self, X: np.ndarray, y: np.ndarray) -> float:
         """Calculates MSE loss for given data."""
         if not self._is_fitted:
             return np.mean(y**2) # Loss if predicting zeros
         y_pred = self.predict(X)
         loss = np.mean((y - y_pred)**2)
         return float(loss) # Ensure float type
