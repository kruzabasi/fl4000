import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.exceptions import NotFittedError
from sklearn.multioutput import MultiOutputRegressor
import logging
from typing import List, Tuple, Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PlaceholderLinearModel:
    """
    A wrapper around SGDRegressor for use as a placeholder model in FL simulation.
    Supports multi-output regression if n_outputs > 1.
    Supports getting/setting parameters and partial fitting for epochs.

    Args:
        n_features (int): Number of features for the model.
        n_outputs (int): Number of outputs/assets for the model.
        random_state (Optional[int]): Random seed for reproducibility.
    """
    def __init__(self, n_features: int, n_outputs: int = 1, random_state: Optional[int] = None):
        """
        Initializes the PlaceholderLinearModel.

        Args:
            n_features (int): Number of features for the model.
            n_outputs (int): Number of outputs/assets for the model.
            random_state (Optional[int]): Random seed for reproducibility.
        """
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.random_state = random_state
        # If n_outputs > 1, use MultiOutputRegressor
        if n_outputs > 1:
            self._model = MultiOutputRegressor(SGDRegressor(
                loss="squared_error",
                penalty=None,
                fit_intercept=True,
                max_iter=1,
                tol=None,
                shuffle=True,
                random_state=random_state,
                warm_start=True
            ))
        else:
            self._model = SGDRegressor(
                loss="squared_error",
                penalty=None,
                fit_intercept=True,
                max_iter=1,
                tol=None,
                shuffle=True,
                random_state=random_state,
                warm_start=True
            )
        self._is_fitted = False # Track if model has been fitted at all

    def get_parameters(self) -> List[np.ndarray]:
        """Returns the model parameters (coefficients and intercepts)."""
        if hasattr(self._model, 'estimators_') and getattr(self._model, 'estimators_', None):
            # MultiOutputRegressor and already fitted
            coefs = np.stack([est.coef_ for est in self._model.estimators_])
            intercepts = np.stack([est.intercept_ for est in self._model.estimators_])
            return [coefs, intercepts]
        elif hasattr(self._model, 'coef_') and hasattr(self._model, 'intercept_'):
            # Single-output and already fitted
            return [self._model.coef_, self._model.intercept_]
        else:
            # Not fitted yet; return zeros with correct shapes
            if self.n_outputs > 1:
                return [np.zeros((self.n_outputs, self.n_features)), np.zeros(self.n_outputs)]
            else:
                return [np.zeros(self.n_features), np.zeros(1)]

    def set_parameters(self, params: List[np.ndarray]) -> None:
        """
        Sets the model parameters.

        Args:
            params (List[np.ndarray]): [coef (n_outputs, n_features) or (n_features,) if single-output, intercept (n_outputs,) or (1,) if single-output]

        Returns:
            None
        """
        if hasattr(self._model, 'estimators_'):
            # MultiOutputRegressor
            for est, coef, intercept in zip(self._model.estimators_, params[0], params[1]):
                est.coef_ = coef
                est.intercept_ = intercept
        else:
            self._model.coef_ = params[0]
            self._model.intercept_ = params[1]
            self._is_fitted = True # Mark as 'fitted' once parameters are set

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Performs one epoch (pass) of training using partial_fit.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training targets.

        Returns:
            None
        """
        # Need to initialize coef_/intercept_ if not already done
        if not hasattr(self._model, 'coef_') or not hasattr(self._model, 'intercept_'):
            # Initialize with zeros or small random values if desired
            if hasattr(self._model, 'estimators_'):
                for est in self._model.estimators_:
                    est.coef_ = np.zeros(self.n_features)
                    est.intercept_ = np.zeros(1)
            else:
                self._model.coef_ = np.zeros(self.n_features)
                self._model.intercept_ = np.zeros(1)

        self._model.partial_fit(X, y)
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the model.

        Args:
            X (np.ndarray): Features to predict on.

        Returns:
            np.ndarray: Predicted values.
        """
        if not self._is_fitted:
             logging.warning("Model predicting based on initial (zero) parameters as it hasn't been fitted or parameters set.")
             # Return zeros or handle as appropriate
             return np.zeros(X.shape[0])
        return self._model.predict(X)

    def calculate_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates MSE loss for given data.

        Args:
            X (np.ndarray): Features.
            y (np.ndarray): True targets.

        Returns:
            float: Mean squared error loss.
        """
        if not self._is_fitted:
             return np.mean(y**2) # Loss if predicting zeros
        y_pred = self.predict(X)
        loss = np.mean((y - y_pred)**2)
        return float(loss) # Ensure float type