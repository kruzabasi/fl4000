import pandas as pd
import numpy as np
import os
import logging
import copy
from typing import List, Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split # Or just use slices
from sklearn.preprocessing import StandardScaler # Assuming scaler is passed or fitted locally
from sklearn.utils import shuffle

# Import the placeholder model (adjust path if needed)
try:
    from models.placeholder import PlaceholderLinearModel
except ImportError:
     logging.error("Could not import PlaceholderLinearModel.")
     # Define a dummy class if import fails, for structure
     class PlaceholderLinearModel:
         def get_parameters(self): return [np.array([0.0]), np.array([0.0])]
         def set_parameters(self, p): pass
         def partial_fit(self, X, y): pass
         def predict(self, X): return np.zeros(len(X))
         def calculate_loss(self, X, y): return 0.0

# Placeholder type for model parameters
ModelParams = List[np.ndarray]


def get_feature_columns(df):
    """
    Returns the list of feature columns to use for FL model training.
    Excludes metadata, target, and non-numeric columns.
    """
    exclude_cols = ['symbol', 'log_return', 'timestamp', 'gics_sector']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    numeric_feature_cols = df[feature_cols].select_dtypes(include=np.number).columns.tolist()
    return numeric_feature_cols


class Client:
    """
    Represents a single client in the Federated Learning simulation.
    Handles local data loading, local training with FedProx logic, and update computation.

    Args:
        client_id (str): Unique identifier for the client.
        model_template (PlaceholderLinearModel): An instance of the model to clone for local use.
        data_path (str): Path to the client's data file (e.g., .parquet).
        random_seed (Optional[int]): Optional random seed for reproducibility.
    """
    def __init__(self, client_id: str, model_template: PlaceholderLinearModel,
                 data_path: str, random_seed: Optional[int] = None):
        """
        Initializes the Client and loads its local data.
        Args:
            client_id (str): Unique identifier for the client.
            model_template (PlaceholderLinearModel): An instance of the model to clone for local use.
            data_path (str): Path to the client's data file (e.g., .parquet).
            random_seed (Optional[int]): Optional random seed for reproducibility.
        """
        self.client_id = client_id
        self.local_model = copy.deepcopy(model_template) # Start with a copy of the initial structure
        self.data_path = data_path
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        # Optional: Load validation data if needed for local monitoring
        # self.X_val: Optional[pd.DataFrame] = None
        # self.y_val: Optional[pd.Series] = None
        self.num_samples: int = 0
        self.random_seed = random_seed
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        self.load_data()

    def load_data(self) -> None:
        """
        Loads and prepares data from the specified path.
        Returns:
            None
        """
        logging.debug(f"Client {self.client_id}: Loading data from {self.data_path}...")
        try:
            df = pd.read_parquet(self.data_path)
            # Use consistent feature selection
            target_col = 'log_return'
            numeric_feature_cols = get_feature_columns(df)
            self.X_train = df[numeric_feature_cols]
            self.y_train = df[target_col]
            self.num_samples = len(df)
            
            # Feature scaling (standardization) for local training
            # Fit scaler only on local data to avoid data leakage
            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            # Standardize target as well
            self.y_mean = self.y_train.mean()
            self.y_std = self.y_train.std() if self.y_train.std() > 0 else 1.0
            self.y_train = (self.y_train - self.y_mean) / self.y_std
            
            logging.info(f"Client {self.client_id}: Loaded {self.num_samples} samples with {len(numeric_feature_cols)} features from {self.data_path}")

        except FileNotFoundError:
            logging.error(f"Client {self.client_id}: Data file not found at {self.data_path}")
            self.num_samples = 0
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error loading data: {e}")
            self.num_samples = 0

    def get_data_loader(self, batch_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Creates a simple batch generator for local training.
        Args:
            batch_size (int): Number of samples per batch.
        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: List of (X_batch, y_batch) tuples.
        """
        if self.X_train is None or self.y_train is None:
             return []

        # Convert to numpy for SGDRegressor compatibility if not already
        X = self.X_train if isinstance(self.X_train, np.ndarray) else self.X_train.to_numpy()
        y = self.y_train.to_numpy() if isinstance(self.y_train, pd.Series) else self.y_train

        # Shuffle data each time loader is called
        X_shuffled, y_shuffled = shuffle(X, y, random_state=self.random_seed)

        data_loader = []
        for i in range(0, self.num_samples, batch_size):
            batch_X = X_shuffled[i : i + batch_size]
            batch_y = y_shuffled[i : i + batch_size]
            data_loader.append((batch_X, batch_y))
        return data_loader

    def set_parameters(self, parameters: ModelParams) -> None:
        """
        Sets the local model parameters from the global model.
        Args:
            parameters (ModelParams): Parameters to set in the local model.
        Returns:
            None
        """
        try:
            self.local_model.set_parameters(parameters)
        except Exception as e:
             logging.error(f"Client {self.client_id}: Error setting parameters - {e}")
             # Fallback: re-initialize model?
             # self.local_model = copy.deepcopy(self.initial_model_template) # Need template stored


    def train(self, global_parameters_t: ModelParams, config: Dict[str, Any]) -> Dict[str, float]:
        """
        Performs local training for E epochs using FedProx logic.
        Uses a manual proximal term update for scikit-learn SGDRegressor.
        Args:
            global_parameters_t (ModelParams): Global model parameters at the start of the round (w^t).
            config (Dict[str, Any]): Dictionary with training hyperparameters:
                'local_epochs', 'batch_size', 'learning_rate', 'mu_prox'.
        Returns:
            Dict[str, float]: Dictionary containing training metrics (e.g., final loss).
        """
        local_epochs = config.get('local_epochs', 1)
        batch_size = config.get('batch_size', 32)
        learning_rate = config.get('learning_rate', 0.01)
        mu_prox = config.get('mu_prox', 0.01) # FedProx mu parameter

        if self.X_train is None or self.y_train is None or self.num_samples == 0:
             logging.warning(f"Client {self.client_id}: No training data available. Skipping training.")
             return {'loss': float('inf')}


        # Use the current global parameters as w^t for prox term calculation
        w_t = [param.copy() for param in global_parameters_t]

        # Set local model to w^t before starting training
        self.set_parameters(w_t)

        logging.debug(f"Client {self.client_id}: Starting local training for {local_epochs} epochs.")

        metrics = {}
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            data_loader = self.get_data_loader(batch_size)
            num_batches = len(data_loader)

            if num_batches == 0: continue # Skip epoch if no data

            for batch_X, batch_y in data_loader:
                 # --- FedProx Modification for SGDRegressor ---
                 # Since SGDRegressor's partial_fit uses its internal optimizer,
                 # we simulate the FedProx update manually.
                 # 1. Get current local parameters (w)
                 w = self.local_model.get_parameters()

                 # 2. Calculate gradient of standard loss (MSE for SGDRegressor)
                 # SGDRegressor doesn't expose gradient easily. We approximate or recalculate.
                 # Simple approximation: Use the update SGDRegressor *would* make internally?
                 # More direct: Calculate gradient manually: grad_L = (predict(batch_X) - batch_y) @ batch_X / len(batch_y)
                 y_pred = self.local_model.predict(batch_X)
                 error = y_pred - batch_y
                 grad_L_coef = (error @ batch_X) / len(batch_y) # Gradient for coefficients
                 grad_L_intercept = np.mean(error) # Gradient for intercept

                 # 3. Calculate gradient of proximal term: mu * (w - w^t)
                 grad_prox_coef = mu_prox * (w[0] - w_t[0])
                 grad_prox_intercept = mu_prox * (w[1] - w_t[1])

                 # 4. Combine gradients
                 combined_grad_coef = grad_L_coef + grad_prox_coef
                 combined_grad_intercept = grad_L_intercept + grad_prox_intercept[0] # Intercept grad is scalar

                 # 5. Apply manual SGD step to update local model parameters
                 new_coef = w[0] - learning_rate * combined_grad_coef
                 new_intercept = w[1] - learning_rate * combined_grad_intercept

                 # Gradient clipping for stability
                 clip_value = 5.0
                 new_coef = np.clip(new_coef, -clip_value, clip_value)
                 new_intercept = np.clip(new_intercept, -clip_value, clip_value)
                 self.local_model._model.coef_ = new_coef
                 if hasattr(self.local_model._model, 'intercept_'):
                    # Ensure intercept is a float or 1D array (for univariate regression)
                    self.local_model._model.intercept_ = np.array([float(new_intercept)])
                 self.local_model._is_fitted = True # Ensure model knows it's 'fitted'

                 # Track loss (using standard MSE before manual update for consistency)
                 # batch_loss = np.mean(error**2) # Use loss before update
                 # Or calculate loss after update:
                 batch_loss = self.local_model.calculate_loss(batch_X, batch_y)
                 epoch_loss += batch_loss

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            logging.debug(f"Client {self.client_id}: Epoch {epoch+1}/{local_epochs} - Avg Batch Loss: {avg_epoch_loss:.4f}")
            metrics[f'epoch_{epoch+1}_loss'] = avg_epoch_loss

        final_loss = self.local_model.calculate_loss(self.X_train, self.y_train) # Overall loss on train set
        # After loss computation, check for NaN/Inf
        if not np.isfinite(final_loss):
            logging.error(f"Client {self.client_id}: Non-finite loss encountered (NaN or Inf). Stopping training.")
            return metrics
        metrics['final_loss'] = final_loss
        logging.debug(f"Client {self.client_id}: Training complete. Final loss on local train data: {final_loss:.4f}")
        return metrics


    def _calculate_l2_norm(self, params: List[np.ndarray]) -> float:
        """Calculates the total L2 norm across a list of parameter arrays."""
        squared_norms = [np.sum(np.square(p)) for p in params]
        return np.sqrt(np.sum(squared_norms))

    def get_update(self, clip_norm: Optional[float]) -> Tuple[Optional[ModelParams], int]:
        """
        Calculates the parameter update (local_params - initial_round_params),
        applies L2 norm clipping if clip_norm is specified, and returns the
        potentially clipped update and the number of samples.

        Differential Privacy:
        - If clip_norm is set, the L2 norm of the update is clipped to clip_norm.
        - This ensures that each client's contribution is bounded.
        - The clipped update is then sent to the aggregator, where noise may be added for DP.

        Args:
            clip_norm (Optional[float]): The L2 norm clipping threshold (C_clip). If None, no clipping is applied.

        Returns:
            Tuple containing the model update (list of numpy arrays) and
            the number of samples used for training. Returns (None, 0) if no training occurred.
        """
        if self.num_samples == 0:
            logging.warning(f"Client {self.client_id}: Cannot get update, no samples trained.")
            return None, 0

        local_params = self.local_model.get_parameters()
        global_parameters_t = getattr(self, 'initial_round_params', None)
        if global_parameters_t is None:
            logging.error(f"Client {self.client_id}: No initial global parameters stored. Cannot calculate update.")
            return None, 0
        if len(local_params) != len(global_parameters_t):
            logging.error(f"Client {self.client_id}: Parameter list length mismatch. Cannot calculate update.")
            return None, 0

        # Calculate raw update
        raw_update = []
        try:
            for local, global_t in zip(local_params, global_parameters_t):
                if local.shape != global_t.shape:
                    raise ValueError(f"Layer shape mismatch: Local {local.shape} vs Global {global_t.shape}")
                raw_update.append(local - global_t)
        except ValueError as e:
            logging.error(f"Client {self.client_id}: Error calculating raw update - {e}")
            return None, 0

        # Apply clipping if C_clip is provided
        if clip_norm is not None and clip_norm > 0:
            l2_norm = self._calculate_l2_norm(raw_update)
            if l2_norm > 0: # Avoid division by zero
                scale = min(1.0, clip_norm / l2_norm)
                clipped_update = [layer * scale for layer in raw_update]
                logging.debug(f"Client {self.client_id}: Clipped update norm from {l2_norm:.4f} to {clip_norm:.4f} (Scale: {scale:.4f})")
                return clipped_update, self.num_samples
            else:
                # Norm is zero, update is zero, return as is
                logging.debug(f"Client {self.client_id}: Update norm is zero, no clipping needed.")
                return raw_update, self.num_samples
        else:
            # No clipping applied
            return raw_update, self.num_samples