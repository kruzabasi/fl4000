import pandas as pd
import numpy as np
import logging
import copy
from typing import List, Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from models.predictive_model import PortfolioPredictiveModel

# Placeholder type for model parameters
ModelParams = List[np.ndarray]

try:
    from portfolio_optimizer import optimize_portfolio, HAS_PYPFOpt
    from pypfopt import risk_models
except ImportError:
    logging.error("Could not import optimize_portfolio function or pypfopt risk_models.")
    optimize_portfolio = None
    HAS_PYPFOpt = False

# --- Canonical FTSE 100 feature list (must match everywhere) ---
CANONICAL_FEATURES = [
    'adjusted_close', 'close', 'dividend_amount', 'high', 'low', 'open', 'split_coefficient',
    'volume', 'sma_5', 'sma_20', 'sma_60', 'volatility_20', 'day_of_week', 'month', 'quarter',
    'log_return_lag_1', 'log_return_lag_2', 'log_return_lag_3', 'log_return_lag_5', 'log_return_lag_10',
    'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5', 'volume_lag_10',
    'adjusted_close_lag_1', 'adjusted_close_lag_2', 'adjusted_close_lag_3', 'adjusted_close_lag_5',
    'adjusted_close_lag_10', 'rsi', 'macd', 'macd_signal', 'macd_diff', 'obv'
]

class Client:
    """Represents a single client in the Federated Learning simulation."""

    def __init__(self, client_id: str, n_features: int, n_outputs: int,
                 data_path: str, random_seed: Optional[int] = None):
        """
        Initializes the Client.

        Args:
            client_id: Unique identifier for the client.
            n_features: Number of input features.
            n_outputs: Number of target assets to predict returns for.
            data_path: Path to the client's data file (e.g., .parquet).
            random_seed: Optional random seed for reproducibility.
        """
        self.client_id = client_id
        self.local_model = PortfolioPredictiveModel(n_features=n_features, n_outputs=n_outputs, model_params={'alpha': 1.0})
        self.data_path = data_path
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.symbols: List[str] = []
        self.num_samples: int = 0
        self.random_seed = random_seed
        self.scaler = None
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self.initial_round_params: Optional[ModelParams] = None
        self.load_data()
        # --- NEW: Ensure model is initialized on real data for parameter transfer robustness ---
        if self.X_train is not None and self.y_train is not None and self.num_samples > 0:
            try:
                # Fit on a small batch (or all data if small) to initialize MultiOutputRegressor estimators
                batch_size = min(10, self.X_train.shape[0])
                self.local_model.fit(self.X_train[:batch_size], self.y_train[:batch_size])
                logging.info(f"Client {self.client_id}: Local model initialized on real data batch for parameter transfer consistency.")
            except Exception as e:
                logging.error(f"Client {self.client_id}: Error during initial model fit for parameter transfer: {e}")

    def load_data(self) -> None:
        """Loads data, prepares features (X) and multi-output target (y)."""
        logging.debug(f"Client {self.client_id}: Loading data from {self.data_path}...")
        try:
            df = pd.read_parquet(self.data_path)
            # Deduplicate on ['symbol', 'timestamp'] if possible
            if 'symbol' in df.columns and 'timestamp' in df.columns:
                before = len(df)
                df = df.drop_duplicates(subset=['symbol', 'timestamp'])
                after = len(df)
                n_dupes = before - after
                if n_dupes > 0:
                    logging.warning(f"Client {self.client_id}: Dropped {n_dupes} duplicate rows based on ['symbol', 'timestamp'].")
                # Set MultiIndex for uniqueness
                df = df.set_index(['symbol', 'timestamp'])
                df = df.sort_index()
            else:
                df = df.sort_index() # Ensure time order

            # Diagnostic logging for index uniqueness
            if not df.index.is_unique:
                logging.error(f"Client {self.client_id}: Duplicate index values before processing: {df.index[df.index.duplicated()].unique()}")
                logging.debug(f"Client {self.client_id}: Head of duplicated index rows:\n{df.loc[df.index.duplicated()].head()}")
                df = df.reset_index(drop=True)
                logging.warning(f"Client {self.client_id}: Index reset to remove duplicates.")
            else:
                logging.debug(f"Client {self.client_id}: Index is unique before processing.")

            # Target: T+1 log return, pivoted to be multi-output (n_samples, n_assets)
            target_col = 'target_log_return'
            # Use index for symbol and timestamp
            if isinstance(df.index, pd.MultiIndex):
                df[target_col] = df.groupby(level='symbol')['log_return'].shift(-1)
                df_pivot_target = df.reset_index().pivot(columns='symbol', values=target_col)
            else:
                df[target_col] = df.groupby('symbol')['log_return'].shift(-1)
                df_pivot_target = df.pivot(columns='symbol', values=target_col)

            # --- Canonical feature enforcement for training ---
            missing_features = [f for f in CANONICAL_FEATURES if f not in df.columns]
            if missing_features:
                raise ValueError(f"Client {self.client_id}: Missing canonical features: {missing_features}")
            X = df[CANONICAL_FEATURES]
            df.dropna(subset=[target_col], inplace=True) # Drop last day
            y = df[target_col]

            # Align X and y by index
            common_index = X.index.intersection(y.index)
            self.X_train = X.loc[common_index].to_numpy()
            self.y_train = y.loc[common_index].to_numpy()
            # Ensure y_train is always 2D for multi-output regression
            if self.y_train.ndim == 1:
                self.y_train = self.y_train.reshape(-1, 1)

            # Feature standardization (fit scaler on X_train)
            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)

            self.num_samples = len(self.X_train)
            logging.info(f"Client {self.client_id}: Loaded data. Train shape X: {self.X_train.shape}, y: {self.y_train.shape}")

        except Exception as e:
            logging.error(f"Client {self.client_id}: Error loading/preparing data: {e}")
            self.num_samples = 0
            self.X_train = None
            self.y_train = None

    def get_data_loader(self, batch_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Creates a simple batch generator for local training from numpy arrays."""
        if self.X_train is None or self.y_train is None or self.num_samples == 0:
             return []
        indices = np.arange(self.num_samples)
        if self.random_seed is not None:
            np.random.seed(self.random_seed + self.num_samples)
        np.random.shuffle(indices)

        data_loader = []
        for i in range(0, self.num_samples, batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_X = self.X_train[batch_indices]
            batch_y = self.y_train[batch_indices]
            data_loader.append((batch_X, batch_y))
        return data_loader

    def set_parameters(self, parameters: ModelParams) -> None:
        """Sets the local model parameters from the global model."""
        try:
            self.local_model.set_parameters(parameters)
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error setting parameters - {e}")

    def train(self, global_parameters_t: ModelParams, config: Dict[str, Any]) -> Dict[str, float]:
        """ Performs local training using the predictive model and FedProx. """
        self.initial_round_params = copy.deepcopy(global_parameters_t)

        local_epochs = config.get('local_epochs', 1)
        batch_size = config.get('batch_size', 32)
        learning_rate = config.get('learning_rate', 0.01)
        mu_prox = config.get('mu_prox', 0.01)

        if self.X_train is None or self.y_train is None or self.num_samples == 0:
            logging.warning(f"Client {self.client_id}: No training data. Skipping.")
            return {'final_loss': float('inf')}

        self.set_parameters(self.initial_round_params)

        metrics = {}
        try:
            self.local_model.fit(self.X_train, self.y_train)
            final_loss = self.local_model.calculate_loss(self.X_train, self.y_train)
            metrics['final_loss'] = final_loss
            logging.info(f"Client {self.client_id}: Training complete (direct fit). Loss: {final_loss:.6f}")
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error during local model fit: {e}")
            metrics['final_loss'] = float('inf')
        return metrics

    def get_update(self, clip_norm: Optional[float]) -> Tuple[Optional[ModelParams], int]:
        """Calculates update (local - initial), clips, returns update and sample count."""
        if self.num_samples == 0 or self.initial_round_params is None:
             logging.warning(f"Client {self.client_id}: Cannot get update, no samples or initial params.")
             return None, 0

        local_params = self.local_model.get_parameters()
        raw_update = []
        try:
            for local, initial in zip(local_params, self.initial_round_params):
                if local.shape != initial.shape:
                     raise ValueError(f"Shape mismatch: Local {local.shape} vs Initial {initial.shape}")
                raw_update.append(local - initial)
        except ValueError as e:
            logging.error(f"Client {self.client_id}: Error calculating raw update - {e}")
            return None, 0

        if clip_norm is not None and clip_norm > 0:
            l2_norm = _calculate_l2_norm(raw_update)
            if l2_norm > 0:
                scale = min(1.0, clip_norm / l2_norm)
                clipped_update = [layer * scale for layer in raw_update]
                logging.debug(f"Client {self.client_id}: Clipped update norm from {l2_norm:.4f}")
                return clipped_update, self.num_samples
            else:
                return raw_update, self.num_samples
        else:
            return raw_update, self.num_samples

    def run_local_optimization(self, current_features: pd.DataFrame,
                               historical_returns: pd.DataFrame,
                               config: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        Uses the client's current local model to predict returns, calculates
        empirical covariance, and runs the MVO solver.

        Args:
            current_features (pd.DataFrame): DataFrame of features for the current
                                             timestamp(s) needed for prediction
                                             (index=timestamp, columns=features).
                                             Needs to be scaled if model expects scaled input.
            historical_returns (pd.DataFrame): DataFrame of recent actual log returns
                                               (index=timestamp, columns=symbols) needed
                                               for empirical covariance calculation.
            config (Dict[str, Any]): Configuration for optimization, e.g.,
                                     {'risk_free_rate': 0.01, 'cov_lookback': 60}.

        Returns:
            Optional[Dict[str, float]]: Dictionary of optimal weights {symbol: weight},
                                        or None if prediction or optimization fails.
        """
        if optimize_portfolio is None:
             logging.error("optimize_portfolio function not available.")
             return None

        # 1. Predict mu (Expected Returns for T+1)
        try:
            if len(current_features.shape) == 1:
                 current_features = current_features.to_numpy().reshape(1, -1)
            elif isinstance(current_features, pd.DataFrame):
                 current_features = current_features.to_numpy()
                 if len(current_features) > 1:
                      logging.warning("Predicting based on multiple feature rows, using last row.")
                      current_features = current_features[-1].reshape(1, -1)

            # Scale features using stored scaler
            if self.scaler is not None:
                current_features = self.scaler.transform(current_features)

            predicted_returns_array = self.local_model.predict(current_features) # Shape (1, n_outputs)
            predicted_returns = predicted_returns_array[0] # Get predictions for the single input timestamp
            mu = pd.Series(predicted_returns, index=self.symbols) # Match symbols

        except Exception as e:
            logging.error(f"Client {self.client_id}: Failed to predict mu - {e}")
            return None

        # 2. Calculate Sigma (Empirical Covariance)
        cov_lookback = config.get('cov_lookback', 60)
        if len(historical_returns) < cov_lookback:
            logging.warning(f"Client {self.client_id}: Not enough historical data ({len(historical_returns)} days) for covariance lookback ({cov_lookback}).")
            return None
        try:
            returns_subset = historical_returns.iloc[-cov_lookback:]
            if HAS_PYPFOpt:
                 Sigma = risk_models.sample_cov(returns_subset, returns_data=True, frequency=252)
            else:
                 Sigma = (returns_subset.cov() * 252)
        except Exception as e:
            logging.error(f"Client {self.client_id}: Failed to calculate Sigma - {e}")
            return None

        # 3. Call Optimizer
        opt_config = {
            'risk_free_rate': config.get('risk_free_rate', 0.01),
            'weight_bounds': config.get('weight_bounds', (0, 1))
        }
        optimal_weights_dict = optimize_portfolio(mu, Sigma, opt_config)

        return optimal_weights_dict

def _calculate_l2_norm(params: List[np.ndarray]) -> float:
    squared_norms = [np.sum(np.square(p)) for p in params]
    return np.sqrt(np.sum(squared_norms))