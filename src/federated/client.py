import pandas as pd
import numpy as np
import os
import logging
import copy
from typing import List, Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.utils import shuffle
from models.predictive_model import PortfolioPredictiveModel 

try:
    from portfolio_optimizer import optimize_portfolio, HAS_PYPFOpt, risk_models
except ImportError:
    logging.error("Could not import optimize_portfolio function.")
    optimize_portfolio = None  # Placeholder
    HAS_PYPFOpt = False
    risk_models = None

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
    def __init__(self, client_id: str, n_features: int, n_outputs: int, data_path: str, random_seed: Optional[int] = None, feature_cols: Optional[List[str]] = None):
        self.client_id = client_id
        self.local_model = PortfolioPredictiveModel(n_features=n_features, n_outputs=n_outputs, model_params={'alpha': 1.0}) 
        self.data_path = data_path
        self.X_train: Optional[np.ndarray] = None 
        self.y_train: Optional[np.ndarray] = None 
        self.symbols: List[str] = [] 
        self.num_samples: int = 0
        self.random_seed = random_seed
        self.feature_cols = feature_cols
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self.initial_round_params: Optional[ModelParams] = None 
        self.load_data()

    def load_data(self) -> None:
        logging.debug(f"Client {self.client_id}: Loading data from {self.data_path}...")
        try:
            df = pd.read_parquet(self.data_path)
            df = df.sort_index()

            target_col = 'target_log_return'
            df[target_col] = df.groupby('symbol')['log_return'].shift(-1)
            df.dropna(subset=[target_col], inplace=True)
            df_pivot_target = df.pivot_table(index=df.index, columns='symbol', values=target_col, aggfunc='first')

            y = df_pivot_target.dropna(axis=0, how='any')
            if self.feature_cols is not None:
                numeric_feature_cols = self.feature_cols
            else:
                numeric_feature_cols = get_feature_columns(df)
            # Group features by index and take the first row for each timestamp
            X_pivot = df[numeric_feature_cols].groupby(df.index).first()
            X_final = X_pivot.loc[y.index]
            y_final = y
            assert X_final.shape[0] == y_final.shape[0], f"X and y row mismatch: {X_final.shape[0]} vs {y_final.shape[0]}"
            self.X_train = X_final.reset_index(drop=True).to_numpy()
            self.y_train = y_final.reset_index(drop=True).to_numpy()
            self.n_features = self.X_train.shape[1]

            # Ensure y_train is 2D: (n_samples, n_outputs)
            if self.y_train.ndim == 1:
                self.y_train = self.y_train.reshape(-1, 1)
            self.n_outputs = self.y_train.shape[1]

            self.num_samples = len(self.X_train)
            logging.info(f"Client {self.client_id}: Loaded data. Train shape X: {self.X_train.shape}, y: {self.y_train.shape}")
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error loading/preparing data: {e}")
            self.num_samples = 0
            self.X_train = None
            self.y_train = None

    def get_data_loader(self, batch_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        if self.X_train is None or self.y_train is None or self.num_samples == 0:
            return []
        indices = np.arange(self.num_samples)
        if self.random_seed is not None:
            np.random.seed(self.random_seed + self.num_samples) 
        np.random.shuffle(indices)

        data_loader = []
        for i in range(0, self.num_samples, batch_size):
            batch_indices = indices[i: i + batch_size]
            batch_X = self.X_train[batch_indices]
            batch_y = self.y_train[batch_indices]
            data_loader.append((batch_X, batch_y))
        return data_loader

    def train(self, global_parameters_t: ModelParams, config: Dict[str, Any]) -> Dict[str, float]:
        self.initial_round_params = copy.deepcopy(global_parameters_t) 

        local_epochs = config.get('local_epochs', 1)
        batch_size = config.get('batch_size', 32)
        learning_rate = config.get('learning_rate', 0.01)
        mu_prox = config.get('mu_prox', 0.01)

        if self.X_train is None or self.y_train is None or self.num_samples == 0:
            logging.warning(f"Client {self.client_id}: No training data. Skipping.")
            return {'loss': float('inf')}

        self.set_parameters(self.initial_round_params)

        logging.debug(f"Client {self.client_id}: Starting local training...")
        metrics = {}
        try:
            self.local_model.fit(self.X_train, self.y_train)
            final_loss = self.local_model.calculate_loss(self.X_train, self.y_train)
            metrics['final_loss'] = final_loss
            logging.debug(f"Client {self.client_id}: Training complete (direct fit). Loss: {final_loss:.6f}")
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error during local model fit: {e}")
            metrics['final_loss'] = float('inf')

        return metrics

    def set_parameters(self, params: List[np.ndarray]) -> None:
        """Sets local model parameters, ensuring the model is initialized with correct shapes."""
        try:
            # Ensure model has correct number of estimators before setting parameters
            if not hasattr(self.local_model._model, 'estimators_') or not self.local_model._model.estimators_:
                X_dummy = np.zeros((1, self.n_features))
                y_dummy = np.zeros((1, self.n_outputs))
                self.local_model._model.fit(X_dummy, y_dummy)
            self.local_model.set_parameters(params)
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error setting parameters - {e}")

    def get_update(self, initial_round_params: Optional[List[np.ndarray]], clip_norm: Optional[float] = None) -> Tuple[Optional[ModelParams], int]:
        if self.num_samples == 0 or initial_round_params is None:
            logging.warning(f"Client {self.client_id}: Cannot get update, no samples or initial params.")
            return None, 0

        self.initial_round_params = copy.deepcopy(initial_round_params)
        local_params = self.local_model.get_parameters()
        raw_update = []
        try:
            for local, initial in zip(local_params, self.initial_round_params):
                if local.shape != initial.shape:
                    raise ValueError(f"Shape mismatch: Local {local.shape} vs Initial {initial.shape}")
                raw_update.append(local - initial)
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error calculating raw update - {e}")
            return None, 0

        if clip_norm is not None and clip_norm > 0:
            l2_norm = self._calculate_l2_norm(raw_update)
            if l2_norm > 0:
                scale = min(1.0, clip_norm / l2_norm)
                clipped_update = [layer * scale for layer in raw_update]
                logging.debug(f"Client {self.client_id}: Clipped update norm from {l2_norm:.4f}")
                return clipped_update, self.num_samples
            else:
                return raw_update, self.num_samples
        else:
            return raw_update, self.num_samples

    def _calculate_l2_norm(self, params: List[np.ndarray]) -> float:
        squared_norms = [np.sum(np.square(p)) for p in params]
        return np.sqrt(np.sum(squared_norms))

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

            predicted_returns_array = self.local_model.predict(current_features)
            predicted_returns = predicted_returns_array[0]
            mu = pd.Series(predicted_returns, index=self.symbols)
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
            if HAS_PYPFOpt and risk_models is not None:
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