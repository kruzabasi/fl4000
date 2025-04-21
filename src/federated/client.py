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
    def __init__(self, client_id: str, n_features: int, n_outputs: int, data_path: str, random_seed: Optional[int] = None):
        self.client_id = client_id
        self.local_model = PortfolioPredictiveModel(n_features=n_features, n_outputs=n_outputs, model_params={'alpha': 1.0}) 
        self.data_path = data_path
        self.X_train: Optional[np.ndarray] = None 
        self.y_train: Optional[np.ndarray] = None 
        self.symbols: List[str] = [] 
        self.num_samples: int = 0
        self.random_seed = random_seed
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
            df_pivot_target = df.pivot(columns='symbol', values=target_col)

            feature_cols = [col for col in df.columns if col not in ['symbol', 'log_return', target_col, 'adjusted_close', 'close', 'open', 'high', 'low', 'gics_sector']]
            numeric_feature_cols = df[feature_cols].select_dtypes(include=np.number).columns.tolist()

            df.dropna(subset=[target_col], inplace=True) 
            X = df[numeric_feature_cols]
            y = df[target_col]

            common_index = X.index.intersection(y.index)
            self.X_train = X.loc[common_index].to_numpy()
            self.y_train = y.loc[common_index].to_numpy()

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

    def set_parameters(self, parameters: ModelParams) -> None:
        try:
            self.local_model.set_parameters(parameters)
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error setting parameters - {e}")

    def get_update(self, clip_norm: Optional[float]) -> Tuple[Optional[ModelParams], int]:
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