import numpy as np
import pandas as pd
import os
import logging
import copy
import random
from typing import List, Dict, Any
import pickle

# Adjust imports based on actual structure
try:
    import config
    from federated.aggregator import Aggregator
    from federated.client import Client
    from models.placeholder import PlaceholderLinearModel
except ImportError:
    logging.error("Failed to import necessary modules. Check paths and file names.")
    exit(1)

# --- DP-accounting import fix ---
try:
    from dp_accounting.dp_event import GaussianDpEvent
    from dp_accounting.rdp import RdpAccountant
    ACCOUNTING_LIB_AVAILABLE = True
except ImportError as e:
    logging.warning(f"dp-accounting library not found or incomplete. ImportError: {e}")
    ACCOUNTING_LIB_AVAILABLE = False
except Exception as e:
    logging.error(f"Unexpected error importing dp-accounting: {e}")
    ACCOUNTING_LIB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Canonical FTSE 100 feature list ---
CANONICAL_FEATURES = [
    'adjusted_close', 'close', 'dividend_amount', 'high', 'low', 'open', 'split_coefficient',
    'volume', 'sma_5', 'sma_20', 'sma_60', 'volatility_20', 'day_of_week', 'month', 'quarter',
    'log_return_lag_1', 'log_return_lag_2', 'log_return_lag_3', 'log_return_lag_5', 'log_return_lag_10',
    'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5', 'volume_lag_10',
    'adjusted_close_lag_1', 'adjusted_close_lag_2', 'adjusted_close_lag_3', 'adjusted_close_lag_5',
    'adjusted_close_lag_10', 'rsi', 'macd', 'macd_signal', 'macd_diff', 'obv'
]
ID_COLS = ['symbol', 'timestamp']
TARGET_COL = 'log_return'
ALL_CANONICAL_COLS = ID_COLS + CANONICAL_FEATURES + [TARGET_COL]

# --- Simulation Configuration ---
# These could also be loaded from a YAML/JSON config file
TOTAL_ROUNDS = 50 # Example: Number of FL rounds
CLIENTS_PER_ROUND = 10 # Example: Number of clients selected each round (K)
# CLIENT_FRACTION = 0.2 # Alternative: Fraction C (select K = max(1, floor(C*N)))
LOCAL_EPOCHS = 5 # E
BATCH_SIZE = 64 # B
LEARNING_RATE = 0.01 # eta
MU_PROX = 0.1 # FedProx mu parameter

# --- Main Simulation Function ---
def run_fl_simulation(fl_params, dp_params, model_params, non_iid_params, seed, experiment_id=None, results_dir=None):
    """Runs the Federated Learning simulation using FedProx/FedAvg/DP with passed config dicts."""
    # Use passed config dicts instead of globals
    # Set up config, logging, and parameters
    import config as global_config
    if results_dir is None:
        results_dir = global_config.RESULTS_DIR
    random.seed(seed)
    np.random.seed(seed)

    # 1. Load Data Info & Determine Feature Size
    try:
        client0_path = os.path.join(global_config.FEDERATED_DATA_DIR, "client_0", "client_data.parquet")
        temp_df = pd.read_parquet(client0_path)
        # --- Canonical feature enforcement ---
        missing_features = [f for f in CANONICAL_FEATURES if f not in temp_df.columns]
        if missing_features:
            raise ValueError(f"Client 0 is missing canonical features: {missing_features}")
        # Use canonical feature order
        X0 = temp_df[CANONICAL_FEATURES]
        n_features = X0.shape[1]
        if n_features == 0:
            raise ValueError("No numeric features found in client data.")
    except Exception as e:
        logging.error(f"Failed to load client 0 data or determine feature size: {e}")
        return None

    # 2. Model Template and Aggregator
    from models.placeholder import PlaceholderLinearModel
    model_template = PlaceholderLinearModel(n_features=n_features, random_state=seed)
    all_client_ids = [f"client_{i}" for i in range(global_config.NUM_CLIENTS)]
    from federated.aggregator import Aggregator
    aggregator = Aggregator(model_template, num_clients=global_config.NUM_CLIENTS, client_ids=all_client_ids)

    # 3. Initialize Clients
    from federated.client import Client
    clients = []
    n_outputs = 1
    try:
        target_col = 'target_log_return'
        if target_col in temp_df.columns:
            n_outputs = 1
    except Exception as e:
        logging.warning(f"Could not infer n_outputs from data: {e}. Defaulting to 1.")
        n_outputs = 1
    for i in range(global_config.NUM_CLIENTS):
        client_id = all_client_ids[i]
        data_path = os.path.join(global_config.FEDERATED_DATA_DIR, client_id, "client_data.parquet")
        if os.path.exists(data_path):
            client = Client(client_id, n_features, n_outputs, data_path, random_seed=seed + i)
            if client.num_samples > 0:
                clients.append(client)
        # else skip
    if not clients:
        logging.error("No clients were successfully initialized. Exiting.")
        return None
    num_active_clients = len(clients)

    # DP/FL/Training config
    DP_PARAMS = dp_params.copy() if dp_params else None
    training_config = fl_params.copy()
    training_config['mu_prox'] = model_params.get('mu_prox', 0.0)

    # Privacy Accountant
    accountant = None
    if DP_PARAMS and ACCOUNTING_LIB_AVAILABLE:
        try:
            accountant = RdpAccountant()
        except Exception as e:
            logging.error(f"Failed to instantiate RdpAccountant: {e}")
            accountant = None
    # Set noise_multiplier if not provided
    if DP_PARAMS and 'noise_multiplier' not in DP_PARAMS:
        DP_PARAMS['noise_multiplier'] = 2.0 # Placeholder

    num_select = max(1, int(fl_params['clients_per_round']))
    history = {'round': [], 'selected_clients': [], 'avg_client_loss': []}

    for t in range(fl_params['total_rounds']):
        available_indices = list(range(num_active_clients))
        selected_indices = random.sample(available_indices, min(num_select, num_active_clients))
        selected_client_ids = [clients[i].client_id for i in selected_indices]
        history['round'].append(t+1)
        history['selected_clients'].append(selected_client_ids)
        global_params_t = aggregator.get_global_parameters()
        client_updates_this_round = []
        round_losses = []
        for client_idx in selected_indices:
            client = clients[client_idx]
            initial_params_for_client = copy.deepcopy(global_params_t)
            client.set_parameters(initial_params_for_client)
            train_metrics = client.train(initial_params_for_client, training_config)
            round_losses.append(train_metrics.get('final_loss', float('inf')))
            update, num_samples = client.get_update(DP_PARAMS['clip_norm'] if DP_PARAMS else None)
            if update is not None and num_samples > 0:
                client_updates_this_round.append((update, num_samples))
        if client_updates_this_round:
            aggregator.aggregate_updates(client_updates_this_round,
                                        num_total_clients=global_config.NUM_CLIENTS,
                                        dp_params=DP_PARAMS)
            # Compose privacy budget
            if DP_PARAMS and ACCOUNTING_LIB_AVAILABLE and accountant and DP_PARAMS.get('noise_multiplier'):
                try:
                    event = GaussianDpEvent(DP_PARAMS['noise_multiplier'])
                    accountant.compose(event, count=1)
                except Exception as acc_e:
                    logging.error(f"Error during privacy accounting: {acc_e}")
            avg_loss = np.mean([loss for loss in round_losses if loss != float('inf')])
            history['avg_client_loss'].append(avg_loss)
        else:
            history['avg_client_loss'].append(np.nan)

    # Final privacy spend
    final_epsilon, final_delta = None, None
    if DP_PARAMS and ACCOUNTING_LIB_AVAILABLE and accountant:
        # Compute ε for given δ threshold and then compute corresponding δ
        final_epsilon = accountant.get_epsilon(DP_PARAMS['target_delta'])
        final_delta = accountant.get_delta(final_epsilon)

    # Save history and model if requested
    if experiment_id:
        history_df = pd.DataFrame(history)
        os.makedirs(results_dir, exist_ok=True)
        history_path = os.path.join(results_dir, f'{experiment_id}_run_history.csv')
        history_df.to_csv(history_path, index=False)
        final_model_params = aggregator.get_global_parameters()
        model_save_path = os.path.join(results_dir, f'{experiment_id}_global_model.pkl')
        with open(model_save_path, 'wb') as f:
            pickle.dump(final_model_params, f)
    # Reconstruct model object with final parameters for downstream evaluation
    from models.placeholder import PlaceholderLinearModel
    global_model = PlaceholderLinearModel(n_features=n_features, random_state=seed)
    global_model.set_parameters(aggregator.get_global_parameters())

    # --- EVALUATE FINAL MODEL (DSR METHODOLOGY: Out-of-sample metrics) ---
    metrics = {}
    try:
        # Use validation set for final evaluation (out-of-sample, as per DSR best practice)
        val_path = getattr(global_config, 'CENTRAL_TEST_PATH', None)
        if val_path is None:
            val_path = os.path.join('data', 'processed', 'central_validation_set.parquet')
        if os.path.exists(val_path):
            logging.info(f"Evaluating final model on {val_path}")
            df_val = pd.read_parquet(val_path)
            # Ensure 'timestamp' is datetime and set as index for portfolio metrics
            if 'timestamp' in df_val.columns:
                df_val['timestamp'] = pd.to_datetime(df_val['timestamp'])
                df_val = df_val.set_index('timestamp')
            # Ensure canonical feature order and target
            X_val = df_val[CANONICAL_FEATURES].to_numpy()
            y_val = df_val[TARGET_COL].to_numpy()
            # --- Apply same standardization as training ---
            from sklearn.preprocessing import StandardScaler
            # Load scaler fit on X_train from any client (all use same scaling logic)
            scaler = None
            for c in clients:
                if hasattr(c, 'scaler') and c.scaler is not None:
                    scaler = c.scaler
                    break
            if scaler is not None:
                X_val = scaler.transform(X_val)
            else:
                logging.warning("No scaler found from clients; X_val not standardized!")
            # Predictive metrics
            from evaluation.metrics import calculate_predictive_metrics, calculate_portfolio_metrics
            pred_metrics = calculate_predictive_metrics(y_val, global_model.predict(X_val))
            # Portfolio metrics (using log returns)
            # If y_pred is 2D, flatten
            if hasattr(global_model.predict(X_val), 'ndim') and global_model.predict(X_val).ndim > 1:
                y_pred = global_model.predict(X_val).ravel()
            else:
                y_pred = global_model.predict(X_val)
            port_metrics = calculate_portfolio_metrics(pd.Series(y_pred, index=df_val.index), risk_free_rate=0.01)
            metrics.update(pred_metrics)
            metrics.update(port_metrics)
        else:
            logging.warning(f"Validation set not found at {val_path}. Skipping final evaluation metrics.")
    except Exception as e:
        logging.error(f"Error during final evaluation: {e}")

    # --- Communication Cost Calculation ---
    from evaluation.metrics import calculate_communication_cost
    from typing import Any
    # Estimate model update size (use aggregator global parameters as proxy)
    model_params = aggregator.get_global_parameters()
    update_size_bytes = 0
    try:
        # Use same estimation logic as in run_experiment.py
        for p in model_params:
            if hasattr(p, 'nbytes'):
                update_size_bytes += p.nbytes
    except Exception as e:
        logging.warning(f"Could not estimate update size: {e}")
    comm_cost = calculate_communication_cost(update_size_bytes, fl_params['clients_per_round'], fl_params['total_rounds'])
    metric_communication_total_MB_uploaded = comm_cost['total_MB_uploaded']

    # --- Convergence Rounds Calculation ---
    from evaluation.metrics import determine_convergence
    # Convert history to list of dicts for determine_convergence
    metrics_history = []
    for i in range(len(history['round'])):
        metrics_history.append({'round': history['round'][i], 'avg_client_loss': history['avg_client_loss'][i]})
    # Relaxed convergence criteria
    convergence_rounds = determine_convergence(metrics_history, metric_key='avg_client_loss', tolerance=1e-3, patience=3)
    if convergence_rounds is not None:
        logging.info(f"[EXPERIMENT LOG] Convergence detected at round: {convergence_rounds}")
    else:
        logging.info(f"[EXPERIMENT LOG] No convergence detected. Last avg_client_loss values: {[m['avg_client_loss'] for m in metrics_history[-6:]]}")

    return {
        'history': history,
        'final_model_params': aggregator.get_global_parameters(),
        'accountant': accountant,
        'final_epsilon': final_epsilon,
        'final_delta': final_delta,
        'global_model': global_model,
        'metrics': metrics,
        'metric_communication_total_MB_uploaded': metric_communication_total_MB_uploaded,
        'convergence_rounds': convergence_rounds
    }

def run_simulation():
    """
    Legacy CLI entry point for running the federated simulation with config module globals.
    Loads config and calls run_fl_simulation with appropriate arguments.
    """
    import config as global_config
    # Extract parameters from config
    fl_params = getattr(global_config, 'FL_PARAMS', {})
    dp_params = getattr(global_config, 'DP_PARAMS', {})
    model_params = getattr(global_config, 'MODEL_PARAMS', {})
    non_iid_params = getattr(global_config, 'NON_IID_PARAMS', {})
    seed = getattr(global_config, 'RANDOM_SEED', 42)
    experiment_id = getattr(global_config, 'EXPERIMENT_ID', None)
    results_dir = getattr(global_config, 'RESULTS_DIR', None)
    return run_fl_simulation(fl_params, dp_params, model_params, non_iid_params, seed, experiment_id, results_dir)

# Keep legacy CLI
if __name__ == "__main__":
    run_simulation()