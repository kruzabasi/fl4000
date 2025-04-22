import numpy as np
import pandas as pd
import os
import logging
import copy
import random
from typing import List, Dict, Any, Optional # Added Optional
import pickle
import warnings # To suppress specific warnings if needed

# Project Modules
try:
    # config is likely still needed for some base paths or constants not in exp_config
    import config
    from federated.aggregator import Aggregator
    from federated.client import Client, get_feature_columns # Moved get_feature_columns here
    from models.predictive_model import PortfolioPredictiveModel
except ImportError as import_err:
    logging.error(f"Failed to import necessary modules: {import_err}. Check paths and file names.")
    # Raising an exception might be better than exit(1) in a library function
    raise ImportError(f"Simulation module import failed: {import_err}")

# --- DP Accounting Imports ---
try:
    from dp_accounting import dp_event
    from dp_accounting.rdp import rdp_privacy_accountant # Use specific accountant if known
    ACCOUNTING_LIB_AVAILABLE = True
    # Define the specific accountant class used
    PrivacyAccountant = rdp_privacy_accountant.RdpAccountant
except ImportError:
    logging.warning("dp-accounting library not found or import failed. DP features will be disabled.")
    ACCOUNTING_LIB_AVAILABLE = False
    PrivacyAccountant = None # Placeholder

# Configure logging - might be better configured by the calling script (run_experiment)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Main Simulation Function ---
def run_simulation(exp_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs a Federated Learning simulation based on the provided configuration.

    Args:
        exp_config (Dict[str, Any]): Dictionary containing experiment parameters.
                                     Expected keys: 'fl_params', 'dp_params', 'model_params',
                                     'data_params', 'model_type', 'random_seed'.

    Returns:
        Dict[str, Any]: Dictionary containing simulation results:
                        'model_params': Final global model parameters.
                        'history': DataFrame of training history (e.g., loss per round).
                        'accountant': Privacy accountant object (if DP used).
    """
    # --- Extract Configuration ---
    fl_params = exp_config.get('fl_params', {})
    dp_params = exp_config.get('dp_params', {})
    model_params_config = exp_config.get('model_params', {}) # Renamed to avoid conflict
    data_params = exp_config.get('data_params', {})
    model_type = exp_config.get('model_type', 'FedAvg').lower() # Default to FedAvg if not specified
    seed = exp_config.get('random_seed', 42)
    results_dir = exp_config.get('results_dir', getattr(config, 'RESULTS_DIR', 'results')) # Use config path

    # FL Parameters
    total_rounds = fl_params.get('total_rounds', 100)
    clients_per_round = fl_params.get('clients_per_round', 10)
    local_epochs = fl_params.get('local_epochs', 5)
    batch_size = fl_params.get('batch_size', 32)
    learning_rate = fl_params.get('learning_rate', 0.01)
    mu_prox = fl_params.get('prox_mu', 0.0) if model_type == 'fedproxdp' else 0.0 # FedProx specific

    # DP Parameters (only relevant if model_type enables DP)
    use_dp = model_type == 'fedproxdp' and ACCOUNTING_LIB_AVAILABLE
    clip_norm = dp_params.get('clip_norm', 1.0) if use_dp else None
    target_delta = dp_params.get('target_delta', 1e-5) if use_dp else None
    # Noise multiplier might be fixed or calculated - assume fixed for now if in config
    noise_multiplier = dp_params.get('noise_multiplier') if use_dp else None

    # Data Parameters
    partitioned_data_dir = data_params.get('partitioned_data_dir', getattr(config, 'FEDERATED_DATA_DIR', 'data/partitioned'))
    num_clients = getattr(config, 'NUM_CLIENTS', 40) # Get total client number, maybe from data dir listing later

    logging.info(f"--- Starting FL Simulation: {exp_config.get('experiment_id', 'N/A')} ---")
    logging.info(f"Model Type: {model_type}, Rounds: {total_rounds}, Clients/Round: {clients_per_round}")
    if use_dp:
        logging.info(f"DP Enabled: Clip={clip_norm}, Delta={target_delta}, NoiseMult={noise_multiplier if noise_multiplier else 'Not Set'}")
    if mu_prox > 0:
        logging.info(f"FedProx Enabled: Mu={mu_prox}")

    logging.info(f"FL Params: total_rounds={total_rounds}, clients_per_round={clients_per_round}, local_epochs={local_epochs}, batch_size={batch_size}, learning_rate={learning_rate}, mu_prox={mu_prox}")
    logging.info(f"DP Params: use_dp={use_dp}, clip_norm={clip_norm}, target_delta={target_delta}, noise_multiplier={noise_multiplier}")
    logging.info(f"Data Params: partitioned_data_dir={partitioned_data_dir}, num_clients={num_clients}")
    logging.info(f"Model Params: {model_params_config}")

    # --- Set Seed ---
    random.seed(seed)
    np.random.seed(seed)
    # Add torch/tf seeds if needed

    # 1. Load Data Info & Determine Feature/Output Size
    # Load one client's data to infer dimensions
    n_features = model_params_config.get('n_features') # Prefer config value if provided
    n_outputs = model_params_config.get('n_outputs')   # Prefer config value if provided

    if n_features is None or n_outputs is None:
        logging.info("Inferring feature/output size from sample client data...")
        try:
            # Find the first available client data directory
            client0_dir = None
            for i in range(num_clients):
                 potential_dir = os.path.join(partitioned_data_dir, f"client_{i}")
                 if os.path.isdir(potential_dir):
                     client0_dir = potential_dir
                     break
            if client0_dir is None:
                 raise FileNotFoundError("Could not find any client data directories.")

            client0_path = os.path.join(client0_dir, "client_data.parquet")
            if not os.path.exists(client0_path):
                 raise FileNotFoundError(f"Sample client data file not found: {client0_path}")

            temp_df = pd.read_parquet(client0_path)
            numeric_feature_cols = get_feature_columns(temp_df) # Use helper from client.py
            if n_features is None:
                 n_features = len(numeric_feature_cols)
            if n_outputs is None:
                 # Infer outputs (e.g., number of unique assets if predicting per asset)
                 # This logic depends heavily on the specific target variable structure
                 # Example: Assuming 'target_log_return' exists and we pivot by 'symbol'
                 target_col = 'target_log_return'
                 if target_col not in temp_df.columns:
                     # Attempt to create target if missing (e.g., shifted log_return)
                     if 'log_return' in temp_df.columns and 'symbol' in temp_df.columns:
                         temp_df[target_col] = temp_df.groupby('symbol')['log_return'].shift(-1)
                     else:
                         raise ValueError(f"Cannot infer outputs: '{target_col}' or source columns missing.")

                 # Check for NaNs introduced by shift
                 temp_df_valid_target = temp_df.dropna(subset=[target_col])
                 if temp_df_valid_target.empty:
                      raise ValueError("No valid target values found after shift/dropna.")

                 # Pivot to get number of unique symbols (outputs)
                 # Use pivot_table for robustness against duplicate index/column entries
                 try:
                     # Ensure index is unique before pivoting if necessary
                     if not temp_df_valid_target.index.is_unique:
                          temp_df_valid_target = temp_df_valid_target.loc[~temp_df_valid_target.index.duplicated(keep='first')]

                     # Use 'symbol' if available, otherwise assume single output
                     if 'symbol' in temp_df_valid_target.columns:
                          df_pivot_target = temp_df_valid_target.pivot_table(index=temp_df_valid_target.index, columns='symbol', values=target_col)
                          n_outputs = df_pivot_target.shape[1]
                     else:
                          n_outputs = 1 # Assume single output if no symbol column
                 except Exception as pivot_err:
                      logging.error(f"Error pivoting data to determine number of outputs: {pivot_err}")
                      raise ValueError("Could not determine number of outputs from data structure.")

            if n_features == 0 or n_outputs == 0:
                 raise ValueError("Inferred zero features or outputs.")
            logging.info(f"Inferred number of features: {n_features}, number of outputs: {n_outputs}")

        except Exception as e:
            logging.error(f"Could not load sample client data or infer dimensions: {e}")
            # Return empty results or raise exception to signal failure
            return {'model_params': None, 'history': pd.DataFrame(), 'accountant': None, 'predictions': None, 'portfolio_returns': None}
    else:
        logging.info(f"Using feature/output size from config: Features={n_features}, Outputs={n_outputs}")
        # Need feature columns if using client helper function
        # This part is tricky if not inferring from data - assume client handles it or pass cols in config
        numeric_feature_cols = None # Or load from config if provided

    # 2. Initialize Model Template and Aggregator
    # Use model parameters from config if provided
    # Only pass valid model_params to PortfolioPredictiveModel
    valid_model_params = model_params_config.copy() if isinstance(model_params_config, dict) else {}
    # Remove keys not accepted by PortfolioPredictiveModel (which only expects e.g. 'alpha' for Ridge)
    for k in list(valid_model_params.keys()):
        if k not in ['alpha', 'fit_intercept', 'normalize', 'copy_X', 'max_iter', 'tol', 'solver', 'random_state', 'positive']:  # Ridge valid params
            valid_model_params.pop(k)
    model_template = PortfolioPredictiveModel(
        n_features=n_features,
        n_outputs=n_outputs,
        model_params=valid_model_params
    )
    # Ensure the model's internal state/estimators are initialized if needed
    # This might depend on the specific model implementation (e.g., sklearn requires fit)
    try:
        # Example: Dummy fit for sklearn models if needed before getting params
        if hasattr(model_template, '_model') and hasattr(model_template._model, 'fit'):
             with warnings.catch_warnings(): # Suppress potential convergence warnings on dummy data
                  warnings.simplefilter("ignore")
                  X_dummy = np.zeros((max(2, n_outputs), n_features)) # Ensure enough samples for some estimators
                  y_dummy = np.zeros((max(2, n_outputs), n_outputs))
                  model_template._model.fit(X_dummy, y_dummy)
    except Exception as model_init_err:
        logging.warning(f"Could not perform initial dummy fit on model template: {model_init_err}")

    all_client_ids = [f"client_{i}" for i in range(num_clients)]
    aggregator = Aggregator(model_template, num_clients=num_clients, client_ids=all_client_ids)

    # 3. Initialize Clients
    clients: List[Client] = []
    successful_client_ids = []
    for i in range(num_clients):
        client_id = all_client_ids[i]
        data_path = os.path.join(partitioned_data_dir, client_id, "client_data.parquet")
        if os.path.exists(data_path):
            try:
                # Pass feature columns if inferred, otherwise client might determine them
                client = Client(client_id, n_features, n_outputs, data_path,
                                random_seed=seed + i, # Use experiment seed + offset
                                feature_cols=numeric_feature_cols) # Pass inferred cols
                if client.num_samples > 0:
                    clients.append(client)
                    successful_client_ids.append(client_id)
                else:
                    logging.warning(f"Client {client_id} initialized but has 0 samples. Skipping.")
            except Exception as client_init_err:
                 logging.error(f"Failed to initialize client {client_id}: {client_init_err}")
        else:
            logging.warning(f"Data path not found for client {client_id}: {data_path}. Skipping.")

    if not clients:
         logging.error("No clients were successfully initialized. Exiting simulation.")
         return {'model_params': None, 'history': pd.DataFrame(), 'accountant': None, 'predictions': None, 'portfolio_returns': None}

    num_active_clients = len(clients)
    logging.info(f"Initialized {num_active_clients} active clients: {successful_client_ids}")
    # Update aggregator if number of active clients differs from total expected
    if hasattr(aggregator, 'client_ids'):
        aggregator.client_ids = successful_client_ids
    # If aggregator tracks num_clients, update as well
    if hasattr(aggregator, 'num_clients'):
        aggregator.num_clients = len(successful_client_ids)

    # --- DP Configuration & Accountant Init ---
    accountant = None
    effective_dp_params = None # Store the params actually used (with calculated noise mult)

    if use_dp:
        if PrivacyAccountant is None:
             logging.error("DP requested but PrivacyAccountant class is not available. Disabling DP.")
             use_dp = False
        else:
            accountant = PrivacyAccountant() # Initialize the specific accountant
            sampling_probability = min(1.0, clients_per_round / num_clients) if num_clients > 0 else 0

            if sampling_probability <= 0:
                logging.warning("Client sampling probability is zero or negative. DP noise calculation skipped.")
                use_dp = False # Cannot apply sampling DP
            elif noise_multiplier is None:
                # If noise multiplier not provided, try to calculate it (requires target epsilon)
                target_epsilon = dp_params.get('target_epsilon')
                if target_epsilon is None:
                     logging.error("DP requested but noise_multiplier and target_epsilon not provided. Cannot calibrate noise. Disabling DP.")
                     use_dp = False
                else:
                     # --- Noise Calibration (Example using RdpAccountant) ---
                     # This is complex and depends on the accountant type.
                     # Placeholder: For now, require noise_multiplier in config if DP is used.
                     logging.warning("Noise calibration from target epsilon not implemented. Please provide 'noise_multiplier' in dp_params for DP.")
                     use_dp = False # Disable DP if noise cannot be determined

            if use_dp:
                 effective_dp_params = {
                     'clip_norm': clip_norm,
                     'noise_multiplier': noise_multiplier,
                     'target_delta': target_delta # Keep for logging/reporting
                 }
                 logging.info(f"DP Initialized: Noise Multiplier={noise_multiplier:.4f}, Clip Norm={clip_norm:.4f}, Target Delta={target_delta:.1E}")
            else:
                 accountant = None # Ensure accountant is None if DP disabled


    # 4. Federation Loop
    training_config = {
        'local_epochs': local_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'mu_prox': mu_prox # Pass mu_prox here
    }
    # Determine actual number of clients to select per round
    num_select = max(1, int(clients_per_round)) if clients_per_round >= 1 else max(1, int(clients_per_round * num_active_clients))
    num_select = min(num_select, num_active_clients) # Cannot select more than available

    history = {'round': [], 'selected_clients': [], 'avg_client_loss': []} # Basic history tracking

    for t in range(total_rounds):
        logging.info(f"--- Round {t+1}/{total_rounds} ---")

        # Select clients for this round (using indices relative to the 'clients' list)
        available_indices = list(range(num_active_clients))
        selected_indices = random.sample(available_indices, num_select)
        selected_client_ids_this_round = [clients[i].client_id for i in selected_indices] # Use correct var name
        logging.info(f"Selected {len(selected_indices)} clients: {selected_client_ids_this_round}")
        history['round'].append(t+1)
        history['selected_clients'].append(selected_client_ids_this_round)

        # Get current global model
        global_params_t = aggregator.get_global_parameters()
        client_updates_this_round = []
        round_losses = []

        # Train selected clients
        for client_idx in selected_indices:
            client = clients[client_idx]
            logging.debug(f"Training client {client.client_id}...")

            # Set client model to global state (make deep copy for prox term)
            initial_params_for_client = copy.deepcopy(global_params_t)
            client.set_parameters(initial_params_for_client)

            # Perform local training with FedProx logic
            try:
                train_metrics = client.train(initial_params_for_client, training_config)
                # Use loss from metrics if available, otherwise default
                final_loss = train_metrics.get('final_loss', train_metrics.get('loss'))
                if final_loss is not None:
                     round_losses.append(final_loss)
                else:
                     round_losses.append(float('inf')) # Indicate failure if no loss reported

                # Compute and collect model updates from clients with L2 norm clipping
                update, num_samples = client.get_update(initial_params_for_client, clip_norm if use_dp else None)
                if update is not None and num_samples > 0:
                    client_updates_this_round.append((update, num_samples))
                else:
                     logging.warning(f"No update generated by client {client.client_id} (Samples: {client.num_samples})")
            except Exception as train_err:
                 logging.error(f"Error during training or update generation for client {client.client_id}: {train_err}")
                 round_losses.append(float('inf')) # Indicate failure

        # Aggregate updates at server
        if client_updates_this_round:
             # Pass effective DP params (with noise multiplier) if DP is used
             aggregator.aggregate_updates(client_updates_this_round,
                                         num_total_clients=num_clients, # Use total expected clients for DP scaling
                                         dp_params=effective_dp_params if use_dp else None)

             valid_losses = [loss for loss in round_losses if loss != float('inf')]
             avg_loss = np.mean(valid_losses) if valid_losses else np.nan
             logging.info(f"Round {t+1} completed. Average client final loss: {avg_loss:.4f}")
             history['avg_client_loss'].append(avg_loss)
        else:
             logging.warning(f"Round {t+1}: No valid updates to aggregate. Global model unchanged.")
             history['avg_client_loss'].append(np.nan)

        # --- Compose Privacy Budget ---
        if use_dp and accountant and effective_dp_params:
            # Calculate sampling probability based on this round's selection
            sampling_probability_this_round = min(1.0, num_select / num_clients) if num_clients > 0 else 0
            noise_mult = effective_dp_params['noise_multiplier']

            if sampling_probability_this_round > 0 and noise_mult is not None:
                try:
                    # Create the appropriate DP event based on the accountant type
                    # Example for RdpAccountant with Gaussian mechanism:
                    event = dp_event.GaussianDpEvent(noise_mult)
                    # Apply sampling amplification
                    # Use Poisson sampling approximation if appropriate, otherwise SampledDpEvent
                    # Assuming Poisson sampling is applicable here:
                    event = dp_event.PoissonSampledDpEvent(sampling_probability_this_round, event)

                    accountant.compose(event, count=1) # Compose one step

                    # Log current privacy spend (optional, can be expensive)
                    if (t + 1) % 10 == 0 or (t + 1) == total_rounds: # Log every 10 rounds and at the end
                         spent_epsilon = accountant.get_epsilon(target_delta)
                         logging.info(f"Privacy Check - Round {t+1}: Spent Epsilon={spent_epsilon:.4f} (Delta={target_delta:.1E})")

                except Exception as acc_e:
                    logging.error(f"Error during privacy accounting composition: {acc_e}")
            else:
                 logging.warning(f"Skipping DP accounting composition in round {t+1} (prob={sampling_probability_this_round}, noise={noise_mult})")


    logging.info("--- Simulation Finished ---")

    # Log final privacy spend
    if use_dp and accountant:
        try:
            final_epsilon = accountant.get_epsilon(target_delta)
            logging.info(f"--- Final Privacy Spend ({total_rounds} rounds) ---")
            logging.info(f"Epsilon: {final_epsilon:.4f}")
            logging.info(f"Delta:   {target_delta:.1E}")
        except Exception as final_acc_e:
             logging.error(f"Could not get final privacy spend: {final_acc_e}")

    # Prepare results dictionary
    history_df = pd.DataFrame(history)
    final_model_params = aggregator.get_global_parameters()

    # --- After federated training, evaluate on central validation set ---
    validation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/ftse_processed_features.csv'))
    predictions_df = None
    portfolio_returns = None
    try:
        if not os.path.exists(validation_path):
            logging.error(f"Validation set not found at {validation_path}")
        else:
            val_df = pd.read_csv(validation_path)
            logging.info(f"Loaded validation set from {validation_path} with shape {val_df.shape}")
            logging.info(f"Validation set columns: {val_df.columns.tolist()}")
            # Use only numeric features for prediction
            exclude_cols = ['symbol', 'log_return', 'timestamp', 'datetime_idx']
            feature_cols = [col for col in val_df.columns if col not in exclude_cols and val_df[col].dtype != 'O']
            logging.info(f"Validation feature columns used for prediction: {feature_cols}")
            X_val = val_df[feature_cols].to_numpy()
            y_val = val_df['log_return'].to_numpy()
            # Predict with final global model
            if hasattr(model_template, 'predict'):
                y_pred = model_template.predict(X_val)
                logging.info(f"Prediction shape: {y_pred.shape}, Actual shape: {y_val.shape}")
                # Patch: If y_pred is 2D and y_val is 1D, use only first column
                if y_pred.ndim == 2 and y_val.ndim == 1 and y_pred.shape[0] == y_val.shape[0]:
                    logging.warning(f"Model is multi-output ({y_pred.shape[1]}), but validation target is 1D. Using only first output for metrics.")
                    y_pred_for_metrics = y_pred[:, 0]
                else:
                    y_pred_for_metrics = y_pred.flatten() if y_pred.ndim > 1 and y_pred.shape[1] == 1 else y_pred
            else:
                y_pred_for_metrics = np.full_like(y_val, np.nan)
                logging.warning("Global model has no predict method. Returning NaN predictions.")
            predictions_df = pd.DataFrame({
                'actual': y_val,
                'prediction': y_pred_for_metrics,
                'symbol': val_df['symbol'],
                'timestamp': val_df['timestamp']
            })
            logging.info(f"Predictions DataFrame head:\n{predictions_df.head()}")
            predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'], errors='coerce')
            logging.info(f"Predictions DataFrame timestamp dtype after conversion: {predictions_df['timestamp'].dtype}")
            # Simple portfolio: equal-weight, sum predictions per day, use as returns
            # This is a placeholder; replace with your own portfolio logic if needed
            returns_by_day = predictions_df.groupby('timestamp')['prediction'].mean()
            portfolio_returns = returns_by_day
            logging.info(f"Portfolio returns head:\n{portfolio_returns.head()}")
    except Exception as e:
        logging.warning(f"Validation/prediction failed: {e}")

    # --- Return results ---
    return {
        'model_params': final_model_params,
        'history': history_df,
        'accountant': accountant,
        'predictions': predictions_df,
        'portfolio_returns': portfolio_returns
    }
