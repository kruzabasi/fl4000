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

try:
    from dp_accounting import dp_event
    from dp_accounting import privacy_accountant
    ACCOUNTING_LIB_AVAILABLE = True
except ImportError:
    logging.warning("dp-accounting library not found. Cannot perform rigorous privacy accounting.")
    ACCOUNTING_LIB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
def run_simulation():
    """Runs the Federated Learning simulation using FedProx."""
    logging.info("--- Starting FedProx Simulation ---")

    # 1. Load Data Info & Determine Feature Size (needed for model init)
    # Requires partitioned data to exist from Sprint 2
    # Load one client's data just to get feature dimension
    try:
         client0_path = os.path.join(config.FEDERATED_DATA_DIR, "client_0", "client_data.parquet")
         temp_df = pd.read_parquet(client0_path)
         # Infer feature columns (excluding metadata, target) - MUST MATCH BASELINE
         feature_cols = [col for col in temp_df.columns if col not in ['symbol', 'log_return', 'target_log_return', 'adjusted_close', 'close', 'open', 'high', 'low', 'gics_sector']]
         numeric_feature_cols = temp_df[feature_cols].select_dtypes(include=np.number).columns.tolist()
         n_features = len(numeric_feature_cols)
         if n_features == 0:
              raise ValueError("No numeric features found in client data.")
         logging.info(f"Determined number of features: {n_features}")
    except Exception as e:
        logging.error(f"Could not load sample client data to determine feature size: {e}")
        return

    # 2. Initialize Model Template and Aggregator
    model_template = PlaceholderLinearModel(n_features=n_features, random_state=config.RANDOM_SEED)
    # Assuming client IDs are 'client_0', 'client_1', ... 'client_39'
    all_client_ids = [f"client_{i}" for i in range(config.NUM_CLIENTS)]
    aggregator = Aggregator(model_template, num_clients=config.NUM_CLIENTS, client_ids=all_client_ids)

    # 3. Initialize Clients
    clients: List[Client] = []
    for i in range(config.NUM_CLIENTS):
        client_id = all_client_ids[i]
        data_path = os.path.join(config.FEDERATED_DATA_DIR, client_id, "client_data.parquet")
        if os.path.exists(data_path):
            client = Client(client_id, model_template, data_path, random_seed=config.RANDOM_SEED + i)
            # Only add client if they successfully loaded data
            if client.num_samples > 0:
                 clients.append(client)
            else:
                 logging.warning(f"Skipping client {client_id} due to data loading issues.")
        else:
            logging.warning(f"Data path not found for client {client_id}: {data_path}")

    if not clients:
         logging.error("No clients were successfully initialized. Exiting.")
         return

    num_active_clients = len(clients)
    logging.info(f"Initialized {num_active_clients} active clients.")

    # --- DP Configuration ---
    DP_PARAMS = {
        'clip_norm': 1.0, # Example C_clip - TUNE THIS
        'target_epsilon': 2.0, # Example total epsilon
        'target_delta': 1e-5, # Example target delta (often related to 1/num_total_samples)
        'total_rounds': TOTAL_ROUNDS,
        # Noise multiplier will be calculated using accountant if library available
    }
    noise_multiplier = None # Placeholder

    # --- Initialize Privacy Accountant ---
    accountant = None
    if ACCOUNTING_LIB_AVAILABLE:
        num_total_clients_for_dp = config.NUM_CLIENTS
        accountant = privacy_accountant.GaussianAccountant(max_compositions=TOTAL_ROUNDS * 2)
        sampling_probability = min(1.0, CLIENTS_PER_ROUND / num_total_clients_for_dp)
        if sampling_probability > 0:
            try:
                noise_multiplier = 2.0 # <<< EXAMPLE - REPLACE WITH ACTUAL CALCULATION using accountant
                logging.info(f"Using PRE-CALCULATED noise multiplier: {noise_multiplier:.4f} (Replace with proper calculation)")
                DP_PARAMS['noise_multiplier'] = noise_multiplier
            except Exception as e:
                logging.error(f"Could not determine noise multiplier using dp-accounting: {e}. DP disabled.")
                DP_PARAMS = None
        else:
            logging.warning("Client sampling probability is zero. DP noise calculation skipped.")
            DP_PARAMS = None

    # 4. Federation Loop
    training_config = {
        'local_epochs': LOCAL_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'mu_prox': MU_PROX
    }
    # Determine actual number of clients to select per round
    num_select = max(1, int(CLIENTS_PER_ROUND)) # Ensure at least 1
    # Or use fraction: num_select = max(1, int(CLIENT_FRACTION * num_active_clients))

    history = {'round': [], 'selected_clients': [], 'avg_client_loss': []} # Basic history tracking

    for t in range(TOTAL_ROUNDS):
        logging.info(f"--- Round {t+1}/{TOTAL_ROUNDS} ---")

        # Select clients for this round (using indices relative to the 'clients' list)
        # Need to adjust aggregator's selection if number of active clients differs
        # Simple approach: select from the active clients
        available_indices = list(range(num_active_clients))
        selected_indices = random.sample(available_indices, min(num_select, num_active_clients))
        num_selected_clients = len(selected_indices)
        sampling_probability_this_round = min(1.0, num_selected_clients / num_total_clients_for_dp)
        selected_client_ids = [clients[i].client_id for i in selected_indices]
        logging.info(f"Selected {len(selected_indices)} clients: {selected_client_ids}")
        history['round'].append(t+1)
        history['selected_clients'].append(selected_client_ids)


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
            train_metrics = client.train(initial_params_for_client, training_config)
            round_losses.append(train_metrics.get('final_loss', float('inf')))

            # Get update (local_params - initial_params_for_client)
            update, num_samples = client.get_update(DP_PARAMS['clip_norm'] if DP_PARAMS else None)

            if update is not None and num_samples > 0:
                client_updates_this_round.append((update, num_samples))
            else:
                 logging.warning(f"No update generated by client {client.client_id}")


        # Aggregate updates at server, passing DP params
        if client_updates_this_round:
            aggregator.aggregate_updates(client_updates_this_round,
                                        num_total_clients=num_total_clients_for_dp,
                                        dp_params=DP_PARAMS)
            # --- Compose Privacy Budget ---
            if ACCOUNTING_LIB_AVAILABLE and accountant and DP_PARAMS and DP_PARAMS.get('noise_multiplier') and sampling_probability_this_round > 0:
                try:
                    event = dp_event.GaussianDpEvent(DP_PARAMS['noise_multiplier'])
                    event = dp_event.SampledDpEvent(sampling_probability_this_round, event)
                    accountant.compose(event, count=1)
                    spent_epsilon, spent_delta = accountant.get_epsilon_and_delta(DP_PARAMS['target_delta'])
                    logging.info(f"Privacy Check - Round {t+1}: Spent Epsilon={spent_epsilon:.4f} (Delta={spent_delta:.1E})")
                except Exception as acc_e:
                    logging.error(f"Error during privacy accounting: {acc_e}")
            avg_loss = np.mean([loss for loss in round_losses if loss != float('inf')])
            logging.info(f"Round {t+1} completed. Average client final loss: {avg_loss:.4f}")
            history['avg_client_loss'].append(avg_loss)
        else:
             logging.warning(f"Round {t+1}: No updates to aggregate. Global model unchanged.")
             history['avg_client_loss'].append(np.nan)


        # Optional: Periodic evaluation on a global validation set
        # if (t + 1) % 5 == 0:
        #     evaluate_global_model(aggregator.get_global_parameters(), global_val_data)

    logging.info("--- Simulation Finished ---")

    # Log final privacy spend
    if ACCOUNTING_LIB_AVAILABLE and accountant and DP_PARAMS:
        final_epsilon, final_delta = accountant.get_epsilon_and_delta(DP_PARAMS['target_delta'])
        logging.info(f"--- Final Privacy Spend ({TOTAL_ROUNDS} rounds) ---")
        logging.info(f"Epsilon: {final_epsilon:.4f}")
        logging.info(f"Delta:   {final_delta:.1E}")

    # Save history/results
    history_df = pd.DataFrame(history)
    history_path = os.path.join(config.RESULTS_DIR, 'fedprox_run_history.csv')
    history_df.to_csv(history_path, index=False)
    logging.info(f"Run history saved to {history_path}")

    # Save final global model
    final_model_params = aggregator.get_global_parameters()
    model_save_path = os.path.join(config.RESULTS_DIR, 'fedprox_global_model.pkl')
    try:
        # Save as list of numpy arrays
        with open(model_save_path, 'wb') as f:
             pickle.dump(final_model_params, f)
        logging.info(f"Final global model parameters saved to {model_save_path}")
    except Exception as e:
        logging.error(f"Could not save final model: {e}")


if __name__ == "__main__":
    run_simulation()