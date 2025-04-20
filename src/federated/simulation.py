import numpy as np
import pandas as pd
import os
import logging
import copy
import random
from typing import List, Dict, Any
import pickle
from federated.client import get_feature_columns

# Adjust imports based on actual structure
try:
    import config
    from .aggregator import Aggregator
    from .client import Client
    from models.placeholder import PlaceholderLinearModel
except ImportError:
    logging.error("Failed to import necessary modules. Check paths and file names.")
    exit(1)

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
         # Use the same feature selection as the clients
         numeric_feature_cols = get_feature_columns(temp_df)
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
            update, num_samples = client.get_update(initial_params_for_client)

            if update is not None and num_samples > 0:
                client_updates_this_round.append((update, num_samples))
            else:
                 logging.warning(f"No update generated by client {client.client_id}")


        # Aggregate updates at server
        if client_updates_this_round:
             aggregator.aggregate_updates(client_updates_this_round)
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

    # Save history/results
    history_path = os.path.join(config.RESULTS_DIR, "fedprox_run_history.csv")
    history_df = pd.DataFrame(history)
    history_df.to_csv(history_path, index=False)
    logging.info(f"Run history saved to {history_path}")

    # Save final global model
    final_model_params = aggregator.get_global_parameters()
    global_model_path = os.path.join(config.RESULTS_DIR, "fedprox_global_model.pkl")
    with open(global_model_path, "wb") as f:
        pickle.dump(final_model_params, f)
    logging.info(f"Final global model parameters saved to {global_model_path}")


if __name__ == "__main__":
    run_simulation()