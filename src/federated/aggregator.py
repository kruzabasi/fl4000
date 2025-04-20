import numpy as np
import random
import copy
import logging
from typing import List, Tuple, Dict, Any

# Placeholder type for model parameters (adjust based on actual model)
ModelParams = List[np.ndarray]

class Aggregator:
    """
    Coordinates the Federated Learning process by maintaining the global model,
    selecting clients, and aggregating client updates using weighted averaging (FedAvg).

    Args:
        model_template (Any): An instance of the model to get parameter structure.
        num_clients (int): Total number of clients available.
        client_ids (List[str]): List of unique client identifiers.
    """

    def __init__(self, model_template: Any, num_clients: int, client_ids: List[str]):
        """
        Initializes the Aggregator with the given model template and client list.
        Args:
            model_template (Any): An instance of the model to get parameter structure.
            num_clients (int): Total number of clients available.
            client_ids (List[str]): List of unique client identifiers.
        """
        self.global_model_params: ModelParams = model_template.get_parameters()
        self.num_clients = num_clients
        self.client_ids = client_ids
        self.current_round = 0
        logging.info(f"Aggregator initialized for {num_clients} clients.")
        # Log initial parameter shapes for debugging
        for i, layer in enumerate(self.global_model_params):
             logging.debug(f"Initial global param layer {i} shape: {layer.shape}")


    def select_clients(self, num_to_select: int) -> List[int]:
        """
        Selects a subset of clients randomly for the current round.
        Args:
            num_to_select (int): Number of clients to select.
        Returns:
            List[int]: Indices of selected clients.
        """
        if num_to_select >= self.num_clients:
            logging.debug("Selecting all clients.")
            return list(range(self.num_clients))
        else:
            selected_indices = random.sample(range(self.num_clients), num_to_select)
            logging.info(f"Round {self.current_round}: Selected {len(selected_indices)} clients: {[self.client_ids[i] for i in selected_indices]}")
            return selected_indices

    def get_global_parameters(self) -> ModelParams:
        """
        Returns a deep copy of the current global model parameters.
        Returns:
            ModelParams: Deep copy of the global model parameters.
        """
        return copy.deepcopy(self.global_model_params)

    def aggregate_updates(self, client_updates: List[Tuple[ModelParams, int]]) -> None:
        """
        Aggregates client updates using weighted averaging (FedAvg principle).
        Updates the global model parameters in-place.
        Args:
            client_updates (List[Tuple[ModelParams, int]]):
                A list of tuples, where each tuple contains (client_update_params, num_samples).
                client_update_params = local_params - global_params_at_start_of_round
        Returns:
            None
        """
        if not client_updates:
            logging.warning(f"Round {self.current_round}: No client updates received for aggregation.")
            return

        total_samples = sum(num_samples for _, num_samples in client_updates)
        if total_samples == 0:
            logging.warning(f"Round {self.current_round}: Total samples from clients is zero. Skipping aggregation.")
            return

        logging.info(f"Round {self.current_round}: Aggregating updates from {len(client_updates)} clients (Total samples: {total_samples}).")

        # Initialize aggregated update with zeros, matching global model structure
        aggregated_update: ModelParams = [np.zeros_like(layer) for layer in self.global_model_params]

        # Perform weighted averaging
        for update_params, num_samples in client_updates:
            weight = num_samples / total_samples
            for i in range(len(aggregated_update)):
                 # Ensure shapes match before adding
                 if update_params[i].shape == aggregated_update[i].shape:
                      aggregated_update[i] += update_params[i] * weight
                 else:
                      logging.error(f"Shape mismatch during aggregation! Layer {i}: Agg={aggregated_update[i].shape}, Update={update_params[i].shape}. Skipping this update.")
                      # Optionally break or handle error differently
                      break # Stop processing this client's update

        # Update the global model: w_{t+1} = w_t + aggregated_update
        # Note: client_update = w_local - w_t
        # Aggregated update = sum( weight * (w_local - w_t) )
        # So, w_{t+1} = w_t + sum( weight * w_local ) - sum( weight * w_t )
        # w_{t+1} = w_t + sum( weight * w_local ) - w_t = sum( weight * w_local ) -> This is FedAvg direct model averaging
        # If updates are parameter diffs (w_local - w_t), then:
        for i in range(len(self.global_model_params)):
             self.global_model_params[i] += aggregated_update[i]

        logging.debug(f"Round {self.current_round}: Global model updated.")
        self.current_round += 1