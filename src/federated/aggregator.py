# FILE: src/federated/aggregator.py

import numpy as np
import random
import copy
import logging
from typing import List, Tuple, Dict, Any, Optional

# Placeholder type for model parameters (adjust based on actual model)
ModelParams = List[np.ndarray]

class Aggregator:
    """Coordinates the Federated Learning process."""

    def __init__(self, model_template: Any, num_clients: int, client_ids: List[str]):
        """
        Initializes the Aggregator.

        Args:
            model_template: An instance of the model to get parameter structure.
            num_clients: Total number of clients available.
            client_ids: List of unique client identifiers.
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
        """Selects a subset of clients randomly for the current round."""
        if num_to_select >= self.num_clients:
            logging.debug("Selecting all clients.")
            return list(range(self.num_clients))
        else:
            selected_indices = random.sample(range(self.num_clients), num_to_select)
            logging.info(f"Round {self.current_round}: Selected {len(selected_indices)} clients: {[self.client_ids[i] for i in selected_indices]}")
            return selected_indices

    def get_global_parameters(self) -> ModelParams:
        """Returns a deep copy of the current global model parameters."""
        return copy.deepcopy(self.global_model_params)

    def aggregate_updates(self,
                         client_updates: List[Tuple[ModelParams, int]],
                         num_total_clients: int,
                         dp_params: Optional[Dict[str, float]] = None) -> None:
        """
        Aggregates client updates (clipped), adds Gaussian noise for Central DP
        if dp_params provided, and updates the global model.

        Args:
            client_updates: List of tuples: (clipped_update_params, num_samples).
            num_total_clients (int): Total number of clients (N) for DP calculation.
            dp_params (Optional[Dict]): DP parameters:
                'target_epsilon': Overall target epsilon for the training.
                'target_delta': Overall target delta.
                'clip_norm': Clipping norm (C_clip) used by clients.
                'total_rounds': Total expected training rounds (T).
                 (Used if calculating noise per round, alternative: pass sigma directly)
                 OR 'noise_multiplier': Pre-calculated noise multiplier (sigma/C_clip).
        """
        if not client_updates:
            logging.warning(f"Round {self.current_round}: No client updates received for aggregation.")
            return

        num_selected_clients = len(client_updates)
        total_samples = sum(num_samples for _, num_samples in client_updates)
        if total_samples == 0:
            logging.warning(f"Round {self.current_round}: Total samples from selected clients is zero. Skipping aggregation.")
            return

        logging.info(f"Round {self.current_round}: Aggregating clipped updates from {num_selected_clients} clients (Total samples: {total_samples}).")

        # Diagnostic: log pre-aggregation global param stats
        pre_params = [p.copy() for p in self.global_model_params]

        # 1. Sum Clipped Updates
        summed_update: ModelParams = [np.zeros_like(layer) for layer in self.global_model_params]
        for update_params, _ in client_updates:
            for i in range(len(summed_update)):
                if update_params[i].shape == summed_update[i].shape:
                    summed_update[i] += update_params[i]
                else:
                    logging.error(f"Shape mismatch during update summation! Layer {i}. Skipping this update.")
                    summed_update = [np.zeros_like(layer) for layer in self.global_model_params]
                    break
            else:
                continue
            break

        # 2. Calculate and Add Noise (if DP enabled)
        noise_std_dev = 0.0
        if dp_params and all(k in dp_params for k in ['clip_norm', 'noise_multiplier']):
            clip_norm = dp_params['clip_norm']
            noise_multiplier = dp_params['noise_multiplier']
            noise_std_dev = noise_multiplier * clip_norm
            logging.info(f"Applying DP noise: C={clip_norm:.2f}, noise_multiplier={noise_multiplier:.4f}, std_dev={noise_std_dev:.4f}")
        elif dp_params:
            logging.warning("DP parameters provided but 'noise_multiplier' is missing. Cannot calculate noise. Consider using a DP library (e.g., dp-accounting) to determine multiplier.")
            pass

        # Add noise to the *summed* updates
        noisy_sum = []
        if noise_std_dev > 0:
            for layer_sum in summed_update:
                noise = np.random.normal(loc=0.0, scale=noise_std_dev, size=layer_sum.shape)
                noisy_sum.append(layer_sum + noise)
        else:
            noisy_sum = summed_update

        # 3. Average the (potentially noisy) sum
        average_update = [layer_sum / num_selected_clients for layer_sum in noisy_sum]

        # 4. Update Global Model: w_{t+1} = w_t + average_update
        for i in range(len(self.global_model_params)):
            if self.global_model_params[i].shape == average_update[i].shape:
                self.global_model_params[i] += average_update[i]
            else:
                logging.error(f"Shape mismatch during final model update! Layer {i}. Skipping layer update.")

        logging.debug(f"Round {self.current_round}: Global model updated.")
        # Diagnostic: log post-aggregation global param stats
        for i, (pre, post) in enumerate(zip(pre_params, self.global_model_params)):
            delta = np.linalg.norm(post - pre)
            logging.info(f"Aggregator: Layer {i} param delta after aggregation: {delta:.6e}")
            logging.info(
                f"Aggregator: Layer {i} stats after aggregation: "
                f"mean={np.mean(post):.6e}, std={np.std(post):.6e}, min={np.min(post):.6e}, max={np.max(post):.6e}"
            )
        self.current_round += 1
