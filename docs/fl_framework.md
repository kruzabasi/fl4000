# Federated Learning Framework: FedProx Simulation

## Overview
This framework implements a simulation of Federated Learning (FL) using the FedProx algorithm. It is designed for research and experimentation with federated model training, client heterogeneity, and aggregation strategies. The simulation orchestrates multiple clients, each training locally on their own data, and a central aggregator that coordinates rounds of model update and aggregation.

## Components
### Aggregator
- Maintains the global model parameters.
- Selects clients for each round.
- Aggregates client updates using weighted averaging (FedAvg principle).

### Client
- Loads and manages its own (partitioned) data.
- Receives global model parameters and trains locally using the FedProx logic.
- Computes and returns model updates to the aggregator.

### PlaceholderLinearModel
- A wrapper around scikit-learn's SGDRegressor.
- Supports parameter extraction, setting, and manual update for FedProx.
- Used as the default model for experimentation.

## Simulation Loop (src/federated/simulation.py)
1. **Initialization**: Loads sample client data to determine feature dimensions. Initializes the global model, aggregator, and all clients.
2. **Round Loop** (for each round):
    - Selects a subset of clients for participation.
    - Each selected client receives the current global model, trains locally (using FedProx), and computes its update.
    - The aggregator collects all updates and performs weighted aggregation to update the global model.
    - Optionally, results and model state are saved after each round.

## FedProx Implementation
- The proximal term logic is implemented in `Client.train()`. For each local batch, the gradient of the proximal term (`mu * (w - w^t)`) is added to the standard gradient.
- Since scikit-learn's SGDRegressor does not natively support FedProx, the update is performed manually by extracting parameters, computing the combined gradient, and updating the model weights accordingly.

## Configuration
Key simulation parameters (can be set in `src/federated/simulation.py` or `config.py`):
- `TOTAL_ROUNDS`: Number of FL rounds to run.
- `CLIENTS_PER_ROUND`: Number of clients selected per round.
- `LOCAL_EPOCHS`: Number of local epochs per client per round.
- `BATCH_SIZE`: Batch size for local training.
- `LEARNING_RATE`: Learning rate for local updates.
- `MU_PROX`: Proximal term coefficient for FedProx.
- Data paths and random seeds are also configurable.

## Running the Simulation
1. Ensure all dependencies are installed and data is partitioned for each client.
2. Adjust configuration parameters as needed in `src/federated/simulation.py` or `config.py`.
3. Run the simulation:

```bash
.venv/bin/python src/federated/simulation.py
```

4. Results (history CSV, global model parameters) will be saved to the configured results directory.

---

*Expand this document as the implementation progresses or as new features are added.*
