# Simulation Harness Usage (`src/run_experiment.py`)

This script runs federated learning and baseline experiments using a configurable YAML file.

## Usage

```bash
.venv/bin/python src/run_experiment.py --config configs/tune_fedprox.yaml --output_dir data/results/
```

- `--config`: Path to the YAML configuration file specifying experiment parameters.
- `--output_dir`: Directory to save experiment logs and results.
- `--seed`: (Optional) Random seed for reproducibility.

## Configuration File Format (YAML)

See `configs/tune_fedprox.yaml` for an example. Key sections:

- `experiment_base_id`: String identifier for the experiment batch.
- `random_seed`: Seed for reproducibility.
- `model_type`: Model/algorithm to run (e.g., FedProxDP, FedAvg, Baseline).
- `fl_params`: Federated learning parameters (rounds, clients, epochs, batch size, etc).
- `model_params`: Model-specific hyperparameters.
- `parameter_ranges`: (Optional) Grid search ranges for hyperparameter tuning.
- `validation_metric`: Metric to optimize during tuning.
- `validation_set_path`: Path to validation data.

Example:
```yaml
experiment_base_id: "TuneFedProxDP"
random_seed: 123
model_type: "FedProxDP"
fl_params:
  total_rounds: 5
  clients_per_round: 10
  local_epochs: 5
  batch_size: 64
model_params:
  alpha: 1.0
parameter_ranges:
  mu_prox: [0.001, 0.01, 0.1, 1.0]
  learning_rate: [0.1, 0.01, 0.001]
  clip_norm: [0.5, 1.0, 5.0]
validation_metric: 'predictive_mse'
validation_set_path: 'data/processed/central_validation_set.parquet'
```

## Script Docstring

```
"""
Run Federated Learning and Baseline Experiments.

Usage:
    .venv/bin/python src/run_experiment.py --config <config.yaml> --output_dir <results_dir> [--seed <int>]

Arguments:
    --config: Path to YAML config file.
    --output_dir: Directory for outputs.
    --seed: (Optional) Random seed.

The config file controls all aspects of the experiment, including FL/DP/model hyperparameters and tuning ranges.
Results are logged in CSV format for downstream analysis.
"""
```

---
For more details, see the docstring at the top of `src/run_experiment.py` and the example YAML in `configs/`.
