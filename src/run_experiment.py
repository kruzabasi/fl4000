import argparse
import yaml # Requires PyYAML: pip install pyyaml
import json
import pandas as pd
import numpy as np
import logging
import os
import time
import random
import pickle
from typing import Dict, Any, List # Added List for type hinting
import datetime
import traceback
import time as time_module
from filelock import FileLock

# Project Modules (adjust paths/imports as necessary)
try:
    import config # Assuming config.py is in src/
    from federated.simulation import run_fl_simulation
    from evaluation.metrics import (calculate_portfolio_metrics, calculate_predictive_metrics,
                                     calculate_communication_cost, get_privacy_cost)
    # Import the predictive model used in the simulation
    from models.predictive_model import PortfolioPredictiveModel # Assuming this is the model class
    # Import baseline trainer if running baselines via harness (optional)
    # from baseline_trainer import main as run_baseline_main # Keep commented unless baseline running is integrated
except ImportError as e:
     logging.error(f"Failed to import required project modules: {e}")
     # Consider adding more specific error handling or guidance
     exit(1)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads experiment configuration from a YAML file."""
    logging.info(f"Loading configuration from: {config_path}")
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found at {config_path}")
    try:
        with open(config_path, 'r') as f:
            exp_config = yaml.safe_load(f)
        if not isinstance(exp_config, dict):
            raise ValueError("Configuration file is not a valid YAML dictionary.")
        return exp_config
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML config file {config_path}: {e}")
        raise # Re-raise YAML parsing errors
    except Exception as e:
        logging.error(f"Error loading config file {config_path}: {e}")
        raise # Re-raise other potential errors


def estimate_update_size(model_params: List[np.ndarray]) -> float:
     """Estimates the size of model parameters in bytes (simple example)."""
     # This is a basic estimation. Real-world size might depend on serialization format.
     if not model_params:
         return 0.0
     total_bytes = sum(p.nbytes for p in model_params if isinstance(p, np.ndarray))
     # Add overhead estimate if needed (e.g., for metadata)
     return float(total_bytes)


def append_to_master_log(row_dict, log_path):
    lock_path = log_path + '.lock'
    from pathlib import Path
    import os
    import pandas as pd
    Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)
    with FileLock(lock_path):
        # Check if file exists and if it is empty
        write_header = False
        if not os.path.exists(log_path):
            write_header = True
        else:
            try:
                if os.path.getsize(log_path) == 0:
                    write_header = True
            except Exception:
                write_header = True
        df = pd.DataFrame([row_dict])
        df.to_csv(log_path, mode='a', header=write_header, index=False)


def run_experiment(exp_config: Dict[str, Any]):
    """Runs a single experiment based on the configuration."""
    # Extract top-level config
    experiment_id = exp_config.get("experiment_id", "unnamed_exp")
    random_seed = exp_config.get("random_seed", 42)
    risk_free_rate = exp_config.get("risk_free_rate", 0.0)
    non_iid_params = exp_config.get("non_iid_params", {})
    fl_params = exp_config.get("fl_params", {})
    model_params = exp_config.get("model_params", {})
    runs = exp_config.get("runs", {})
    results_dir = getattr(__import__('config'), 'RESULTS_DIR', 'data/results')
    master_full_log = os.path.join(results_dir, 'experiments_full_log.csv')
    master_comp_log = os.path.join(results_dir, 'experiments_comparison_log.csv')

    # Run each requested FL config (e.g., fedprox_dp_run, fedavg_run)
    for run_name, run_cfg in runs.items():
        logging.info(f"--- Running {run_name} ---")
        this_model_params = model_params.copy()
        this_model_params.update({k: v for k, v in run_cfg.items() if k in ["mu_prox", "model_type"]})
        dp_params = run_cfg.get("dp_params", None)
        this_seed = random_seed if isinstance(random_seed, int) else int(time.time())
        logging.info(f"[EXPERIMENT] Using random seed: {this_seed} for run {run_name}")
        start_time = time.time()
        run_status = "success"
        error_message = ""
        sim_result = None
        try:
            sim_result = run_fl_simulation(
                fl_params=fl_params,
                dp_params=dp_params,
                model_params=this_model_params,
                non_iid_params=non_iid_params,
                seed=this_seed,
                experiment_id=experiment_id,
                results_dir=os.path.join(results_dir, "FL")
            )
        except Exception as e:
            run_status = "failure"
            error_message = str(e)
            logging.error(f"Experiment {experiment_id} failed: {e}")
        elapsed_time = time.time() - start_time
        # Prepare log row (minimal example, expand as needed)
        row = {
            'experiment_id': experiment_id,
            'random_seed': exp_config.get('random_seed', ''),
            'model_type': run_cfg.get('model_type', ''),
            'run_status': run_status,
            'error_message': error_message,
            'elapsed_time_sec': elapsed_time,
            'final_epsilon': sim_result.get('final_epsilon') if sim_result else None,
            'final_delta': sim_result.get('final_delta') if sim_result else None,
            'mse': sim_result['metrics'].get('mse') if sim_result and 'metrics' in sim_result else None,
            'r2': sim_result['metrics'].get('r2') if sim_result and 'metrics' in sim_result else None,
            'sharpe': sim_result['metrics'].get('sharpe') if sim_result and 'metrics' in sim_result else None,
            'max_drawdown': sim_result['metrics'].get('max_drawdown') if sim_result and 'metrics' in sim_result else None,
            'var_95': sim_result['metrics'].get('var_95') if sim_result and 'metrics' in sim_result else None,
            'var_99': sim_result['metrics'].get('var_99') if sim_result and 'metrics' in sim_result else None,
            'cvar_95': sim_result['metrics'].get('cvar_95') if sim_result and 'metrics' in sim_result else None,
            'metric_communication_total_MB_uploaded': sim_result.get('metric_communication_total_MB_uploaded') if sim_result else None,
            'convergence_rounds': sim_result.get('convergence_rounds') if sim_result else None,
        }
        # Optionally, update with any additional metrics
        if sim_result and 'metrics' in sim_result:
            for k, v in sim_result['metrics'].items():
                if k not in row:
                    row[k] = v
        # Append to master logs (thread/process safe)
        append_to_master_log(row, master_comp_log)
        # Append full config/results to master_full_log
        append_to_master_log({**row, **exp_config}, master_full_log)


# --- Command Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Federated Learning Experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    exp_config = load_config(args.config)
    run_experiment(exp_config)
