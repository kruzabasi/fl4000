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
    results_dir = getattr(config, "RESULTS_DIR", "results")

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
                experiment_id=f"{experiment_id}_{run_name}",
                results_dir=results_dir
            )
        except Exception as e:
            run_status = "failure"
            error_message = str(e)
            logging.error(f"Experiment {experiment_id}_{run_name} failed: {error_message}")
            traceback.print_exc()
        elapsed_time = time.time() - start_time
        if sim_result is not None:
            if sim_result.get('final_epsilon') is not None:
                logging.info(f"{run_name}: Final epsilon={sim_result['final_epsilon']} delta={sim_result['final_delta']}")
            logging.info(f"--- Experiment {experiment_id}_{run_name} Finished ---")
            if 'metrics' in sim_result:
                logging.info(f"Final Metrics: {sim_result['metrics']}")
            else:
                logging.info("No metrics found in simulation result.")
        results_full_log_path = os.path.join(results_dir, 'experiments_full_log.csv')
        flat_result = {}
        flat_result.update(exp_config)
        flat_result['experiment_id'] = f"{experiment_id}_{run_name}"
        flat_result['model_type'] = this_model_params.get('model_type', 'unknown')
        flat_result['run_status'] = run_status
        flat_result['error_message'] = error_message
        flat_result['elapsed_time_sec'] = elapsed_time
        if sim_result is not None:
            flat_result['final_epsilon'] = sim_result.get('final_epsilon', None)
            flat_result['final_delta'] = sim_result.get('final_delta', None)
            if 'metrics' in sim_result:
                for metric_type, metrics_dict in sim_result['metrics'].items():
                    if isinstance(metrics_dict, dict):
                        for k, v in metrics_dict.items():
                            flat_result[f"{metric_type}_{k}"] = v
                    else:
                        flat_result[metric_type] = metrics_dict
        results_df = pd.DataFrame([flat_result])
        if os.path.exists(results_full_log_path):
            results_df.to_csv(results_full_log_path, mode='a', header=False, index=False)
        else:
            results_df.to_csv(results_full_log_path, mode='w', header=True, index=False)
        logging.info(f"Full config results logged to {results_full_log_path}")
        results_comp_log_path = os.path.join(results_dir, 'experiments_comparison_log.csv')
        comp_row = {
            'experiment_id': f"{experiment_id}_{run_name}",
            'model_type': this_model_params.get('model_type', 'unknown'),
            'run_status': run_status,
            'error_message': error_message,
            'elapsed_time_sec': elapsed_time,
            'final_delta': flat_result.get('final_delta', None),
        }
        metric_map = {
            'Sharpe': None,
            'Max Drawdown': None,
            'VaR_95': None,
            'CVaR_95': None
        }
        metric_key_map = {
            'Sharpe': 'sharpe',
            'Max Drawdown': 'max_drawdown',
            'VaR_95': 'var_95',
            'CVaR_95': 'cvar_95',
        }
        if sim_result is not None and 'metrics' in sim_result:
            metrics_dict = sim_result['metrics']
            for csv_key, metric_key in metric_key_map.items():
                v = None
                for k in metrics_dict:
                    if k.lower() == metric_key:
                        v = metrics_dict[k]
                        break
                if v is None:
                    for mval in metrics_dict.values():
                        if isinstance(mval, dict):
                            for k2 in mval:
                                if k2.lower() == metric_key:
                                    v = mval[k2]
                                    break
                        if v is not None:
                            break
                metric_map[csv_key] = v
        comp_row.update(metric_map)
        # --- Write header if file does not exist or is empty ---
        write_header = not os.path.exists(results_comp_log_path) or os.path.getsize(results_comp_log_path) == 0
        comp_columns = [
            'experiment_id', 'model_type', 'run_status', 'error_message', 'elapsed_time_sec',
            'final_delta', 'Sharpe', 'Max Drawdown', 'VaR_95', 'CVaR_95'
        ]
        comp_row_ordered = {col: comp_row.get(col, None) for col in comp_columns}
        comp_df = pd.DataFrame([comp_row_ordered])
        comp_df = comp_df[comp_columns]  # Ensure column order
        comp_df.to_csv(results_comp_log_path, mode='a', header=write_header, index=False)
        logging.info(f"Comparison metrics logged to {results_comp_log_path}")


# --- Command Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Federated Learning Experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    exp_config = load_config(args.config)
    run_experiment(exp_config)
