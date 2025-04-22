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
    from federated.simulation import run_simulation # Assume run_simulation is refactored/callable
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

# Check for dp-accounting library availability
try:
    from dp_accounting.accountant import GaussianAccountant # Example accountant
    from dp_accounting.dp_event import DpEvent, SampledDpEvent, GaussianDpEvent # Example events
    ACCOUNTING_LIB_AVAILABLE = True
except ImportError:
    logging.warning("dp-accounting library not installed. Privacy cost calculation will be skipped for DP models.")
    ACCOUNTING_LIB_AVAILABLE = False
    # Define placeholder classes if needed to avoid runtime errors later
    class GaussianAccountant: pass
    class DpEvent: pass
    class SampledDpEvent: pass
    class GaussianDpEvent: pass


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
    exp_id = exp_config.get('experiment_id', f"exp_{int(time_module.time())}")
    model_type = exp_config.get('model_type', 'FedProxDP').lower() # Default to FedProxDP
    seed = exp_config.get('random_seed', 42)
    results_dir = getattr(config, 'RESULTS_DIR', 'results') # Use results dir from config or default
    os.makedirs(results_dir, exist_ok=True) # Ensure results directory exists

    results = {'config': exp_config, 'metrics': {}} # Store results

    logging.info(f"--- Starting Experiment: {exp_id} ---")
    logging.info(f"Model Type: {model_type}")
    logging.info(f"Random Seed: {seed}")
    logging.debug(f"Full Configuration: {json.dumps(exp_config, indent=2)}")

    start_time = time_module.time()

    # --- Set Seeds ---
    random.seed(seed)
    np.random.seed(seed)
    # Add torch/tf seeds if applicable and using those libraries
    # try:
    #     import torch
    #     torch.manual_seed(seed)
    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed_all(seed)
    # except ImportError:
    #     pass
    # try:
    #     import tensorflow as tf
    #     tf.random.set_seed(seed)
    # except ImportError:
    #     pass

    # --- Load Data ---
    # Data loading/partitioning logic should be handled here or called from here.
    # This depends heavily on how data is managed (e.g., pre-partitioned files, dynamic partitioning).
    # For now, assume data is available/loaded within the simulation or baseline functions.
    logging.info("Data loading/partitioning step assumed to be handled within simulation/baseline.")

    # --- Run Training/Loading ---
    final_model_params = None # Store final global model params if applicable
    history = None # Store training history (e.g., loss per round)
    accountant = None # Store privacy accountant object for DP models
    test_predictions = None # Store predictions on the test set (pd.DataFrame expected)
    portfolio_returns = None # Store calculated portfolio returns (pd.Series expected)

    if model_type == 'centralized':
        logging.info("Running/Loading Centralized Baseline...")
        # Option 1: Run baseline_trainer directly if integrated
        # try:
        #     baseline_args = exp_config.get('baseline_params', {}) # Pass relevant params
        #     # Assuming run_baseline_main returns paths or results dict
        #     baseline_results = run_baseline_main(**baseline_args)
        #     # Extract metrics, predictions, returns from baseline_results
        #     # ...
        # except Exception as e:
        #     logging.exception(f"Failed to run baseline trainer: {e}")

        # Option 2: Load pre-calculated results (as in the example)
        try:
            # Define expected paths based on config/conventions
            preds_path = os.path.join(results_dir, exp_config.get('baseline_preds_file', 'baseline_predictions.csv'))
            metrics_path = os.path.join(results_dir, exp_config.get('baseline_metrics_file', 'baseline_portfolio_metrics.csv'))
            returns_path = os.path.join(results_dir, exp_config.get('baseline_returns_file', 'baseline_portfolio_returns.csv')) # Added returns path

            if not os.path.exists(metrics_path) or not os.path.exists(preds_path) or not os.path.exists(returns_path):
                 logging.warning(f"Baseline result files not found in {results_dir}. Skipping baseline loading.")
                 raise FileNotFoundError("Baseline result files missing.")

            logging.info(f"Loading baseline metrics from: {metrics_path}")
            portfolio_metrics_df = pd.read_csv(metrics_path, index_col=0)
            # Select the first row/best model assumed saved by baseline_trainer
            results['metrics']['portfolio'] = portfolio_metrics_df.iloc[0].to_dict()

            logging.info(f"Loading baseline predictions from: {preds_path}")
            test_predictions = pd.read_csv(preds_path, index_col=0, parse_dates=True)
            # Assuming baseline saved 'actual' and one prediction column
            pred_col = [c for c in test_predictions.columns if c.endswith('_pred')]
            if not pred_col: raise ValueError("No prediction column found in baseline predictions file.")
            results['metrics']['predictive'] = calculate_predictive_metrics(
                test_predictions['actual'].values, test_predictions[pred_col[0]].values
            )

            logging.info(f"Loading baseline portfolio returns from: {returns_path}")
            portfolio_returns = pd.read_csv(returns_path, index_col=0, parse_dates=True).squeeze("columns") # Load as Series

            # Set placeholders for non-applicable metrics
            results['metrics']['communication'] = {'total_MB_uploaded': 0.0, 'num_rounds': 0}
            results['metrics']['privacy'] = {'epsilon': float('inf'), 'delta': 1.0} # No privacy for centralized

        except FileNotFoundError:
             # Handle missing files gracefully if loading pre-calculated results
             results['metrics']['portfolio'] = {'sharpe': np.nan, 'var_95': np.nan, 'var_99': np.nan} # Use keys from metrics.py
             results['metrics']['predictive'] = {'mse': np.nan, 'r2': np.nan}
             results['metrics']['communication'] = {'total_MB_uploaded': 0.0, 'num_rounds': 0}
             results['metrics']['privacy'] = {'epsilon': float('inf'), 'delta': 1.0}
        except Exception as e:
             logging.exception(f"Failed to load or process baseline results: {e}")
             # Assign NaNs or defaults if loading fails unexpectedly
             results['metrics']['portfolio'] = {'sharpe': np.nan, 'var_95': np.nan, 'var_99': np.nan}
             results['metrics']['predictive'] = {'mse': np.nan, 'r2': np.nan}
             results['metrics']['communication'] = {'total_MB_uploaded': 0.0, 'num_rounds': 0}
             results['metrics']['privacy'] = {'epsilon': float('inf'), 'delta': 1.0}


    elif model_type in ['fedproxdp', 'fedavg', 'localonly']: # Handle FL models
        logging.info(f"Running Federated Learning Simulation: {model_type}...")
        try:
            # --- Actual FL Simulation Call ---
            # This assumes federated.simulation.run_simulation is refactored to:
            # 1. Accept the experiment configuration dictionary (exp_config)
            # 2. Return a dictionary containing:
            #    'model_params': final global model parameters (list of numpy arrays)
            #    'history': training history (e.g., pandas DataFrame with loss/metrics per round)
            #    'accountant': the privacy accountant object (if DP is used)
            #    'predictions': pandas DataFrame with 'actual' and 'prediction' columns on test set
            #    'portfolio_returns': pandas Series of portfolio returns calculated from predictions

            logging.info("Calling run_simulation function...")
            simulation_results = run_simulation(exp_config) # Pass the whole config

            # --- Extract results from simulation ---
            final_model_params = simulation_results.get('model_params')
            history = simulation_results.get('history')
            accountant = simulation_results.get('accountant')
            test_predictions = simulation_results.get('predictions') # DataFrame: index=time, cols=['actual', 'prediction', ...]
            portfolio_returns = simulation_results.get('portfolio_returns') # Series: index=time, values=returns

            if test_predictions is None or portfolio_returns is None or final_model_params is None:
                 logging.warning("Simulation did not return all expected outputs (model_params, predictions, portfolio_returns). Evaluation might be incomplete.")
                 # Handle missing outputs gracefully if possible

            # --- Evaluation Post-FL Run ---
            if test_predictions is not None and 'actual' in test_predictions.columns and 'prediction' in test_predictions.columns:
                 results['metrics']['predictive'] = calculate_predictive_metrics(
                      test_predictions['actual'].values, test_predictions['prediction'].values
                 )
            else:
                 logging.warning("Predictive metrics could not be calculated due to missing prediction data.")
                 results['metrics']['predictive'] = {'mse': np.nan, 'r2': np.nan}

            if portfolio_returns is not None:
                 results['metrics']['portfolio'] = calculate_portfolio_metrics(
                      portfolio_returns, risk_free_rate=exp_config.get('risk_free_rate', 0.01)
                 )
            else:
                 logging.warning("Portfolio metrics could not be calculated due to missing returns data.")
                 results['metrics']['portfolio'] = {'sharpe': np.nan, 'var_95': np.nan, 'var_99': np.nan} # Use keys from metrics.py

            # Estimate comm cost
            if final_model_params is not None:
                update_size = estimate_update_size(final_model_params) # Estimate size
                # Get FL specific params from config
                fl_params = exp_config.get('fl_params', {})
                clients_round = fl_params.get('clients_per_round', getattr(config, 'NUM_CLIENTS', 10)) # Default if not in config
                num_rounds = fl_params.get('total_rounds', 100) # Default if not in config
                results['metrics']['communication'] = calculate_communication_cost(update_size, clients_round, num_rounds)
            else:
                logging.warning("Communication cost could not be estimated due to missing model parameters.")
                results['metrics']['communication'] = {'total_MB_uploaded': np.nan, 'num_rounds': np.nan}


            # Get privacy cost
            if model_type == 'fedproxdp':
                 if ACCOUNTING_LIB_AVAILABLE and accountant is not None:
                     dp_params = exp_config.get('dp_params', {})
                     target_delta = dp_params.get('target_delta', 1e-5) # Default delta
                     results['metrics']['privacy'] = get_privacy_cost(accountant, target_delta)
                 else:
                     logging.warning("Privacy cost calculation skipped for FedProxDP (dp-accounting unavailable or accountant object missing).")
                     results['metrics']['privacy'] = {'epsilon': np.nan, 'delta': np.nan}
            else: # FedAvg / LocalOnly = no privacy cost calculated by design
                 results['metrics']['privacy'] = {'epsilon': float('inf'), 'delta': 1.0}


        except Exception as e:
            logging.exception(f"FL Simulation or evaluation failed for {model_type}: {e}")
            # Assign NaNs for all metrics in case of failure
            results['metrics'] = {'predictive': {'mse': np.nan, 'r2': np.nan},
                                 'portfolio': {'sharpe': np.nan, 'var_95': np.nan, 'var_99': np.nan}, # Use keys from metrics.py
                                 'communication': {'total_MB_uploaded': np.nan, 'num_rounds': np.nan},
                                 'privacy': {'epsilon': np.nan, 'delta': np.nan}}

    else:
        logging.error(f"Unknown model_type specified in config: {model_type}")
        return None # Indicate failure for this experiment run

    logging.info(f"--- Experiment {exp_id} Finished ---")
    # Log only the metrics part for brevity, full config is logged at start
    logging.info(f"Final Metrics: {json.dumps(results.get('metrics', {}), indent=2)}")

    # --- Log results ---
    results_log_path = os.path.join(results_dir, exp_config.get('log_file', 'experiments_log.csv'))
    # Define fixed schema columns
    expected_columns = [
        "experiment_id", "model_type", "timestamp", "config_total_rounds", "config_clients_per_round", "config_local_epochs", "config_prox_mu", "config_noise_multiplier", "config_clip_norm", "config_target_epsilon", "config_target_delta", "config_random_seed", "metric_predictive_mse", "metric_predictive_r2", "metric_portfolio_sharpe", "metric_portfolio_var_95", "metric_portfolio_var_99", "metric_communication_total_MB_uploaded", "metric_communication_num_rounds", "metric_privacy_epsilon", "metric_privacy_delta"
    ]
    # Flatten results for CSV logging
    flat_result = {}
    # Add identifying info
    flat_result['experiment_id'] = exp_id
    flat_result['model_type'] = model_type
    flat_result['timestamp'] = pd.Timestamp.now()

    # Add config parameters (selectively, or all)
    fl_params = exp_config.get('fl_params', {})
    dp_params = exp_config.get('dp_params', {})
    flat_result['config_total_rounds'] = fl_params.get('total_rounds')
    flat_result['config_clients_per_round'] = fl_params.get('clients_per_round')
    flat_result['config_local_epochs'] = fl_params.get('local_epochs')
    flat_result['config_prox_mu'] = fl_params.get('prox_mu')
    flat_result['config_noise_multiplier'] = dp_params.get('noise_multiplier')
    flat_result['config_clip_norm'] = dp_params.get('clip_norm')
    flat_result['config_target_epsilon'] = dp_params.get('target_epsilon')
    flat_result['config_target_delta'] = dp_params.get('target_delta')
    flat_result['config_random_seed'] = seed

    # Flatten nested metrics
    for metric_type, metrics_dict in results.get('metrics', {}).items():
         if isinstance(metrics_dict, dict):
             for k, v in metrics_dict.items():
                 flat_result[f"metric_{metric_type}_{k}"] = v
         else:
             flat_result[f"metric_{metric_type}"] = metrics_dict

    # Enforce fixed schema: fill missing keys with empty string, drop extras
    row_data = [flat_result.get(col, "") for col in expected_columns]
    results_df = pd.DataFrame([row_data], columns=expected_columns)

    try:
        if os.path.exists(results_log_path):
             results_df.to_csv(results_log_path, mode='a', header=False, index=False)
        else:
             results_df.to_csv(results_log_path, mode='w', header=True, index=False)
        logging.info(f"Results logged to {results_log_path}")
    except Exception as e:
        logging.error(f"Failed to log results to CSV {results_log_path}: {e}")

    # --- Enhanced logging ---
    end_time = time_module.time()
    elapsed = end_time - start_time
    run_summary = {
        'experiment_id': exp_id,
        'timestamp': datetime.datetime.now().isoformat(),
        'status': 'success',
        'elapsed_seconds': elapsed,
        'final_privacy_epsilon': flat_result.get('metric_privacy_epsilon', None),
        'final_privacy_delta': flat_result.get('metric_privacy_delta', None),
        'error': None
    }
    logging.info(f"Experiment summary: {run_summary}")
    if 'error' in flat_result:
        run_summary['status'] = 'fail'
        run_summary['error'] = flat_result['error']
        logging.error(f"Experiment failed: {run_summary}")

    # --- Optional: Save detailed results (history, predictions) ---
    save_details = exp_config.get('save_detailed_results', False)
    if save_details:
        details_dir = os.path.join(results_dir, exp_id)
        os.makedirs(details_dir, exist_ok=True)
        try:
            if history is not None:
                history.to_csv(os.path.join(details_dir, 'training_history.csv'), index=False)
            if test_predictions is not None:
                test_predictions.to_csv(os.path.join(details_dir, 'test_predictions.csv'))
            if portfolio_returns is not None:
                portfolio_returns.to_csv(os.path.join(details_dir, 'portfolio_returns.csv'))
            # Save final model params (e.g., using pickle or framework-specific save)
            if final_model_params is not None:
                 with open(os.path.join(details_dir, 'final_model_params.pkl'), 'wb') as f:
                     pickle.dump(final_model_params, f)
            logging.info(f"Detailed results saved to {details_dir}")
        except Exception as e:
            logging.error(f"Failed to save detailed results for {exp_id}: {e}")

    return results


# --- Command Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Federated Learning Experiments")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the experiment YAML configuration file.")
    parser.add_argument("--run", type=str, default=None,
                        help="Optional: Specify which run key in the config to execute (e.g., fedprox_dp_run, fedavg_run)")
    # Add optional arguments, e.g., for overriding config values
    # parser.add_argument("--output-dir", type=str, help="Override results directory")

    args = parser.parse_args()

    try:
        config_data = load_config(args.config)

        # If the config has a 'runs' dict and a --run argument, select that sub-config
        if 'runs' in config_data and args.run:
            if args.run in config_data['runs']:
                run_conf = config_data.copy()
                run_conf.update(config_data['runs'][args.run])
                run_conf['experiment_id'] = f"{config_data.get('experiment_id', 'exp')}_{args.run}_{int(time_module.time())}"
                logging.info(f"Running experiment for run: {args.run}")
                run_experiment(run_conf)
                logging.info("\n--- Experiment Completed ---")
            else:
                logging.error(f"Run key '{args.run}' not found in config 'runs'. Available: {list(config_data['runs'].keys())}")
                exit(1)
        # Existing multi-experiment logic
        elif isinstance(config_data.get('experiments'), list):
            logging.info(f"Found {len(config_data['experiments'])} experiments defined in the config file.")
            all_results = []
            for i, exp_conf in enumerate(config_data['experiments']):
                logging.info(f"\n--- Running Experiment {i+1} of {len(config_data['experiments'])} ---")
                if 'experiment_id' not in exp_conf:
                    base_id = os.path.splitext(os.path.basename(args.config))[0]
                    exp_conf['experiment_id'] = f"{base_id}_run_{i+1}_{int(time_module.time())}"
                exp_result = run_experiment(exp_conf)
                all_results.append(exp_result)
            logging.info("\n--- All Experiments Completed ---")
        else: # Single experiment
            logging.info("Running single experiment defined in the config file.")
            if 'experiment_id' not in config_data:
                base_id = os.path.splitext(os.path.basename(args.config))[0]
                config_data['experiment_id'] = f"{base_id}_{int(time_module.time())}"
            run_experiment(config_data)
            logging.info("\n--- Experiment Completed ---")

    except FileNotFoundError as e:
        logging.error(f"Configuration file error: {e}")
        exit(1)
    except (ValueError, yaml.YAMLError) as e:
        logging.error(f"Configuration file format error: {e}")
        exit(1)
    except Exception as e:
        logging.exception(f"An unexpected error occurred during experiment execution: {e}")
        exit(1)
