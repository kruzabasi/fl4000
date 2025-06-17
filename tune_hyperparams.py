import itertools
import yaml
import copy
import pandas as pd
import numpy as np
import os
import logging
import sys

# Ensure src/ is in the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.run_experiment import run_experiment, load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for FL/DP experiments")
    parser.add_argument('--config', type=str, required=True, help='Path to tuning config YAML')
    parser.add_argument('--output', type=str, default='tuning_results.csv', help='CSV to log results')
    args = parser.parse_args()

    config = load_config(args.config)
    param_grid = config.get('parameter_ranges', {})
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = []
    for i, combo in enumerate(combinations):
        logging.info(f"\n=== Tuning Run {i+1}/{len(combinations)}: {combo} ===")
        exp_config = copy.deepcopy(config)
        exp_config['experiment_id'] = f"{config['experiment_base_id']}_tune_{i+1}"
        # Set FL/DP params
        exp_config['fl_params'] = exp_config.get('fl_params', {})
        exp_config['fl_params']['learning_rate'] = combo['learning_rate']
        exp_config['fl_params']['prox_mu'] = combo['mu_prox']
        exp_config['dp_params'] = exp_config.get('dp_params', {})
        exp_config['dp_params']['clip_norm'] = combo['clip_norm']
        # Run the experiment
        try:
            result = run_experiment(exp_config)
            # Evaluate on validation set if possible
            val_metric = np.nan
            if result and 'metrics' in result and 'predictive' in result['metrics']:
                if config.get('validation_metric') == 'predictive_mse':
                    val_metric = result['metrics']['predictive'].get('mse', np.nan)
                # Add more metrics as needed
            row = {**combo, 'val_metric': val_metric}
            results.append(row)
            logging.info(f"Validation {config['validation_metric']}: {val_metric}")
        except Exception as e:
            logging.error(f"Tuning run failed for {combo}: {e}")
            row = {**combo, 'val_metric': np.nan, 'error': str(e)}
            results.append(row)
    # Save all results
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    logging.info(f"Tuning results saved to {args.output}")

if __name__ == "__main__":
    main()
