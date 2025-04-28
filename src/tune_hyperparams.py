import os
import yaml
import itertools
import pandas as pd
import numpy as np
import logging
import importlib
from copy import deepcopy

from federated.simulation import run_fl_simulation
from evaluation.metrics import calculate_predictive_metrics, calculate_portfolio_metrics


def load_yaml_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def grid_search_param_combinations(param_ranges):
    keys = list(param_ranges.keys())
    values = [param_ranges[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tune FL/DP hyperparameters via grid search.")
    parser.add_argument('--config', type=str, required=True, help='Path to tuning YAML config')
    parser.add_argument('--results_csv', type=str, default='tuning_results.csv', help='Where to save tuning results')
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    param_ranges = config['parameter_ranges']
    validation_metric = config['validation_metric']
    validation_set_path = config['validation_set_path']
    base_fl_params = deepcopy(config['fl_params'])
    base_model_params = deepcopy(config.get('model_params', {}))
    dp_params = deepcopy(config.get('dp_params', {}))
    seed = config.get('random_seed', 42)
    experiment_base_id = config.get('experiment_base_id', 'TuningRun')

    # Load validation set
    val_df = pd.read_parquet(validation_set_path)
    # Use canonized feature list and target
    CANONICAL_FEATURES = [
        'adjusted_close', 'close', 'dividend_amount', 'high', 'low', 'open', 'split_coefficient', 'volume', 'log_return',
        'sma_5', 'sma_20', 'sma_60', 'volatility_20', 'day_of_week', 'month', 'quarter',
        'log_return_lag_1', 'log_return_lag_2', 'log_return_lag_3', 'log_return_lag_5', 'log_return_lag_10',
        'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5', 'volume_lag_10',
        'adjusted_close_lag_1', 'adjusted_close_lag_2', 'adjusted_close_lag_3', 'adjusted_close_lag_5', 'adjusted_close_lag_10',
        'rsi', 'macd', 'macd_signal', 'macd_diff', 'obv'
    ]
    CANONICAL_FEATURES = [f for f in CANONICAL_FEATURES if f not in ['log_return', 'symbol', 'timestamp']]
    target_col = 'log_return'
    id_cols = ['symbol', 'timestamp']
    # Remove target and ID columns if present
    drop_cols = [c for c in [target_col] + id_cols if c in val_df.columns]
    X_val = val_df.drop(columns=drop_cols)
    # --- Canonical feature enforcement for validation set ---
    missing = [f for f in CANONICAL_FEATURES if f not in X_val.columns]
    if missing:
        raise ValueError(f"Validation set missing canonical features: {missing}")
    X_val = X_val[CANONICAL_FEATURES]
    y_val = val_df[target_col].values.flatten()
    # Ensure y_val is 1D to match model prediction output

    # Diagnostic logging for validation features and targets
    logging.info(f"Validation X stats -- mean: {np.mean(X_val.values, axis=0)}, std: {np.std(X_val.values, axis=0)}, min: {np.min(X_val.values, axis=0)}, max: {np.max(X_val.values, axis=0)}")
    logging.info(f"Validation y stats -- mean: {np.mean(y_val)}, std: {np.std(y_val)}, min: {np.min(y_val)}, max: {np.max(y_val)}")

    # Standardize validation features using the first client's scaler (assume all clients use the same canonical features)
    from federated.client import Client
    if hasattr(Client, 'scaler') and Client.scaler is not None:
        X_val_scaled = Client.scaler.transform(X_val.values)
    else:
        # Fallback: fit scaler on validation (not ideal, but prevents crash)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_val_scaled = scaler.fit_transform(X_val.values)
        logging.warning("No client scaler found, fitting StandardScaler on validation set.")
    logging.info(f"Validation X stats (scaled) -- mean: {np.mean(X_val_scaled, axis=0)}, std: {np.std(X_val_scaled, axis=0)}")
    X_val = X_val_scaled

    results = []
    best_metric = None
    best_params = None

    for param_combo in grid_search_param_combinations(param_ranges):
        fl_params = deepcopy(base_fl_params)
        model_params = deepcopy(base_model_params)
        dp_params_run = deepcopy(dp_params)

        # Inject tunable params into correct dicts
        for k, v in param_combo.items():
            if k in fl_params:
                fl_params[k] = v
            elif k in model_params:
                model_params[k] = v
            elif k in dp_params_run:
                dp_params_run[k] = v
            else:
                # Try to infer based on name
                if k == 'mu_prox':
                    fl_params['mu_prox'] = v
                elif k == 'learning_rate':
                    fl_params['learning_rate'] = v
                elif k == 'clip_norm':
                    dp_params_run['clip_norm'] = v
                else:
                    fl_params[k] = v  # fallback

        experiment_id = f"{experiment_base_id}_" + "_".join(f"{k}{v}" for k,v in param_combo.items())
        print(f"\n=== Running: {experiment_id} ===")
        logging.info(f"Running experiment with params: {param_combo}")
        sim_result = run_fl_simulation(fl_params, dp_params_run, model_params, None, seed, experiment_id=experiment_id)
        if sim_result is None:
            logging.warning(f"Simulation failed for params: {param_combo}")
            metric_score = np.nan
        else:
            # Load global model (assume returned or saved as sim_result['global_model'])
            global_model = sim_result.get('global_model') if isinstance(sim_result, dict) else None
            if global_model is None:
                logging.warning(f"No global model returned for params: {param_combo}")
                metric_score = np.nan
            else:
                if validation_metric == 'predictive_mse':
                    # global_model is a model object; use its predict method
                    y_pred = global_model.predict(X_val)
                    metric_score = calculate_predictive_metrics(y_val, y_pred)['mse']
                    # After evaluation, print a few predictions and targets for inspection
                    try:
                        print("Sample predictions vs. targets:")
                        for i in range(min(5, len(y_val))):
                            print(f"y_pred: {y_pred[i]:.6f}, y_true: {y_val[i]:.6f}")
                    except Exception as e:
                        print(f"Error printing predictions/targets: {e}")
                elif validation_metric == 'portfolio_sharpe':
                    # TODO: Implement portfolio evaluation logic
                    metric_score = calculate_portfolio_metrics(global_model, X_val, y_val)['sharpe']
                else:
                    raise ValueError(f"Unknown validation metric: {validation_metric}")
        results.append({**param_combo, validation_metric: metric_score})
        print(f"Params: {param_combo} -> {validation_metric}: {metric_score}")
        if best_metric is None or (not np.isnan(metric_score) and (
            (validation_metric == 'predictive_mse' and metric_score < best_metric) or
            (validation_metric == 'portfolio_sharpe' and metric_score > best_metric)
        )):
            best_metric = metric_score
            best_params = deepcopy(param_combo)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.results_csv, index=False)
    print(f"\n=== Hyperparameter Tuning Complete ===")
    print(f"Best params: {best_params}")
    print(f"Best {validation_metric}: {best_metric}")
    print(f"All results saved to: {args.results_csv}")

if __name__ == "__main__":
    main()
