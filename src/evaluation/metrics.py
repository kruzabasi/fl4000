import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional

try:
    import quantstats as qs
    HAS_QUANTSTATS = True
except ImportError:
    logging.warning("quantstats not installed. Portfolio metrics will be limited.")
    HAS_QUANTSTATS = False

try:
    from sklearn.metrics import mean_squared_error, r2_score
except ImportError:
     logging.error("scikit-learn not installed. Predictive metrics cannot be calculated.")
     # Define dummy functions if needed
     def mean_squared_error(y_true, y_pred): return np.nan
     def r2_score(y_true, y_pred): return np.nan


# Assumes privacy accountant object structure from dp-accounting library
try:
    # Adjust based on the specific accountant class used (e.g., RdpAccountant)
    from dp_accounting.privacy_accountant import NeighboringRelation
    from dp_accounting.rdp import rdp_privacy_accountant # Example
    PrivacyAccountant = rdp_privacy_accountant.RdpAccountant
except ImportError:
     logging.warning("dp-accounting not installed. Privacy cost cannot be calculated.")
     PrivacyAccountant = None # Placeholder type


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_predictive_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate predictive metrics: MSE, R2.

    Args:
        y_true (array-like): Ground truth values.
        y_pred (array-like): Predicted values.
    Returns:
        dict: {'mse': float, 'r2': float}
    """
    metrics = {}
    try:
        # Ensure inputs are numpy arrays
        y_true_np = np.asarray(y_true)
        y_pred_np = np.asarray(y_pred)

        if y_true_np.shape != y_pred_np.shape:
             raise ValueError(f"Shape mismatch: y_true {y_true_np.shape}, y_pred {y_pred_np.shape}")
        if len(y_true_np) == 0:
             logging.warning("Empty arrays provided for predictive metrics.")
             return {'mse': np.nan, 'r2': np.nan}

        metrics['mse'] = mean_squared_error(y_true_np, y_pred_np)
        metrics['r2'] = r2_score(y_true_np, y_pred_np)
        logging.info(f"Calculated predictive metrics: MSE={metrics['mse']:.6f}, R2={metrics['r2']:.4f}")
    except Exception as e:
        logging.error(f"Error calculating predictive metrics: {e}")
        metrics['mse'] = np.nan
        metrics['r2'] = np.nan
    return metrics


def calculate_portfolio_metrics(portfolio_returns: pd.Series, risk_free_rate: float = 0.01, required_metrics: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Calculate portfolio evaluation metrics: Sharpe, Max Drawdown, VaR, CVaR.

    Args:
        portfolio_returns (array-like): Portfolio returns.
        risk_free_rate (float): Risk-free rate for Sharpe ratio.
        required_metrics (Optional[List[str]]): Specific metrics to calculate
                                                (e.g., ['sharpe', 'max_drawdown', 'var_95', 'var_99', 'cvar_95']).
                                                If None, calculates all listed default metrics.
    Returns:
        dict: {'sharpe': float, 'max_drawdown': float, 'var_95': float, 'var_99': float, 'cvar_95': float}
    """
    metrics = {}
    if not HAS_QUANTSTATS:
        logging.warning("Quantstats needed for portfolio metrics.")
        return {'error': 'quantstats not installed'}

    # Define default metrics including VaR 99%
    default_metrics = ['sharpe', 'max_drawdown', 'var_95', 'var_99', 'cvar_95']
    default_nan_return = {m: np.nan for m in default_metrics} # Use keys from default_metrics

    if portfolio_returns.empty or portfolio_returns.isnull().all():
        logging.warning("Portfolio returns series is empty or all NaN.")
        return default_nan_return

    # Use default metrics if none specified
    if required_metrics is None:
        required_metrics = default_metrics

    try:
        if 'sharpe' in required_metrics:
            metrics['sharpe'] = qs.stats.sharpe(portfolio_returns, rf=risk_free_rate, annualize=True)
        if 'max_drawdown' in required_metrics:
            metrics['max_drawdown'] = qs.stats.max_drawdown(portfolio_returns)
        if 'var_95' in required_metrics:
            # Patch: quantstats uses 'sigma' not 'confidence_value' for qs.stats.var, and pandas uses neither
            # Use quantstats API: qs.stats.var(series, sigma=0.05) for 95% VaR
            metrics['var_95'] = qs.stats.var(portfolio_returns, sigma=0.05)
        if 'var_99' in required_metrics: # Added VaR 99% calculation
            metrics['var_99'] = qs.stats.var(portfolio_returns, sigma=0.01)
        if 'cvar_95' in required_metrics:
            metrics['cvar_95'] = qs.stats.cvar(portfolio_returns, sigma=0.05) # Expected Shortfall

        logging.info(f"Calculated portfolio metrics: { {k: f'{v:.4f}' for k, v in metrics.items()} }")
    except Exception as e:
        logging.error(f"Error calculating portfolio metrics: {e}")
        # Return NaNs for metrics that failed or weren't calculated
        for m in required_metrics:
            metric_key = m # Handle var_95, var_99, cvar_95 keys correctly
            if m == 'var' and 'var_95' not in metrics: metric_key = 'var_95'
            if m == 'var' and 'var_99' not in metrics: metric_key = 'var_99' # Check for var_99
            if m == 'cvar' and 'cvar_95' not in metrics: metric_key = 'cvar_95'

            if metric_key not in metrics:
                 metrics[metric_key] = np.nan
        # Ensure all keys from default_nan_return exist, even if not required, setting to NaN if missing
        for key in default_nan_return:
            if key not in metrics:
                metrics[key] = np.nan


    return metrics


def calculate_communication_cost(update_size_bytes: float, clients_per_round: int, num_rounds: int) -> Dict[str, float]:
    """
    Calculate total communication cost in MB.

    Args:
        update_size_bytes (int): Size of one update in bytes.
        clients_per_round (int): Number of clients per round.
        num_rounds (int): Number of rounds.
    Returns:
        dict: {'total_MB_uploaded': float, 'num_rounds': float}
    """
    # Simplistic model: assumes only client->server upload cost dominates
    # Does not account for model broadcast size (server->client) or ACKs.
    # Add download cost if needed: cost += model_size_bytes * num_rounds
    total_bytes = update_size_bytes * clients_per_round * num_rounds
    total_mb = total_bytes / (1024 * 1024)
    metrics = {'total_MB_uploaded': total_mb, 'num_rounds': float(num_rounds)}
    logging.info(f"Estimated communication cost: {total_mb:.2f} MB uploaded over {num_rounds} rounds.")
    return metrics


def get_privacy_cost(accountant: Optional[Any], target_delta: float) -> Dict[str, float]:
    """
    Compute (epsilon, delta) for DP accounting.

    Args:
        accountant: The dp-accounting object used during simulation.
        target_delta (float): The target delta for which to calculate epsilon.
    Returns:
        dict: {'epsilon': float, 'delta': float}
    """
    if accountant is None or PrivacyAccountant is None:
        logging.warning("Privacy accountant not available. Cannot report privacy cost.")
        return {'epsilon': np.nan, 'delta': target_delta}
    if not isinstance(accountant, PrivacyAccountant):
         logging.warning(f"Invalid accountant type: {type(accountant)}. Cannot get privacy cost.")
         return {'epsilon': np.nan, 'delta': target_delta}

    try:
        epsilon = accountant.get_epsilon(target_delta)
        logging.info(f"Calculated privacy cost: Epsilon={epsilon:.4f} for Delta={target_delta:.1E}")
        return {'epsilon': epsilon, 'delta': target_delta}
    except Exception as e:
        logging.error(f"Error getting privacy cost from accountant: {e}")
        return {'epsilon': np.nan, 'delta': target_delta}


def determine_convergence(metrics_history: List[Dict[str, Any]],
                          metric_key: str = 'validation_loss', # Or e.g., validation_sharpe
                          tolerance: float = 1e-4,
                          patience: int = 5) -> Optional[int]:
    """
    Checks if a metric has stabilized based on tolerance and patience.

    Args:
        metrics_history (List[Dict]): List of metric dictionaries, one per round.
                                      Each dict should contain the metric_key.
        metric_key (str): The key of the metric to monitor for convergence.
        tolerance (float): The minimum change considered significant.
        patience (int): The number of consecutive rounds the metric must stabilize for.

    Returns:
        Optional[int]: The round number where convergence was detected, or None.
    """
    if len(metrics_history) < patience + 1:
        return None # Not enough history

    recent_metrics = [m.get(metric_key) for m in metrics_history[-patience-1:]]

    # Check if metric exists and is numeric in recent history
    if any(m is None or not isinstance(m, (int, float)) for m in recent_metrics):
         # logging.debug(f"Metric '{metric_key}' missing or non-numeric in recent history.")
         return None

    # Check if absolute changes are within tolerance for 'patience' rounds
    changes = np.abs(np.diff(recent_metrics)) # Changes between consecutive rounds
    if np.all(changes < tolerance):
        # Ensure 'round' key exists, otherwise estimate based on list length
        convergence_round = metrics_history[-patience-1].get('round', len(metrics_history) - patience)
        logging.info(f"Convergence detected at round {convergence_round} based on '{metric_key}'.")
        return convergence_round
    else:
        return None
