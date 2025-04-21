import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any

try:
    from pypfopt import EfficientFrontier, objective_functions, risk_models
    HAS_PYPFOpt = True
except ImportError:
    logging.warning("pypfopt not installed. Portfolio optimization functionality will be disabled.")
    HAS_PYPFOpt = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def optimize_portfolio(mu: pd.Series,
                       Sigma: pd.DataFrame,
                       config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, float]]:
    """
    Optimizes portfolio weights to maximize the Sharpe ratio using MVO.

    Args:
        mu (pd.Series): Predicted expected returns (index=symbols).
        Sigma (pd.DataFrame): Covariance matrix (index/columns=symbols).
        config (Optional[Dict]): Configuration dictionary, e.g.,
                                 {'risk_free_rate': 0.01, 'weight_bounds': (0, 1)}.

    Returns:
        Optional[Dict[str, float]]: Dictionary of optimal weights {symbol: weight},
                                    or None if optimization fails or library missing.
    """
    if not HAS_PYPFOpt:
        logging.error("PyPortfolioOpt is required for optimization.")
        return None

    if mu.empty or Sigma.empty:
        logging.warning("Empty expected returns (mu) or covariance matrix (Sigma). Cannot optimize.")
        return None

    # Ensure alignment
    mu_aligned, S_aligned = mu.align(pd.Series(np.diag(Sigma), index=Sigma.columns), join='inner')
    if mu_aligned.empty:
         logging.warning("No common symbols between mu and Sigma after alignment.")
         return None
    S_aligned = Sigma.loc[mu_aligned.index, mu_aligned.index]

    # Default config
    cfg = {
        'risk_free_rate': 0.01,
        'weight_bounds': (0, 1) # Example: Long only, max 100% allocation
    }
    if config:
        cfg.update(config)

    try:
        # Check for positive semi-definite - PyPortfolioOpt often handles small issues
        # if np.linalg.det(S_aligned) < 1e-8: # Check determinant
        #      logging.warning("Covariance matrix may be singular.")
             # Add jitter or use shrinkage if needed

        ef = EfficientFrontier(mu_aligned, S_aligned, weight_bounds=cfg['weight_bounds'])

        # Add regularization (optional, can help stability)
        # ef.add_objective(objective_functions.L2_reg, gamma=0.1)

        # Optimize for max Sharpe
        weights = ef.max_sharpe(risk_free_rate=cfg['risk_free_rate']) # Assumes mu and rf are annualized if Sigma is
        cleaned_weights = ef.clean_weights() # Rounds small weights, ensures sum ~= 1

        logging.debug(f"Optimization successful. Weights: {cleaned_weights}")
        return cleaned_weights

    except ValueError as ve:
        logging.warning(f"Portfolio optimization ValueError: {ve}. Returning None.")
        return None
    except Exception as e:
        logging.error(f"Portfolio optimization failed: {e}")
        return None
