# Portfolio Optimizer: maximize Sharpe ratio (PyPortfolioOpt)

## Function: `optimize_portfolio`
- **Location:** `src/portfolio_optimizer.py`

## Purpose
- Solves the mean-variance optimization (MVO) problem to maximize the portfolio Sharpe ratio.
- Uses PyPortfolioOpt's `EfficientFrontier` for robust, industry-standard optimization.

## Inputs
- `mu` (`pd.Series`): Predicted expected returns for each asset (index = asset symbols)
- `Sigma` (`pd.DataFrame`): Covariance matrix of asset returns (index/columns = asset symbols)
- `config` (`dict`): Optional configuration (e.g., risk-free rate, weight bounds)

## Output
- `weights` (`dict`): Optimal portfolio weights for each asset (keys = asset symbols, values = weights)

## Library Used
- [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/)

## Objective
- Maximize the Sharpe ratio (risk-adjusted return)
- Handles symbol alignment, missing data, and robust error handling
