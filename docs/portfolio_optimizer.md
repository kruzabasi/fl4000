# Portfolio Optimizer: maximize Sharpe ratio (MVO)

## Function: `optimize_portfolio`
- **Location:** `src/portfolio_optimizer.py`
- **Objective:** Maximize portfolio Sharpe ratio using Mean-Variance Optimization (MVO).
- **Library:** [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt)

## Inputs
- **mu:** `pd.Series` of predicted expected returns (index = asset symbols).
- **Sigma:** `pd.DataFrame` covariance matrix (index/columns = symbols).
- **config:** Optional `dict` with keys such as `risk_free_rate` and `weight_bounds`.

## Output
- **weights:** `dict` mapping asset symbols to optimal portfolio weights (sums to 1).

## Details
- Uses PyPortfolioOpt's `EfficientFrontier` for optimization.
- Handles missing data and alignment between mu and Sigma.
- Returns `None` if optimization fails or if PyPortfolioOpt is not installed.
- See function docstring for further details.
