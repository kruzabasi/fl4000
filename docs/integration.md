# Integration: FL Predictive Model to Portfolio Optimizer

## Method: `Client.run_local_optimization`
- **Location:** `src/federated/client.py`

## Workflow
1. **Obtain global model** (after FL training).
2. **Predict expected returns (μ):** Use the local FL model to predict T+1 log returns for all assets, given the current feature vector.
3. **Calculate empirical covariance (Σ):** Use recent historical log returns to compute the rolling covariance matrix (empirical or using PyPortfolioOpt risk models).
4. **Call optimizer:** Pass μ and Σ to the portfolio optimizer (`optimize_portfolio`) to get optimal weights.
5. **Receive weights:** Output is a dictionary mapping asset symbols to portfolio weights (sums to 1).

## When to Run
- This workflow is intended for the evaluation phase, after federated learning training has converged and the global model is distributed to clients.
- Each client uses their local data and the final model to compute personalized portfolio allocations for the next period.

## See Also
- `docs/predictive_model.md`
- `docs/portfolio_optimizer.md`
- Implementation in `src/federated/client.py` and `src/portfolio_optimizer.py`
