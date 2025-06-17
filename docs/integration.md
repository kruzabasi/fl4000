# FL Client Integration: Predictive Model to Portfolio Optimization

## Method: `Client.run_local_optimization`
- **Location:** `src/federated/client.py`

## Workflow
1. **Receive global model parameters** (from FL server after training round/convergence)
2. **Predict expected returns (μ):** Use the local predictive model to forecast T+1 log returns for all assets, given the most recent features.
3. **Calculate empirical covariance (Σ):** Compute rolling or sample covariance from recent historical log returns (using PyPortfolioOpt if available).
4. **Run portfolio optimizer:** Call `optimize_portfolio` to solve for optimal asset weights, maximizing the Sharpe ratio.
5. **Return optimal weights:** Output is a dictionary mapping asset symbols to portfolio weights.

## When to Use
- This workflow is intended for the **evaluation phase** (post-FL training), not during local model training.
- Used to simulate how each client would construct their portfolio using the final global model and current/historical market data.
