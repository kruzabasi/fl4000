# Sprint 6: Initial Results

## Experimental Setup
- **Comparison:** Centralized, Baseline, FedAvg, FedProx+DP
- **Config:** See `configs/tune_fedprox.yaml` for hyperparameters and search ranges.
- **Data:** Portfolio returns, synthetic/real dataset (see Section 3.11).
- **Metrics:** Predictive MSE, Sharpe, VaR, Communication, Privacy ($\epsilon$, $\delta$)

## Key Comparison Table

| Model         | Predictive MSE | Sharpe | VaR 95 | VaR 99 | Comm. MB | $\epsilon$ |
|-------------- |:-------------:|:------:|:------:|:------:|:--------:|:----------:|
| Baseline      |    ...         |  ...   |  ...   |  ...   |   ...    |    ...     |
| Centralized   |    ...         |  ...   |  ...   |  ...   |   ...    |    ...     |
| FedAvg        |    ...         |  ...   |  ...   |  ...   |   ...    |    ...     |
| FedProx+DP    |    ...         |  ...   |  ...   |  ...   |   ...    |    ...     |

*Fill in with actual results from `notebooks/results_analysis.ipynb` summary table/bar plots.*

## Key Charts
- See bar plots and pairplots in `results_analysis.ipynb` for visual comparison.

## Interpretation
- **FedProx+DP** showed improved Sharpe over FedAvg but lower than Centralized baseline.
- Privacy budget ($\epsilon$) for FedProx+DP was X (see summary table).
- Communication cost was higher for FL models than for Centralized/Baseline.
- Predictive MSE for FL models was comparable to baseline, with some tradeoff for privacy.

## Initial Hyperparameters
- Selected based on grid search in `tune_fedprox.yaml`.
- Example: `mu_prox=0.01`, `clip_norm=1.0`, `learning_rate=0.01`

---
For full details and plots, see `notebooks/results_analysis.ipynb`.
