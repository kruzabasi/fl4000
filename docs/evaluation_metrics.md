# Evaluation Metrics

This document summarizes the evaluation metrics used in the project, their calculation, and relevance. For detailed methodology, see Section 3.11 of the Methodology document.

## Predictive Metrics

### Mean Squared Error (MSE)
- **Calculation:**
    \[ \text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 \]
- **Function:** `calculate_predictive_metrics` in `src/evaluation/metrics.py`
- **Relevance:** Measures the average squared difference between predicted and true values; lower is better. Used to assess model prediction accuracy.

### R-squared (R2)
- **Calculation:**
    \[ R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} \]
- **Function:** `calculate_predictive_metrics`
- **Relevance:** Proportion of variance explained by the model. Higher values indicate better fit.

## Portfolio Metrics

### Sharpe Ratio
- **Calculation:**
    \[ \text{Sharpe} = \frac{\mathbb{E}[R_p - R_f]}{\sigma_p} \]
    where $R_p$ is portfolio return, $R_f$ is risk-free rate, $\sigma_p$ is return std. dev.
- **Function:** `calculate_portfolio_metrics`
- **Relevance:** Measures risk-adjusted return; higher is better.

### Maximum Drawdown
- **Calculation:** Largest observed loss from a peak to a trough of a portfolio.
- **Function:** `calculate_portfolio_metrics`
- **Relevance:** Indicates worst-case loss; lower is safer.

### Value at Risk (VaR 95, VaR 99)
- **Calculation:** 5th/1st percentile of return distribution (i.e., loss not exceeded with 95%/99% confidence).
- **Function:** `calculate_portfolio_metrics`
- **Relevance:** Measures tail risk.

### Conditional Value at Risk (CVaR 95)
- **Calculation:** Average loss in worst 5% of cases.
- **Function:** `calculate_portfolio_metrics`
- **Relevance:** Captures expected loss in extreme scenarios.

## Communication Cost

### Total MB Uploaded
- **Calculation:**
    \[ \text{Total MB} = \frac{\text{update size (bytes)} \times \text{clients per round} \times \text{num rounds}}{1024^2} \]
- **Function:** `calculate_communication_cost`
- **Relevance:** Quantifies communication overhead in FL.

## Privacy Metrics

### Epsilon (ε) and Delta (δ)
- **Calculation:** Computed using DP accountant for a given $\delta$.
- **Function:** `get_privacy_cost`
- **Relevance:** Lower epsilon means stronger privacy. See Section 3.11 for DP background.

---

For implementation details, see docstrings in `src/evaluation/metrics.py`.
