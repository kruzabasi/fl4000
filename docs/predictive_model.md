# Predictive Model: MultiOutput Ridge Regression

## Architecture
- **Model:** MultiOutput Ridge Regression (scikit-learn)
- **Rationale:**
  - Provides a simple, interpretable linear baseline for federated learning (FL) simulations.
  - Supports simultaneous prediction of multiple asset returns (multi-output).
  - Well-suited for initial FL experiments due to stability and low risk of overfitting.

## Input/Output
- **Input:** Scaled feature matrix (`X`), shape `(n_samples, n_features)`
- **Output:** Predicted T+1 log returns matrix (`y_pred`), shape `(n_samples, n_assets)`

## Loss Function
- **Loss:** Mean Squared Error (MSE)
- **Definition:** Average squared difference between predicted and true T+1 log returns, averaged across all assets and samples.

## Usage Context
- Used as the client-side predictive model in the federated learning framework.
- Predicts the expected return (`μ`) for each asset for the next period, which is then used for portfolio optimization.
