# Predictive Model: MultiOutput Ridge Regression

## Architecture
- **Model:** MultiOutputRegressor wrapping Ridge regression (`sklearn.multioutput.MultiOutputRegressor`, `sklearn.linear_model.Ridge`).
- **Purpose:** Predicts expected T+1 log returns for all assets in a single step.
- **Rationale:**
  - Linear baseline model for interpretability and benchmarking.
  - Supports multiple outputs (assets) simultaneously, matching the portfolio prediction use case.
  - Efficient and robust for federated learning experiments.

## Inputs
- **Feature Matrix:** Scaled numpy array or DataFrame of shape `(n_samples, n_features)`.
  - Features should be numeric and appropriately preprocessed/scaled.

## Outputs
- **Predicted Returns Matrix:** Numpy array of shape `(n_samples, n_assets)`.
  - Each row contains predicted T+1 log returns for all assets.

## Loss Function
- **Mean Squared Error (MSE):**
  - The model is trained to minimize the average MSE across all outputs (assets).

## Usage
- See `src/models/predictive_model.py` for implementation details and API.
