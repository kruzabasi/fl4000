import pytest
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
import sys
import random
from unittest.mock import MagicMock, patch

# --- Add src directory to Python path ---
# This assumes your tests directory is parallel to your src directory
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, src_path)

# Adjust imports based on actual structure
from federated.aggregator import Aggregator
from federated.client import Client # Import Client to potentially test prox grad
from models.placeholder import PlaceholderLinearModel

# --- Fixtures ---
@pytest.fixture
def mock_model_template():
    """Creates a mock model template with parameter structure."""
    template = PlaceholderLinearModel(n_features=2)
    # Initialize parameters to non-zero for testing updates
    template.set_parameters([np.array([0.1, 0.2]), np.array([0.5])])
    return template

@pytest.fixture
def simple_aggregator(mock_model_template):
    """Creates a simple Aggregator instance."""
    return Aggregator(mock_model_template, num_clients=2, client_ids=['client_0', 'client_1'])

# --- Tests ---
def test_aggregator_initialization(simple_aggregator, mock_model_template):
    """Test if aggregator initializes with correct parameters."""
    assert simple_aggregator.num_clients == 2
    assert simple_aggregator.current_round == 0
    initial_global_params = simple_aggregator.get_global_parameters()
    expected_initial_params = mock_model_template.get_parameters()
    assert len(initial_global_params) == len(expected_initial_params)
    for layer_idx in range(len(initial_global_params)):
        np.testing.assert_array_equal(initial_global_params[layer_idx], expected_initial_params[layer_idx])

def test_aggregator_select_clients(simple_aggregator):
    """Test client selection logic."""
    # Select 1 client
    selected = simple_aggregator.select_clients(num_to_select=1)
    assert len(selected) == 1
    assert selected[0] in [0, 1]

    # Select all clients
    selected_all = simple_aggregator.select_clients(num_to_select=2)
    assert len(selected_all) == 2
    assert set(selected_all) == {0, 1}

    # Select more than available
    selected_over = simple_aggregator.select_clients(num_to_select=3)
    assert len(selected_over) == 2
    assert set(selected_over) == {0, 1}


def test_aggregator_weighted_average(simple_aggregator, mock_model_template):
    """Test the weighted averaging aggregation."""
    initial_params = mock_model_template.get_parameters() # [array([0.1, 0.2]), array([0.5])]

    # Mock client updates (update = local_params - initial_params)
    # Ensure updates and initial_params use 2D arrays for weights (shape (1, 2))
    print('DEBUG initial_params shapes:', [p.shape for p in initial_params])
    update0 = [np.array([[0.1, 0.1]]), np.array([0.1])]  # weights: shape (1, 2), bias: shape (1,)
    print('DEBUG update0 shapes:', [u.shape for u in update0])
    update1 = [np.array([[-0.1, -0.1]]), np.array([0.2])]  # weights: shape (1, 2), bias: shape (1,)
    print('DEBUG update1 shapes:', [u.shape for u in update1])
    samples0 = 100
    samples1 = 300

    client_updates = [(update0, samples0), (update1, samples1)]

    # Calculate expected aggregated update (sample-weighted average)
    total_samples = samples0 + samples1
    expected_agg_update_coef = (update0[0] * samples0 + update1[0] * samples1) / total_samples
    expected_agg_update_int = (update0[1] * samples0 + update1[1] * samples1) / total_samples
    expected_agg_update = [expected_agg_update_coef, expected_agg_update_int]

    # Calculate expected final global params: initial_params + expected_agg_update
    expected_final_coef = initial_params[0] + expected_agg_update[0]
    expected_final_int = initial_params[1] + expected_agg_update[1]

    simple_aggregator.aggregate_updates(client_updates, num_total_clients=2)
    final_global_params = simple_aggregator.get_global_parameters()

    # Assert
    assert len(final_global_params) == 2
    np.testing.assert_allclose(final_global_params[0], expected_final_coef, rtol=1e-5)
    np.testing.assert_allclose(final_global_params[1], expected_final_int, rtol=1e-5)
    assert simple_aggregator.current_round == 1 # Round should advance

# Test for proximal gradient calculation (if Client.train logic is complex enough)
# def test_client_proximal_gradient_calculation():
#     # Setup mock client and model
#     # Define w and w_t parameters
#     # Define mu_prox
#     # Call a hypothetical function _calculate_proximal_gradient(w, w_t, mu_prox)
#     # Assert expected_gradient == mu_prox * (w - w_t)
#     pass

# Add to FILE: tests/test_federated_logic.py

@pytest.fixture
def mock_client_data():
    """Creates dummy data files for mock clients with all canonical FTSE 100 features."""
    temp_dir = tempfile.mkdtemp()
    client_dirs = {}
    # Canonical FTSE 100 features
    canon_features = [
        'adjusted_close', 'close', 'dividend_amount', 'high', 'low', 'open', 'split_coefficient',
        'volume', 'sma_5', 'sma_20', 'sma_60', 'volatility_20', 'day_of_week', 'month', 'quarter',
        'log_return_lag_1', 'log_return_lag_2', 'log_return_lag_3', 'log_return_lag_5', 'log_return_lag_10',
        'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5', 'volume_lag_10',
        'adjusted_close_lag_1', 'adjusted_close_lag_2', 'adjusted_close_lag_3', 'adjusted_close_lag_5',
        'adjusted_close_lag_10', 'rsi', 'macd', 'macd_signal', 'macd_diff', 'obv'
    ]
    id_cols = ['symbol', 'timestamp']
    target_col = 'log_return'
    n_features = len(canon_features)
    for i in range(2): # Create data for 2 mock clients
        client_id = f"client_{i}"
        client_path = os.path.join(temp_dir, client_id)
        os.makedirs(client_path)
        # Create dummy data
        dates = pd.date_range(start='2023-01-01', periods=10)
        data = {feat: np.random.rand(10) for feat in canon_features}
        data['symbol'] = [f'SYM{i+1}']*10
        data['timestamp'] = dates
        data['log_return'] = np.random.randn(10)*0.01
        data['target_log_return'] = np.random.randn(10)*0.01
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        df.to_parquet(os.path.join(client_path, 'client_data.parquet'))
        client_dirs[client_id] = client_path
    # Yield base temp dir and n_features
    yield temp_dir, n_features
    # Teardown: remove temporary directory
    shutil.rmtree(temp_dir)

@patch('models.placeholder.PlaceholderLinearModel')
def test_simulation_completes_rounds(mock_placeholder_model, mock_model_template, mock_client_data):
    """Test if the main simulation loop runs for a few rounds."""
    import federated.simulation as simulation  # Import here to patch config before use
    temp_data_dir, n_features = mock_client_data
    import importlib
    import federated.simulation
    import federated
    import config
    # Patch config.RESULTS_DIR before simulation runs
    temp_results_dir = tempfile.mkdtemp()
    config.RESULTS_DIR = temp_results_dir
    federated.simulation.config.RESULTS_DIR = temp_results_dir

    # Configure the mock config object
    federated.simulation.config.FEDERATED_DATA_DIR = temp_data_dir
    federated.simulation.config.NUM_CLIENTS = 2
    federated.simulation.config.RANDOM_SEED = 42

    # Patch FL_PARAMS with required keys in the config module used by federated.simulation
    federated.simulation.config.FL_PARAMS = {
        'clients_per_round': 1,
        'total_rounds': 2,
        # Add any other required FL params here
    }
    federated.simulation.config.EXPERIMENT_ID = "fedprox"

    # Mock model template needs correct feature number
    mock_placeholder_model.return_value = PlaceholderLinearModel(n_features=n_features, random_state=42)

    # Configure simulation parameters for a short run
    original_rounds = simulation.TOTAL_ROUNDS
    original_clients_per_round = simulation.CLIENTS_PER_ROUND
    simulation.TOTAL_ROUNDS = 2
    simulation.CLIENTS_PER_ROUND = 1 # Select 1 client per round

    try:
        # Run the simulation
        simulation.run_simulation()

        # Basic assertions: Check if it completed and if model likely changed
        assert os.path.exists(os.path.join(temp_results_dir, 'fedprox_run_history.csv'))
    finally:
        # Restore original values
        simulation.TOTAL_ROUNDS = original_rounds
        simulation.CLIENTS_PER_ROUND = original_clients_per_round
        shutil.rmtree(temp_results_dir)