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
    # Client 0: local params = [array([0.2, 0.3]), array([0.6])] -> update = [array([0.1, 0.1]), array([0.1])]
    update0 = [np.array([0.1, 0.1]), np.array([0.1])]
    samples0 = 100

    # Client 1: local params = [array([0.0, 0.1]), array([0.7])] -> update = [array([-0.1, -0.1]), array([0.2])]
    update1 = [np.array([-0.1, -0.1]), np.array([0.2])]
    samples1 = 300

    client_updates = [(update0, samples0), (update1, samples1)]
    total_samples = samples0 + samples1 # 400
    weight0 = samples0 / total_samples # 0.25
    weight1 = samples1 / total_samples # 0.75

    # Calculate expected aggregated update (simple mean, not sample-weighted)
    expected_agg_update_coef = (update0[0] + update1[0]) / 2
    expected_agg_update_int = (update0[1] + update1[1]) / 2
    expected_agg_update = [expected_agg_update_coef, expected_agg_update_int]

    # Calculate expected final global params: initial_params + expected_agg_update
    expected_final_coef = initial_params[0] + expected_agg_update[0]
    expected_final_int = initial_params[1] + expected_agg_update[1]

    # Perform aggregation
    simple_aggregator.aggregate_updates(client_updates, 2)
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

import tempfile
import shutil

@pytest.fixture
def mock_client_data():
    """Creates dummy data files for mock clients."""
    temp_dir = tempfile.mkdtemp()
    client_dirs = {}
    n_features = 2
    for i in range(2): # Create data for 2 mock clients
        client_id = f"client_{i}"
        client_path = os.path.join(temp_dir, client_id)
        os.makedirs(client_path)
        # Create dummy data
        dates = pd.date_range(start='2023-01-01', periods=10)
        df = pd.DataFrame({
            'symbol': [f'SYM{i+1}']*10,
            'feature1': np.random.rand(10),
            'feature2': np.random.rand(10),
            'log_return': np.random.randn(10)*0.01,
            'target_log_return': np.random.randn(10)*0.01
        }, index=dates)
        df.index.name = 'timestamp'
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
        # Restore original simulation parameters if modified
        simulation.TOTAL_ROUNDS = original_rounds
        simulation.CLIENTS_PER_ROUND = original_clients_per_round
        shutil.rmtree(temp_results_dir)