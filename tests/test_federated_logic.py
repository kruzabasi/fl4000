# tests/test_federated_logic.py

import unittest
from src.federated.aggregator import Aggregator
from src.federated.client import Client
from src.federated.simulation import Simulation
from src.federated.utils import get_model_parameters, set_model_parameters
from src.models.placeholder import PlaceholderModel

class TestFederatedLogic(unittest.TestCase):
    def test_aggregator_exists(self):
        self.assertIsInstance(Aggregator(), Aggregator)

    def test_client_exists(self):
        self.assertIsInstance(Client(), Client)

    def test_simulation_exists(self):
        self.assertIsInstance(Simulation(), Simulation)

    def test_placeholder_model(self):
        model = PlaceholderModel()
        self.assertIsInstance(model, PlaceholderModel)

if __name__ == "__main__":
    unittest.main()
