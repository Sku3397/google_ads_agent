import unittest
import numpy as np
from unittest.mock import MagicMock
from google.ads.googleads.client import GoogleAdsClient
from services.reinforcement_learning_service import ReinforcementLearningService

class TestReinforcementLearningService(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_client = MagicMock(spec=GoogleAdsClient)
        self.customer_id = "1234567890"
        self.service = ReinforcementLearningService(self.mock_client, self.customer_id)

    def test_initialization(self):
        """Test that the service initializes correctly."""
        self.assertIsNotNone(self.service.model)
        self.assertEqual(self.service.epsilon, 1.0)
        self.assertEqual(self.service.epsilon_min, 0.01)
        self.assertEqual(self.service.epsilon_decay, 0.995)
        self.assertEqual(self.service.gamma, 0.95)
        self.assertEqual(self.service.batch_size, 32)
        self.assertEqual(self.service.max_memory_size, 10000)
        self.assertEqual(len(self.service.memory), 0)

    def test_build_model(self):
        """Test that the model is built with the correct architecture."""
        model = self.service._build_model()
        self.assertEqual(len(model.layers), 3)
        self.assertEqual(model.layers[0].units, 64)
        self.assertEqual(model.layers[1].units, 64)
        self.assertEqual(model.layers[2].units, 3)
        self.assertEqual(model.layers[0].activation.__name__, 'relu')
        self.assertEqual(model.layers[1].activation.__name__, 'relu')
        self.assertEqual(model.layers[2].activation.__name__, 'linear')

    def test_get_action_explore(self):
        """Test that get_action returns a random action when exploring."""
        state = np.zeros((1, 10))
        self.service.epsilon = 1.0  # Force exploration
        action = self.service.get_action(state)
        self.assertTrue(0 <= action <= 2)

    def test_get_action_exploit(self):
        """Test that get_action returns the best action when exploiting."""
        state = np.zeros((1, 10))
        self.service.epsilon = 0.0  # Force exploitation
        # Mock model prediction to return a specific action
        self.service.model.predict = MagicMock(return_value=np.array([[0.1, 0.2, 0.7]]))
        action = self.service.get_action(state)
        self.assertEqual(action, 2)

    def test_store_experience(self):
        """Test that experiences are stored correctly in memory."""
        state = np.zeros((1, 10))
        action = 1
        reward = 10.0
        next_state = np.ones((1, 10))
        done = False
        self.service.store_experience(state, action, reward, next_state, done)
        self.assertEqual(len(self.service.memory), 1)
        self.assertEqual(self.service.memory[0], (state, action, reward, next_state, done))

    def test_store_experience_max_memory(self):
        """Test that memory does not exceed max size."""
        state = np.zeros((1, 10))
        action = 1
        reward = 10.0
        next_state = np.ones((1, 10))
        done = False
        self.service.max_memory_size = 2
        for i in range(3):
            self.service.store_experience(state, action, reward, next_state, done)
        self.assertEqual(len(self.service.memory), 2)

    def test_train_policy_empty_data(self):
        """Test training policy with empty historical data."""
        result = self.service.train_policy([])
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['message'], 'Policy training completed')

    def test_simulate_strategy(self):
        """Test simulating a strategy for a campaign."""
        campaign_id = "12345"
        simulation_params = {}
        result = self.service.simulate_strategy(campaign_id, simulation_params)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['message'], f"Simulation completed for campaign {campaign_id}")
        self.assertEqual(result['results']['campaign_id'], campaign_id)

    def test_safe_deploy_policy(self):
        """Test safe deployment of a policy to a campaign."""
        campaign_id = "12345"
        result = self.service.safe_deploy_policy(campaign_id)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['message'], f"Policy deployed to campaign {campaign_id} with epsilon-greedy exploration")
        self.assertEqual(result['campaign_id'], campaign_id)

if __name__ == '__main__':
    unittest.main() 