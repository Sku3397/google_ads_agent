"""
Tests for Self Play Service

This module contains tests for the Self Play Service, which implements
agent vs agent competitive simulations to discover optimal bidding strategies.
"""

import os
import json
import unittest
from unittest.mock import patch, MagicMock, Mock
import numpy as np
import pandas as pd
from datetime import datetime

# Import the service to test
from services.self_play_service import SelfPlayService
from services.reinforcement_learning_service import ReinforcementLearningService, AdsEnvironment


class TestSelfPlayService(unittest.TestCase):
    """Test cases for SelfPlayService"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a mock ads_api
        self.mock_ads_api = MagicMock()

        # Create a mock optimizer
        self.mock_optimizer = MagicMock()

        # Create a mock logger
        self.mock_logger = MagicMock()

        # Test configuration
        self.test_config = {
            "model_save_path": "test_models/self_play",
            "algorithm": "pbt",
            "population_size": 5,
            "tournament_size": 2,
            "elitism_count": 1,
            "mutation_rate": 0.2,
            "historical_data_days": 7,
        }

        # Create the service instance
        self.service = SelfPlayService(
            ads_api=self.mock_ads_api,
            optimizer=self.mock_optimizer,
            config=self.test_config,
            logger=self.mock_logger,
        )

        # Mock RL service
        self.mock_rl_service = MagicMock(spec=ReinforcementLearningService)

        # Create test directory if it doesn't exist
        if not os.path.exists("test_models/self_play"):
            os.makedirs("test_models/self_play", exist_ok=True)

    def tearDown(self):
        """Clean up after tests"""
        # Remove test files
        if os.path.exists("test_models/self_play/agent_population.json"):
            os.remove("test_models/self_play/agent_population.json")

    def test_initialization(self):
        """Test that the service initializes correctly"""
        self.assertEqual(self.service.algorithm, "pbt")
        self.assertEqual(self.service.population_size, 5)
        self.assertEqual(self.service.tournament_size, 2)
        self.assertEqual(self.service.elitism_count, 1)
        self.assertEqual(self.service.mutation_rate, 0.2)
        self.assertEqual(len(self.service.agent_population), 0)
        self.assertEqual(len(self.service.tournament_history), 0)

    def test_initialize_rl_service(self):
        """Test initializing the RL service reference"""
        self.service.initialize_rl_service(self.mock_rl_service)
        self.assertEqual(self.service.rl_service, self.mock_rl_service)

    def test_generate_agent_hyperparameters(self):
        """Test generating agent hyperparameters"""
        hyperparams = self.service._generate_agent_hyperparameters()

        # Check that required keys exist
        self.assertIn("learning_rate", hyperparams)
        self.assertIn("gamma", hyperparams)
        self.assertIn("hidden_layers", hyperparams)
        self.assertIn("batch_size", hyperparams)
        self.assertIn("reward_weights", hyperparams)

        # Check value ranges
        self.assertTrue(0.0005 <= hyperparams["learning_rate"] <= 0.005)
        self.assertTrue(0.9 <= hyperparams["gamma"] <= 0.99)
        self.assertIn(hyperparams["batch_size"], [16, 32, 64, 128])

        # Check reward weights
        reward_weights = hyperparams["reward_weights"]
        self.assertIsInstance(reward_weights, dict)
        self.assertIn("conversions", reward_weights)
        self.assertIn("cost", reward_weights)

    @patch("services.self_play_service.self_play_service.AdsEnvironment")
    def test_initialize_population(self, mock_env_class):
        """Test initializing the agent population"""
        # Set up mocks
        self.service.initialize_rl_service(self.mock_rl_service)
        self.service._get_historical_data = MagicMock(
            return_value=[{"campaign_id": "123", "keyword_text": "test", "impressions": 100}]
        )

        # Call method
        result = self.service.initialize_population(campaign_id="123")

        # Check results
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(self.service.agent_population), 5)  # population_size from config

        # Check that each agent has the required fields
        for agent_id, agent in self.service.agent_population.items():
            self.assertIn("id", agent)
            self.assertIn("hyperparameters", agent)
            self.assertIn("fitness", agent)
            self.assertIn("generation", agent)
            self.assertEqual(agent["generation"], 1)  # First generation

    @patch("services.self_play_service.self_play_service.AdsEnvironment")
    def test_run_tournament(self, mock_env_class):
        """Test running a tournament between agents"""
        # Set up mocks
        self.service.initialize_rl_service(self.mock_rl_service)
        self.service._get_historical_data = MagicMock(
            return_value=[{"campaign_id": "123", "keyword_text": "test", "impressions": 100}]
        )

        # Create mock environment
        mock_env = MagicMock()
        mock_env.max_steps = 10
        mock_env.action_space = 3
        mock_env_class.return_value = mock_env

        # Initialize population first
        self.service.initialize_population(campaign_id="123")

        # Mock the _run_match method to avoid randomness in tests
        self.service._run_match = MagicMock(
            return_value={
                "match_id": "test_match",
                "agent1_id": "agent_1",
                "agent2_id": "agent_2",
                "agent1_reward": 100,
                "agent2_reward": 50,
                "winner": "agent_1",
                "margin": 50,
                "steps": 10,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Call method
        result = self.service.run_tournament(campaign_id="123")

        # Check results
        self.assertEqual(result["status"], "success")
        self.assertIn("matches", result)
        self.assertIn("leaderboard", result)

        # Check that tournament was saved
        self.assertEqual(len(self.service.tournament_history), 1)

    def test_evolve_population(self):
        """Test evolving the agent population"""
        # Set up a test population
        self.service.agent_population = {
            "agent_1": {
                "id": "agent_1",
                "fitness": 10.0,
                "hyperparameters": {"learning_rate": 0.001, "gamma": 0.95},
                "win_count": 5,
                "match_count": 10,
                "generation": 1,
                "created_at": datetime.now().isoformat(),
            },
            "agent_2": {
                "id": "agent_2",
                "fitness": 8.0,
                "hyperparameters": {"learning_rate": 0.002, "gamma": 0.96},
                "win_count": 4,
                "match_count": 10,
                "generation": 1,
                "created_at": datetime.now().isoformat(),
            },
            "agent_3": {
                "id": "agent_3",
                "fitness": 5.0,
                "hyperparameters": {"learning_rate": 0.003, "gamma": 0.97},
                "win_count": 2,
                "match_count": 10,
                "generation": 1,
                "created_at": datetime.now().isoformat(),
            },
            "agent_4": {
                "id": "agent_4",
                "fitness": 3.0,
                "hyperparameters": {"learning_rate": 0.004, "gamma": 0.98},
                "win_count": 1,
                "match_count": 10,
                "generation": 1,
                "created_at": datetime.now().isoformat(),
            },
        }

        # Mock the _save_population method to avoid file operations
        self.service._save_population = MagicMock()

        # Call method
        result = self.service.evolve_population()

        # Check results
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["elite_count"], 1)  # From config
        self.assertEqual(len(self.service.agent_population), 4)  # Should remain the same

        # The elite agent should still be present
        self.assertIn("agent_1", self.service.agent_population)

        # Some agents should be from generation 2
        gen2_count = sum(1 for a in self.service.agent_population.values() if a["generation"] == 2)
        self.assertGreater(gen2_count, 0)

    def test_tournament_selection(self):
        """Test tournament selection method"""
        # Create sorted agents list
        sorted_agents = [
            ("agent_1", {"fitness": 10.0}),
            ("agent_2", {"fitness": 8.0}),
            ("agent_3", {"fitness": 5.0}),
            ("agent_4", {"fitness": 3.0}),
        ]

        # Call method multiple times (with seed for reproducibility)
        np.random.seed(42)
        selected_ids = [self.service._tournament_selection(sorted_agents) for _ in range(10)]

        # The best agent should be selected often but not always
        self.assertIn("agent_1", selected_ids)
        # Not always the best agent due to tournament selection randomness
        self.assertTrue(len(set(selected_ids)) > 1)

    def test_crossover(self):
        """Test crossover method"""
        parent1 = {
            "learning_rate": 0.001,
            "gamma": 0.95,
            "hidden_layers": [64, 32],
            "reward_weights": {"conversions": 20.0, "cost": -0.1},
        }

        parent2 = {
            "learning_rate": 0.005,
            "gamma": 0.99,
            "hidden_layers": [128, 64],
            "reward_weights": {"conversions": 10.0, "cost": -0.2},
        }

        # Call method
        child = self.service._crossover(parent1, parent2)

        # Check that child has all expected keys
        for key in parent1:
            self.assertIn(key, child)

        # Check that values are properly crossed over
        # For numerical values, child should be between parents
        self.assertTrue(
            parent1["learning_rate"] <= child["learning_rate"] <= parent2["learning_rate"]
            or parent2["learning_rate"] <= child["learning_rate"] <= parent1["learning_rate"]
        )

        # Nested dictionaries should also be crossed over
        for key in parent1["reward_weights"]:
            self.assertIn(key, child["reward_weights"])

    def test_mutate(self):
        """Test mutation method"""
        # Set a high mutation rate for testing
        self.service.mutation_rate = 1.0

        params = {
            "learning_rate": 0.001,
            "gamma": 0.95,
            "hidden_layers": [64, 32],
            "reward_weights": {"conversions": 20.0, "cost": -0.1},
        }

        # Call method
        mutated = self.service._mutate(params)

        # Check that mutated has all expected keys
        for key in params:
            self.assertIn(key, mutated)

        # At least some values should be different due to mutation
        differences = sum(params[key] != mutated[key] for key in ["learning_rate", "gamma"])
        self.assertGreater(differences, 0)

    def test_get_elite_strategy(self):
        """Test getting the elite strategy"""
        # Set up a test population
        self.service.agent_population = {
            "agent_1": {
                "id": "agent_1",
                "fitness": 10.0,
                "hyperparameters": {"learning_rate": 0.001, "gamma": 0.95},
                "win_count": 5,
                "match_count": 10,
                "generation": 1,
                "created_at": datetime.now().isoformat(),
            },
            "agent_2": {
                "id": "agent_2",
                "fitness": 8.0,
                "hyperparameters": {"learning_rate": 0.002, "gamma": 0.96},
                "win_count": 4,
                "match_count": 10,
                "generation": 1,
                "created_at": datetime.now().isoformat(),
            },
        }

        # Call method
        result = self.service.get_elite_strategy()

        # Check results
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["agent_id"], "agent_1")  # Highest fitness
        self.assertEqual(result["fitness"], 10.0)
        self.assertEqual(result["win_rate"], 0.5)  # 5 wins out of 10 matches

    def test_generate_mock_data(self):
        """Test generating mock data"""
        # Set seed for reproducibility
        np.random.seed(42)

        # Call method
        mock_data = self.service._generate_mock_data(campaign_id="123", days=7)

        # Check results
        self.assertIsInstance(mock_data, list)
        self.assertGreater(len(mock_data), 0)

        # Check fields in first item
        first_item = mock_data[0]
        self.assertEqual(first_item["campaign_id"], "123")
        self.assertIn("keyword_text", first_item)
        self.assertIn("impressions", first_item)
        self.assertIn("clicks", first_item)
        self.assertIn("conversions", first_item)
        self.assertIn("cost", first_item)

        # Check derived fields
        self.assertIn("ctr", first_item)
        self.assertIn("conv_rate", first_item)
        self.assertIn("cpc", first_item)
        self.assertIn("conv_value", first_item)

    def test_failure_cases(self):
        """Test various failure cases"""
        # Test running tournament with empty population
        self.service.agent_population = {}
        result = self.service.run_tournament()
        self.assertEqual(result["status"], "failed")
        self.assertIn("Need at least 2 agents", result["message"])

        # Test evolving with empty population
        result = self.service.evolve_population()
        self.assertEqual(result["status"], "failed")
        self.assertIn("Need at least 2 agents", result["message"])

        # Test getting elite strategy with empty population
        result = self.service.get_elite_strategy()
        self.assertEqual(result["status"], "failed")
        self.assertIn("No agent population", result["message"])


if __name__ == "__main__":
    unittest.main()
