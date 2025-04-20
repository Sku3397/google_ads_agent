"""
Tests for the enhanced Reinforcement Learning Service.

This module contains comprehensive tests for the advanced policy gradient methods
and environment implementations.
"""

import unittest
import numpy as np
import os
import tempfile
import shutil
import tensorflow as tf
from unittest.mock import MagicMock, patch, ANY
from google.ads.googleads.client import GoogleAdsClient
from services.reinforcement_learning_service import (
    ReinforcementLearningService,
    AdsEnvironment,
    PolicyConfig,
    PPOPolicy,
    A2CPolicy,
    SACPolicy,
)
from datetime import datetime


class TestReinforcementLearningService(unittest.TestCase):
    """Test cases for the enhanced ReinforcementLearningService."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temp directory for models
        self.temp_dir = tempfile.mkdtemp()

        # Mock Google Ads client
        self.mock_client = MagicMock()
        self.customer_id = "1234567890"

        # Test configuration
        self.config = {
            "model_save_path": self.temp_dir,
            "algorithm": "ppo",  # Using PPO as default
            "use_stable_baselines": False,
            "environment": {
                "observation_space_dim": 20,
                "action_space": 11,
                "action_space_type": "discrete",
                "max_steps": 10,
                "use_market_dynamics": True,
                "feature_engineering": "standard",
                "reward_weights": {
                    "conversions": 10.0,
                    "cost": -0.1,
                    "clicks": 0.5,
                    "conv_value": 5.0,
                    "roas": 20.0,
                    "quality_score": 2.0,
                    "ctr": 15.0,
                    "avg_position": -0.5,
                    "constraint_violation": -50.0,
                },
            },
            "policy": {
                "hidden_layers": [256, 128, 64],
                "learning_rate": 3e-4,
                "clip_ratio": 0.2,
                "target_kl": 0.01,
                "entropy_coef": 0.01,
                "value_coef": 0.5,
                "use_gae": True,
                "gae_lambda": 0.95,
                "gamma": 0.99,
                "normalize_advantages": True,
                "use_critic_ensemble": True,
                "num_critics": 2,
            },
            "training": {
                "batch_size": 64,
                "num_epochs": 10,
                "max_grad_norm": 0.5,
                "target_update_interval": 100,
                "warmup_steps": 1000,
            },
            "deployment_safety": {
                "max_bid_change": 0.25,
                "rollout_percentage": 0.2,
                "performance_threshold": 0.05,
            },
        }

        # Initialize service
        self.service = ReinforcementLearningService(
            client=self.mock_client, customer_id=self.customer_id, config=self.config
        )

        # Sample test data
        self.sample_keywords = [
            {
                "criterion_id": "123456",
                "keyword_text": "test keyword",
                "match_type": "EXACT",
                "campaign_id": "7890",
                "ad_group_id": "12345",
                "current_bid": 1.5,
                "impressions": 1000,
                "clicks": 100,
                "conversions": 10,
                "cost": 150.0,
                "quality_score": 7,
                "conv_value": 500.0,
                "ctr": 0.1,
                "conv_rate": 0.1,
                "avg_position": 2.0,
            },
            {
                "criterion_id": "123457",
                "keyword_text": "another test",
                "match_type": "PHRASE",
                "campaign_id": "7890",
                "ad_group_id": "12345",
                "current_bid": 2.0,
                "impressions": 500,
                "clicks": 50,
                "conversions": 5,
                "cost": 100.0,
                "quality_score": 5,
                "conv_value": 300.0,
                "ctr": 0.1,
                "conv_rate": 0.1,
                "avg_position": 3.0,
            },
        ]

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)

    def test_environment_initialization(self):
        """Test that the AdsEnvironment initializes correctly."""
        env = AdsEnvironment(self.config["environment"], self.sample_keywords)

        # Check spaces
        self.assertEqual(env.observation_space_dim, 20)
        self.assertEqual(env.action_space, 11)

        # Check reward weights
        self.assertEqual(env.reward_weights["conversions"], 10.0)
        self.assertEqual(env.reward_weights["cost"], -0.1)

        # Check constraints
        self.assertEqual(env.constraints.max_bid_change, 0.25)
        self.assertTrue(env.use_market_dynamics)

    def test_environment_reset(self):
        """Test environment reset."""
        env = AdsEnvironment(self.config["environment"], self.sample_keywords)
        state = env.reset()

        # Check state shape and bounds
        self.assertEqual(state.shape, (20,))
        self.assertTrue(np.all(state >= 0))
        self.assertTrue(np.all(state <= 1))

        # Check internal state
        self.assertEqual(env.current_step, 0)
        self.assertEqual(len(env.constraint_violations), 0)

    def test_environment_step(self):
        """Test environment step function."""
        env = AdsEnvironment(self.config["environment"], self.sample_keywords)
        state = env.reset()

        # Take a step with a moderate bid adjustment
        action = 5  # Middle action (no change)
        next_state, reward, done, info = env.step(action)

        # Check outputs
        self.assertEqual(next_state.shape, (20,))
        self.assertIsInstance(reward, float)
        self.assertFalse(done)
        self.assertIn("metrics", info)
        self.assertIn("bid_adjustment", info)

        # Check metrics
        metrics = info["metrics"]
        self.assertIn("impressions", metrics)
        self.assertIn("clicks", metrics)
        self.assertIn("conversions", metrics)
        self.assertIn("cost", metrics)
        self.assertIn("roas", metrics)

    def test_ppo_policy(self):
        """Test PPO policy network."""
        config = PolicyConfig(
            state_dim=20, action_dim=11, hidden_layers=[64, 64], learning_rate=3e-4
        )

        policy = PPOPolicy(config)

        # Test forward pass
        batch_size = 32
        states = tf.random.normal((batch_size, 20))
        logits, values = policy(states)

        # Check output shapes
        self.assertEqual(logits.shape, (batch_size, 11))
        self.assertEqual(values.shape, (batch_size, 1))

        # Test action distribution
        dist = policy.get_action_distribution(logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        self.assertEqual(actions.shape, (batch_size,))
        self.assertEqual(log_probs.shape, (batch_size,))

    def test_a2c_policy(self):
        """Test A2C policy network."""
        config = PolicyConfig(
            state_dim=20, action_dim=11, hidden_layers=[64, 64], learning_rate=3e-4
        )

        policy = A2CPolicy(config)

        # Test forward pass
        batch_size = 32
        states = tf.random.normal((batch_size, 20))
        logits, values = policy(states)

        # Check output shapes
        self.assertEqual(logits.shape, (batch_size, 11))
        self.assertEqual(values.shape, (batch_size, 1))

    def test_sac_policy(self):
        """Test SAC policy network."""
        config = PolicyConfig(
            state_dim=20,
            action_dim=1,  # Continuous action space
            hidden_layers=[64, 64],
            learning_rate=3e-4,
        )

        policy = SACPolicy(config)

        # Test actor network
        batch_size = 32
        states = tf.random.normal((batch_size, 20))
        mean, log_std = policy.actor(states)

        # Check output shapes
        self.assertEqual(mean.shape, (batch_size, 1))
        self.assertEqual(log_std.shape, (batch_size, 1))

        # Test critic networks
        actions = tf.random.normal((batch_size, 1))
        q1 = policy.critic_1([states, actions])
        q2 = policy.critic_2([states, actions])

        self.assertEqual(q1.shape, (batch_size, 1))
        self.assertEqual(q2.shape, (batch_size, 1))

    def test_training_loop(self):
        """Test the complete training loop."""
        # Configure for a short training run
        self.config["training"]["num_epochs"] = 2
        self.config["training"]["batch_size"] = 8

        # Initialize environment
        env = AdsEnvironment(self.config["environment"], self.sample_keywords)

        # Initialize policy
        policy_config = PolicyConfig(
            state_dim=20, action_dim=11, hidden_layers=[32, 32], learning_rate=3e-4
        )
        policy = PPOPolicy(policy_config)

        # Run a few episodes
        total_episodes = 5
        for episode in range(total_episodes):
            state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Get action from policy
                state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
                logits, _ = policy(state_tensor)
                dist = policy.get_action_distribution(logits)
                action = dist.sample().numpy()[0]

                # Take step in environment
                next_state, reward, done, info = env.step(action)
                episode_reward += reward

                # Update state
                state = next_state

            self.assertIsInstance(episode_reward, float)

    def test_safe_deployment(self):
        """Test safe policy deployment with rollout controls."""
        # Initialize policy and environment
        env = AdsEnvironment(self.config["environment"], self.sample_keywords)

        policy_config = PolicyConfig(
            state_dim=20, action_dim=11, hidden_layers=[32, 32], learning_rate=3e-4
        )
        policy = PPOPolicy(policy_config)

        # Mock the deployment functions
        self.service._backup_current_bids = MagicMock(
            return_value={
                "status": "success",
                "backup_id": "test_backup",
                "timestamp": datetime.now().isoformat(),
            }
        )

        self.service._apply_bid_recommendations = MagicMock(
            return_value={"successes": [{"keyword_id": "123456"}], "failures": []}
        )

        # Test progressive rollout
        result = self.service.safe_deploy_policy(
            campaign_id="7890", rollout_percentage=0.1, monitor_window_hours=24  # Start with 10%
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["rollout_percentage"], 0.1)
        self.assertGreater(len(result["metrics"]), 0)

    def test_market_dynamics(self):
        """Test market dynamics simulation."""
        env = AdsEnvironment(self.config["environment"], self.sample_keywords)

        # Run multiple episodes to check market dynamics
        num_episodes = 5
        market_states = []

        for _ in range(num_episodes):
            env.reset()
            market_states.append(
                {
                    "competition_level": env.market_state["competition_level"],
                    "seasonality": env.market_state["seasonality"],
                    "trend": env.market_state["trend"],
                }
            )

        # Check that market states vary
        competition_levels = [state["competition_level"] for state in market_states]
        self.assertTrue(len(set(competition_levels)) > 1)

    def test_constraint_handling(self):
        """Test constraint violation handling."""
        env = AdsEnvironment(self.config["environment"], self.sample_keywords)
        env.constraints.budget_limit = 100.0  # Set a low budget limit

        state = env.reset()

        # Take an aggressive action that should violate budget constraint
        action = 10  # Maximum bid increase
        _, reward, done, info = env.step(action)

        # Check that constraint violation was detected
        self.assertTrue(done)  # Episode should end on violation
        self.assertIn("violations", info)
        self.assertGreater(len(info["violations"]), 0)
        self.assertLess(reward, 0)  # Should receive negative reward

    def test_reward_engineering(self):
        """Test sophisticated reward engineering."""
        env = AdsEnvironment(self.config["environment"], self.sample_keywords)

        # Test different scenarios
        scenarios = [
            {
                "metrics": {
                    "conversions": 10,
                    "cost": 100,
                    "clicks": 100,
                    "conv_value": 500,
                    "quality_score": 8,
                    "ctr": 0.1,
                    "roas": 5.0,
                },
                "violations": [],
            },
            {
                "metrics": {
                    "conversions": 5,
                    "cost": 200,
                    "clicks": 50,
                    "conv_value": 200,
                    "quality_score": 4,
                    "ctr": 0.05,
                    "roas": 1.0,
                },
                "violations": ["budget_limit"],
            },
        ]

        rewards = []
        for scenario in scenarios:
            reward = env._calculate_reward(scenario["metrics"], scenario["violations"])
            rewards.append(reward)

        # First scenario should have higher reward
        self.assertGreater(rewards[0], rewards[1])

    def test_model_persistence(self):
        """Test model saving and loading."""
        # Initialize policy
        policy_config = PolicyConfig(
            state_dim=20, action_dim=11, hidden_layers=[32, 32], learning_rate=3e-4
        )
        policy = PPOPolicy(policy_config)

        # Save model
        save_path = os.path.join(self.temp_dir, "test_model")
        policy.save_weights(save_path)

        # Load model
        new_policy = PPOPolicy(policy_config)
        new_policy.load_weights(save_path)

        # Compare outputs
        test_input = tf.random.normal((1, 20))
        output1, value1 = policy(test_input)
        output2, value2 = new_policy(test_input)

        np.testing.assert_array_almost_equal(output1.numpy(), output2.numpy())
        np.testing.assert_array_almost_equal(value1.numpy(), value2.numpy())


if __name__ == "__main__":
    unittest.main()
