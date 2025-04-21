"""
Unit tests for the MetaLearningService.

Tests the core functionality of the MetaLearningService including:
- Strategy execution recording
- Strategy recommendation
- Context similarity calculations
- Hyperparameter optimization
- Transfer learning
"""

import unittest
import os
import json
import tempfile
import shutil
from datetime import datetime
from unittest.mock import patch, MagicMock
import pickle

from services.meta_learning_service import MetaLearningService


class TestMetaLearningService(unittest.TestCase):
    """Test cases for the MetaLearningService."""

    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for test data
        self.test_data_dir = tempfile.mkdtemp()

        # Create subdirectory for meta learning data
        os.makedirs(os.path.join(self.test_data_dir, "meta_learning"), exist_ok=True)

        # Patch the os.path.join to use the test directory
        self.path_join_patcher = patch("os.path.join")
        self.mock_path_join = self.path_join_patcher.start()
        self.mock_path_join.side_effect = self._mock_path_join

        # Initialize service
        self.service = MetaLearningService(
            ads_api=MagicMock(),
            optimizer=MagicMock(),
            config={"test_config": True},
            logger=MagicMock(),
        )

        # Add some test data to the service
        self._add_test_strategy_data()

    def tearDown(self):
        """Clean up after tests."""
        # Stop patchers
        self.path_join_patcher.stop()

        # Remove test directory
        shutil.rmtree(self.test_data_dir)

    def _mock_path_join(self, *args):
        """Mock implementation of os.path.join that redirects to the test directory."""
        if args[0] == "data" and args[1] == "meta_learning":
            return os.path.join(self.test_data_dir, "meta_learning")
        return os.path.join(*args)

    def _add_test_strategy_data(self):
        """Add test strategy execution data to the service."""
        # Add some test strategy history
        self.service.strategy_history = []
        self.service.performance_history = {}

        # Strategy 1: bid_service/performance_bidding
        self.service.record_strategy_execution(
            service_name="bid_service",
            strategy_name="performance_bidding",
            context={"campaign_type": "search", "industry": "retail", "budget_level": "high"},
            parameters={"target_cpa": 15.0, "max_cpc_increase": 0.5},
            results={
                "before": {"ctr": 0.02, "conversion_rate": 0.01, "cpa": 25.0, "roas": 3.0},
                "after": {"ctr": 0.025, "conversion_rate": 0.015, "cpa": 20.0, "roas": 4.0},
            },
        )

        # Strategy 2: bid_service/target_cpa_bidding
        self.service.record_strategy_execution(
            service_name="bid_service",
            strategy_name="target_cpa_bidding",
            context={"campaign_type": "search", "industry": "retail", "budget_level": "medium"},
            parameters={"target_cpa": 18.0, "max_bid_change": 0.3},
            results={
                "before": {"ctr": 0.018, "conversion_rate": 0.008, "cpa": 28.0, "roas": 2.5},
                "after": {"ctr": 0.019, "conversion_rate": 0.012, "cpa": 22.0, "roas": 3.2},
            },
        )

        # Strategy 3: bid_service/position_based_bidding
        self.service.record_strategy_execution(
            service_name="bid_service",
            strategy_name="position_based_bidding",
            context={"campaign_type": "search", "industry": "finance", "budget_level": "high"},
            parameters={"target_position": 2.0, "max_bid_change": 0.4},
            results={
                "before": {"ctr": 0.015, "conversion_rate": 0.007, "cpa": 35.0, "roas": 1.8},
                "after": {"ctr": 0.022, "conversion_rate": 0.009, "cpa": 32.0, "roas": 2.1},
            },
        )

    def test_record_strategy_execution(self):
        """Test recording a strategy execution."""
        # Record a new strategy execution
        result = self.service.record_strategy_execution(
            service_name="keyword_service",
            strategy_name="keyword_expansion",
            context={"campaign_type": "search", "industry": "education", "budget_level": "low"},
            parameters={"max_keywords": 50, "min_search_volume": 100},
            results={
                "before": {
                    "keyword_count": 25,
                    "impressions": 1000,
                    "clicks": 50,
                    "conversions": 5,
                },
                "after": {"keyword_count": 75, "impressions": 1500, "clicks": 80, "conversions": 8},
            },
        )

        # Check that the result has the expected structure
        self.assertIn("id", result)
        self.assertIn("timestamp", result)
        self.assertEqual(result["service_name"], "keyword_service")
        self.assertEqual(result["strategy_name"], "keyword_expansion")

        # Check that metrics were extracted correctly
        self.assertIn("metrics", result)
        self.assertIn("impressions_improvement", result["metrics"])
        self.assertIn("clicks_improvement", result["metrics"])
        self.assertIn("conversions_improvement", result["metrics"])

        # Check that it was added to the history
        self.assertEqual(len(self.service.strategy_history), 4)  # 3 from setup + 1 new

        # Check that it was added to the performance history
        self.assertIn("keyword_service_keyword_expansion", self.service.performance_history)
        self.assertEqual(
            len(self.service.performance_history["keyword_service_keyword_expansion"]), 1
        )

    def test_recommend_strategy(self):
        """Test recommending a strategy based on historical performance."""
        # Request a recommendation for a context similar to one in history
        recommendation = self.service.recommend_strategy(
            service_name="bid_service",
            context={"campaign_type": "search", "industry": "retail", "budget_level": "medium"},
            available_strategies=[
                "performance_bidding",
                "target_cpa_bidding",
                "position_based_bidding",
            ],
        )

        # Check that the recommendation has the expected structure
        self.assertEqual(recommendation["service"], "bid_service")
        self.assertIn("recommended_strategy", recommendation)
        self.assertIn("parameters", recommendation)
        self.assertIn("confidence_score", recommendation)
        self.assertIn("alternatives", recommendation)

        # The target_cpa_bidding strategy should be recommended since the context matches exactly
        self.assertEqual(recommendation["recommended_strategy"], "target_cpa_bidding")

        # Check that parameters were returned
        self.assertIn("target_cpa", recommendation["parameters"])
        self.assertIn("max_bid_change", recommendation["parameters"])

        # Check that alternatives include other strategies
        self.assertEqual(len(recommendation["alternatives"]), 2)

    def test_recommend_strategy_new_context(self):
        """Test recommending a strategy for a new context not in history."""
        # Request a recommendation for a context not in history
        recommendation = self.service.recommend_strategy(
            service_name="bid_service",
            context={"campaign_type": "display", "industry": "healthcare", "budget_level": "low"},
            available_strategies=[
                "performance_bidding",
                "target_cpa_bidding",
                "position_based_bidding",
            ],
        )

        # Should still return a recommendation, but with lower confidence
        self.assertIn("recommended_strategy", recommendation)
        self.assertLess(recommendation["confidence_score"], 0.5)

    def test_calculate_context_similarity(self):
        """Test calculating similarity between contexts."""
        context1 = {
            "campaign_type": "search",
            "industry": "retail",
            "budget_level": "high",
            "impression_volume": 10000,
        }

        context2 = {
            "campaign_type": "search",
            "industry": "retail",
            "budget_level": "medium",
            "impression_volume": 8000,
        }

        context3 = {
            "campaign_type": "display",
            "industry": "healthcare",
            "budget_level": "low",
            "impression_volume": 2000,
        }

        # Test similarity between similar contexts
        similarity1 = self.service._calculate_context_similarity(context1, context2)
        self.assertGreater(similarity1, 0.7)  # Should be quite similar

        # Test similarity between different contexts
        similarity2 = self.service._calculate_context_similarity(context1, context3)
        self.assertLess(similarity2, 0.3)  # Should be quite different

        # Test similarity with empty context
        similarity3 = self.service._calculate_context_similarity(context1, {})
        self.assertEqual(similarity3, 0)  # Should be zero

    def test_transfer_learning(self):
        """Test transfer learning between contexts."""
        # Test transfer learning between similar industries
        transfer_result = self.service.transfer_learning(
            source_context={"campaign_type": "search", "industry": "retail"},
            target_context={"campaign_type": "search", "industry": "e-commerce"},
        )

        # Check that the result has the expected structure
        self.assertIn("source_context", transfer_result)
        self.assertIn("target_context", transfer_result)
        self.assertIn("context_similarity", transfer_result)
        self.assertIn("adapted_strategies", transfer_result)

        # Check that strategies were found and adapted
        self.assertGreater(len(transfer_result["adapted_strategies"]), 0)

    def test_learn_hyperparameters(self):
        """Test learning hyperparameters for a strategy."""

        # Define a simple evaluation function for testing
        def eval_func(params):
            # Prefer higher learning rates and larger batch sizes for this test
            return params["learning_rate"] * 2 + params["batch_size"] / 100

        # Test hyperparameter optimization
        result = self.service.learn_hyperparameters(
            service_name="reinforcement_learning",
            strategy_name="ppo_bidding",
            param_grid={"learning_rate": [0.001, 0.01, 0.1], "batch_size": [32, 64, 128]},
            evaluation_function=eval_func,
            n_trials=5,
        )

        # Check that the result has the expected structure
        self.assertEqual(result["service"], "reinforcement_learning")
        self.assertEqual(result["strategy"], "ppo_bidding")
        self.assertIn("best_parameters", result)
        self.assertIn("best_score", result)

        # Check that parameters were optimized correctly
        # The best parameters should be learning_rate=0.1, batch_size=128
        self.assertEqual(result["best_parameters"]["learning_rate"], 0.1)
        self.assertEqual(result["best_parameters"]["batch_size"], 128)

    def test_analyze_cross_service_patterns(self):
        """Test analyzing patterns across services."""
        # Test cross-service pattern analysis
        analysis = self.service.analyze_cross_service_patterns()

        # Check that the result has the expected structure
        self.assertIn("synergies", analysis)
        self.assertIn("conflicts", analysis)
        self.assertIn("service_correlations", analysis)


if __name__ == "__main__":
    unittest.main()
