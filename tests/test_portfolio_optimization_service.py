"""
Tests for the Portfolio Optimization Service.

This module contains unit tests for the Portfolio Optimization Service.
"""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import shutil

from services.portfolio_optimization_service.portfolio_optimization_service import (
    PortfolioOptimizationService,
)

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestPortfolioOptimizationService(unittest.TestCase):
    """Test suite for PortfolioOptimizationService."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a test directory for output
        self.test_output_dir = "test_output/portfolio_optimization"
        os.makedirs(self.test_output_dir, exist_ok=True)

        # Mock Google Ads API client
        self.mock_ads_api = MagicMock()

        # Initialize the service
        self.service = PortfolioOptimizationService(ads_api=self.mock_ads_api, logger=logger)

        # Generate test data
        self.test_data = self._generate_test_data()

    def tearDown(self):
        """Clean up after tests."""
        # Remove test output directory
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    def _generate_test_data(self):
        """Generate test data for portfolio optimization."""
        # Create sample campaign data
        np.random.seed(42)

        campaigns = []
        for i in range(5):
            campaign = {
                "campaign_id": f"campaign_{i}",
                "campaign_name": f"Test Campaign {i}",
                "impressions": np.random.randint(1000, 100000),
                "clicks": np.random.randint(100, 5000),
                "conversions": np.random.randint(5, 200),
                "cost": np.random.uniform(500, 5000),
                "conversion_value": np.random.uniform(1000, 10000),
                "budget": np.random.uniform(300, 3000),
            }
            # Add derived metrics
            campaign["ctr"] = (
                campaign["clicks"] / campaign["impressions"] if campaign["impressions"] > 0 else 0
            )
            campaign["cvr"] = (
                campaign["conversions"] / campaign["clicks"] if campaign["clicks"] > 0 else 0
            )
            campaign["cpc"] = campaign["cost"] / campaign["clicks"] if campaign["clicks"] > 0 else 0
            campaign["cpa"] = (
                campaign["cost"] / campaign["conversions"] if campaign["conversions"] > 0 else 0
            )
            campaign["roas"] = (
                campaign["conversion_value"] / campaign["cost"] if campaign["cost"] > 0 else 0
            )

            campaigns.append(campaign)

        return campaigns

    def test_initialization(self):
        """Test correct initialization of the service."""
        self.assertIsNotNone(self.service)

    def test_optimize_campaign_portfolio_convex(self):
        """Test optimizing campaign portfolio using convex optimization."""
        # Mock the _get_campaign_performance method to return test data
        with patch.object(self.service, "_get_campaign_performance", return_value=self.test_data):
            # Test with conversions objective
            result = self.service.optimize_campaign_portfolio(
                days=30, objective="conversions", constraint="budget", algorithm="convex"
            )

            # Check results
            self.assertEqual(result["status"], "success")
            self.assertIn("recommendations", result)
            self.assertEqual(len(result["recommendations"]), len(self.test_data))
            self.assertEqual(result["optimization_details"]["algorithm"], "convex")
            self.assertIn("expected_improvement", result["optimization_details"])

    def test_optimize_campaign_portfolio_efficient_frontier(self):
        """Test optimizing campaign portfolio using efficient frontier."""
        # Mock the _get_campaign_performance method to return test data
        with patch.object(self.service, "_get_campaign_performance", return_value=self.test_data):
            # Test with roas objective and efficient frontier
            result = self.service.optimize_campaign_portfolio(
                days=30,
                objective="roas",
                constraint="budget",
                algorithm="efficient_frontier",
                risk_tolerance=0.7,
            )

            # Check results
            self.assertEqual(result["status"], "success")
            self.assertIn("recommendations", result)
            self.assertEqual(result["optimization_details"]["algorithm"], "efficient_frontier")
            self.assertEqual(result["optimization_details"]["risk_tolerance"], 0.7)

            # Check if sharpe ratio is included
            self.assertIn("sharpe_ratio", result["optimization_details"])

    def test_optimize_campaign_portfolio_multi_objective(self):
        """Test optimizing campaign portfolio using multi-objective optimization."""
        # Mock the _get_campaign_performance method to return test data
        with patch.object(self.service, "_get_campaign_performance", return_value=self.test_data):
            # Test with multi-objective
            multi_objective_weights = {"conversions": 0.6, "clicks": 0.3, "roas": 0.1}

            result = self.service.optimize_campaign_portfolio(
                days=30,
                objective="multi",
                constraint="budget",
                algorithm="multi_objective",
                multi_objective_weights=multi_objective_weights,
            )

            # Check results
            self.assertEqual(result["status"], "success")
            self.assertIn("recommendations", result)
            self.assertEqual(result["optimization_details"]["algorithm"], "multi_objective")

            # Check expected metrics
            self.assertIn("expected_improvement", result["optimization_details"])

    def test_optimize_campaign_portfolio_with_visualization(self):
        """Test optimizing campaign portfolio with visualization enabled."""
        # Skip test if matplotlib not available
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            self.skipTest("Matplotlib or seaborn not available")

        # Mock the _get_campaign_performance method to return test data
        with patch.object(self.service, "_get_campaign_performance", return_value=self.test_data):
            # Test with visualization
            result = self.service.optimize_campaign_portfolio(
                days=30,
                objective="conversions",
                constraint="budget",
                algorithm="convex",
                visualize=True,
                output_dir=self.test_output_dir,
            )

            # Check results
            self.assertEqual(result["status"], "success")
            self.assertIn("recommendations", result)

            # Check for visualization paths
            self.assertIn("visualization_paths", result)
            self.assertGreater(len(result["visualization_paths"]), 0)

            # Check if files exist
            for vis_path in result["visualization_paths"]:
                self.assertTrue(os.path.exists(vis_path))

    def test_apply_portfolio_recommendations(self):
        """Test applying portfolio recommendations."""
        # Create sample recommendations
        recommendations = [
            {"campaign_id": "campaign_1", "current_budget": 1000.0, "recommended_budget": 1500.0},
            {"campaign_id": "campaign_2", "current_budget": 2000.0, "recommended_budget": 1800.0},
        ]

        # Mock the ads_api.update_campaign_budget method
        self.mock_ads_api.update_campaign_budget.side_effect = [
            {"status": "success"},
            {"status": "success"},
        ]

        # Test applying recommendations
        result = self.service.apply_portfolio_recommendations(recommendations)

        # Check results
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["applied_count"], 2)
        self.assertEqual(result["failed_count"], 0)

        # Verify API calls
        self.assertEqual(self.mock_ads_api.update_campaign_budget.call_count, 2)
        self.mock_ads_api.update_campaign_budget.assert_any_call(
            campaign_id="campaign_1", new_budget=1500.0
        )
        self.mock_ads_api.update_campaign_budget.assert_any_call(
            campaign_id="campaign_2", new_budget=1800.0
        )

    def test_invalid_algorithm(self):
        """Test error handling for invalid algorithm."""
        # Mock the _get_campaign_performance method to return test data
        with patch.object(self.service, "_get_campaign_performance", return_value=self.test_data):
            # Test with invalid algorithm
            result = self.service.optimize_campaign_portfolio(
                days=30, objective="conversions", algorithm="invalid_algorithm"
            )

            # Check error response
            self.assertEqual(result["status"], "error")
            self.assertIn("Invalid algorithm", result["message"])

    def test_invalid_objective(self):
        """Test error handling for invalid objective."""
        # Mock the _get_campaign_performance method to return test data
        with patch.object(self.service, "_get_campaign_performance", return_value=self.test_data):
            # Test with invalid objective
            result = self.service.optimize_campaign_portfolio(
                days=30, objective="invalid_objective", algorithm="convex"
            )

            # Check error response
            self.assertEqual(result["status"], "error")
            self.assertIn("Invalid objective", result["message"])

    def test_insufficient_data(self):
        """Test error handling for insufficient data."""
        # Mock the _get_campaign_performance method to return insufficient data
        with patch.object(self.service, "_get_campaign_performance", return_value=[]):
            # Test with insufficient data
            result = self.service.optimize_campaign_portfolio(
                days=30, objective="conversions", algorithm="convex"
            )

            # Check error response
            self.assertEqual(result["status"], "error")
            self.assertIn("Insufficient campaign data", result["message"])

    def test_cross_campaign_keyword_analysis(self):
        """Test cross-campaign keyword analysis."""
        # Generate sample keyword data
        keywords = []
        for i in range(20):
            campaign_id = f"campaign_{np.random.randint(0, 3)}"
            keyword = {
                "keyword_text": f"keyword_{i // 5}",  # Create some overlapping keywords
                "campaign_id": campaign_id,
                "ad_group_id": f"adgroup_{np.random.randint(0, 5)}",
                "clicks": np.random.randint(10, 500),
                "impressions": np.random.randint(100, 5000),
                "conversions": np.random.randint(0, 20),
                "cost": np.random.uniform(50, 500),
                "average_cpc": np.random.uniform(0.5, 2.0),
                "ctr": np.random.uniform(0.01, 0.1),
                "conversion_rate": np.random.uniform(0.01, 0.05),
                "quality_score": np.random.randint(1, 10),
            }
            keywords.append(keyword)

        # Mock the _get_keyword_performance method
        with patch.object(self.service, "_get_keyword_performance", return_value=keywords):
            # Mock the analysis methods to return some data
            with patch.object(
                self.service,
                "_identify_keyword_overlaps",
                return_value=[{"keyword": "keyword_1", "campaigns": ["campaign_0", "campaign_1"]}],
            ):
                with patch.object(
                    self.service,
                    "_identify_cannibalization",
                    return_value=[
                        {
                            "keyword": "keyword_2",
                            "primary_campaign": "campaign_0",
                            "secondary_campaign": "campaign_1",
                        }
                    ],
                ):
                    with patch.object(
                        self.service,
                        "_identify_performance_disparities",
                        return_value=[
                            {
                                "keyword": "keyword_3",
                                "best_campaign": "campaign_2",
                                "worst_campaign": "campaign_0",
                            }
                        ],
                    ):
                        # Test cross-campaign analysis
                        result = self.service.cross_campaign_keyword_analysis(days=30)

                        # Check results
                        self.assertEqual(result["status"], "success")
                        self.assertIn("overlapping_keywords", result)
                        self.assertIn("potential_cannibalization", result)
                        self.assertIn("performance_disparities", result)

    def test_optimize_budget_allocation_over_time(self):
        """Test time-based budget allocation optimization."""
        # Generate sample daily performance data
        days = 30
        campaigns = self.test_data
        daily_data = {}

        for campaign in campaigns:
            campaign_id = campaign["campaign_id"]
            daily_stats = []

            # Base values
            base_impressions = campaign["impressions"] / days
            base_clicks = campaign["clicks"] / days
            base_conversions = campaign["conversions"] / days
            base_cost = campaign["cost"] / days
            base_value = campaign["conversion_value"] / days

            # Add some randomness and trends
            for i in range(days):
                # Add day-of-week patterns (weekends lower)
                dow_factor = 0.7 if i % 7 >= 5 else 1.0

                # Add slight upward trend
                trend_factor = 1.0 + (i / days) * 0.2

                # Add randomness
                random_factor = np.random.uniform(0.8, 1.2)

                # Combine factors
                daily_factor = dow_factor * trend_factor * random_factor

                # Create daily stat
                daily_stat = {
                    "date": (datetime.now() - timedelta(days=days - i)).strftime("%Y-%m-%d"),
                    "impressions": int(base_impressions * daily_factor),
                    "clicks": int(base_clicks * daily_factor),
                    "conversions": int(base_conversions * daily_factor)
                    + 1,  # Ensure at least 1 conversion
                    "cost": base_cost * daily_factor,
                    "conversion_value": base_value * daily_factor,
                }
                daily_stats.append(daily_stat)

            daily_data[campaign_id] = daily_stats

        # Mock the _get_campaign_daily_performance method
        with patch.object(self.service, "_get_campaign_daily_performance", return_value=daily_data):
            # Mock the forecasting and optimization methods
            forecasts = {
                campaign_id: pd.DataFrame(data) for campaign_id, data in daily_data.items()
            }
            with patch.object(self.service, "_generate_campaign_forecasts", return_value=forecasts):
                daily_allocations = [
                    {
                        "date": (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d"),
                        "allocations": [
                            {
                                "campaign_id": campaign["campaign_id"],
                                "budget": campaign["budget"] * np.random.uniform(0.8, 1.2),
                            }
                            for campaign in campaigns
                        ],
                    }
                    for i in range(7)  # 7 days of allocations
                ]
                with patch.object(
                    self.service, "_optimize_daily_allocations", return_value=daily_allocations
                ):
                    # Test time-based optimization
                    result = self.service.optimize_budget_allocation_over_time(
                        days=30, forecast_days=7, objective="conversions"
                    )

                    # Check results
                    self.assertEqual(result["status"], "success")
                    self.assertIn("daily_budget_allocations", result)
                    self.assertEqual(
                        len(result["daily_budget_allocations"]), 7
                    )  # 7 days of allocations

                    # Check optimization details
                    self.assertEqual(result["optimization_details"]["objective"], "conversions")
                    self.assertEqual(result["optimization_details"]["forecast_days"], 7)
                    self.assertEqual(
                        result["optimization_details"]["algorithm"],
                        "time_series_portfolio_optimization",
                    )


if __name__ == "__main__":
    unittest.main()
