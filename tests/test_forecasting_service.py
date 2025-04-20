"""
Unit tests for the ForecastingService.

Tests the core functionality of the ForecastingService including:
- Performance metric forecasting
- Budget forecasting
- Trend detection
- Model training and evaluation
"""

import unittest
import os
import json
import tempfile
import shutil
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, ANY

from services.forecasting_service import ForecastingService


class TestForecastingService(unittest.TestCase):
    """Test cases for the ForecastingService."""

    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for test data
        self.test_data_dir = tempfile.mkdtemp()

        # Create subdirectory for forecasting data
        os.makedirs(os.path.join(self.test_data_dir, "forecasting", "models"), exist_ok=True)

        # Patch the os.path.join to use the test directory for data/forecasting paths
        self.path_join_patcher = patch("os.path.join")
        self.mock_path_join = self.path_join_patcher.start()
        self.mock_path_join.side_effect = self._mock_path_join

        # Create mock ads_api with sample data
        self.mock_ads_api = MagicMock()
        self.mock_ads_api.get_keyword_performance.return_value = self._generate_mock_keyword_data()

        # Initialize service
        self.service = ForecastingService(
            ads_api=self.mock_ads_api,
            optimizer=MagicMock(),
            config={"forecasting": {"horizon_days": 14, "confidence_level": 0.9}},
            logger=MagicMock(),
        )

    def tearDown(self):
        """Clean up after tests."""
        # Stop patchers
        self.path_join_patcher.stop()

        # Remove test directory
        shutil.rmtree(self.test_data_dir)

    def _mock_path_join(self, *args):
        """Mock implementation of os.path.join that redirects data/forecasting paths to test directory."""
        if len(args) >= 2 and args[0] == "data" and args[1] == "forecasting":
            # Redirect to test directory
            return os.path.join(self.test_data_dir, "forecasting", *args[2:])
        return os.path.join(*args)

    def _generate_mock_keyword_data(self):
        """Generate mock keyword performance data for testing."""
        # Generate dates for the last 90 days
        today = datetime.now().date()
        dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(90)]

        # Generate sample keywords
        keywords = [
            "sample keyword 1",
            "sample keyword 2",
            "trending keyword 1",
            "trending keyword 2",
            "declining keyword 1",
            "declining keyword 2",
        ]

        # Generate sample data with trends
        data = []
        for date in dates:
            for keyword in keywords:
                # Base metrics
                impressions = np.random.randint(100, 1000)
                clicks = np.random.randint(5, 50)
                conversions = np.random.randint(0, 5)
                cost = np.random.uniform(10, 100)

                # Add trends for specific keywords
                days_ago = (today - datetime.strptime(date, "%Y-%m-%d").date()).days

                if "trending" in keyword and days_ago < 30:
                    # Make trending keywords have increasing metrics in recent days
                    multiplier = 1 + (30 - days_ago) * 0.05
                    impressions = int(impressions * multiplier)
                    clicks = int(clicks * multiplier)

                if "declining" in keyword and days_ago >= 60:
                    # Make declining keywords have higher metrics in the past
                    multiplier = 1 + (days_ago - 60) * 0.05
                    impressions = int(impressions * multiplier)
                    clicks = int(clicks * multiplier)

                data.append(
                    {
                        "date": date,
                        "campaign_id": "123456789",
                        "ad_group_id": "987654321",
                        "keyword_text": keyword,
                        "match_type": "EXACT",
                        "impressions": impressions,
                        "clicks": clicks,
                        "conversions": conversions,
                        "cost": cost,
                        "ctr": clicks / impressions if impressions > 0 else 0,
                        "conversion_rate": conversions / clicks if clicks > 0 else 0,
                        "cpa": cost / conversions if conversions > 0 else 0,
                    }
                )

        return data

    def test_fetch_historical_data(self):
        """Test fetching historical data."""
        # Call the method with test parameters
        df = self.service.fetch_historical_data(
            metrics=["clicks", "impressions", "conversions", "cost"], days=90, aggregate_by="day"
        )

        # Check that the API was called correctly
        self.mock_ads_api.get_keyword_performance.assert_called_once_with(90, None)

        # Check that a DataFrame was returned
        self.assertIsInstance(df, pd.DataFrame)

        # Check that the DataFrame has the expected columns
        expected_columns = ["clicks", "impressions", "conversions", "cost"]
        for col in expected_columns:
            self.assertIn(col, df.columns)

        # Check that metrics_history was updated
        self.assertIn("all_day", self.service.metrics_history)

    def test_forecast_metrics(self):
        """Test forecasting performance metrics."""
        # Patch the _train_and_evaluate_models method to avoid actual model training
        with patch.object(self.service, "_train_and_evaluate_models") as mock_train_eval:
            # Set up the mock to return a model that will generate simple forecasts
            mock_model = MagicMock()
            mock_forecast = np.array(
                [
                    100.0,
                    101.0,
                    102.0,
                    103.0,
                    104.0,
                    105.0,
                    106.0,
                    107.0,
                    108.0,
                    109.0,
                    110.0,
                    111.0,
                    112.0,
                    113.0,
                ]
            )
            mock_lower = mock_forecast * 0.9
            mock_upper = mock_forecast * 1.1

            # Setup mock _generate_forecast method
            with patch.object(
                self.service,
                "_generate_forecast",
                return_value=(mock_forecast, mock_lower, mock_upper),
            ):
                # Setup mock _train_and_evaluate_models to return a simple model
                mock_train_eval.return_value = {
                    "best_model": mock_model,
                    "best_model_type": "arima",
                    "performance": {"mse": 10.0, "mae": 2.0, "mape": 5.0},
                }

                # Call the method with test parameters
                result = self.service.forecast_metrics(
                    metrics=["clicks", "impressions"],
                    days_to_forecast=14,
                    campaign_id="123456789",
                    model_type="auto",
                )

                # Check that the result has the expected structure
                self.assertIn("metrics", result)
                self.assertIn("campaign_id", result)
                self.assertEqual(result["campaign_id"], "123456789")
                self.assertEqual(result["forecast_horizon"], 14)

                # Check metrics data
                self.assertIn("clicks", result["metrics"])
                self.assertIn("impressions", result["metrics"])

                # Check each metric's forecast data
                for metric in ["clicks", "impressions"]:
                    metric_data = result["metrics"][metric]
                    self.assertIn("forecast", metric_data)
                    self.assertIn("lower_bound", metric_data)
                    self.assertIn("upper_bound", metric_data)
                    self.assertIn("dates", metric_data)

                    # Check arrays have the right length
                    self.assertEqual(len(metric_data["forecast"]), 14)
                    self.assertEqual(len(metric_data["lower_bound"]), 14)
                    self.assertEqual(len(metric_data["upper_bound"]), 14)
                    self.assertEqual(len(metric_data["dates"]), 14)

    def test_forecast_budget(self):
        """Test forecasting budget requirements."""
        # First mock the forecast_metrics method to return controlled data
        with patch.object(self.service, "forecast_metrics") as mock_forecast:
            # Create mock forecast results for conversions and cost
            conv_forecast = np.arange(10, 24).tolist()  # 14 days of increasing conversions
            cost_forecast = np.arange(100, 240, 10).tolist()  # 14 days of increasing costs

            mock_forecast.side_effect = [
                # First call for target metric (conversions)
                {
                    "metrics": {
                        "conversions": {
                            "forecast": conv_forecast,
                            "lower_bound": [x * 0.9 for x in conv_forecast],
                            "upper_bound": [x * 1.1 for x in conv_forecast],
                            "dates": [
                                (datetime.now().date() + timedelta(days=i + 1)).isoformat()
                                for i in range(14)
                            ],
                        }
                    }
                },
                # Second call for cost
                {
                    "metrics": {
                        "cost": {
                            "forecast": cost_forecast,
                            "lower_bound": [x * 0.9 for x in cost_forecast],
                            "upper_bound": [x * 1.1 for x in cost_forecast],
                            "dates": [
                                (datetime.now().date() + timedelta(days=i + 1)).isoformat()
                                for i in range(14)
                            ],
                        }
                    }
                },
            ]

            # Also mock the fetch_historical_data method to return controlled data
            with patch.object(self.service, "fetch_historical_data") as mock_fetch:
                # Create a DataFrame with historical data
                dates = pd.date_range(end=datetime.now(), periods=90)
                mock_df = pd.DataFrame(
                    {
                        "cost": np.random.uniform(100, 200, 90),
                        "clicks": np.random.randint(50, 150, 90),
                        "conversions": np.random.randint(5, 15, 90),
                        "impressions": np.random.randint(1000, 2000, 90),
                    },
                    index=dates,
                )

                # Make historical_key point to this DataFrame
                self.service.metrics_history["123456789_day"] = mock_df

                # Call the method with test parameters
                result = self.service.forecast_budget(
                    campaign_id="123456789",
                    days_to_forecast=14,
                    target_metric="conversions",
                    target_value=200,
                )

                # Check that the result has the expected structure
                self.assertIn("campaign_id", result)
                self.assertEqual(result["campaign_id"], "123456789")
                self.assertIn("forecast_horizon", result)
                self.assertEqual(result["forecast_horizon"], 14)

                # Check historical performance metrics
                self.assertIn("historical_performance", result)
                self.assertIn("cpc", result["historical_performance"])
                self.assertIn("ctr", result["historical_performance"])
                self.assertIn("cpa", result["historical_performance"])

                # Check forecasts
                self.assertIn("metric_forecast", result)
                self.assertIn("cost_forecast", result)
                self.assertIn("total_forecasted_cost", result)

                # If target value was provided, check for required budget
                self.assertIn("required_budget_for_target", result)
                self.assertIn("suggested_daily_budget", result)

    def test_detect_search_trends(self):
        """Test detecting search trends."""
        result = self.service.detect_search_trends(days_lookback=90, min_growth_rate=0.1)

        # Check that the result has the expected structure
        self.assertIn("analysis_date", result)
        self.assertIn("days_analyzed", result)
        self.assertEqual(result["days_analyzed"], 90)

        # Check trend lists
        self.assertIn("trending_keywords", result)
        self.assertIn("declining_keywords", result)
        self.assertIn("stable_volume_keywords", result)

        # Check that trending keywords include the expected keywords
        trending_found = False
        for keyword in result["trending_keywords"]:
            if "trending" in keyword["keyword"]:
                trending_found = True
                break
        self.assertTrue(trending_found, "Expected to find trending keywords in the results")

        # Check that declining keywords include the expected keywords
        declining_found = False
        for keyword in result["declining_keywords"]:
            if "declining" in keyword["keyword"]:
                declining_found = True
                break
        self.assertTrue(declining_found, "Expected to find declining keywords in the results")

    def test_get_demand_forecasts(self):
        """Test retrieving demand forecasts."""
        # Test case when the API client doesn't have get_demand_forecasts method
        result = self.service.get_demand_forecasts()

        # Check that the result has the expected structure for unsupported feature
        self.assertIn("status", result)
        self.assertEqual(result["status"], "warning")
        self.assertIn("message", result)
        self.assertIn("documentation_url", result)

        # Now test with a mock API that supports demand forecasts
        self.mock_ads_api.get_demand_forecasts = MagicMock(
            return_value=[
                {
                    "keyword": "holiday gifts",
                    "forecast_start_date": "2025-05-01",
                    "predicted_search_interest": 150,
                    "peak_date": "2025-05-15",
                }
            ]
        )

        result = self.service.get_demand_forecasts()

        # Check that the result has the expected structure
        self.assertIn("status", result)
        self.assertEqual(result["status"], "success")
        self.assertIn("forecast_date", result)
        self.assertIn("demand_forecasts", result)

    def test_run_method(self):
        """Test the run method with different operations."""
        # Test forecast_metrics operation
        with patch.object(self.service, "forecast_metrics") as mock_forecast:
            mock_forecast.return_value = {"status": "success"}

            result = self.service.run(
                operation="forecast_metrics",
                metrics=["clicks", "impressions"],
                days=14,
                campaign_id="123456789",
            )

            mock_forecast.assert_called_once_with(
                ["clicks", "impressions"], 14, "123456789", "auto"
            )
            self.assertEqual(result, {"status": "success"})

        # Test forecast_budget operation
        with patch.object(self.service, "forecast_budget") as mock_budget:
            mock_budget.return_value = {"status": "success"}

            result = self.service.run(
                operation="forecast_budget",
                campaign_id="123456789",
                days=30,
                target_metric="conversions",
                target_value=100,
            )

            mock_budget.assert_called_once_with("123456789", 30, "conversions", 100)
            self.assertEqual(result, {"status": "success"})

        # Test detect_trends operation
        with patch.object(self.service, "detect_search_trends") as mock_trends:
            mock_trends.return_value = {"status": "success"}

            result = self.service.run(
                operation="detect_trends", days_lookback=60, min_growth_rate=0.15
            )

            mock_trends.assert_called_once_with(60, 0.15)
            self.assertEqual(result, {"status": "success"})

        # Test demand_forecasts operation
        with patch.object(self.service, "get_demand_forecasts") as mock_demand:
            mock_demand.return_value = {"status": "success"}

            result = self.service.run(operation="demand_forecasts")

            mock_demand.assert_called_once()
            self.assertEqual(result, {"status": "success"})

        # Test invalid operation
        result = self.service.run(operation="invalid_operation")
        self.assertEqual(result["status"], "error")


if __name__ == "__main__":
    unittest.main()
