"""
Unit tests for the Trend Forecasting Service
"""

from services.trend_forecasting_service import TrendForecastingService
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import json
from datetime import datetime, timedelta
import tempfile

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestTrendForecastingService(unittest.TestCase):
    """Test cases for the Trend Forecasting Service"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock ads_api
        self.mock_ads_api = MagicMock()

        # Mock keyword performance data
        self.mock_keyword_data = [
            {
                "keyword_text": "test keyword",
                "clicks": 100,
                "impressions": 1000,
                "cost": 50.0,
                "conversions": 5,
                "date": "2023-01-01",
            },
            {
                "keyword_text": "test keyword",
                "clicks": 120,
                "impressions": 1100,
                "cost": 55.0,
                "conversions": 6,
                "date": "2023-01-02",
            },
            {
                "keyword_text": "test keyword",
                "clicks": 110,
                "impressions": 1050,
                "cost": 52.5,
                "conversions": 5.5,
                "date": "2023-01-03",
            },
            {
                "keyword_text": "another keyword",
                "clicks": 50,
                "impressions": 500,
                "cost": 25.0,
                "conversions": 2.5,
                "date": "2023-01-01",
            },
            {
                "keyword_text": "another keyword",
                "clicks": 60,
                "impressions": 600,
                "cost": 30.0,
                "conversions": 3,
                "date": "2023-01-02",
            },
        ]

        # Mock campaign performance data
        self.mock_campaign_data = [
            {
                "id": "123456789",
                "name": "Test Campaign",
                "clicks": 500,
                "impressions": 5000,
                "cost": 250.0,
                "conversions": 25,
                "date": "2023-01-01",
            },
            {
                "id": "123456789",
                "name": "Test Campaign",
                "clicks": 520,
                "impressions": 5200,
                "cost": 260.0,
                "conversions": 26,
                "date": "2023-01-02",
            },
            {
                "id": "123456789",
                "name": "Test Campaign",
                "clicks": 540,
                "impressions": 5400,
                "cost": 270.0,
                "conversions": 27,
                "date": "2023-01-03",
            },
        ]

        # Configure mock responses
        self.mock_ads_api.get_keyword_performance.return_value = self.mock_keyword_data
        self.mock_ads_api.get_campaign_performance.return_value = self.mock_campaign_data

        # Use temp directory for test data files
        self.test_dir = tempfile.mkdtemp()

        # Create service with mocked API
        self.service = TrendForecastingService(ads_api=self.mock_ads_api)

        # Override service directories to use temp directory
        self.service.model_dir = os.path.join(self.test_dir, "models")
        self.service.forecast_dir = os.path.join(self.test_dir, "forecasts")

        # Create directories
        os.makedirs(self.service.model_dir, exist_ok=True)
        os.makedirs(self.service.forecast_dir, exist_ok=True)

        # Patch matplotlib to avoid displaying plots during tests
        plt.show = lambda: None

    def tearDown(self):
        """Clean up after tests"""
        # Remove temp directories
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_initialization(self):
        """Test service initialization"""
        self.assertEqual(self.service.ads_api, self.mock_ads_api)
        self.assertEqual(self.service.forecast_horizons["short_term"], 7)
        self.assertEqual(self.service.forecast_horizons["medium_term"], 30)
        self.assertEqual(self.service.forecast_horizons["long_term"], 90)
        self.assertIsNotNone(self.service.model_configs)
        self.assertIsNotNone(self.service.forecast_history)

    def test_get_keyword_historical_data(self):
        """Test getting keyword historical data"""
        # Call method
        df = self.service._get_keyword_historical_data("test keyword", "123456789", 30)

        # Check that API was called
        self.mock_ads_api.get_keyword_performance.assert_called_with(
            days_ago=30, campaign_id="123456789"
        )

        # Check results
        self.assertEqual(len(df), 3)  # Only "test keyword" entries
        self.assertIn("keyword_text", df.columns)
        self.assertIn("clicks", df.columns)
        self.assertIn("impressions", df.columns)
        self.assertIn("date", df.columns)

    def test_get_campaign_keyword_data(self):
        """Test getting campaign keyword data"""
        # Call method
        df = self.service._get_campaign_keyword_data("123456789", 30)

        # Check that API was called
        self.mock_ads_api.get_keyword_performance.assert_called_with(
            days_ago=30, campaign_id="123456789"
        )

        # Check results
        self.assertEqual(len(df), 5)  # All keyword entries
        self.assertIn("keyword", df.columns)  # Renamed from keyword_text
        self.assertIn("clicks", df.columns)
        self.assertIn("impressions", df.columns)
        self.assertIn("date", df.columns)

    def test_get_campaign_historical_data(self):
        """Test getting campaign historical data"""
        # Call method
        df = self.service._get_campaign_historical_data("123456789", 30)

        # Check that API was called
        self.mock_ads_api.get_campaign_performance.assert_called_with(days_ago=30)

        # Check results
        self.assertEqual(len(df), 3)  # All campaign entries
        self.assertIn("id", df.columns)
        self.assertIn("name", df.columns)
        self.assertIn("clicks", df.columns)
        self.assertIn("impressions", df.columns)
        self.assertIn("date", df.columns)

    def test_get_account_historical_data(self):
        """Test getting account historical data"""
        # Call method
        df = self.service._get_account_historical_data(30)

        # Check that API was called
        self.mock_ads_api.get_campaign_performance.assert_called_with(days_ago=30)

        # Check results
        self.assertEqual(len(df), 3)  # Unique dates from campaign data
        self.assertIn("clicks", df.columns)
        self.assertIn("impressions", df.columns)
        self.assertIn("date", df.columns)

    def test_prepare_data_for_forecasting(self):
        """Test preparing data for forecasting"""
        # Create test data
        df = pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-01", periods=10),
                "clicks": np.random.randint(50, 200, 10),
                "impressions": np.random.randint(500, 2000, 10),
            }
        )

        # Call method
        forecast_df = self.service._prepare_data_for_forecasting(df, "clicks")

        # Check results
        self.assertIn("ds", forecast_df.columns)  # Renamed date column
        self.assertIn("y", forecast_df.columns)  # Renamed clicks column
        self.assertEqual(len(forecast_df), 10)

    @patch("fbprophet.Prophet")
    def test_prophet_forecast(self, mock_prophet_class):
        """Test Prophet forecasting"""
        # Create mock Prophet instance
        mock_prophet = MagicMock()
        mock_prophet_class.return_value = mock_prophet

        # Mock Prophet fit and predict methods
        mock_prophet.fit.return_value = None

        # Create mock forecast
        mock_forecast = pd.DataFrame(
            {
                "ds": pd.date_range(start="2023-01-01", periods=13),
                "yhat": np.random.randint(100, 200, 13),
                "yhat_lower": np.random.randint(50, 100, 13),
                "yhat_upper": np.random.randint(200, 300, 13),
            }
        )
        mock_prophet.predict.return_value = mock_forecast

        # Create mock future dataframe
        mock_future = pd.DataFrame({"ds": pd.date_range(start="2023-01-01", periods=13)})
        mock_prophet.make_future_dataframe.return_value = mock_future

        # Create test data
        df = pd.DataFrame(
            {
                "ds": pd.date_range(start="2023-01-01", periods=10),
                "y": np.random.randint(50, 200, 10),
            }
        )

        # Call method
        result = self.service._prophet_forecast(df, 3)

        # Check result structure
        self.assertIn("forecasted_value", result)
        self.assertIn("lower_bound", result)
        self.assertIn("upper_bound", result)
        self.assertIn("confidence", result)
        self.assertIn("features_used", result)

        # Check that Prophet was called correctly
        mock_prophet.fit.assert_called_once()
        mock_prophet.make_future_dataframe.assert_called_once_with(periods=3)
        mock_prophet.predict.assert_called_once()

    def test_simple_forecast(self):
        """Test simple forecasting"""
        # Create test data
        df = pd.DataFrame(
            {
                "ds": pd.date_range(start="2023-01-01", periods=10),
                "y": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],  # Linear trend
            }
        )

        # Call method
        result = self.service._simple_forecast(df, 5)

        # Check result structure
        self.assertIn("forecasted_value", result)
        self.assertIn("lower_bound", result)
        self.assertIn("upper_bound", result)
        self.assertIn("confidence", result)

        # Check that forecast follows the trend
        self.assertGreater(result["forecasted_value"], 190)  # Last value was 190

    @patch(
        "services.trend_forecasting_service.trend_forecasting_service.TrendForecastingService._prophet_forecast"
    )
    def test_forecast_keyword_performance(self, mock_prophet_forecast):
        """Test forecasting keyword performance"""
        # Mock Prophet forecast result
        mock_prophet_forecast.return_value = {
            "forecasted_value": 150.0,
            "lower_bound": 120.0,
            "upper_bound": 180.0,
            "confidence": 0.8,
            "features_used": ["clicks"],
        }

        # Call method
        forecast = self.service.forecast_keyword_performance(
            keyword="test keyword",
            campaign_id="123456789",
            horizon="medium_term",
            metric="clicks",
            model_type="prophet",
        )

        # Check results
        self.assertEqual(forecast.keyword, "test keyword")
        self.assertEqual(forecast.forecasted_value, 150.0)
        self.assertEqual(forecast.lower_bound, 120.0)
        self.assertEqual(forecast.upper_bound, 180.0)
        self.assertEqual(forecast.confidence, 0.8)
        self.assertEqual(forecast.model_type, "prophet")

        # Check that the API was called
        self.mock_ads_api.get_keyword_performance.assert_called()

    @patch(
        "services.trend_forecasting_service.trend_forecasting_service.TrendForecastingService._calculate_trend_metrics"
    )
    def test_detect_emerging_trends(self, mock_calculate_trend_metrics):
        """Test detecting emerging trends"""
        # Mock trend metrics calculation
        mock_calculate_trend_metrics.return_value = {
            "recent_growth_rate": 0.25,
            "current_volume": 100,
            "predicted_volume": 125,
            "trend_strength": 0.8,
            "seasonality_impact": 1.2,
            "confidence": 0.75,
        }

        # Call method
        trends = self.service.detect_emerging_trends(
            campaign_id="123456789", lookback_days=90, min_growth_rate=0.2
        )

        # Check that API was called
        self.mock_ads_api.get_keyword_performance.assert_called()

        # Check results
        self.assertEqual(len(trends), 2)  # Two unique keywords in our mock data
        self.assertIn("keyword", trends[0])
        self.assertIn("growth_rate", trends[0])
        self.assertIn("confidence", trends[0])

    @patch("services.trend_forecasting_service.trend_forecasting_service.seasonal_decompose")
    def test_identify_seasonal_patterns(self, mock_seasonal_decompose):
        """Test identifying seasonal patterns"""
        # Create mock decomposition
        mock_decomposition = MagicMock()
        mock_seasonal = pd.Series(
            np.sin(np.arange(14) * 2 * np.pi / 7) * 0.2 + 1,  # Weekly pattern
            index=pd.date_range(start="2023-01-01", periods=14),
        )
        mock_decomposition.seasonal = mock_seasonal
        mock_decomposition.resid = pd.Series(
            np.random.normal(0, 0.05, 14), index=pd.date_range(start="2023-01-01", periods=14)
        )

        mock_seasonal_decompose.return_value = mock_decomposition

        # Call method
        patterns = self.service.identify_seasonal_patterns(
            campaign_id="123456789", lookback_days=90, metric="clicks"
        )

        # Check that API was called
        self.mock_ads_api.get_campaign_performance.assert_called()

        # This might not find patterns with our limited test data
        # But at least check it runs without errors
        self.assertIsInstance(patterns, list)

    @patch(
        "services.trend_forecasting_service.trend_forecasting_service.TrendForecastingService.detect_emerging_trends"
    )
    @patch(
        "services.trend_forecasting_service.trend_forecasting_service.TrendForecastingService.identify_seasonal_patterns"
    )
    @patch(
        "services.trend_forecasting_service.trend_forecasting_service.TrendForecastingService.forecast_keyword_performance"
    )
    def test_generate_trend_report(self, mock_forecast, mock_seasonal, mock_trends):
        """Test generating trend report"""
        # Mock method results
        mock_trends.return_value = [{"keyword": "test", "growth_rate": 0.2}]
        mock_seasonal.return_value = [{"period": 7, "period_name": "Weekly", "strength": 0.8}]

        from services.trend_forecasting_service.trend_forecasting_service import TrendForecast

        mock_forecast.return_value = TrendForecast(
            keyword="test",
            forecast_date=datetime.now() + timedelta(days=30),
            forecasted_value=150.0,
            lower_bound=120.0,
            upper_bound=180.0,
            confidence=0.8,
            model_type="prophet",
            features_used=["clicks"],
        )

        # Mock top keywords method
        self.service._get_top_keywords_for_campaign = MagicMock(return_value=["test keyword"])

        # Create simpler _save_trend_visualizations to avoid PIL dependency in tests
        self.service._save_trend_visualizations = MagicMock()

        # Call method
        report = self.service.generate_trend_report(
            campaign_id="123456789", lookback_days=90, forecast_horizon="medium_term"
        )

        # Check results
        self.assertIn("emerging_trends", report)
        self.assertIn("seasonal_patterns", report)
        self.assertIn("top_keyword_forecasts", report)
        self.assertIn("visualizations", report)

        # Check that API was called
        self.mock_ads_api.get_campaign_performance.assert_called()

    def test_discover_trending_keywords(self):
        """Test discovering trending keywords"""
        # Call method
        keywords = self.service.discover_trending_keywords(
            industry="retail", location="New York", limit=5
        )

        # Check results
        self.assertEqual(len(keywords), 5)
        self.assertIn("keyword", keywords[0])
        self.assertIn("trend_score", keywords[0])
        self.assertIn("growth_rate", keywords[0])
        self.assertIn("volume", keywords[0])

    def test_calculate_trend_metrics(self):
        """Test calculating trend metrics"""
        # Create test data
        kw_data = pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-01", periods=14),
                "clicks": [
                    100,
                    105,
                    110,
                    115,
                    120,
                    125,
                    130,
                    135,
                    140,
                    145,
                    150,
                    155,
                    160,
                    165,
                ],  # Linear trend
            }
        )

        # Call method
        metrics = self.service._calculate_trend_metrics(kw_data)

        # Check results
        self.assertIn("recent_growth_rate", metrics)
        self.assertIn("current_volume", metrics)
        self.assertIn("predicted_volume", metrics)
        self.assertIn("trend_strength", metrics)
        self.assertIn("confidence", metrics)

        # Growth rate should be positive
        self.assertGreater(metrics["recent_growth_rate"], 0)

        # Trend strength should be high for linear data
        self.assertGreater(metrics["trend_strength"], 0.8)

    def test_analyze_overall_trend(self):
        """Test analyzing overall trend"""
        # Create test data
        df = pd.DataFrame(
            {
                "date": pd.date_range(start="2023-01-01", periods=14),
                "clicks": [
                    100,
                    105,
                    110,
                    115,
                    120,
                    125,
                    130,
                    135,
                    140,
                    145,
                    150,
                    155,
                    160,
                    165,
                ],  # Linear trend
                "impressions": [
                    1000,
                    1050,
                    1100,
                    1150,
                    1200,
                    1250,
                    1300,
                    1350,
                    1400,
                    1450,
                    1500,
                    1550,
                    1600,
                    1650,
                ],
            }
        )

        # Call method
        analysis = self.service._analyze_overall_trend(df)

        # Check results
        self.assertIn("clicks", analysis)
        self.assertIn("impressions", analysis)
        self.assertIn("stability", analysis)

        # Trend direction should be up for both metrics
        self.assertEqual(analysis["clicks"]["direction"], "up")
        self.assertEqual(analysis["impressions"]["direction"], "up")

        # Trend strength should be high for linear data
        self.assertGreater(analysis["clicks"]["strength"], 0.8)
        self.assertGreater(analysis["impressions"]["strength"], 0.8)


if __name__ == "__main__":
    unittest.main()
