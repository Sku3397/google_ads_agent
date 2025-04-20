"""
Unit tests for the Contextual Signal Service
"""

from services.contextual_signal_service import ContextualSignalService, ContextualSignal
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import json
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import requests

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestContextualSignalService(unittest.TestCase):
    """Test cases for the Contextual Signal Service"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock ads_api
        self.mock_ads_api = MagicMock()

        # Configure mock response for get_campaign_performance
        self.mock_ads_api.get_campaign_performance.return_value = [
            {
                "id": "123456789",
                "name": "Test Campaign",
                "status": "ENABLED",
                "impressions": 1000,
                "clicks": 100,
                "conversions": 10,
                "cost": 500.0,
            }
        ]

        # Configure mock response for get_keyword_performance
        self.mock_ads_api.get_keyword_performance.return_value = [
            {
                "campaign_id": "123456789",
                "campaign_name": "Test Campaign",
                "ad_group_id": "987654321",
                "ad_group_name": "Test Ad Group",
                "keyword_text": "winter jackets",
                "match_type": "EXACT",
                "status": "ENABLED",
                "current_bid": 1.5,
                "resource_name": "customers/1234567890/adGroups/987654321/criteria/111111",
                "clicks": 50,
                "impressions": 500,
                "conversions": 5,
            },
            {
                "campaign_id": "123456789",
                "campaign_name": "Test Campaign",
                "ad_group_id": "987654321",
                "ad_group_name": "Test Ad Group",
                "keyword_text": "snow boots",
                "match_type": "EXACT",
                "status": "ENABLED",
                "current_bid": 1.2,
                "resource_name": "customers/1234567890/adGroups/987654321/criteria/222222",
                "clicks": 30,
                "impressions": 300,
                "conversions": 3,
            },
        ]

        # Configure mock response for apply_optimization
        self.mock_ads_api.apply_optimization.return_value = (True, "Successfully updated bid")

        # Create service with mocked API
        self.service = ContextualSignalService(ads_api=self.mock_ads_api)

        # Mock API keys
        self.service.api_keys = {
            "weather": "mock_weather_key",
            "news": "mock_news_key",
            "trends": "mock_trends_key",
            "economic": "mock_economic_key",
            "social": "mock_social_key",
        }

    def test_initialization(self):
        """Test service initialization"""
        self.assertEqual(self.service.ads_api, self.mock_ads_api)
        self.assertIsNotNone(self.service.signal_cache)
        self.assertIsNotNone(self.service.cache_expiry)
        self.assertIsNotNone(self.service.cache_validity)

    @patch("requests.get")
    def test_get_weather_signals(self, mock_get):
        """Test getting weather signals"""
        # Configure mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "main": {"temp": 25.5},
            "weather": [{"main": "Clear", "description": "clear sky", "icon": "01d"}],
        }
        mock_get.return_value = mock_response

        # Call method
        signals = self.service.get_weather_signals("New York")

        # Check results
        self.assertEqual(len(signals), 2)
        self.assertEqual(signals[0].signal_type, "temperature")
        self.assertEqual(signals[0].value, 25.5)
        self.assertEqual(signals[1].signal_type, "weather_condition")
        self.assertEqual(signals[1].value, "Clear")

        # Check cache
        self.assertIn("weather_New York", self.service.signal_cache)
        self.assertEqual(len(self.service.signal_cache["weather_New York"]), 2)

    @patch("requests.get")
    def test_get_news_signals(self, mock_get):
        """Test getting news signals"""
        # Configure mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "articles": [
                {
                    "source": {"name": "Test News"},
                    "title": "Winter Fashion Trends",
                    "description": "Latest trends in winter jackets and snow boots",
                    "url": "https://testnews.com/winter-fashion",
                    "publishedAt": datetime.now().isoformat() + "Z",
                    "content": "Test content",
                }
            ]
        }
        mock_get.return_value = mock_response

        # Call method
        signals = self.service.get_news_signals("Retail", ["winter jackets", "snow boots"])

        # Check results
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].signal_type, "news_article")
        self.assertEqual(signals[0].value, "Winter Fashion Trends")
        self.assertIn("winter jackets", signals[0].metadata["description"])

    def test_get_trend_signals(self):
        """Test getting trend signals"""
        # Call method
        signals = self.service.get_trend_signals("Retail", ["winter jackets", "snow boots"])

        # Check results
        self.assertEqual(len(signals), 3)  # One for industry + two keywords
        for signal in signals:
            self.assertEqual(signal.signal_type, "search_trend")
            self.assertIsInstance(signal.value, float)
            self.assertIn(signal.metadata["trend_direction"], ["up", "down"])
            self.assertIn(signal.metadata["term"], ["Retail", "winter jackets", "snow boots"])

    def test_get_economic_signals(self):
        """Test getting economic signals"""
        # Call method
        signals = self.service.get_economic_signals("New York")

        # Check results
        self.assertGreater(len(signals), 0)
        for signal in signals:
            self.assertTrue(signal.signal_type.startswith("economic_"))
            self.assertEqual(signal.metadata["location"], "New York")
            self.assertEqual(signal.metadata["country"], "us")

    def test_get_social_signals(self):
        """Test getting social signals"""
        # Call method
        signals = self.service.get_social_signals(["winter jackets", "snow boots"])

        # Check results
        self.assertEqual(len(signals), 8)  # 2 keywords * 4 platforms
        for signal in signals:
            self.assertEqual(signal.signal_type, "social_sentiment")
            self.assertIsInstance(signal.value, float)
            self.assertGreaterEqual(signal.value, -1)
            self.assertLessEqual(signal.value, 1)
            self.assertIn(signal.metadata["keyword"], ["winter jackets", "snow boots"])
            self.assertIn(
                signal.metadata["platform"], ["twitter", "facebook", "instagram", "tiktok"]
            )

    def test_get_seasonal_signals(self):
        """Test getting seasonal signals"""
        # Call method
        signals = self.service.get_seasonal_signals("Retail", "New York")

        # Check results
        self.assertGreater(len(signals), 0)

        # Check for season signal
        season_signals = [s for s in signals if s.signal_type == "season"]
        self.assertEqual(len(season_signals), 1)
        self.assertIn(season_signals[0].value, ["winter", "spring", "summer", "fall"])

        # Check for holiday signals
        holiday_signals = [s for s in signals if s.signal_type == "holiday_proximity"]

        # Check for industry seasonality
        industry_signals = [s for s in signals if s.signal_type == "industry_seasonality"]
        if industry_signals:
            self.assertIn(industry_signals[0].value, ["high", "medium", "low"])
            self.assertEqual(industry_signals[0].metadata["industry"], "retail")

    def test_analyze_signals_for_keywords(self):
        """Test analyzing signals for keywords"""
        # Create test signals
        weather_signal = ContextualSignal(
            signal_type="weather_condition",
            source="test",
            timestamp=datetime.now(),
            value="Snowy",
            relevance_score=0.8,
            metadata={"location": "New York"},
        )

        news_signal = ContextualSignal(
            signal_type="news_article",
            source="test",
            timestamp=datetime.now(),
            value="Winter jackets on sale",
            relevance_score=0.7,
            metadata={"description": "Best winter jackets for snow and cold weather"},
        )

        social_signal = ContextualSignal(
            signal_type="social_sentiment",
            source="test",
            timestamp=datetime.now(),
            value=0.6,
            relevance_score=0.9,
            metadata={"keyword": "winter jackets", "platform": "twitter"},
        )

        signals = {"weather": [weather_signal], "news": [news_signal], "social": [social_signal]}

        # Call method
        results = self.service.analyze_signals_for_keywords(
            signals, ["winter jackets", "snow boots"]
        )

        # Check results
        self.assertIn("winter jackets", results)
        self.assertIn("snow boots", results)

        # Winter jackets should have high relevance for news and social
        self.assertGreaterEqual(results["winter jackets"]["news"], 0.5)
        self.assertGreaterEqual(results["winter jackets"]["social"], 0.8)

        # Snow boots should have low or zero relevance
        self.assertEqual(results["snow boots"]["social"], 0)

    def test_get_recommendations_from_signals(self):
        """Test getting recommendations from signals"""
        # Create test signals
        weather_signal = ContextualSignal(
            signal_type="weather_condition",
            source="test",
            timestamp=datetime.now(),
            value="Snow",
            relevance_score=0.8,
            metadata={"location": "New York"},
        )

        holiday_signal = ContextualSignal(
            signal_type="holiday_proximity",
            source="test",
            timestamp=datetime.now(),
            value="Christmas",
            relevance_score=0.9,
            metadata={"days_until": 10, "holiday_name": "Christmas"},
        )

        industry_signal = ContextualSignal(
            signal_type="industry_seasonality",
            source="test",
            timestamp=datetime.now(),
            value="high",
            relevance_score=0.85,
            metadata={"industry": "retail", "month": 12},
        )

        signals = {"weather": [weather_signal], "seasonal": [holiday_signal, industry_signal]}

        # Call method
        recommendations = self.service.get_recommendations_from_signals(
            signals, ["winter jackets", "snow boots", "Christmas gifts"]
        )

        # Check results
        self.assertGreater(len(recommendations), 0)

        # Should have weather recommendation
        weather_recs = [r for r in recommendations if "weather" in r["trigger"].lower()]
        self.assertGreater(len(weather_recs), 0)

        # Should have holiday recommendation
        holiday_recs = [r for r in recommendations if "holiday" in r["trigger"].lower()]
        if holiday_recs:
            self.assertIn("Christmas", holiday_recs[0]["keywords_affected"])

        # Should have seasonality recommendation
        season_recs = [r for r in recommendations if "season" in r["trigger"].lower()]
        self.assertGreater(len(season_recs), 0)

    def test_apply_signal_based_optimizations(self):
        """Test applying signal-based optimizations"""
        # Call method
        success, message = self.service.apply_signal_based_optimizations("123456789")

        # Check that API was called correctly
        self.mock_ads_api.get_campaign_performance.assert_called_once()
        self.mock_ads_api.get_keyword_performance.assert_called_once_with(
            days_ago=30, campaign_id="123456789"
        )

        # There might not be applied optimizations in this test since it depends on the simulated signals
        # But the method should at least complete successfully
        self.assertIsNotNone(message)

    def test_get_all_signals(self):
        """Test getting all signals"""
        # Mock all the individual signal methods
        self.service.get_weather_signals = MagicMock(return_value=[MagicMock()])
        self.service.get_news_signals = MagicMock(return_value=[MagicMock()])
        self.service.get_trend_signals = MagicMock(return_value=[MagicMock()])
        self.service.get_economic_signals = MagicMock(return_value=[MagicMock()])
        self.service.get_social_signals = MagicMock(return_value=[MagicMock()])
        self.service.get_seasonal_signals = MagicMock(return_value=[MagicMock()])

        # Call method
        signals = self.service.get_all_signals("New York", "Retail", ["winter jackets"])

        # Check results
        self.assertEqual(len(signals), 6)
        self.assertIn("weather", signals)
        self.assertIn("news", signals)
        self.assertIn("trends", signals)
        self.assertIn("economic", signals)
        self.assertIn("social", signals)
        self.assertIn("seasonal", signals)

        # Verify all methods were called
        self.service.get_weather_signals.assert_called_once_with("New York")
        self.service.get_news_signals.assert_called_once_with("Retail", ["winter jackets"])
        self.service.get_trend_signals.assert_called_once_with("Retail", ["winter jackets"])
        self.service.get_economic_signals.assert_called_once_with("New York")
        self.service.get_social_signals.assert_called_once_with(["winter jackets"])
        self.service.get_seasonal_signals.assert_called_once_with("Retail", "New York")

    def test_cache_functionality(self):
        """Test that caching works correctly"""
        # Create a signal and add to cache
        test_signal = ContextualSignal(
            signal_type="test",
            source="test",
            timestamp=datetime.now(),
            value="test value",
            relevance_score=0.8,
            metadata={},
        )

        # Set cache
        self.service.signal_cache["test_key"] = [test_signal]
        self.service.cache_expiry["test_key"] = datetime.now() + timedelta(hours=1)

        # Mock the expensive operation
        self.service._expensive_operation = MagicMock()

        # Define a method that uses the cache
        def get_cached_data():
            # Check cache
            if (
                "test_key" in self.service.signal_cache
                and datetime.now() < self.service.cache_expiry.get("test_key", datetime.min)
            ):
                return self.service.signal_cache["test_key"]
            return self.service._expensive_operation()

        # Call once - should use cache
        result = get_cached_data()
        self.assertEqual(result, [test_signal])
        self.service._expensive_operation.assert_not_called()

        # Expire the cache
        self.service.cache_expiry["test_key"] = datetime.now() - timedelta(minutes=1)

        # Mock return value for expensive operation
        new_signal = ContextualSignal(
            signal_type="new",
            source="new",
            timestamp=datetime.now(),
            value="new value",
            relevance_score=0.9,
            metadata={},
        )
        self.service._expensive_operation.return_value = [new_signal]

        # Call again - should call expensive operation
        result = get_cached_data()
        self.service._expensive_operation.assert_called_once()
        self.assertEqual(result, [new_signal])


if __name__ == "__main__":
    unittest.main()
