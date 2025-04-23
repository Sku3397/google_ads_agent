"""Tests for the SimulationService."""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Correct import path relative to project root
from services.simulation_service.simulation_service import SimulationService


class TestSimulationService(unittest.TestCase):
    """Test cases for SimulationService."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_ads_client = Mock()
        self.customer_id = "123456789"
        self.service = SimulationService(self.mock_ads_client, self.customer_id)

    def test_init(self):
        """Test service initialization."""
        self.assertEqual(self.service.customer_id, self.customer_id)
        self.assertIsNotNone(self.service.simulation_service)

    def test_simulate_bid_changes(self):
        """Test simulating bid changes."""
        # Mock keyword data
        mock_keyword_data = {
            "criterion_id": "123",
            "keyword_text": "test keyword",
            "current_bid": 1.0,
            "clicks": 100,
            "impressions": 1000,
            "cost": 100.0,
        }

        # Mock simulation data
        mock_simulation = {
            "estimated_clicks": 120,
            "estimated_impressions": 1200,
            "estimated_cost": 120.0,
            "estimated_conversions": 10,
            "simulation_date_range": {"start_date": "2024-01-01", "end_date": "2024-01-31"},
        }

        # Mock the internal methods
        self.service._get_keyword_data = Mock(return_value=mock_keyword_data)
        self.service._get_bid_simulation = Mock(return_value=mock_simulation)

        # Test the simulation
        results = self.service.simulate_bid_changes(
            keyword_ids=["123"], bid_modifiers=[1.2]  # 20% increase
        )

        # Verify results
        self.assertIn("123", results)
        self.assertEqual(results["123"]["current_bid"], 1.0)
        self.assertEqual(results["123"]["simulated_bid"], 1.2)
        self.assertEqual(results["123"]["estimated_metrics"], mock_simulation)

    def test_get_keyword_data(self):
        """Test getting keyword data."""
        # Mock the Google Ads API response
        mock_response = Mock()
        mock_row = Mock()
        mock_row.ad_group_criterion.criterion_id = "123"
        mock_row.ad_group_criterion.keyword.text = "test keyword"
        mock_row.ad_group_criterion.effective_cpc_bid_micros = 1000000  # $1.00
        mock_row.metrics.clicks = 100
        mock_row.metrics.impressions = 1000
        mock_row.metrics.cost_micros = 100000000  # $100.00
        mock_response.__iter__ = Mock(return_value=iter([mock_row]))
        self.mock_ads_client.get_service().search.return_value = mock_response

        # Test getting data
        result = self.service._get_keyword_data("123")

        # Verify the result
        self.assertEqual(result["criterion_id"], "123")
        self.assertEqual(result["keyword_text"], "test keyword")
        self.assertEqual(result["current_bid"], 1.0)
        self.assertEqual(result["clicks"], 100)
        self.assertEqual(result["impressions"], 1000)
        self.assertEqual(result["cost"], 100.0)

    def test_get_bid_simulation(self):
        """Test getting bid simulation data."""
        # Mock the Google Ads API response
        mock_response = Mock()
        mock_row = Mock()
        mock_row.ad_group_criterion_simulation.criterion_id = "123"
        mock_row.ad_group_criterion_simulation.start_date = "2024-01-01"
        mock_row.ad_group_criterion_simulation.end_date = "2024-01-31"

        # Mock simulation points
        mock_point = Mock()
        mock_point.cpc_bid_micros = 1200000  # $1.20
        mock_point.clicks = 120
        mock_point.impressions = 1200
        mock_point.cost_micros = 120000000  # $120.00
        mock_point.conversions = 10

        mock_row.ad_group_criterion_simulation.cpc_bid_point_list.points = [mock_point]
        mock_response.__iter__ = Mock(return_value=iter([mock_row]))
        self.mock_ads_client.get_service().search.return_value = mock_response

        # Test getting simulation
        result = self.service._get_bid_simulation("123", 1.2)

        # Verify the result
        self.assertEqual(result["estimated_clicks"], 120)
        self.assertEqual(result["estimated_impressions"], 1200)
        self.assertEqual(result["estimated_cost"], 120.0)
        self.assertEqual(result["estimated_conversions"], 10)
        self.assertEqual(result["simulation_date_range"]["start_date"], "2024-01-01")
        self.assertEqual(result["simulation_date_range"]["end_date"], "2024-01-31")

    def test_get_performance_forecast(self):
        """Test getting performance forecast."""
        # Mock the Google Ads API response
        mock_response = Mock()
        mock_row = Mock()
        mock_row.campaign.id = "123"
        mock_row.metrics.clicks = 100
        mock_row.metrics.impressions = 1000
        mock_row.metrics.cost_micros = 100000000  # $100.00
        mock_row.metrics.conversions = 10
        mock_row.segments.date = "2024-01-01"
        mock_response.__iter__ = Mock(return_value=iter([mock_row]))
        self.mock_ads_client.get_service().search.return_value = mock_response

        # Test getting forecast
        result = self.service.get_performance_forecast("123", days=30)

        # Verify the result
        self.assertIn("forecast_metrics", result)
        self.assertIn("confidence_intervals", result)
        self.assertIn("forecast_dates", result)

    def test_error_handling(self):
        """Test error handling in the service."""
        # Test with invalid input
        with self.assertRaises(ValueError):
            self.service.simulate_bid_changes(
                keyword_ids=["123", "456"], bid_modifiers=[1.2]  # Mismatched length
            )

        # Test API error
        self.mock_ads_client.get_service().search.side_effect = Exception("API Error")
        result = self.service._get_keyword_data("123")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
