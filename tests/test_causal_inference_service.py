"""Tests for the CausalInferenceService."""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from services.causal_inference_service.causal_inference_service import CausalInferenceService


class TestCausalInferenceService(unittest.TestCase):
    """Test cases for CausalInferenceService."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_ads_client = Mock()
        self.config = {
            "min_pre_period_days": 30,
            "min_post_period_days": 14,
            "significance_level": 0.05,
        }
        self.service = CausalInferenceService(self.mock_ads_client, self.config)

    def test_init(self):
        """Test service initialization."""
        self.assertEqual(self.service.min_pre_period_days, 30)
        self.assertEqual(self.service.min_post_period_days, 14)
        self.assertEqual(self.service.significance_level, 0.05)

    @patch("services.causal_inference_service.causal_inference_service.CausalImpact")
    def test_analyze_campaign_change_impact(self, mock_causal_impact):
        """Test analyzing campaign change impact."""
        # Mock the Google Ads API response
        mock_response = Mock()
        mock_row = Mock()
        mock_row.segments.date = "2024-01-01"
        mock_row.metrics.clicks = 100
        mock_response.__iter__ = Mock(return_value=iter([mock_row]))
        self.mock_ads_client.get_service().search.return_value = mock_response

        # Mock CausalImpact results
        mock_ci = Mock()
        mock_summary = {
            "AbsEffect": [10.0],
            "AbsEffect.lower": [5.0],
            "AbsEffect.upper": [15.0],
            "p": [0.01],
            "RelEffect": [0.1],
        }
        mock_ci.summary = Mock(side_effect=[mock_summary, "Test Report"])
        mock_causal_impact.return_value = mock_ci

        # Test the analysis
        result = self.service.analyze_campaign_change_impact(
            campaign_id="123456789", change_date=datetime(2024, 1, 1), metric="clicks"
        )

        # Verify results
        self.assertEqual(result["estimated_effect"], 10.0)
        self.assertEqual(result["confidence_interval"], (5.0, 15.0))
        self.assertEqual(result["p_value"], 0.01)
        self.assertTrue(result["is_significant"])
        self.assertEqual(result["relative_effect"], 0.1)

    def test_get_campaign_data(self):
        """Test getting campaign performance data."""
        # Mock the Google Ads API response
        mock_response = Mock()
        mock_row = Mock()
        mock_row.segments.date = "2024-01-01"
        mock_row.metrics.clicks = 100
        mock_response.__iter__ = Mock(return_value=iter([mock_row]))
        self.mock_ads_client.get_service().search.return_value = mock_response

        # Test getting data
        df = self.service._get_campaign_data(
            campaign_id="123456789",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            metric="clicks",
        )

        # Verify the DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["clicks"], 100)

    def test_get_control_campaign_data(self):
        """Test getting control campaign data."""
        # Mock the Google Ads API response
        mock_response = Mock()
        mock_row = Mock()
        mock_row.segments.date = "2024-01-01"
        mock_row.campaign.id = "123456789"
        mock_row.metrics.clicks = 100
        mock_response.__iter__ = Mock(return_value=iter([mock_row]))
        self.mock_ads_client.get_service().search.return_value = mock_response

        # Test getting control data
        df = self.service._get_control_campaign_data(
            campaign_ids=["123456789"],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            metric="clicks",
        )

        # Verify the DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["clicks"], 100)
        self.assertEqual(df.iloc[0]["campaign_id"], "123456789")

    @patch("services.causal_inference_service.causal_inference_service.CausalImpact")
    def test_run_causal_impact_analysis(self, mock_causal_impact):
        """Test running causal impact analysis."""
        # Create test data
        pre_data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", "2024-01-10"),
                "clicks": np.random.randint(80, 120, 10),
            }
        )
        post_data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-11", "2024-01-20"),
                "clicks": np.random.randint(100, 140, 10),
            }
        )

        # Mock CausalImpact results
        mock_ci = Mock()
        mock_summary = {
            "AbsEffect": [10.0],
            "AbsEffect.lower": [5.0],
            "AbsEffect.upper": [15.0],
            "p": [0.01],
            "RelEffect": [0.1],
        }
        mock_ci.summary = Mock(side_effect=[mock_summary, "Test Report"])
        mock_causal_impact.return_value = mock_ci

        # Test the analysis
        result = self.service._run_causal_impact_analysis(pre_data, post_data)

        # Verify results
        self.assertEqual(result["estimated_effect"], 10.0)
        self.assertEqual(result["confidence_interval"], (5.0, 15.0))
        self.assertEqual(result["p_value"], 0.01)
        self.assertTrue(result["is_significant"])
        self.assertEqual(result["relative_effect"], 0.1)

    def test_error_handling(self):
        """Test error handling in the service."""
        # Test with invalid campaign ID
        self.mock_ads_client.get_service().search.side_effect = Exception("API Error")

        result = self.service.analyze_campaign_change_impact(
            campaign_id="invalid_id", change_date=datetime(2024, 1, 1), metric="clicks"
        )

        # Verify error handling
        self.assertIn("error", result)
        self.assertEqual(result["estimated_effect"], 0)
        self.assertEqual(result["confidence_interval"], (0, 0))
        self.assertEqual(result["p_value"], 1.0)
        self.assertFalse(result["is_significant"])


if __name__ == "__main__":
    unittest.main()
