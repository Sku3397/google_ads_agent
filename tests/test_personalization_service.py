"""
Unit tests for the Personalization Service.

This module contains tests for the PersonalizationService class and its methods.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import os
import json
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import the services
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.personalization_service import PersonalizationService


class TestPersonalizationService(unittest.TestCase):
    """Test cases for PersonalizationService."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock ads_api
        self.ads_api = MagicMock()

        # Create a mock optimizer
        self.optimizer = MagicMock()

        # Create test config
        self.config = {
            "segment_count": 3,
            "min_observations": 50,
            "update_frequency_days": 7,
            "data_lookback_days": 30,
        }

        # Create the service
        self.service = PersonalizationService(
            ads_api=self.ads_api, optimizer=self.optimizer, config=self.config
        )

        # Create test user data
        self.user_data = pd.DataFrame(
            {
                "user_id": range(100),
                "device": np.random.choice(["mobile", "desktop", "tablet"], 100, p=[0.6, 0.3, 0.1]),
                "location": np.random.choice(
                    ["urban", "suburban", "rural"], 100, p=[0.5, 0.3, 0.2]
                ),
                "time_of_day": np.random.choice(["morning", "afternoon", "evening", "night"], 100),
                "day_of_week": np.random.choice(["weekday", "weekend"], 100, p=[0.7, 0.3]),
                "query_category": np.random.choice(
                    ["commercial", "informational", "navigational"], 100
                ),
                "ctr": np.random.beta(2, 10, 100),
                "conversion_rate": np.random.beta(1, 20, 100),
                "cpc": np.random.gamma(2, 0.5, 100),
            }
        )

        # Create test performance data
        self.performance_data = pd.DataFrame(
            {
                "segment_id": [str(x) for x in np.random.randint(0, 3, 50)],
                "campaign_id": [f"campaign_{x}" for x in np.random.randint(1, 3, 50)],
                "ad_group_id": [f"adgroup_{x}" for x in np.random.randint(1, 3, 50)],
                "device": np.random.choice(["mobile", "desktop", "tablet"], 50),
                "impressions": np.random.randint(10, 1000, 50),
                "clicks": np.random.randint(0, 100, 50),
                "conversions": np.random.randint(0, 10, 50),
                "cost": np.random.uniform(10, 1000, 50),
            }
        )

    def test_init(self):
        """Test initialization of the service."""
        # Check that the service is initialized with the correct values
        self.assertEqual(self.service.segment_count, 3)
        self.assertEqual(self.service.min_observations, 50)
        self.assertEqual(self.service.update_frequency_days, 7)
        self.assertEqual(self.service.data_lookback_days, 30)

        # Check that data structures are initialized
        self.assertIsInstance(self.service.user_segments, dict)
        self.assertIsInstance(self.service.personalization_models, dict)
        self.assertIsInstance(self.service.segment_performance, dict)

    @patch("services.personalization_service.personalization_service.pd.DataFrame")
    @patch("services.personalization_service.personalization_service.KMeans")
    def test_create_user_segments(self, mock_kmeans, mock_df):
        """Test creating user segments."""
        # Set up the mocks
        mock_kmeans_instance = MagicMock()
        mock_kmeans_instance.labels_ = np.array([0, 1, 2] * 33 + [0])  # 100 labels
        mock_kmeans.return_value = mock_kmeans_instance

        # Call the method
        segments = self.service.create_user_segments(self.user_data)

        # Check results
        self.assertIsInstance(segments, dict)
        self.assertEqual(len(segments), 3)  # Should have 3 segments

        # Check that _save_state was called
        self.service._save_state = MagicMock()
        self.service.create_user_segments(self.user_data)
        self.service._save_state.assert_called_once()

    def test_get_segment_for_user(self):
        """Test getting segment for a user."""
        # Set up test segments
        self.service.user_segments = {
            "0": {
                "segment_id": "0",
                "features": {
                    "device": {
                        "type": "categorical",
                        "distribution": {"mobile": 0.7, "desktop": 0.2, "tablet": 0.1},
                    },
                    "location": {
                        "type": "categorical",
                        "distribution": {"urban": 0.6, "suburban": 0.3, "rural": 0.1},
                    },
                },
            },
            "1": {
                "segment_id": "1",
                "features": {
                    "device": {
                        "type": "categorical",
                        "distribution": {"mobile": 0.2, "desktop": 0.7, "tablet": 0.1},
                    },
                    "location": {
                        "type": "categorical",
                        "distribution": {"urban": 0.3, "suburban": 0.6, "rural": 0.1},
                    },
                },
            },
        }

        # Test with user that should match segment 0
        user_data = {"device": "mobile", "location": "urban"}
        segment = self.service.get_segment_for_user(user_data)
        self.assertEqual(segment, "0")

        # Test with user that should match segment 1
        user_data = {"device": "desktop", "location": "suburban"}
        segment = self.service.get_segment_for_user(user_data)
        self.assertEqual(segment, "1")

        # Test with empty segments
        self.service.user_segments = {}
        segment = self.service.get_segment_for_user(user_data)
        self.assertEqual(segment, "0")  # Default segment

    def test_get_personalized_bid_adjustments(self):
        """Test getting personalized bid adjustments."""
        # Set up test data
        self.service.segment_performance = {
            "campaign_1_adgroup_1_0": {
                "conversion_rate": 0.05,
                "base_conversion_rate": 0.025,
                "device_performance": {"mobile": {"conversion_rate": 0.06, "conversions": 20}},
            }
        }

        # Test with data that should return adjustments
        adjustments = self.service.get_personalized_bid_adjustments(
            campaign_id="campaign_1", ad_group_id="adgroup_1", user_segment="0"
        )

        # Check results
        self.assertIsInstance(adjustments, dict)
        self.assertIn("device", adjustments)
        self.assertIn("location", adjustments)
        self.assertIn("audience", adjustments)
        self.assertGreater(
            adjustments["audience"], 1.0
        )  # Should be higher due to better conversion rate

        # Test with missing data
        adjustments = self.service.get_personalized_bid_adjustments(
            campaign_id="campaign_2", ad_group_id="adgroup_2", user_segment="0"
        )

        # Check default values returned
        self.assertEqual(adjustments["device"], 1.0)
        self.assertEqual(adjustments["location"], 1.0)
        self.assertEqual(adjustments["audience"], 1.0)

    def test_get_personalized_ads(self):
        """Test getting personalized ads."""
        # Set up test data
        self.service.segment_performance = {
            "adgroup_1_ad1_0": {"ctr": 0.05, "conversion_rate": 0.02},
            "adgroup_1_ad2_0": {"ctr": 0.03, "conversion_rate": 0.01},
        }

        # Set up user segments
        self.service.user_segments = {"0": {"segment_id": "0", "features": {}}}

        # Test ads
        available_ads = [
            {"id": "ad1", "headline": "Test Ad 1"},
            {"id": "ad2", "headline": "Test Ad 2"},
        ]

        # Test with data that should return ranked ads
        ranked_ads = self.service.get_personalized_ads(
            ad_group_id="adgroup_1", user_segment="0", available_ads=available_ads
        )

        # Check results
        self.assertEqual(len(ranked_ads), 2)
        self.assertEqual(ranked_ads[0]["id"], "ad1")  # Best performing ad should be first

        # Test with empty ads list
        ranked_ads = self.service.get_personalized_ads(
            ad_group_id="adgroup_1", user_segment="0", available_ads=[]
        )
        self.assertEqual(len(ranked_ads), 0)

        # Test with missing segment
        ranked_ads = self.service.get_personalized_ads(
            ad_group_id="adgroup_1", user_segment="99", available_ads=available_ads
        )
        self.assertEqual(len(ranked_ads), 2)  # Should return original ads

    def test_update_segment_performance(self):
        """Test updating segment performance."""
        # Call the method
        updated_segments = self.service.update_segment_performance(self.performance_data)

        # Check results
        self.assertIsInstance(updated_segments, dict)
        self.assertGreater(len(updated_segments), 0)

        # Check that metrics were calculated
        for segment_id in np.unique(self.performance_data["segment_id"]):
            self.assertIn(segment_id, updated_segments)

            segment = updated_segments[segment_id]
            self.assertIn("impressions", segment)
            self.assertIn("clicks", segment)
            self.assertIn("conversions", segment)
            self.assertIn("ctr", segment)
            self.assertIn("conversion_rate", segment)
            self.assertIn("last_updated", segment)

    def test_recommend_ad_customizers(self):
        """Test recommending ad customizers."""
        # Set up test segments
        self.service.user_segments = {
            "0": {
                "segment_id": "0",
                "features": {
                    "device": {
                        "type": "categorical",
                        "distribution": {"mobile": 0.7, "desktop": 0.2, "tablet": 0.1},
                    },
                    "location": {
                        "type": "categorical",
                        "distribution": {"urban": 0.6, "suburban": 0.3, "rural": 0.1},
                    },
                },
            }
        }

        # Test with valid segment
        customizers = self.service.recommend_ad_customizers(
            ad_group_id="adgroup_1", user_segment="0"
        )

        # Check results
        self.assertIsInstance(customizers, dict)
        self.assertIn("headline_customizers", customizers)
        self.assertIn("description_customizers", customizers)
        self.assertGreater(len(customizers["headline_customizers"]), 0)

        # Test with invalid segment
        customizers = self.service.recommend_ad_customizers(
            ad_group_id="adgroup_1", user_segment="99"
        )
        self.assertEqual(customizers, {})

    @patch("services.personalization_service.personalization_service.pd.DataFrame")
    def test_run_personalization_update(self, mock_df):
        """Test running a full personalization update."""
        # Mock the dependencies
        self.service.create_user_segments = MagicMock(return_value={"0": {}, "1": {}, "2": {}})
        self.service.update_segment_performance = MagicMock(
            return_value={"0": {}, "1": {}, "2": {}}
        )

        # Call the method
        result = self.service.run_personalization_update()

        # Check results
        self.assertTrue(result)
        self.service.create_user_segments.assert_called_once()
        self.service.update_segment_performance.assert_called_once()

        # Test failure case
        self.service.ads_api = None
        result = self.service.run_personalization_update()
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
