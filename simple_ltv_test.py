import unittest
import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock


# Create a simplified version of BaseService
class MockBaseService:
    def __init__(self, ads_api=None, optimizer=None, config=None, logger=None):
        self.ads_api = ads_api
        self.optimizer = optimizer
        self.config = config or {}
        self.logger = logger or MagicMock()
        self.metrics = {
            "invocations": 0,
            "success_count": 0,
            "failure_count": 0,
            "last_run": None,
            "avg_execution_time_ms": 0,
        }

    def _track_execution(self, start_time, success):
        pass


# Create a simplified version of LTVBiddingService
class SimpleLTVBiddingService(MockBaseService):
    def __init__(self, ads_api=None, optimizer=None, config=None, logger=None):
        super().__init__(ads_api, optimizer, config, logger)
        self.ltv_model = None
        self.feature_columns = None
        self.model_path = os.path.join("data", "ltv_model.joblib")
        self.scaler_path = os.path.join("data", "ltv_scaler.joblib")
        self.feature_importance = None

    def _generate_mock_data(self, days):
        """Generate mock customer data for development/testing"""
        n_samples = min(days * 10, 10000)  # Scale with days but cap at reasonable size

        np.random.seed(42)  # For reproducibility

        data = {
            "geo_location": np.random.choice(["US", "UK", "CA", "AU", "DE"], n_samples),
            "device": np.random.choice(["mobile", "desktop", "tablet"], n_samples),
            "keyword_id": np.random.randint(1000, 9999, n_samples),
            "campaign_id": np.random.choice([f"campaign_{i}" for i in range(1, 6)], n_samples),
            "ad_group_id": np.random.choice([f"adgroup_{i}" for i in range(1, 11)], n_samples),
            "match_type": np.random.choice(["exact", "phrase", "broad"], n_samples),
            "conversion_lag_days": np.random.randint(0, 30, n_samples),
            "clicks_before_conversion": np.random.randint(1, 10, n_samples),
            "impressions_before_conversion": np.random.randint(1, 50, n_samples),
            "first_conversion_value": np.random.uniform(10, 500, n_samples),
            "user_recency_days": np.random.randint(0, 365, n_samples),
            "user_frequency": np.random.randint(1, 20, n_samples),
            "average_time_on_site": np.random.uniform(30, 600, n_samples),
            "pages_per_session": np.random.uniform(1, 10, n_samples),
        }

        # Generate LTV with some relationship to features
        ltv_base = data["first_conversion_value"] * (1 + 0.1 * data["user_frequency"])
        device_factor = np.where(
            data["device"] == "desktop", 1.2, np.where(data["device"] == "mobile", 0.8, 1.0)
        )
        geo_factor = np.where(
            data["geo_location"] == "US", 1.3, np.where(data["geo_location"] == "UK", 1.1, 0.9)
        )

        # Add some noise
        noise = np.random.normal(1, 0.3, n_samples)

        data["customer_ltv"] = ltv_base * device_factor * geo_factor * noise

        return pd.DataFrame(data)


# Define test case
class TestSimpleLTVBiddingService(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mock dependencies
        self.mock_ads_api = MagicMock()
        self.mock_optimizer = MagicMock()
        self.mock_config = {
            "ltv_bidding": {
                "min_data_points": 50,
                "min_confidence": 0.6,
                "max_bid_adjustment": 0.3,
                "reallocation_percent": 0.1,
            }
        }
        self.mock_logger = MagicMock()

        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)

        # Initialize service with mock dependencies
        self.service = SimpleLTVBiddingService(
            ads_api=self.mock_ads_api,
            optimizer=self.mock_optimizer,
            config=self.mock_config,
            logger=self.mock_logger,
        )

    def test_initialization(self):
        """Test that the service initializes correctly."""
        self.assertIsNone(self.service.ltv_model)
        self.assertIsNone(self.service.feature_columns)
        self.assertEqual(self.service.model_path, os.path.join("data", "ltv_model.joblib"))
        self.assertEqual(self.service.scaler_path, os.path.join("data", "ltv_scaler.joblib"))
        print("Initialization test passed!")

    def test_mock_data_generation(self):
        """Test the mock data generation methods."""
        # Test _generate_mock_data
        mock_data = self.service._generate_mock_data(10)
        self.assertIsInstance(mock_data, pd.DataFrame)
        self.assertGreater(len(mock_data), 0)
        self.assertIn("customer_ltv", mock_data.columns)
        print("Mock data generation test passed!")

        # Print some sample data
        print("\nSample data:")
        print(mock_data.head(3))
        print("\nData shape:", mock_data.shape)
        print("\nColumn names:", mock_data.columns.tolist())


if __name__ == "__main__":
    unittest.main(verbosity=2)
