from services.ltv_bidding_service.ltv_bidding_service import LTVBiddingService
from services.base_service import BaseService
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add the current directory to the Python path
sys.path.append(os.path.abspath("."))

# Mock the google.ads.googleads.client module
sys.modules["google.ads.googleads.client"] = MagicMock()
sys.modules["google.ads.googleads"] = MagicMock()
sys.modules["google.ads"] = MagicMock()
sys.modules["google"] = MagicMock()

# Create a direct import of the LTVBiddingService class with mocked base service


# Define a simplified test case
class SimpleLTVBiddingServiceTest(unittest.TestCase):
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
        self.service = LTVBiddingService(
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


if __name__ == "__main__":
    # Import pandas here to avoid import before mocking
    import pandas as pd

    # Run tests
    unittest.main(verbosity=2)
