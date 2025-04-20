"""
Tests for the LTVBiddingService.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import shutil
from unittest.mock import patch, MagicMock

# Import the service to test
from services.ltv_bidding_service.ltv_bidding_service import LTVBiddingService

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestLTVBiddingService(unittest.TestCase):
    """Test suite for LTVBiddingService."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a test directory for model outputs
        self.test_data_dir = "test_data"
        os.makedirs(self.test_data_dir, exist_ok=True)

        # Create a configuration for testing
        self.test_config = {
            "model_config": {
                "model_type": "gradient_boosting",  # Use simpler model for faster tests
                "hyperparams": {
                    "gradient_boosting": {
                        "n_estimators": 10,  # Small number for faster tests
                        "learning_rate": 0.1,
                        "max_depth": 3,
                        "random_state": 42,
                    }
                },
                "feature_selection": True,
            }
        }

        # Mock Google Ads API client
        self.mock_ads_api = MagicMock()

        # Initialize the service with the test config
        # Override the model_path to use test directory
        with patch.object(LTVBiddingService, "__init__", return_value=None):
            self.service = LTVBiddingService()
            self.service.__init__(ads_api=self.mock_ads_api, config=self.test_config, logger=logger)
            self.service.model_path = os.path.join(self.test_data_dir, "ltv_model.joblib")

        # Generate test data
        self.test_data = self._generate_test_data()

    def tearDown(self):
        """Clean up after tests."""
        # Remove test output directory
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)

    def _generate_test_data(self):
        """Generate test data for LTV model training and analysis."""
        # Create a simple dataset with customer conversion data
        np.random.seed(42)
        n_samples = 200

        # Basic features
        geo_locations = ["US", "UK", "CA", "DE", "FR"]
        devices = ["mobile", "desktop", "tablet"]
        campaign_ids = ["campaign_1", "campaign_2", "campaign_3"]
        ad_group_ids = ["adgroup_1", "adgroup_2", "adgroup_3", "adgroup_4"]
        match_types = ["exact", "phrase", "broad"]

        # Generate data with some patterns for LTV
        data = {
            "customer_id": [f"user_{i}" for i in range(n_samples)],
            "geo_location": np.random.choice(geo_locations, n_samples),
            "device": np.random.choice(devices, n_samples),
            "campaign_id": np.random.choice(campaign_ids, n_samples),
            "ad_group_id": np.random.choice(ad_group_ids, n_samples),
            "match_type": np.random.choice(match_types, n_samples),
            "clicks_before_conversion": np.random.randint(1, 10, n_samples),
            "impressions_before_conversion": np.random.randint(5, 50, n_samples),
            "conversion_lag_days": np.random.randint(0, 30, n_samples),
            "first_conversion_value": np.random.uniform(10, 100, n_samples),
            "user_recency_days": np.random.randint(0, 90, n_samples),
            "user_frequency": np.random.randint(1, 10, n_samples),
            "average_time_on_site": np.random.uniform(60, 300, n_samples),
            "pages_per_session": np.random.uniform(1, 8, n_samples),
            "acquisition_cost": np.random.uniform(5, 50, n_samples),
        }

        # Add some feature relationships to LTV
        # More pages & longer session time = higher LTV
        base_ltv = 100 + data["average_time_on_site"] * 0.5 + data["pages_per_session"] * 10

        # Device impacts (desktop tends to have higher LTV)
        device_multipliers = {"desktop": 1.3, "mobile": 0.9, "tablet": 1.0}
        device_factors = np.array([device_multipliers[d] for d in data["device"]])

        # Geographic impacts
        geo_multipliers = {"US": 1.2, "UK": 1.0, "CA": 1.1, "DE": 0.9, "FR": 0.8}
        geo_factors = np.array([geo_multipliers[g] for g in data["geo_location"]])

        # Calculate LTV with some randomness
        data["customer_ltv"] = (
            base_ltv * device_factors * geo_factors * np.random.uniform(0.8, 1.2, n_samples)
        )

        # Add dates for cohort analysis
        today = datetime.now()

        # First conversion date (between 1 and 365 days ago)
        first_dates = [today - timedelta(days=np.random.randint(1, 365)) for _ in range(n_samples)]
        data["first_conversion_date"] = first_dates

        # Conversion date (either same as first date or later)
        conversion_dates = []
        for i, first_date in enumerate(first_dates):
            # Some customers have repeat conversions
            if np.random.random() < 0.3:  # 30% have repeat conversions
                days_later = np.random.randint(1, 180)  # Up to 180 days later
                conversion_dates.append(first_date + timedelta(days=days_later))
            else:
                conversion_dates.append(first_date)  # Same as first conversion

        data["conversion_date"] = conversion_dates

        # Additional conversions for cohort analysis
        additional_rows = []
        for i in range(n_samples):
            if np.random.random() < 0.4:  # 40% of customers have multiple conversions
                num_additional = np.random.randint(1, 4)  # 1-3 additional conversions
                for j in range(num_additional):
                    new_row = {k: data[k][i] for k in data.keys()}
                    # Update conversion-specific data
                    days_later = np.random.randint(10, 300)
                    new_row["conversion_date"] = first_dates[i] + timedelta(days=days_later)
                    new_row["conversion_value"] = np.random.uniform(10, 100)
                    additional_rows.append(new_row)

        # Add conversion_value for original rows
        data["conversion_value"] = np.random.uniform(10, 100, n_samples)

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Add additional rows if any
        if additional_rows:
            additional_df = pd.DataFrame(additional_rows)
            df = pd.concat([df, additional_df], ignore_index=True)

        return df

    def test_initialization(self):
        """Test correct initialization of the service."""
        self.assertIsNotNone(self.service)
        self.assertEqual(self.service.model_config["model_type"], "gradient_boosting")
        self.assertIsNone(self.service.ltv_model)

    def test_train_ltv_model_gradient_boosting(self):
        """Test training an LTV model with gradient boosting."""
        result = self.service.train_ltv_model(
            historical_data=self.test_data, days=365, model_type="gradient_boosting"
        )

        # Check the result
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["model_type"], "gradient_boosting")
        self.assertIsNotNone(result["model_metrics"])
        self.assertIsNotNone(result["feature_importance"])

        # Check that model was saved
        self.assertIsNotNone(self.service.ltv_model)
        self.assertTrue(os.path.exists(self.service.model_path))

    @unittest.skipIf(
        not hasattr(LTVBiddingService, "xgboost_available")
        or not LTVBiddingService.xgboost_available,
        "XGBoost not available",
    )
    def test_train_ltv_model_xgboost(self):
        """Test training an LTV model with XGBoost."""
        result = self.service.train_ltv_model(
            historical_data=self.test_data, days=365, model_type="xgboost"
        )

        # Check the result
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["model_type"], "xgboost")
        self.assertIsNotNone(result["model_metrics"])

    @unittest.skipIf(
        not hasattr(LTVBiddingService, "lightgbm_available")
        or not LTVBiddingService.lightgbm_available,
        "LightGBM not available",
    )
    def test_train_ltv_model_lightgbm(self):
        """Test training an LTV model with LightGBM."""
        result = self.service.train_ltv_model(
            historical_data=self.test_data, days=365, model_type="lightgbm"
        )

        # Check the result
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["model_type"], "lightgbm")
        self.assertIsNotNone(result["model_metrics"])

    def test_train_ltv_model_with_hyperparameter_tuning(self):
        """Test training an LTV model with hyperparameter tuning."""
        # Skip if hyperparameter tuning is not available
        if (
            not hasattr(self.service, "hyperparam_tuning_available")
            or not self.service.hyperparam_tuning_available
        ):
            self.skipTest("Hyperparameter tuning not available")

        result = self.service.train_ltv_model(
            historical_data=self.test_data,
            days=365,
            hyperparam_tuning=True,
            model_type="gradient_boosting",
        )

        # Check the result
        self.assertEqual(result["status"], "success")
        self.assertIsNotNone(result["hyperparameters"])

    def test_predict_customer_ltv(self):
        """Test predicting LTV for a customer."""
        # First train a model
        self.service.train_ltv_model(
            historical_data=self.test_data, days=365, model_type="gradient_boosting"
        )

        # Create a test customer
        test_customer = {
            "geo_location": "US",
            "device": "desktop",
            "campaign_id": "campaign_1",
            "ad_group_id": "adgroup_1",
            "match_type": "exact",
            "clicks_before_conversion": 5,
            "impressions_before_conversion": 20,
            "conversion_lag_days": 2,
            "first_conversion_value": 50.0,
            "user_recency_days": 10,
            "user_frequency": 3,
            "average_time_on_site": 180.0,
            "pages_per_session": 4.0,
        }

        # Get prediction
        result = self.service.predict_customer_ltv(test_customer)

        # Check the result
        self.assertEqual(result["status"], "success")
        self.assertIsNotNone(result["predicted_ltv"])
        self.assertIsNotNone(result["confidence"])

    def test_predict_customer_ltv_with_confidence_interval(self):
        """Test predicting LTV with confidence intervals."""
        # First train a model
        self.service.train_ltv_model(
            historical_data=self.test_data, days=365, model_type="gradient_boosting"
        )

        # Create a test customer
        test_customer = {
            "geo_location": "US",
            "device": "desktop",
            "campaign_id": "campaign_1",
            "ad_group_id": "adgroup_1",
            "match_type": "exact",
            "clicks_before_conversion": 5,
            "impressions_before_conversion": 20,
            "conversion_lag_days": 2,
            "first_conversion_value": 50.0,
            "user_recency_days": 10,
            "user_frequency": 3,
            "average_time_on_site": 180.0,
            "pages_per_session": 4.0,
        }

        # Get prediction with confidence interval
        result = self.service.predict_customer_ltv(test_customer, return_confidence_interval=True)

        # Check if confidence interval is available
        # Note: It might not be available for all model types
        if "confidence_interval" in result:
            self.assertEqual(len(result["confidence_interval"]), 2)
            self.assertLess(result["confidence_interval"][0], result["predicted_ltv"])
            self.assertGreater(result["confidence_interval"][1], result["predicted_ltv"])

    def test_predict_customer_ltv_with_feature_contribution(self):
        """Test predicting LTV with feature contributions."""
        # Skip if SHAP not available
        try:
            import shap
        except ImportError:
            self.skipTest("SHAP library not available")

        # First train a model
        self.service.train_ltv_model(
            historical_data=self.test_data, days=365, model_type="gradient_boosting"
        )

        # Create a test customer
        test_customer = {
            "geo_location": "US",
            "device": "desktop",
            "campaign_id": "campaign_1",
            "ad_group_id": "adgroup_1",
            "match_type": "exact",
            "clicks_before_conversion": 5,
            "impressions_before_conversion": 20,
            "conversion_lag_days": 2,
            "first_conversion_value": 50.0,
            "user_recency_days": 10,
            "user_frequency": 3,
            "average_time_on_site": 180.0,
            "pages_per_session": 4.0,
        }

        # Get prediction with feature contributions
        result = self.service.predict_customer_ltv(test_customer, return_feature_contribution=True)

        # Check if feature contributions are available
        if "feature_contributions" in result:
            self.assertIsInstance(result["feature_contributions"], dict)
            self.assertGreater(len(result["feature_contributions"]), 0)

            # Check if explanation is available
            if "explanation" in result:
                self.assertIsInstance(result["explanation"], list)

    def test_generate_ltv_bid_adjustments(self):
        """Test generating bid adjustments based on LTV predictions."""
        # First train a model
        self.service.train_ltv_model(
            historical_data=self.test_data, days=365, model_type="gradient_boosting"
        )

        # Mock the _fetch_conversion_data method to return test data
        with patch.object(
            self.service, "_fetch_conversion_data", return_value=self.test_data.to_dict("records")
        ):
            # Generate bid adjustments
            result = self.service.generate_ltv_bid_adjustments(
                min_data_points=10,  # Lower threshold for testing
                min_confidence=0.5,
                max_bid_adjustment=0.5,
            )

            # Check the result
            self.assertEqual(result["status"], "success")
            self.assertIsInstance(result["adjustments"], list)

    def test_perform_ltv_cohort_analysis(self):
        """Test performing cohort analysis on LTV data."""
        # Skip if the data doesn't have the required columns
        required_columns = [
            "customer_id",
            "first_conversion_date",
            "conversion_date",
            "conversion_value",
        ]
        for col in required_columns:
            if col not in self.test_data.columns:
                self.skipTest(f"Test data missing required column: {col}")

        # Mock the _fetch_historical_data method to return test data
        with patch.object(self.service, "_fetch_historical_data", return_value=self.test_data):
            # Perform cohort analysis
            result = self.service.perform_ltv_cohort_analysis(
                days=365,
                cohort_period="month",
                min_cohort_size=5,  # Lower threshold for testing
                retention_type="revenue",
            )

            # Check the result
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["retention_type"], "revenue")

            # Check if key metrics are available
            if "max_ltv" in result:
                self.assertIsInstance(result["max_ltv"], float)

            # Check if insights are available
            self.assertIsInstance(result["insights"], list)

            # Check if recommendations are available
            self.assertIsInstance(result["recommendations"], list)

    def test_save_and_load_model(self):
        """Test saving and loading an LTV model."""
        # First train and save a model
        self.service.train_ltv_model(
            historical_data=self.test_data, days=365, model_type="gradient_boosting"
        )

        # Verify the model was saved
        self.assertTrue(os.path.exists(self.service.model_path))

        # Create a new service instance
        with patch.object(LTVBiddingService, "__init__", return_value=None):
            new_service = LTVBiddingService()
            new_service.__init__(ads_api=self.mock_ads_api, config=self.test_config, logger=logger)
            new_service.model_path = self.service.model_path

        # Load the model
        loaded = new_service._load_model()

        # Verify the model was loaded
        self.assertTrue(loaded)
        self.assertIsNotNone(new_service.ltv_model)

        # Try to make a prediction with the loaded model
        test_customer = {
            "geo_location": "US",
            "device": "desktop",
            "campaign_id": "campaign_1",
            "ad_group_id": "adgroup_1",
            "match_type": "exact",
            "clicks_before_conversion": 5,
            "impressions_before_conversion": 20,
            "conversion_lag_days": 2,
            "first_conversion_value": 50.0,
            "user_recency_days": 10,
            "user_frequency": 3,
            "average_time_on_site": 180.0,
            "pages_per_session": 4.0,
        }

        result = new_service.predict_customer_ltv(test_customer)
        self.assertEqual(result["status"], "success")


if __name__ == "__main__":
    unittest.main()
