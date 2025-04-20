"""
Tests for the Landing Page Optimization Service.

This module contains tests for the LandingPageOptimizationService, which
provides tools for analyzing and optimizing landing pages to improve
conversion rates.
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import json
import tempfile
import shutil
from datetime import datetime

from services.landing_page_optimization_service import LandingPageOptimizationService


class TestLandingPageOptimizationService(unittest.TestCase):
    """Test cases for the Landing Page Optimization Service."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create mock dependencies
        self.mock_ads_api = MagicMock()
        self.mock_optimizer = MagicMock()

        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.temp_dir, "data/landing_page_analysis"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "data/landing_page_tests"), exist_ok=True)

        # Configuration for testing
        self.config = {
            "landing_page_optimization.min_traffic_threshold": 50,
            "landing_page_optimization.analytics_source": "test_analytics",
            "landing_page_optimization.testing_framework": "test_framework",
        }

        # Create service instance
        self.service = LandingPageOptimizationService(
            ads_api=self.mock_ads_api, optimizer=self.mock_optimizer, config=self.config
        )

        # Replace save and load methods for testing
        self.service.save_data = MagicMock(return_value=True)
        self.service.load_data = MagicMock(return_value=None)

    def tearDown(self):
        """Clean up after each test."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test service initialization."""
        self.assertIsNotNone(self.service)
        self.assertEqual(self.service.ads_api, self.mock_ads_api)
        self.assertEqual(self.service.optimizer, self.mock_optimizer)
        self.assertEqual(self.service.config["landing_page_optimization.min_traffic_threshold"], 50)
        self.assertEqual(
            self.service.config["landing_page_optimization.analytics_source"], "test_analytics"
        )

    def test_validate_config(self):
        """Test configuration validation."""
        # Test with empty config
        service = LandingPageOptimizationService(config={})
        self.assertIn("landing_page_optimization.min_traffic_threshold", service.config)
        self.assertIn("landing_page_optimization.analytics_source", service.config)

        # Test with partial config
        service = LandingPageOptimizationService(
            config={"landing_page_optimization.min_traffic_threshold": 200}
        )
        self.assertEqual(service.config["landing_page_optimization.min_traffic_threshold"], 200)
        self.assertIn("landing_page_optimization.analytics_source", service.config)

    def test_analyze_landing_page(self):
        """Test landing page analysis method."""
        # Call the method
        result = self.service.analyze_landing_page("https://example.com/landing-page")

        # Check that the result has the expected structure
        self.assertEqual(result["url"], "https://example.com/landing-page")
        self.assertEqual(result["status"], "success")
        self.assertIn("performance", result)
        self.assertIn("speed_analysis", result)
        self.assertIn("content_analysis", result)
        self.assertIn("form_analysis", result)
        self.assertIn("recommendations", result)

        # Verify that save_data was called
        self.service.save_data.assert_called_once()

    def test_create_a_b_test(self):
        """Test A/B test creation method."""
        # Call the method
        result = self.service.create_a_b_test(
            url="https://example.com/original",
            variant_urls=["https://example.com/variant1", "https://example.com/variant2"],
            test_name="Test Homepage Headline",
            duration_days=7,
            traffic_split=[0.34, 0.33, 0.33],
        )

        # Check that the result has the expected structure
        self.assertEqual(result["status"], "success")
        self.assertIn("test_id", result)
        self.assertIn("test_config", result)

        # Check test configuration
        test_config = result["test_config"]
        self.assertEqual(test_config["name"], "Test Homepage Headline")
        self.assertEqual(test_config["control_url"], "https://example.com/original")
        self.assertEqual(len(test_config["variant_urls"]), 2)
        self.assertEqual(test_config["duration_days"], 7)
        self.assertEqual(test_config["status"], "created")

        # Verify that save_data was called
        self.service.save_data.assert_called_once()

    def test_get_a_b_test_results_not_found(self):
        """Test getting A/B test results when test doesn't exist."""
        # Set up load_data to return None (test not found)
        self.service.load_data.return_value = None

        # Call the method
        result = self.service.get_a_b_test_results("nonexistent_test_id")

        # Check that the result indicates error
        self.assertEqual(result["status"], "error")
        self.assertIn("message", result)

    def test_get_a_b_test_results_found(self):
        """Test getting A/B test results when test exists."""
        # Create a mock test configuration
        mock_test_config = {
            "id": "test_123",
            "name": "Test Homepage",
            "control_url": "https://example.com/original",
            "variant_urls": ["https://example.com/variant"],
            "traffic_split": [0.5, 0.5],
            "start_date": (datetime.now().isoformat()),
            "end_date": (datetime.now().isoformat()),
            "duration_days": 7,
            "status": "running",
            "metrics": ["conversion_rate"],
            "primary_metric": "conversion_rate",
        }

        # Set up load_data to return the test config
        self.service.load_data.return_value = mock_test_config

        # Call the method
        result = self.service.get_a_b_test_results("test_123")

        # Check that the result has the expected structure
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["test_id"], "test_123")
        self.assertEqual(result["test_config"], mock_test_config)
        self.assertIn("results", result)

    def test_analyze_page_elements(self):
        """Test page element analysis method."""
        # Call the method
        result = self.service.analyze_page_elements("https://example.com/landing-page")

        # Check that the result has the expected structure
        self.assertEqual(result["url"], "https://example.com/landing-page")
        self.assertEqual(result["status"], "success")
        self.assertIn("elements", result)
        self.assertIn("critical_elements", result)
        self.assertIn("underperforming_elements", result)
        self.assertIn("recommendations", result)

    def test_optimize_for_page_speed(self):
        """Test page speed optimization method."""
        # Call the method
        result = self.service.optimize_for_page_speed("https://example.com/landing-page")

        # Check that the result has the expected structure
        self.assertEqual(result["url"], "https://example.com/landing-page")
        self.assertEqual(result["status"], "success")
        self.assertIn("current_speed_metrics", result)
        self.assertIn("recommendations", result)
        self.assertIn("estimated_impact", result)

    def test_optimize_form_conversion(self):
        """Test form conversion optimization method."""
        # Mock _analyze_forms to return a form analysis
        self.service._analyze_forms = MagicMock(
            return_value={
                "has_form": True,
                "form_fields": 8,
                "form_completion_rate": 0.3,
                "required_fields_ratio": 0.9,
                "issues": [{"name": "Too many required fields", "impact": "high"}],
            }
        )

        # Call the method
        result = self.service.optimize_form_conversion("https://example.com/landing-page")

        # Check that the result has the expected structure
        self.assertEqual(result["url"], "https://example.com/landing-page")
        self.assertEqual(result["status"], "success")
        self.assertIn("current_form_metrics", result)
        self.assertIn("recommendations", result)
        self.assertIn("estimated_impact", result)

    def test_optimize_form_conversion_no_form(self):
        """Test form conversion optimization when no form is present."""
        # Mock _analyze_forms to return no form
        self.service._analyze_forms = MagicMock(return_value={"has_form": False})

        # Call the method
        result = self.service.optimize_form_conversion("https://example.com/landing-page")

        # Check that the result indicates error
        self.assertEqual(result["status"], "error")
        self.assertIn("message", result)
        self.assertEqual(result["message"], "No form detected on this page")


if __name__ == "__main__":
    unittest.main()
