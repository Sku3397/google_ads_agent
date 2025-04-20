# tests/test_reporting_service.py
import unittest
from unittest.mock import MagicMock
import logging
from services.reporting_service.reporting_service import ReportingService

# Disable logging for tests
logging.disable(logging.CRITICAL)


class TestReportingService(unittest.TestCase):
    """Test cases for the ReportingService."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.mock_ads_client = MagicMock()
        self.config = {
            "reporting_settings": {
                "default_days": 30,
                "report_format": "json",
            }
        }
        self.service = ReportingService(self.mock_ads_client, self.config)

    def test_initialization(self) -> None:
        """Test service initialization."""
        self.assertIsNotNone(self.service)
        self.assertEqual(self.service.ads_client, self.mock_ads_client)
        self.assertEqual(self.service.config, self.config)

    def test_generate_performance_report_placeholder(self) -> None:
        """Test the placeholder implementation of generate_performance_report."""
        campaign_ids = ["123", "456"]
        days_ago = 7

        # Call the method
        report = self.service.generate_performance_report(campaign_ids, days_ago)

        # Assert that the report contains expected placeholder structure
        self.assertIsInstance(report, dict)
        self.assertEqual(report["report_title"], f"Performance Report ({days_ago} days)")
        self.assertEqual(report["campaigns_analyzed"], campaign_ids)
        self.assertIn("metrics", report)
        self.assertIsInstance(report["metrics"], dict)
        self.assertIn("insights", report)
        self.assertIsInstance(report["insights"], list)
        self.assertTrue(len(report["insights"]) > 0)

        # Check if metrics are present (values are placeholders)
        self.assertIn("impressions", report["metrics"])
        self.assertIn("clicks", report["metrics"])
        self.assertIn("cost", report["metrics"])
        self.assertIn("conversions", report["metrics"])

    def test_run_placeholder(self) -> None:
        """Test the placeholder run method."""
        # Since the run method currently does nothing but log,
        # we just call it to ensure it doesn't raise an error.
        try:
            self.service.run()
        except Exception as e:
            self.fail(f"ReportingService run() method raised an exception unexpectedly: {e}")


if __name__ == "__main__":
    unittest.main()
