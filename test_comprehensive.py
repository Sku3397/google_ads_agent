from chat_interface import ChatInterface
from optimizer import AdsOptimizer
from ads_api import GoogleAdsAPI
from logger import AdsAgentLogger
import unittest
import sys
import os
from datetime import datetime, timedelta
import json
import re

# Add project root to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


class TestGoogleAdsAgentComprehensive(unittest.TestCase):
    """Comprehensive tests for the Google Ads Optimization Agent"""

    def setUp(self):
        """Set up test environment and create mock objects"""
        self.logger = AdsAgentLogger(log_dir="test_logs")

        # Mock config for testing (using dummy values)
        self.config = {
            "google_ads": {
                "client_id": "test_client_id",
                "client_secret": "test_client_secret",
                "developer_token": "test_developer_token",
                "refresh_token": "test_refresh_token",
                "login_customer_id": "1234567890",
                "customer_id": "1234567890",
            },
            "openai": {"api_key": "test_api_key"},
            "google_ai": {"api_key": "test_google_ai_key"},
        }

        # We'll mock these API calls rather than actually calling external services
        self.ads_api = self._create_mock_ads_api()
        self.optimizer = self._create_mock_optimizer()

    def _create_mock_ads_api(self):
        """Create a mock Google Ads API class that returns test data"""

        class MockGoogleAdsAPI:
            def __init__(self, config):
                self.config = config

            def _get_date_range_clause(self, days_ago):
                # Test proper date range calculation
                if not isinstance(days_ago, int) or days_ago < 1 or days_ago > 365:
                    raise ValueError("days_ago must be an integer between 1 and 365")

                end_date = datetime.now().date() - timedelta(days=1)
                start_date = end_date - timedelta(days=days_ago - 1)

                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")

                return f"segments.date BETWEEN '{start_str}' AND '{end_str}'"

            def get_campaign_performance(self, days_ago=30):
                # Return mock campaign data
                return [
                    {
                        "id": "123456789",
                        "name": "Test Campaign 1",
                        "status": "ENABLED",
                        "channel_type": "SEARCH",
                        "bidding_strategy": "MAXIMIZE_CONVERSIONS",
                        "budget": 50.0,
                        "clicks": 1000,
                        "impressions": 10000,
                        "ctr": 10.0,
                        "average_cpc": 0.5,
                        "conversions": 20,
                        "cost": 500.0,
                        "conversion_rate": 2.0,
                        "cost_per_conversion": 25.0,
                        "days": days_ago,
                    },
                    {
                        "id": "987654321",
                        "name": "Test Campaign 2",
                        "status": "ENABLED",
                        "channel_type": "SEARCH",
                        "bidding_strategy": "MANUAL_CPC",
                        "budget": 25.0,
                        "clicks": 500,
                        "impressions": 7500,
                        "ctr": 6.67,
                        "average_cpc": 0.3,
                        "conversions": 5,
                        "cost": 150.0,
                        "conversion_rate": 1.0,
                        "cost_per_conversion": 30.0,
                        "days": days_ago,
                    },
                ]

            def get_keyword_performance(self, days_ago=30, campaign_id=None):
                # Return mock keyword data
                keywords = [
                    {
                        "campaign_id": "123456789",
                        "campaign_name": "Test Campaign 1",
                        "ad_group_id": "11111",
                        "ad_group_name": "Test Ad Group 1",
                        "keyword_text": "test keyword 1",
                        "match_type": "EXACT",
                        "status": "ENABLED",
                        "system_serving_status": "ELIGIBLE",
                        "quality_score": 8,
                        "current_bid": 1.0,
                        "clicks": 200,
                        "impressions": 1000,
                        "ctr": 20.0,
                        "average_cpc": 0.5,
                        "conversions": 10,
                        "cost": 100.0,
                        "conversion_rate": 5.0,
                        "cost_per_conversion": 10.0,
                        "top_impression_percentage": 75.0,
                        "search_impression_share": 0.85,
                        "search_top_impression_share": 0.65,
                        "days": days_ago,
                        "id": "adgroups/11111/criteria/test keyword 1",
                    },
                    {
                        "campaign_id": "123456789",
                        "campaign_name": "Test Campaign 1",
                        "ad_group_id": "11111",
                        "ad_group_name": "Test Ad Group 1",
                        "keyword_text": "test keyword 2",
                        "match_type": "PHRASE",
                        "status": "ENABLED",
                        "system_serving_status": "ELIGIBLE",
                        "quality_score": 5,
                        "current_bid": 0.8,
                        "clicks": 100,
                        "impressions": 800,
                        "ctr": 12.5,
                        "average_cpc": 0.4,
                        "conversions": 1,
                        "cost": 40.0,
                        "conversion_rate": 1.0,
                        "cost_per_conversion": 40.0,
                        "top_impression_percentage": 45.0,
                        "search_impression_share": 0.55,
                        "search_top_impression_share": 0.35,
                        "days": days_ago,
                        "id": "adgroups/11111/criteria/test keyword 2",
                    },
                    {
                        "campaign_id": "987654321",
                        "campaign_name": "Test Campaign 2",
                        "ad_group_id": "22222",
                        "ad_group_name": "Test Ad Group 2",
                        "keyword_text": "test keyword 3",
                        "match_type": "BROAD",
                        "status": "ENABLED",
                        "system_serving_status": "ELIGIBLE",
                        "quality_score": 3,
                        "current_bid": 0.6,
                        "clicks": 150,
                        "impressions": 1500,
                        "ctr": 10.0,
                        "average_cpc": 0.3,
                        "conversions": 0,
                        "cost": 45.0,
                        "conversion_rate": 0.0,
                        "cost_per_conversion": 0.0,
                        "top_impression_percentage": 25.0,
                        "search_impression_share": 0.35,
                        "search_top_impression_share": 0.15,
                        "days": days_ago,
                        "id": "adgroups/22222/criteria/test keyword 3",
                    },
                ]

                # If campaign_id is provided, filter the keywords
                if campaign_id:
                    return [k for k in keywords if str(k["campaign_id"]) == str(campaign_id)]

                return keywords

            def apply_optimization(self, optimization_type, entity_type, entity_id, changes):
                # Mock successful optimization application
                return (
                    True,
                    f"Successfully applied {optimization_type} to {entity_type} {entity_id}",
                )

        return MockGoogleAdsAPI(self.config["google_ads"])

    def _create_mock_optimizer(self):
        """Create a mock Optimizer class that returns test suggestions"""

        class MockAdsOptimizer:
            def __init__(self, config):
                self.config = config

            def format_campaign_data(self, campaigns):
                return "Formatted campaign data for testing"

            def format_keyword_data(self, keywords):
                return "Formatted keyword data for testing"

            def get_optimization_suggestions(self, campaigns, keywords=None):
                # Return mock optimization suggestions
                suggestions = [
                    {
                        "index": 1,
                        "title": 'Increase bid for high-converting keyword "test keyword 1"',
                        "action_type": "BID_ADJUSTMENT",
                        "entity_type": "keyword",
                        "entity_id": "test keyword 1",
                        "change": "Increase bid by 20% from $1.00 to $1.20",
                        "rationale": "This keyword has a high conversion rate (5.0%) and low cost per conversion ($10.00).",
                        "original_text": "Full suggestion text here",
                        "edited": False,
                        "applied": False,
                        "status": "pending",
                        "result_message": "",
                        "current_value": 1.0,
                        "change_value": {"type": "percentage_increase", "value": 20.0},
                    },
                    {
                        "index": 2,
                        "title": 'Pause poorly performing keyword "test keyword 3"',
                        "action_type": "STATUS_CHANGE",
                        "entity_type": "keyword",
                        "entity_id": "test keyword 3",
                        "change": "Pause this keyword",
                        "rationale": "This keyword has spent $45.00 with 0 conversions.",
                        "original_text": "Full suggestion text here",
                        "edited": False,
                        "applied": False,
                        "status": "pending",
                        "result_message": "",
                        "current_value": "ENABLED",
                        "change_value": {"type": "status", "value": "PAUSED"},
                    },
                    {
                        "index": 3,
                        "title": "Increase budget for high-performing Test Campaign 1",
                        "action_type": "BUDGET_ADJUSTMENT",
                        "entity_type": "campaign",
                        "entity_id": "123456789",
                        "change": "Increase daily budget from $50.00 to $75.00",
                        "rationale": "This campaign has a good conversion rate (2.0%) and relatively low cost per conversion ($25.00).",
                        "original_text": "Full suggestion text here",
                        "edited": False,
                        "applied": False,
                        "status": "pending",
                        "result_message": "",
                        "current_value": 50.0,
                        "change_value": {"type": "absolute", "value": 75.0},
                    },
                ]
                return suggestions

        return MockAdsOptimizer(self.config["openai"])

    def test_date_range_conversion(self):
        """Test that date range inputs are properly converted to valid date literals"""
        # Test valid date ranges
        self.logger.info("Testing date range conversion...")

        for days in [1, 7, 30, 90, 365]:
            date_clause = self.ads_api._get_date_range_clause(days)

            # Test that the clause is properly formatted
            self.assertIsInstance(date_clause, str)
            self.assertTrue(date_clause.startswith("segments.date BETWEEN"))

            # Extract dates and validate format
            match = re.search(
                r"BETWEEN '(\d{4}-\d{2}-\d{2})' AND '(\d{4}-\d{2}-\d{2})'", date_clause
            )
            self.assertIsNotNone(match, f"Date clause format is incorrect: {date_clause}")

            start_date_str = match.group(1)
            end_date_str = match.group(2)

            # Validate date string formats
            try:
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
            except ValueError:
                self.fail(f"Date string format is incorrect: {start_date_str}, {end_date_str}")

            # Validate the date range is correct
            expected_days = days - 1  # -1 because both start and end dates are inclusive
            actual_days = (end_date - start_date).days
            self.assertEqual(
                actual_days,
                expected_days,
                f"Expected date range of {expected_days} days, got {actual_days}",
            )

        # Test invalid date ranges
        with self.assertRaises(ValueError):
            self.ads_api._get_date_range_clause(0)  # Too small

        with self.assertRaises(ValueError):
            self.ads_api._get_date_range_clause(366)  # Too large

        with self.assertRaises(ValueError):
            self.ads_api._get_date_range_clause("30")  # Wrong type

        self.logger.info("Date range conversion tests passed!")

    def test_keyword_data_fetching(self):
        """Test that keyword data is correctly fetched, stored, and displayed"""
        self.logger.info("Testing keyword data fetching...")

        # Test fetching all keywords
        keywords = self.ads_api.get_keyword_performance(days_ago=30)

        # Verify data structure and content
        self.assertIsInstance(keywords, list)
        self.assertGreater(len(keywords), 0)

        for keyword in keywords:
            # Check required fields
            self.assertIn("campaign_id", keyword)
            self.assertIn("campaign_name", keyword)
            self.assertIn("ad_group_id", keyword)
            self.assertIn("ad_group_name", keyword)
            self.assertIn("keyword_text", keyword)
            self.assertIn("match_type", keyword)
            self.assertIn("status", keyword)
            self.assertIn("clicks", keyword)
            self.assertIn("impressions", keyword)
            self.assertIn("ctr", keyword)
            self.assertIn("conversions", keyword)
            self.assertIn("cost", keyword)

        # Test filtering by campaign_id
        campaign_id = "123456789"
        filtered_keywords = self.ads_api.get_keyword_performance(
            days_ago=30, campaign_id=campaign_id
        )

        # Verify filtering works
        self.assertIsInstance(filtered_keywords, list)
        self.assertGreater(len(filtered_keywords), 0)
        for keyword in filtered_keywords:
            self.assertEqual(keyword["campaign_id"], campaign_id)

        self.logger.info("Keyword data fetching tests passed!")

    def test_optimizer_comprehensive_input(self):
        """Test that the optimizer receives comprehensive input including both campaign and keyword data"""
        self.logger.info("Testing optimizer comprehensive input...")

        # Get test data
        campaigns = self.ads_api.get_campaign_performance(days_ago=30)
        keywords = self.ads_api.get_keyword_performance(days_ago=30)

        # Create a real optimizer (not the mock) to test its formatting
        real_optimizer = AdsOptimizer(self.config["google_ai"])

        # Test campaign data formatting
        campaign_data_formatted = real_optimizer.format_campaign_data(campaigns)
        self.assertIsInstance(campaign_data_formatted, str)
        self.assertIn("Google Ads Campaign Performance Data", campaign_data_formatted)

        # Check if all campaign names are included
        for campaign in campaigns:
            self.assertIn(campaign["name"], campaign_data_formatted)
            self.assertIn(str(campaign["clicks"]), campaign_data_formatted)
            self.assertIn(str(campaign["impressions"]), campaign_data_formatted)

        # Test keyword data formatting
        keyword_data_formatted = real_optimizer.format_keyword_data(keywords)
        self.assertIsInstance(keyword_data_formatted, str)
        self.assertIn("Google Ads Keyword Performance Data", keyword_data_formatted)

        # Check if all keyword texts are included
        for keyword in keywords:
            self.assertIn(keyword["keyword_text"], keyword_data_formatted)
            self.assertIn(str(keyword["clicks"]), keyword_data_formatted)

        self.logger.info("Optimizer comprehensive input tests passed!")

    def test_chat_interface_commands(self):
        """Test that chat interface correctly processes user commands"""
        self.logger.info("Testing chat interface commands...")

        # Create a mock chat interface
        chat_interface = ChatInterface(self.ads_api, self.optimizer, self.config, self.logger)

        # Test command processing
        commands = [
            "Analyze my campaigns",
            "Give me keyword recommendations",
            "How are my campaigns performing?",
            "Find my best keywords",
            "What should I optimize?",
        ]

        for command in commands:
            # Process the command
            response = chat_interface.process_user_message(command)

            # Verify response
            self.assertIsInstance(response, str)
            self.assertNotEqual(response, "")

        self.logger.info("Chat interface command tests passed!")

    def test_suggestion_parsing_and_application(self):
        """Test that optimization suggestions are correctly parsed and can be applied"""
        self.logger.info("Testing suggestion parsing and application...")

        # Get test data and suggestions
        campaigns = self.ads_api.get_campaign_performance(days_ago=30)
        keywords = self.ads_api.get_keyword_performance(days_ago=30)
        suggestions = self.optimizer.get_optimization_suggestions(campaigns, keywords)

        # Verify suggestions structure
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)

        for suggestion in suggestions:
            # Check required fields
            self.assertIn("title", suggestion)
            self.assertIn("action_type", suggestion)
            self.assertIn("entity_type", suggestion)
            self.assertIn("entity_id", suggestion)
            self.assertIn("change", suggestion)
            self.assertIn("status", suggestion)

            # Test applying a suggestion
            if (
                suggestion["action_type"] == "BID_ADJUSTMENT"
                and suggestion["entity_type"] == "keyword"
            ):
                success, message = self.ads_api.apply_optimization(
                    "bid_adjustment",
                    "keyword",
                    suggestion["entity_id"],
                    {"bid_micros": 1200000},  # $1.20
                )
                self.assertTrue(success)
                self.assertIsInstance(message, str)

            elif (
                suggestion["action_type"] == "STATUS_CHANGE"
                and suggestion["entity_type"] == "keyword"
            ):
                success, message = self.ads_api.apply_optimization(
                    "status_change", "keyword", suggestion["entity_id"], {"status": "PAUSED"}
                )
                self.assertTrue(success)
                self.assertIsInstance(message, str)

            elif (
                suggestion["action_type"] == "BUDGET_ADJUSTMENT"
                and suggestion["entity_type"] == "campaign"
            ):
                success, message = self.ads_api.apply_optimization(
                    "budget_adjustment",
                    "campaign",
                    suggestion["entity_id"],
                    {"budget_micros": 75000000},  # $75.00
                )
                self.assertTrue(success)
                self.assertIsInstance(message, str)

        self.logger.info("Suggestion parsing and application tests passed!")

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        self.logger.info("Testing edge cases and error handling...")

        # Test empty data
        empty_campaigns = []
        empty_keywords = []

        # Creating actual optimizer for this test
        real_optimizer = AdsOptimizer(self.config["google_ai"])

        # Test optimizer response with empty campaigns
        campaign_data_formatted = real_optimizer.format_campaign_data(empty_campaigns)
        self.assertEqual(campaign_data_formatted, "No campaign data available.")

        # Test optimizer response with empty keywords
        keyword_data_formatted = real_optimizer.format_keyword_data(empty_keywords)
        self.assertEqual(keyword_data_formatted, "No keyword data available.")

        self.logger.info("Edge case tests passed!")

    def test_all_functionality(self):
        """Comprehensive test of all functionality working together"""
        self.logger.info("Running comprehensive functionality test...")

        # Test the full optimization flow
        campaigns = self.ads_api.get_campaign_performance(days_ago=30)
        keywords = self.ads_api.get_keyword_performance(days_ago=30)

        # Verify campaign data
        self.assertIsInstance(campaigns, list)
        self.assertGreater(len(campaigns), 0)

        # Verify keyword data
        self.assertIsInstance(keywords, list)
        self.assertGreater(len(keywords), 0)

        # Get optimization suggestions
        suggestions = self.optimizer.get_optimization_suggestions(campaigns, keywords)

        # Verify suggestions
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)

        # Apply one suggestion of each type
        applied_count = 0
        for suggestion in suggestions:
            if suggestion["status"] == "pending":
                if (
                    suggestion["action_type"] == "BID_ADJUSTMENT"
                    and suggestion["entity_type"] == "keyword"
                    and applied_count == 0
                ):
                    success, message = self.ads_api.apply_optimization(
                        "bid_adjustment",
                        "keyword",
                        suggestion["entity_id"],
                        {"bid_micros": 1200000},  # $1.20
                    )
                    self.assertTrue(success)
                    applied_count += 1

                elif (
                    suggestion["action_type"] == "STATUS_CHANGE"
                    and suggestion["entity_type"] == "keyword"
                    and applied_count == 1
                ):
                    success, message = self.ads_api.apply_optimization(
                        "status_change", "keyword", suggestion["entity_id"], {"status": "PAUSED"}
                    )
                    self.assertTrue(success)
                    applied_count += 1

                elif (
                    suggestion["action_type"] == "BUDGET_ADJUSTMENT"
                    and suggestion["entity_type"] == "campaign"
                    and applied_count == 2
                ):
                    success, message = self.ads_api.apply_optimization(
                        "budget_adjustment",
                        "campaign",
                        suggestion["entity_id"],
                        {"budget_micros": 75000000},  # $75.00
                    )
                    self.assertTrue(success)
                    applied_count += 1

        self.assertGreater(applied_count, 0, "No suggestions were applied")

        self.logger.info("Comprehensive functionality test passed!")


if __name__ == "__main__":
    unittest.main()
