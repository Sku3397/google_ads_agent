import unittest
import logging
from unittest.mock import MagicMock, patch
from services.creative_service.creative_service import CreativeService
import pytest
from unittest.mock import Mock
from datetime import datetime
import numpy as np
from services.creative_service import CreativeElement, CreativeTest

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestCreativeService(unittest.TestCase):
    """Test case for the CreativeService class."""

    def setUp(self):
        """Set up the test fixtures before each test."""
        # Create mock dependencies
        self.mock_ads_api = MagicMock()
        self.mock_optimizer = MagicMock()
        self.mock_config = {}
        self.mock_logger = MagicMock()

        # Create instance of CreativeService with mock dependencies
        self.service = CreativeService(
            ads_api=self.mock_ads_api,
            optimizer=self.mock_optimizer,
            config=self.mock_config,
            logger=self.mock_logger,
        )
        logger.info("Set up test environment for CreativeService")

    def test_initialization(self):
        """Test that the service initializes correctly."""
        self.assertIsInstance(self.service, CreativeService)
        self.assertEqual(self.service.headline_max_length, 30)
        self.assertEqual(self.service.description_max_length, 90)
        self.assertEqual(self.service.max_headlines_per_ad, 15)
        self.assertEqual(self.service.max_descriptions_per_ad, 4)
        logger.info("Initialization test passed")

    def test_custom_config(self):
        """Test that custom configuration is applied correctly."""
        # Create a service with custom configuration
        custom_config = {
            "creative_service": {
                "headline_max_length": 25,
                "description_max_length": 80,
                "max_headlines_per_ad": 10,
                "max_descriptions_per_ad": 3,
            }
        }

        service = CreativeService(
            ads_api=self.mock_ads_api,
            optimizer=self.mock_optimizer,
            config=custom_config,
            logger=self.mock_logger,
        )

        # Verify custom settings were applied
        self.assertEqual(service.headline_max_length, 25)
        self.assertEqual(service.description_max_length, 80)
        self.assertEqual(service.max_headlines_per_ad, 10)
        self.assertEqual(service.max_descriptions_per_ad, 3)
        logger.info("Custom configuration test passed")

    def test_generate_ad_content(self):
        """Test the generate_ad_content method."""
        # Call the method
        result = self.service.generate_ad_content(
            campaign_id="123456",
            ad_group_id="654321",
            keywords=["test keyword"],
            product_info={"name": "Test Product"},
        )

        # Verify the result
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["campaign_id"], "123456")
        self.assertEqual(result["ad_group_id"], "654321")
        self.assertIn("headlines", result)
        self.assertIn("descriptions", result)
        self.assertTrue(len(result["headlines"]) > 0)
        self.assertTrue(len(result["descriptions"]) > 0)
        logger.info("Generate ad content test passed")

    def test_analyze_ad_performance(self):
        """Test the analyze_ad_performance method."""
        # Call the method with campaign filter
        result = self.service.analyze_ad_performance(campaign_id="123456", days=15)

        # Verify the result
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["scope"], "campaign 123456")
        self.assertEqual(result["days_analyzed"], 15)
        self.assertIn("top_performing_headlines", result)
        self.assertIn("improvement_suggestions", result)
        logger.info("Analyze ad performance test passed")

    def test_create_responsive_search_ad(self):
        """Test the create_responsive_search_ad method."""
        # Create test data
        headlines = [{"text": "Test Headline", "pinned_position": 1}]
        descriptions = [{"text": "Test Description", "pinned_position": 1}]

        # Call the method
        result = self.service.create_responsive_search_ad(
            campaign_id="123456",
            ad_group_id="654321",
            headlines=headlines,
            descriptions=descriptions,
            final_url="https://example.com",
        )

        # Verify the result
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["campaign_id"], "123456")
        self.assertEqual(result["ad_group_id"], "654321")
        self.assertIn("ad_id", result)
        logger.info("Create responsive search ad test passed")

    def test_setup_ad_testing(self):
        """Test the setup_ad_testing method."""
        # Create test variants
        test_variants = [
            {"headline": "Variant 1", "description": "Test Description 1"},
            {"headline": "Variant 2", "description": "Test Description 2"},
        ]

        # Call the method
        result = self.service.setup_ad_testing(
            campaign_id="123456",
            ad_group_id="654321",
            test_variants=test_variants,
            test_duration_days=7,
        )

        # Verify the result
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["campaign_id"], "123456")
        self.assertEqual(result["ad_group_id"], "654321")
        self.assertEqual(result["variant_count"], 2)
        self.assertIn("test_id", result)
        logger.info("Setup ad testing test passed")

    def test_analyze_ad_test_results(self):
        """Test the analyze_ad_test_results method."""
        # Call the method
        result = self.service.analyze_ad_test_results(test_id="test_123")

        # Verify the result
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["test_id"], "test_123")
        self.assertIn("winner_variant_id", result)
        self.assertIn("winner_confidence", result)
        logger.info("Analyze ad test results test passed")

    def test_get_creative_quality_metrics(self):
        """Test the get_creative_quality_metrics method."""
        # Call the method
        result = self.service.get_creative_quality_metrics(
            campaign_id="123456", ad_group_id="654321"
        )

        # Verify the result
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["scope"], "ad group 654321")
        self.assertIn("quality_metrics", result)
        self.assertIn("improvement_suggestions", result)
        logger.info("Get creative quality metrics test passed")

    def test_run_method_generate_content(self):
        """Test the run method with generate_ad_content action."""
        with patch.object(self.service, "generate_ad_content") as mock_method:
            mock_method.return_value = {"status": "success", "message": "test"}

            result = self.service.run(
                action="generate_ad_content", campaign_id="123456", ad_group_id="654321"
            )

            # Verify the correct method was called with parameters
            mock_method.assert_called_once_with(
                campaign_id="123456",
                ad_group_id="654321",
                keywords=None,
                product_info=None,
                tone="professional",
            )
            self.assertEqual(result["status"], "success")
        logger.info("Run method with generate_ad_content test passed")

    def test_run_method_unknown_action(self):
        """Test the run method with an unknown action."""
        result = self.service.run(action="unknown_action")

        # Verify error response
        self.assertEqual(result["status"], "error")
        self.assertIn("Unknown action", result["message"])
        logger.info("Run method with unknown action test passed")


@pytest.fixture
def mock_google_ads_client():
    """Create a mock Google Ads client."""
    mock_client = Mock()
    mock_client.get_service.return_value = Mock()
    return mock_client


@pytest.fixture
def creative_service(mock_google_ads_client):
    """Create a CreativeService instance with mock client."""
    return CreativeService(mock_google_ads_client, "123456789")


def test_analyze_creative_elements(creative_service, mock_google_ads_client):
    """Test creative element analysis functionality."""
    # Mock Google Ads API response
    mock_response = Mock()
    mock_row = Mock()
    mock_row.ad_group_ad.ad.id = "123"
    mock_row.ad_group_ad.ad.expanded_text_ad.headline_part1 = "Amazing Offer"
    mock_row.ad_group_ad.ad.expanded_text_ad.headline_part2 = "50% Off Today"
    mock_row.ad_group_ad.ad.expanded_text_ad.description = "Shop our collection now"
    mock_row.metrics.impressions = 1000
    mock_row.metrics.clicks = 50
    mock_row.metrics.conversions = 5
    mock_row.metrics.cost_micros = 1000000  # $1.00
    mock_response.__iter__ = Mock(return_value=iter([mock_row]))

    mock_google_ads_client.get_service().search.return_value = mock_response

    # Test analysis
    results = creative_service.analyze_creative_elements(["123"])

    assert "headline_analysis" in results
    assert "description_analysis" in results
    assert "performance_data" in results
    assert "recommendations" in results

    # Verify headline analysis
    headline_analysis = results["headline_analysis"]
    assert "characteristics" in headline_analysis
    assert "common_terms" in headline_analysis

    # Verify performance metrics
    performance = results["performance_data"][0]
    assert performance["creative_id"] == "123"
    assert performance["ctr"] == 0.05  # 50 clicks / 1000 impressions
    assert performance["conv_rate"] == 0.1  # 5 conversions / 50 clicks


def test_setup_creative_experiment(creative_service):
    """Test creative experiment setup."""
    elements = {
        "headlines": ["Test Headline 1", "Test Headline 2"],
        "descriptions": ["Test Description"],
    }

    test = creative_service.setup_creative_experiment("123", elements)

    assert isinstance(test, CreativeTest)
    assert test.ad_group_id == "123"
    assert test.confidence_level == 0.95
    assert test.status == "INITIALIZED"
    assert test.end_date is None
    assert isinstance(test.start_date, datetime)

    # Verify elements were properly converted
    assert "headlines" in test.elements
    assert len(test.elements["headlines"]) == 2
    assert isinstance(test.elements["headlines"][0], CreativeElement)


def test_monitor_creative_test(creative_service, mock_google_ads_client):
    """Test creative test monitoring."""
    # Create a test object
    test = CreativeTest(
        test_id="test_123",
        ad_group_id="123",
        elements={},
        start_date=datetime.now(),
        end_date=None,
        confidence_level=0.95,
        status="RUNNING",
        results=None,
    )

    # Mock API response
    mock_response = Mock()
    mock_row = Mock()
    mock_row.ad_group_ad.ad.id = "123"
    mock_row.metrics.impressions = 2000
    mock_row.metrics.clicks = 100
    mock_row.metrics.conversions = 10
    mock_row.metrics.cost_micros = 2000000  # $2.00
    mock_response.__iter__ = Mock(return_value=iter([mock_row]))

    mock_google_ads_client.get_service().search.return_value = mock_response

    # Test monitoring
    results = creative_service.monitor_creative_test(test)

    assert results["test_id"] == "test_123"
    assert results["status"] == "RUNNING"
    assert "results" in results
    assert "significant_differences" in results
    assert "recommendations" in results

    # Verify significance calculations
    significance = results["significant_differences"]
    assert "123" in significance
    assert "ctr" in significance["123"]
    assert "confidence_interval" in significance["123"]
    assert "sample_size" in significance["123"]


def test_calculate_significance(creative_service):
    """Test statistical significance calculations."""
    results = [
        {"ad_id": "123", "impressions": 1000, "clicks": 50, "conversions": 5, "cost": 1.0},
        {"ad_id": "456", "impressions": 1000, "clicks": 30, "conversions": 3, "cost": 0.8},
    ]

    significance = creative_service._calculate_significance(results, 0.95)

    assert "123" in significance
    assert "456" in significance

    # Verify CTR calculations
    assert significance["123"]["ctr"] == 0.05  # 50/1000
    assert significance["456"]["ctr"] == 0.03  # 30/1000

    # Verify confidence intervals
    assert len(significance["123"]["confidence_interval"]) == 2
    assert significance["123"]["confidence_interval"][0] < significance["123"]["ctr"]
    assert significance["123"]["confidence_interval"][1] > significance["123"]["ctr"]


def test_generate_creative_recommendations(creative_service):
    """Test recommendation generation."""
    headline_analysis = {
        "characteristics": {
            "avg_length": 3,
            "contains_numbers": 0.1,
            "contains_question": 0,
            "contains_exclamation": 0,
            "similarity_score": 0.5,
        },
        "common_terms": ["test", "offer"],
    }

    description_analysis = {
        "characteristics": {
            "avg_length": 5,
            "contains_numbers": 0,
            "contains_question": 0,
            "contains_exclamation": 0,
            "similarity_score": 0.3,
        },
        "common_terms": ["shop", "now"],
    }

    performance_data = [{"creative_id": "123", "ctr": 0.01, "conv_rate": 0.1, "cpa": 5.0}]

    recommendations = creative_service._generate_creative_recommendations(
        headline_analysis, description_analysis, performance_data
    )

    assert isinstance(recommendations, list)
    assert len(recommendations) > 0

    # Verify recommendation structure
    for rec in recommendations:
        assert "type" in rec
        assert "action" in rec
        assert "message" in rec


def test_analyze_text_elements(creative_service):
    """Test text element analysis."""
    texts = ["Amazing Offer - 50% Off", "Limited Time Deal - Save Now", "Best Prices - Shop Today"]

    analysis = creative_service._analyze_text_elements(texts, "headline")

    assert "characteristics" in analysis
    assert "common_terms" in analysis

    chars = analysis["characteristics"]
    assert "avg_length" in chars
    assert "contains_numbers" in chars
    assert "contains_question" in chars
    assert "contains_exclamation" in chars
    assert "similarity_score" in chars

    # Verify text analysis
    assert chars["avg_length"] > 0
    assert 0 <= chars["similarity_score"] <= 1
    assert isinstance(analysis["common_terms"], list)


def test_generate_test_recommendations(creative_service):
    """Test test-specific recommendation generation."""
    results = [
        {"ad_id": "123", "impressions": 2000, "clicks": 100, "conversions": 10, "cost": 2.0},
        {
            "ad_id": "456",
            "impressions": 500,  # Low sample size
            "clicks": 20,
            "conversions": 2,
            "cost": 0.5,
        },
    ]

    significance_results = {
        "123": {"ctr": 0.05, "confidence_interval": (0.04, 0.06), "sample_size": 2000},
        "456": {"ctr": 0.04, "confidence_interval": (0.02, 0.06), "sample_size": 500},
    }

    recommendations = creative_service._generate_test_recommendations(results, significance_results)

    assert isinstance(recommendations, list)
    assert len(recommendations) > 0

    # Verify recommendations
    has_winner = False
    has_sample_size = False

    for rec in recommendations:
        if rec["type"] == "winner":
            has_winner = True
        elif rec["type"] == "sample_size":
            has_sample_size = True

    assert has_winner  # Should identify best performer
    assert has_sample_size  # Should flag low sample size


if __name__ == "__main__":
    unittest.main()
