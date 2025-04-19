import unittest
import logging
from unittest.mock import MagicMock, patch
from services.creative_service.creative_service import CreativeService

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
            logger=self.mock_logger
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
                "max_descriptions_per_ad": 3
            }
        }
        
        service = CreativeService(
            ads_api=self.mock_ads_api,
            optimizer=self.mock_optimizer,
            config=custom_config,
            logger=self.mock_logger
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
            product_info={"name": "Test Product"}
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
        result = self.service.analyze_ad_performance(
            campaign_id="123456",
            days=15
        )
        
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
            final_url="https://example.com"
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
            {"headline": "Variant 2", "description": "Test Description 2"}
        ]
        
        # Call the method
        result = self.service.setup_ad_testing(
            campaign_id="123456",
            ad_group_id="654321",
            test_variants=test_variants,
            test_duration_days=7
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
        result = self.service.analyze_ad_test_results(
            test_id="test_123"
        )
        
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
            campaign_id="123456",
            ad_group_id="654321"
        )
        
        # Verify the result
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["scope"], "ad group 654321")
        self.assertIn("quality_metrics", result)
        self.assertIn("improvement_suggestions", result)
        logger.info("Get creative quality metrics test passed")
    
    def test_run_method_generate_content(self):
        """Test the run method with generate_ad_content action."""
        with patch.object(self.service, 'generate_ad_content') as mock_method:
            mock_method.return_value = {"status": "success", "message": "test"}
            
            result = self.service.run(
                action="generate_ad_content",
                campaign_id="123456",
                ad_group_id="654321"
            )
            
            # Verify the correct method was called with parameters
            mock_method.assert_called_once_with(
                campaign_id="123456",
                ad_group_id="654321",
                keywords=None,
                product_info=None,
                tone="professional"
            )
            self.assertEqual(result["status"], "success")
        logger.info("Run method with generate_ad_content test passed")
    
    def test_run_method_unknown_action(self):
        """Test the run method with an unknown action."""
        result = self.service.run(
            action="unknown_action"
        )
        
        # Verify error response
        self.assertEqual(result["status"], "error")
        self.assertIn("Unknown action", result["message"])
        logger.info("Run method with unknown action test passed")

if __name__ == '__main__':
    unittest.main() 