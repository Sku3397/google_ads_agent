"""
Tests for the SERP Scraper Service

This module contains tests for the SERPScraperService component.
"""

import json
import os
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from services.serp_scraper_service import SERPScraperService
from services.serp_scraper_service.serp_scraper_service import SERPResult


class TestSERPScraperService(unittest.TestCase):
    """Test cases for the SERP Scraper Service"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_ads_api = MagicMock()
        self.mock_optimizer = MagicMock()
        self.config = {
            "webdriver_path": "path/to/chromedriver",
            "use_proxy": False
        }
        self.logger = MagicMock()
        
        # Create a mock for the WebDriver
        self.patcher_webdriver = patch('services.serp_scraper_service.serp_scraper_service.webdriver')
        self.mock_webdriver = self.patcher_webdriver.start()
        
        # Create a mock for WebDriverWait
        self.patcher_wait = patch('services.serp_scraper_service.serp_scraper_service.WebDriverWait')
        self.mock_wait = self.patcher_wait.start()
        
        # Create a mock for time.sleep to avoid delays in tests
        self.patcher_sleep = patch('services.serp_scraper_service.serp_scraper_service.time.sleep')
        self.mock_sleep = self.patcher_sleep.start()
        
        # Create a mock for the Service class
        self.patcher_service = patch('services.serp_scraper_service.serp_scraper_service.Service')
        self.mock_service = self.patcher_service.start()
        
        # Create the service with mocks
        self.serp_service = SERPScraperService(
            ads_api=self.mock_ads_api,
            optimizer=self.mock_optimizer,
            config=self.config,
            logger=self.logger
        )
        
        # Set up a mock driver
        self.mock_driver = MagicMock()
        self.serp_service.driver = self.mock_driver

    def tearDown(self):
        """Tear down test fixtures"""
        self.patcher_webdriver.stop()
        self.patcher_wait.stop()
        self.patcher_sleep.stop()
        self.patcher_service.stop()

    def test_initialization(self):
        """Test the initialization of the service"""
        self.assertEqual(self.serp_service.webdriver_path, self.config["webdriver_path"])
        self.assertEqual(self.serp_service.use_proxy, self.config["use_proxy"])
        self.assertEqual(self.serp_service.ads_api, self.mock_ads_api)
        self.assertEqual(self.serp_service.optimizer, self.mock_optimizer)
        
        # Check that the results directory exists
        results_dir = os.path.join("data", "serp_results")
        self.assertTrue(os.path.exists(results_dir))

    def test_initialize_driver(self):
        """Test initializing the WebDriver"""
        # Reset the driver
        self.serp_service.driver = None
        
        # Set up mocks for Chrome options
        mock_options = MagicMock()
        self.mock_webdriver.Chrome.options.Options.return_value = mock_options
        
        # Call method
        self.serp_service._initialize_driver()
        
        # Check that options were set correctly
        mock_options.add_argument.assert_any_call("--headless")
        mock_options.add_argument.assert_any_call("--no-sandbox")
        mock_options.add_argument.assert_any_call("--disable-dev-shm-usage")
        
        # Check that the driver was created with options
        self.mock_webdriver.Chrome.assert_called_once()
        
        # Check that the logger was called
        self.logger.info.assert_called_with("WebDriver initialized successfully")

    def test_close_driver(self):
        """Test closing the WebDriver"""
        # Call method
        self.serp_service._close_driver()
        
        # Check that the driver was quit
        self.mock_driver.quit.assert_called_once()
        
        # Check that the driver was set to None
        self.assertIsNone(self.serp_service.driver)
        
        # Check that the logger was called
        self.logger.info.assert_called_with("WebDriver closed")

    @patch('services.serp_scraper_service.serp_scraper_service.os.path.exists')
    @patch('services.serp_scraper_service.serp_scraper_service.open')
    @patch('services.serp_scraper_service.serp_scraper_service.json.dump')
    def test_save_serp_result(self, mock_json_dump, mock_open, mock_exists):
        """Test saving a SERP result to a file"""
        # Create a mock file handle
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Create a test result
        test_result = SERPResult(
            query="test query",
            timestamp=datetime.now().isoformat(),
            ads_top=[{"title": "Test Ad", "display_url": "test.com", "description": "Test description"}],
            ads_bottom=[],
            organic_results=[{"position": 1, "title": "Test Result", "url": "test.com", "description": "Test description"}],
            related_searches=["test related search"]
        )
        
        # Call method
        self.serp_service._save_serp_result(test_result)
        
        # Check that the file was opened for writing
        mock_open.assert_called_once()
        
        # Check that json.dump was called with the result data
        mock_json_dump.assert_called_once()
        
        # Check that the logger was called
        self.logger.info.assert_called_once()

    @patch('services.serp_scraper_service.serp_scraper_service.SERPScraperService._extract_ads')
    @patch('services.serp_scraper_service.serp_scraper_service.SERPScraperService._extract_organic_results')
    @patch('services.serp_scraper_service.serp_scraper_service.SERPScraperService._extract_related_searches')
    @patch('services.serp_scraper_service.serp_scraper_service.SERPScraperService._extract_knowledge_panel')
    @patch('services.serp_scraper_service.serp_scraper_service.SERPScraperService._extract_local_pack')
    @patch('services.serp_scraper_service.serp_scraper_service.SERPScraperService._extract_shopping_results')
    @patch('services.serp_scraper_service.serp_scraper_service.SERPScraperService._save_serp_result')
    def test_scrape_serp(self, mock_save, mock_shopping, mock_local, mock_knowledge, 
                        mock_related, mock_organic, mock_ads):
        """Test scraping a SERP"""
        # Set up return values for mocks
        mock_ads.side_effect = [
            [{"title": "Test Ad", "display_url": "test.com", "description": "Test description"}],  # top ads
            []  # bottom ads
        ]
        mock_organic.return_value = [{"position": 1, "title": "Test Result", "url": "test.com", "description": "Test description"}]
        mock_related.return_value = ["test related search"]
        mock_knowledge.return_value = None
        mock_local.return_value = None
        mock_shopping.return_value = None
        
        # Call method
        result = self.serp_service.scrape_serp("test query")
        
        # Check the result
        self.assertEqual(result.query, "test query")
        self.assertEqual(len(result.ads_top), 1)
        self.assertEqual(len(result.ads_bottom), 0)
        self.assertEqual(len(result.organic_results), 1)
        self.assertEqual(len(result.related_searches), 1)
        
        # Check that the driver methods were called
        self.mock_driver.get.assert_called_once_with("https://www.google.com/search?q=test+query")
        
        # Check that the result was saved
        mock_save.assert_called_once()

    def test_extract_ads_top(self):
        """Test extracting top ads from a SERP"""
        # Set up mocks for driver elements
        mock_ad_container = MagicMock()
        self.mock_driver.find_element.return_value = mock_ad_container
        
        mock_ad_element = MagicMock()
        mock_ad_container.find_elements.return_value = [mock_ad_element]
        
        # Set up text values for the ad elements
        mock_ad_element.find_element.side_effect = lambda by, value: {
            "div.CCgQ5": MagicMock(text="Test Ad Title"),
            "span.x2VHCd": MagicMock(text="test.com"),
            "div.MUxGbd": MagicMock(text="Test Ad Description")
        }[value]
        
        # Extensions
        mock_extension = MagicMock(text="Test Extension")
        mock_ad_element.find_elements.return_value = [mock_extension]
        
        # Call method
        ads = self.serp_service._extract_ads("top")
        
        # Check the results
        self.assertEqual(len(ads), 1)
        self.assertEqual(ads[0]["title"], "Test Ad Title")
        self.assertEqual(ads[0]["display_url"], "test.com")
        self.assertEqual(ads[0]["description"], "Test Ad Description")
        self.assertEqual(ads[0]["position"], "top")
        self.assertEqual(len(ads[0]["extensions"]), 1)
        self.assertEqual(ads[0]["extensions"][0], "Test Extension")

    def test_extract_organic_results(self):
        """Test extracting organic results from a SERP"""
        # Set up mocks for driver elements
        mock_org_element = MagicMock()
        self.mock_driver.find_elements.return_value = [mock_org_element]
        
        # Set up text values for the organic elements
        mock_title = MagicMock(text="Test Organic Title")
        mock_url = MagicMock()
        mock_url.get_attribute.return_value = "https://test.com"
        mock_desc = MagicMock(text="Test Organic Description")
        
        mock_org_element.find_element.side_effect = lambda by, value: {
            "h3": mock_title,
            "a": mock_url,
            "div.VwiC3b": mock_desc
        }[value]
        
        # Call method
        results = self.serp_service._extract_organic_results()
        
        # Check the results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["position"], 1)
        self.assertEqual(results[0]["title"], "Test Organic Title")
        self.assertEqual(results[0]["url"], "https://test.com")
        self.assertEqual(results[0]["description"], "Test Organic Description")

    def test_analyze_competitor_ads(self):
        """Test analyzing competitor ads"""
        # Mock the scrape_serp method
        self.serp_service.scrape_serp = MagicMock()
        
        # Create mock results
        mock_result1 = MagicMock()
        mock_result1.ads_top = [
            {"title": "Ad 1", "display_url": "competitor1.com", "description": "Description 1", "position": "top", "extensions": []}
        ]
        mock_result1.ads_bottom = []
        
        mock_result2 = MagicMock()
        mock_result2.ads_top = [
            {"title": "Ad 2", "display_url": "competitor1.com", "description": "Description 2", "position": "top", "extensions": []}
        ]
        mock_result2.ads_bottom = [
            {"title": "Ad 3", "display_url": "competitor2.com", "description": "Description 3", "position": "bottom", "extensions": []}
        ]
        
        # Set return values for scrape_serp
        self.serp_service.scrape_serp.side_effect = [mock_result1, mock_result2]
        
        # Mock the open function for saving results
        with patch('services.serp_scraper_service.serp_scraper_service.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            # Call method
            result = self.serp_service.analyze_competitor_ads(["query1", "query2"])
            
            # Check the results
            self.assertEqual(result["total_ads_found"], 3)
            self.assertEqual(result["queries_analyzed"], 2)
            self.assertIn("competitors", result)
            self.assertEqual(len(result["competitors"]), 2)
            
            # Check that scrape_serp was called twice
            self.assertEqual(self.serp_service.scrape_serp.call_count, 2)
            
            # Check that the result was saved
            mock_open.assert_called_once()

    def test_track_keyword_rankings(self):
        """Test tracking keyword rankings"""
        # Mock the scrape_serp method
        self.serp_service.scrape_serp = MagicMock()
        
        # Create mock results
        mock_result1 = MagicMock()
        mock_result1.organic_results = [
            {"position": 1, "title": "Result 1", "url": "https://test.com/page1", "description": "Description 1"}
        ]
        
        mock_result2 = MagicMock()
        mock_result2.organic_results = [
            {"position": 1, "title": "Result 2", "url": "https://competitor.com", "description": "Description 2"},
            {"position": 2, "title": "Result 3", "url": "https://test.com/page2", "description": "Description 3"}
        ]
        
        # Set return values for scrape_serp
        self.serp_service.scrape_serp.side_effect = [mock_result1, mock_result2]
        
        # Mock the open function for saving results
        with patch('services.serp_scraper_service.serp_scraper_service.open', create=True) as mock_open:
            # Mock the json.dump function
            with patch('services.serp_scraper_service.serp_scraper_service.json.dump') as mock_json_dump:
                # Mock os.path.exists to return False (no existing history file)
                with patch('services.serp_scraper_service.serp_scraper_service.os.path.exists', return_value=False):
                    # Call method
                    result = self.serp_service.track_keyword_rankings(
                        ["keyword1", "keyword2"], "test.com", store_history=True
                    )
                    
                    # Check the results
                    self.assertEqual(result["domain"], "test.com")
                    self.assertEqual(result["keywords_tracked"], 2)
                    self.assertIn("rankings", result)
                    self.assertEqual(len(result["rankings"]), 2)
                    
                    # Check keyword1 rankings (should be position 1)
                    self.assertTrue(result["rankings"]["keyword1"]["found"])
                    self.assertEqual(result["rankings"]["keyword1"]["position"], 1)
                    
                    # Check keyword2 rankings (should be position 2)
                    self.assertTrue(result["rankings"]["keyword2"]["found"])
                    self.assertEqual(result["rankings"]["keyword2"]["position"], 2)
                    
                    # Check that scrape_serp was called twice
                    self.assertEqual(self.serp_service.scrape_serp.call_count, 2)
                    
                    # Check that results were saved
                    self.assertEqual(mock_open.call_count, 2)  # once for rankings, once for history
                    self.assertEqual(mock_json_dump.call_count, 2)


# Run the tests
if __name__ == "__main__":
    unittest.main() 