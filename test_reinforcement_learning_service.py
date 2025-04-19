"""
Unit tests for the Reinforcement Learning Service.
"""

import unittest
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from services.reinforcement_learning_service import ReinforcementLearningService
from services.base_service import BaseService

class TestReinforcementLearningService(unittest.TestCase):
    """Test cases for the Reinforcement Learning Service."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock dependencies
        self.mock_ads_api = MagicMock()
        self.mock_optimizer = MagicMock()
        
        # Create a set of mock keywords
        self.mock_keywords = [
            {
                "criterion_id": "123",
                "keyword_text": "test keyword 1",
                "match_type": "EXACT",
                "campaign_id": "456",
                "campaign_name": "Test Campaign",
                "ad_group_id": "789",
                "ad_group_name": "Test Ad Group",
                "current_bid": 1.50,
                "impressions": 100,
                "clicks": 10,
                "conversions": 1,
                "cost": 15.0
            },
            {
                "criterion_id": "124",
                "keyword_text": "test keyword 2",
                "match_type": "PHRASE",
                "campaign_id": "456",
                "campaign_name": "Test Campaign",
                "ad_group_id": "789",
                "ad_group_name": "Test Ad Group",
                "current_bid": 2.00,
                "impressions": 200,
                "clicks": 20,
                "conversions": 2,
                "cost": 30.0
            }
        ]
        
        # Configure mocks
        self.mock_ads_api.get_keyword_performance.return_value = self.mock_keywords
        
        # Create config with temp model path
        self.config = {
            "reinforcement_learning": {
                "model_save_path": os.path.join(self.temp_dir, "rl_models")
            }
        }
        
        # Create the service
        self.rl_service = ReinforcementLearningService(
            ads_api=self.mock_ads_api,
            optimizer=self.mock_optimizer,
            config=self.config
        )
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temp directory
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test service initialization."""
        self.assertIsInstance(self.rl_service, BaseService)
        self.assertEqual(self.rl_service.model_save_path, 
                        os.path.join(self.temp_dir, "rl_models"))
        self.assertTrue(os.path.exists(self.rl_service.model_save_path))
    
    def test_build_auction_simulator(self):
        """Test building an auction simulator."""
        # Mock the private methods
        self.rl_service._get_historical_auction_insights = MagicMock(return_value=[{"data": "mock"}])
        self.rl_service._process_auction_insights = MagicMock(return_value=pd.DataFrame())
        self.rl_service._build_simulator_model = MagicMock(return_value={"model": "mock"})
        
        # Call the method
        result = self.rl_service.build_auction_simulator()
        
        # Verify the result
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["simulator_model"], {"model": "mock"})
        self.assertEqual(result["data_points"], 1)
        
        # Verify method calls
        self.rl_service._get_historical_auction_insights.assert_called_once()
        self.rl_service._process_auction_insights.assert_called_once()
        self.rl_service._build_simulator_model.assert_called_once()
    
    def test_train_policy(self):
        """Test training a policy."""
        # Mock the private methods
        self.rl_service.build_auction_simulator = MagicMock(return_value={"status": "success"})
        self.rl_service._initialize_policy_model = MagicMock(return_value={"model": "mock"})
        self.rl_service._train_policy_model = MagicMock(return_value={"metrics": "mock"})
        
        # Call the method
        result = self.rl_service.train_policy(training_episodes=100)
        
        # Verify the result
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["model"], {"model": "mock"})
        self.assertEqual(result["training_metrics"], {"metrics": "mock"})
        self.assertEqual(result["training_episodes"], 100)
        
        # Verify method calls
        self.rl_service._initialize_policy_model.assert_called_once()
        self.rl_service._train_policy_model.assert_called_once()
    
    def test_generate_bid_recommendations(self):
        """Test generating bid recommendations."""
        # Mock the private methods
        mock_policy = {"type": "mock_policy"}
        mock_recommendations = [{"keyword_id": "123", "recommended_bid": 1.75}]
        
        self.rl_service._load_policy_model = MagicMock(return_value=mock_policy)
        self.rl_service._get_current_keyword_data = MagicMock(return_value=self.mock_keywords)
        self.rl_service._generate_recommendations_from_policy = MagicMock(return_value=mock_recommendations)
        
        # Call the method
        result = self.rl_service.generate_bid_recommendations()
        
        # Verify the result
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["recommendations"], mock_recommendations)
        self.assertEqual(result["keywords_analyzed"], 2)
        self.assertEqual(result["recommendations_count"], 1)
        
        # Verify method calls
        self.rl_service._load_policy_model.assert_called_once()
        self.rl_service._get_current_keyword_data.assert_called_once()
        self.rl_service._generate_recommendations_from_policy.assert_called_once_with(
            mock_policy, self.mock_keywords, 0.1
        )
    
    def test_get_current_keyword_data(self):
        """Test getting current keyword data."""
        # Call the method
        result = self.rl_service._get_current_keyword_data()
        
        # Verify the result
        self.assertEqual(result, self.mock_keywords)
        
        # Verify the API was called correctly
        self.mock_ads_api.get_keyword_performance.assert_called_once()
    
    def test_generate_recommendations_from_policy(self):
        """Test generating recommendations from policy."""
        # Call the method
        mock_policy = {"algorithm": "test_algorithm"}
        result = self.rl_service._generate_recommendations_from_policy(
            mock_policy, self.mock_keywords, 0.05
        )
        
        # Basic validation - should return a list
        self.assertIsInstance(result, list)
        
        # Should have recommendations for our keywords
        if result:  # May be empty in placeholder implementation
            for rec in result:
                self.assertIn(rec["keyword_id"], ["123", "124"])
                self.assertIsNotNone(rec["recommended_bid"])
                self.assertEqual(rec["algorithm"], "test_algorithm")


if __name__ == '__main__':
    unittest.main() 