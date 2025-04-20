"""
Tests for the ExperimentationService integration with AdsAgent.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import os
import tempfile
from datetime import datetime, timedelta

from ads_agent import AdsAgent
from services.experimentation_service import ExperimentationService


class TestExperimentationServiceIntegration(unittest.TestCase):
    """Test cases for ExperimentationService integration with AdsAgent."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for experiment data
        self.temp_dir = tempfile.TemporaryDirectory()

        # Mock config
        self.config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "developer_token": "test_developer_token",
            "refresh_token": "test_refresh_token",
            "login_customer_id": "1234567890",
            "use_proto_plus": False,
            "experiment_data_path": os.path.join(self.temp_dir.name, "experiments.json"),
        }

        # Mock API and optimizer
        self.mock_ads_api = Mock()
        self.mock_optimizer = Mock()

        # Create a logger that doesn't output anything for tests
        self.mock_logger = Mock()

        # Create a patched AdsAgent
        with patch("ads_agent.load_config", return_value=self.config):
            with patch("ads_agent.GoogleAdsAPI", return_value=self.mock_ads_api):
                with patch("ads_agent.AdsOptimizer", return_value=self.mock_optimizer):
                    with patch("ads_agent.logging.getLogger", return_value=self.mock_logger):
                        self.agent = AdsAgent()

    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()

    def test_create_experiment(self):
        """Test creating an experiment."""
        # Mock the ExperimentationService.create_experiment method
        mock_experiment_id = "test-experiment-id-123"
        mock_experiment = {
            "id": mock_experiment_id,
            "name": "Test Experiment",
            "type": "A/B Test",
            "status": "draft",
        }

        # Set up mocks
        self.agent.services["experimentation"].create_experiment = Mock(
            return_value=mock_experiment_id
        )
        self.agent.services["experimentation"].get_experiment = Mock(return_value=mock_experiment)

        # Call the method
        result = self.agent.create_experiment(
            name="Test Experiment",
            type="A/B Test",
            hypothesis="Testing improves results",
            control_group="campaign-1",
            treatment_groups=["campaign-2", "campaign-3"],
            metrics=["clicks", "conversions"],
            duration_days=30,
        )

        # Verify the results
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["message"], "Created experiment 'Test Experiment'")
        self.assertEqual(result["experiment"], mock_experiment)

        # Verify the expected method was called with correct arguments
        self.agent.services["experimentation"].create_experiment.assert_called_once_with(
            name="Test Experiment",
            type="A/B Test",
            hypothesis="Testing improves results",
            control_group="campaign-1",
            treatment_groups=["campaign-2", "campaign-3"],
            metrics=["clicks", "conversions"],
            duration_days=30,
            traffic_split=None,
            custom_parameters=None,
        )

    def test_list_experiments(self):
        """Test listing experiments."""
        # Mock data
        mock_experiments = [
            {"id": "exp-1", "name": "Experiment 1", "status": "running"},
            {"id": "exp-2", "name": "Experiment 2", "status": "completed"},
        ]

        # Set up mock
        self.agent.services["experimentation"].list_experiments = Mock(
            return_value=mock_experiments
        )

        # Call the method
        result = self.agent.list_experiments()

        # Verify the results
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["experiments"], mock_experiments)

        # Verify the method was called
        self.agent.services["experimentation"].list_experiments.assert_called_once_with(
            status=None, limit=100, offset=0
        )

    def test_start_experiment(self):
        """Test starting an experiment."""
        # Mock data
        mock_experiment_id = "exp-test-123"
        mock_experiment = {"id": mock_experiment_id, "name": "Test Experiment", "status": "running"}

        # Set up mock
        self.agent.services["experimentation"].start_experiment = Mock(return_value=mock_experiment)

        # Call the method
        result = self.agent.start_experiment(mock_experiment_id)

        # Verify the results
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["message"], "Started experiment 'Test Experiment'")
        self.assertEqual(result["experiment"], mock_experiment)

        # Verify the method was called
        self.agent.services["experimentation"].start_experiment.assert_called_once_with(
            mock_experiment_id
        )

    def test_stop_experiment(self):
        """Test stopping an experiment."""
        # Mock data
        mock_experiment_id = "exp-test-123"
        mock_experiment = {"id": mock_experiment_id, "name": "Test Experiment", "status": "stopped"}

        # Set up mock
        self.agent.services["experimentation"].stop_experiment = Mock(return_value=mock_experiment)

        # Call the method
        result = self.agent.stop_experiment(mock_experiment_id)

        # Verify the results
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["message"], "Stopped experiment 'Test Experiment'")
        self.assertEqual(result["experiment"], mock_experiment)

        # Verify the method was called
        self.agent.services["experimentation"].stop_experiment.assert_called_once_with(
            mock_experiment_id
        )

    def test_analyze_experiment(self):
        """Test analyzing an experiment."""
        # Mock data
        mock_experiment_id = "exp-test-123"
        mock_experiment = {
            "id": mock_experiment_id,
            "name": "Test Experiment",
            "status": "completed",
        }
        mock_results = {
            "winner": "campaign-2",
            "metrics": {
                "clicks": {
                    "winner": "campaign-2",
                    "treatments": [{"name": "campaign-2", "lift": 0.15, "significant": True}],
                }
            },
        }

        # Set up mocks
        self.agent.services["experimentation"].analyze_experiment = Mock(return_value=mock_results)
        self.agent.services["experimentation"].get_experiment = Mock(return_value=mock_experiment)

        # Call the method
        result = self.agent.analyze_experiment(mock_experiment_id, confidence_level=0.95)

        # Verify the results
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["message"], "Analyzed experiment 'Test Experiment'")
        self.assertEqual(result["experiment_name"], "Test Experiment")
        self.assertEqual(result["results"], mock_results)

        # Verify the method was called
        self.agent.services["experimentation"].analyze_experiment.assert_called_once_with(
            experiment_id=mock_experiment_id, confidence_level=0.95
        )

    def test_apply_winning_variation(self):
        """Test applying the winning variation."""
        # Mock data
        mock_experiment_id = "exp-test-123"
        mock_experiment = {"id": mock_experiment_id, "name": "Test Experiment", "status": "applied"}

        # Set up mocks
        self.agent.services["experimentation"].apply_winning_variation = Mock(return_value=True)
        self.agent.services["experimentation"].get_experiment = Mock(return_value=mock_experiment)

        # Call the method
        result = self.agent.apply_winning_variation(mock_experiment_id)

        # Verify the results
        self.assertEqual(result["status"], "success")
        self.assertEqual(
            result["message"], "Applied winning variation for experiment 'Test Experiment'"
        )
        self.assertEqual(result["experiment"], mock_experiment)

        # Verify the method was called
        self.agent.services["experimentation"].apply_winning_variation.assert_called_once_with(
            mock_experiment_id
        )

    def test_get_experiment_recommendations(self):
        """Test getting experiment recommendations."""
        # Mock data
        mock_experiment_id = "exp-test-123"
        mock_experiment = {
            "id": mock_experiment_id,
            "name": "Test Experiment",
            "status": "completed",
        }
        mock_recommendations = [
            {
                "type": "apply_winner",
                "description": "Apply the winning variation 'campaign-2' to the original campaign",
                "importance": "high",
            }
        ]

        # Set up mocks
        self.agent.services["experimentation"].get_experiment_recommendations = Mock(
            return_value=mock_recommendations
        )
        self.agent.services["experimentation"].get_experiment = Mock(return_value=mock_experiment)

        # Call the method
        result = self.agent.get_experiment_recommendations(mock_experiment_id)

        # Verify the results
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["experiment_name"], "Test Experiment")
        self.assertEqual(result["recommendations"], mock_recommendations)

        # Verify the method was called
        self.agent.services[
            "experimentation"
        ].get_experiment_recommendations.assert_called_once_with(mock_experiment_id)

    def test_error_handling(self):
        """Test error handling."""
        # Mock data
        mock_experiment_id = "exp-test-123"
        mock_error = ValueError("Experiment not found")

        # Set up mock to raise an exception
        self.agent.services["experimentation"].get_experiment = Mock(side_effect=mock_error)

        # Call the method and expect an error response
        result = self.agent.get_experiment_recommendations(mock_experiment_id)

        # Verify the error response
        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            result["message"], "Error getting experiment recommendations: Experiment not found"
        )


if __name__ == "__main__":
    unittest.main()
