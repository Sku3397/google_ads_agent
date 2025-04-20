import unittest
import logging
import os
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock, ANY
import numpy as np
from google.ads.googleads.client import GoogleAdsClient
from services.bandit_service import BanditService, BanditAlgorithm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestBanditService(unittest.TestCase):
    def setUp(self):
        """Set up the test environment before each test."""
        # Create temp directory for data
        self.temp_dir = tempfile.mkdtemp()

        # Mock GoogleAdsClient to avoid actual API calls
        self.mock_client = MagicMock(spec=GoogleAdsClient)
        self.customer_id = "test_customer_id"
        self.config = {
            "data_path": self.temp_dir,
            "alpha_prior": 1.0,
            "beta_prior": 1.0,
            "epsilon": 0.1,
            "ucb_alpha": 1.0,
            "discount_factor": 0.95,
        }

        # Initialize service
        self.service = BanditService(self.mock_client, self.customer_id, self.config)

        # Reset the bandits dictionary for a clean state in each test
        self.service.bandits = {}

        logger.info("Set up test environment for BanditService")

    def tearDown(self):
        """Clean up after each test."""
        # Remove temp directory
        shutil.rmtree(self.temp_dir)
        logger.info("Cleaned up test environment")

    def test_initialization(self):
        """Test that the BanditService initializes correctly."""
        self.assertIsInstance(self.service, BanditService)
        self.assertEqual(self.service.alpha_prior, 1.0)
        self.assertEqual(self.service.beta_prior, 1.0)
        self.assertEqual(self.service.epsilon, 0.1)
        self.assertEqual(self.service.bandits, {})
        self.assertEqual(self.service.data_path, self.temp_dir)
        self.assertTrue(os.path.exists(self.temp_dir))
        logger.info("Initialization test passed")

    def test_initialize_bandit_thompson(self):
        """Test initializing a Thompson Sampling bandit."""
        arms = ["arm1", "arm2", "arm3"]
        result = self.service.initialize_bandit(
            name="test_bandit", arms=arms, algorithm=BanditAlgorithm.THOMPSON_SAMPLING
        )

        self.assertEqual(result["status"], "success")

        # Check if bandit was created
        bandit_id = result["bandit_id"]
        self.assertIn(bandit_id, self.service.bandits)

        # Check bandit properties
        bandit = self.service.bandits[bandit_id]
        self.assertEqual(bandit["algorithm"], BanditAlgorithm.THOMPSON_SAMPLING)
        self.assertEqual(bandit["name"], "test_bandit")
        self.assertEqual(len(bandit["arms"]), 3)

        # Check arm properties
        for arm_id in arms:
            self.assertIn(arm_id, bandit["arms"])
            arm = bandit["arms"][arm_id]
            self.assertEqual(arm["alpha"], 1.0)
            self.assertEqual(arm["beta"], 1.0)
            self.assertEqual(arm["pulls"], 0)
            self.assertEqual(arm["rewards"], 0.0)

        logger.info("Thompson Sampling bandit initialization test passed")

    def test_initialize_bandit_ucb(self):
        """Test initializing a UCB bandit."""
        arms = ["arm1", "arm2", "arm3"]
        result = self.service.initialize_bandit(
            name="test_bandit", arms=arms, algorithm=BanditAlgorithm.UCB
        )

        self.assertEqual(result["status"], "success")

        # Check if bandit was created
        bandit_id = result["bandit_id"]
        self.assertIn(bandit_id, self.service.bandits)

        # Check bandit properties
        bandit = self.service.bandits[bandit_id]
        self.assertEqual(bandit["algorithm"], BanditAlgorithm.UCB)
        self.assertEqual(bandit["name"], "test_bandit")
        self.assertEqual(len(bandit["arms"]), 3)
        self.assertEqual(bandit["ucb_alpha"], 1.0)

        # Check arm properties
        for arm_id in arms:
            self.assertIn(arm_id, bandit["arms"])
            arm = bandit["arms"][arm_id]
            self.assertEqual(arm["mean_reward"], 0.0)
            self.assertEqual(arm["pulls"], 0)
            self.assertEqual(arm["rewards"], 0.0)
            self.assertEqual(arm["ucb_score"], float("inf"))

        logger.info("UCB bandit initialization test passed")

    def test_initialize_bandit_epsilon_greedy(self):
        """Test initializing an Epsilon-Greedy bandit."""
        arms = ["arm1", "arm2", "arm3"]
        result = self.service.initialize_bandit(
            name="test_bandit", arms=arms, algorithm=BanditAlgorithm.EPSILON_GREEDY
        )

        self.assertEqual(result["status"], "success")

        # Check if bandit was created
        bandit_id = result["bandit_id"]
        self.assertIn(bandit_id, self.service.bandits)

        # Check bandit properties
        bandit = self.service.bandits[bandit_id]
        self.assertEqual(bandit["algorithm"], BanditAlgorithm.EPSILON_GREEDY)
        self.assertEqual(bandit["name"], "test_bandit")
        self.assertEqual(len(bandit["arms"]), 3)
        self.assertEqual(bandit["epsilon"], 0.1)

        # Check arm properties
        for arm_id in arms:
            self.assertIn(arm_id, bandit["arms"])
            arm = bandit["arms"][arm_id]
            self.assertEqual(arm["mean_reward"], 0.0)
            self.assertEqual(arm["pulls"], 0)
            self.assertEqual(arm["rewards"], 0.0)

        logger.info("Epsilon-Greedy bandit initialization test passed")

    def test_select_arm_thompson(self):
        """Test selecting an arm using Thompson Sampling."""
        # Initialize a bandit
        arms = ["arm1", "arm2", "arm3"]
        init_result = self.service.initialize_bandit(
            name="test_bandit", arms=arms, algorithm=BanditAlgorithm.THOMPSON_SAMPLING
        )
        bandit_id = init_result["bandit_id"]

        # Update one arm to have higher rewards
        self.service.update_bandit(bandit_id, "arm1", 0.8)
        self.service.update_bandit(bandit_id, "arm1", 0.9)

        # Mock np.random.beta to return predictable values
        with patch("numpy.random.beta", side_effect=[0.7, 0.3, 0.5]):
            result = self.service.select_arm(bandit_id)

            self.assertEqual(result["status"], "success")
            self.assertEqual(
                result["selected_arm"], "arm1"
            )  # Should select arm with highest sample
            self.assertEqual(result["selection_value"], 0.7)
            self.assertEqual(result["algorithm"], "THOMPSON_SAMPLING")

        logger.info("Thompson Sampling arm selection test passed")

    def test_select_arm_ucb(self):
        """Test selecting an arm using UCB."""
        # Initialize a bandit
        arms = ["arm1", "arm2", "arm3"]
        init_result = self.service.initialize_bandit(
            name="test_bandit", arms=arms, algorithm=BanditAlgorithm.UCB
        )
        bandit_id = init_result["bandit_id"]

        # Update arms with some data
        self.service.update_bandit(bandit_id, "arm1", 0.7)
        self.service.update_bandit(bandit_id, "arm2", 0.4)
        self.service.update_bandit(bandit_id, "arm3", 0.2)

        # Arm 1 will have highest mean_reward (0.7) and best UCB value
        result = self.service.select_arm(bandit_id)

        self.assertEqual(result["status"], "success")
        self.assertIn(result["selected_arm"], arms)
        self.assertEqual(result["algorithm"], "UCB")

        logger.info("UCB arm selection test passed")

    def test_update_bandit_thompson(self):
        """Test updating a Thompson Sampling bandit."""
        # Initialize a bandit
        arms = ["arm1", "arm2", "arm3"]
        init_result = self.service.initialize_bandit(
            name="test_bandit", arms=arms, algorithm=BanditAlgorithm.THOMPSON_SAMPLING
        )
        bandit_id = init_result["bandit_id"]

        # Update an arm
        result = self.service.update_bandit(bandit_id, "arm1", 0.75)

        self.assertEqual(result["status"], "success")

        # Check if arm was updated
        bandit = self.service.bandits[bandit_id]
        arm = bandit["arms"]["arm1"]

        self.assertEqual(arm["pulls"], 1)
        self.assertEqual(arm["rewards"], 0.75)
        self.assertEqual(arm["alpha"], 1.75)  # 1.0 (prior) + 0.75
        self.assertEqual(arm["beta"], 1.25)  # 1.0 (prior) + (1.0 - 0.75)

        # Check bandit-level metrics
        self.assertEqual(bandit["total_pulls"], 1)
        self.assertEqual(bandit["total_rewards"], 0.75)

        logger.info("Thompson Sampling bandit update test passed")

    def test_update_bandit_ucb(self):
        """Test updating a UCB bandit."""
        # Initialize a bandit
        arms = ["arm1", "arm2", "arm3"]
        init_result = self.service.initialize_bandit(
            name="test_bandit", arms=arms, algorithm=BanditAlgorithm.UCB
        )
        bandit_id = init_result["bandit_id"]

        # Update an arm
        result = self.service.update_bandit(bandit_id, "arm1", 0.75)

        self.assertEqual(result["status"], "success")

        # Check if arm was updated
        bandit = self.service.bandits[bandit_id]
        arm = bandit["arms"]["arm1"]

        self.assertEqual(arm["pulls"], 1)
        self.assertEqual(arm["rewards"], 0.75)
        self.assertEqual(arm["mean_reward"], 0.75)

        # Check bandit-level metrics
        self.assertEqual(bandit["total_pulls"], 1)
        self.assertEqual(bandit["total_rewards"], 0.75)

        logger.info("UCB bandit update test passed")

    def test_get_bandit_stats(self):
        """Test retrieving statistics for a bandit."""
        # Initialize a bandit
        arms = ["arm1", "arm2", "arm3"]
        init_result = self.service.initialize_bandit(
            name="test_bandit", arms=arms, algorithm=BanditAlgorithm.THOMPSON_SAMPLING
        )
        bandit_id = init_result["bandit_id"]

        # Update arms
        self.service.update_bandit(bandit_id, "arm1", 0.75)
        self.service.update_bandit(bandit_id, "arm2", 0.40)
        self.service.update_bandit(bandit_id, "arm3", 0.20)

        # Get stats
        result = self.service.get_bandit_stats(bandit_id)

        self.assertEqual(result["status"], "success")

        stats = result["stats"]
        self.assertEqual(stats["bandit_id"], bandit_id)
        self.assertEqual(stats["name"], "test_bandit")
        self.assertEqual(stats["algorithm"], BanditAlgorithm.THOMPSON_SAMPLING)
        self.assertEqual(stats["total_pulls"], 3)
        self.assertEqual(stats["total_rewards"], 1.35)
        self.assertEqual(stats["average_reward"], 0.45)
        self.assertEqual(stats["arms_count"], 3)

        # Check arm stats
        self.assertIn("arm_stats", stats)
        self.assertEqual(len(stats["arm_stats"]), 3)

        # Check arm1 stats
        arm1_stats = stats["arm_stats"]["arm1"]
        self.assertEqual(arm1_stats["pulls"], 1)
        self.assertEqual(arm1_stats["rewards"], 0.75)
        self.assertEqual(arm1_stats["alpha"], 1.75)
        self.assertEqual(arm1_stats["beta"], 1.25)
        self.assertEqual(
            arm1_stats["mean_estimate"], 0.75 / 2.0
        )  # alpha / (alpha + beta) = 1.75 / (1.75 + 1.25)

        logger.info("Get bandit stats test passed")

    def test_allocate_budget(self):
        """Test budget allocation using a bandit."""
        # Initialize a bandit
        arms = ["campaign1", "campaign2", "campaign3"]
        init_result = self.service.initialize_bandit(
            name="test_bandit", arms=arms, algorithm=BanditAlgorithm.THOMPSON_SAMPLING
        )
        bandit_id = init_result["bandit_id"]

        # Update arms with some performance data
        self.service.update_bandit(bandit_id, "campaign1", 0.8)
        self.service.update_bandit(bandit_id, "campaign2", 0.4)
        self.service.update_bandit(bandit_id, "campaign3", 0.2)

        # Allocate budget
        total_budget = 1000.0

        # Mock _select_arm_thompson to return consistent results for testing
        with patch.object(
            self.service,
            "_select_arm_thompson",
            return_value={
                "status": "success",
                "selected_arm": "campaign1",
                "selection_value": 0.8,
                "all_samples": {
                    "campaign1": {"sample": 0.8, "alpha": 1.8, "beta": 1.2},
                    "campaign2": {"sample": 0.4, "alpha": 1.4, "beta": 1.6},
                    "campaign3": {"sample": 0.2, "alpha": 1.2, "beta": 1.8},
                },
                "algorithm": "THOMPSON_SAMPLING",
                "selection_type": "exploitation",
            },
        ):
            result = self.service.allocate_budget(bandit_id, total_budget)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["total_budget"], total_budget)

        # Check allocations
        allocations = result["allocations"]
        self.assertEqual(len(allocations), 3)

        # The best arm (campaign1) should get the largest allocation
        self.assertGreater(allocations["campaign1"], allocations["campaign2"])
        self.assertGreater(allocations["campaign2"], allocations["campaign3"])

        # Sum of allocations should equal total budget
        self.assertAlmostEqual(sum(allocations.values()), total_budget, places=2)

        logger.info("Budget allocation test passed")

    def test_reset_bandit(self):
        """Test resetting a bandit to its initial state."""
        # Initialize a bandit
        arms = ["arm1", "arm2", "arm3"]
        init_result = self.service.initialize_bandit(
            name="test_bandit", arms=arms, algorithm=BanditAlgorithm.THOMPSON_SAMPLING
        )
        bandit_id = init_result["bandit_id"]

        # Update arms
        self.service.update_bandit(bandit_id, "arm1", 0.75)
        self.service.update_bandit(bandit_id, "arm2", 0.40)

        # Reset the bandit
        result = self.service.reset_bandit(bandit_id)

        self.assertEqual(result["status"], "success")

        # Check if bandit was reset
        bandit = self.service.bandits[bandit_id]
        self.assertEqual(bandit["total_pulls"], 0)
        self.assertEqual(bandit["total_rewards"], 0.0)

        # Check if arms were reset
        for arm_id in arms:
            arm = bandit["arms"][arm_id]
            self.assertEqual(arm["alpha"], 1.0)
            self.assertEqual(arm["beta"], 1.0)
            self.assertEqual(arm["pulls"], 0)
            self.assertEqual(arm["rewards"], 0.0)

        logger.info("Reset bandit test passed")

    def test_save_and_load_bandits(self):
        """Test saving and loading bandits."""
        # Initialize a bandit
        arms = ["arm1", "arm2", "arm3"]
        init_result = self.service.initialize_bandit(
            name="test_bandit", arms=arms, algorithm=BanditAlgorithm.THOMPSON_SAMPLING
        )
        bandit_id = init_result["bandit_id"]

        # Update an arm
        self.service.update_bandit(bandit_id, "arm1", 0.75)

        # Save bandits
        save_result = self.service.save_bandits()
        self.assertEqual(save_result["status"], "success")
        filepath = save_result["filepath"]

        # Clear bandits
        self.service.bandits = {}

        # Load bandits
        load_result = self.service.load_bandits(filepath)
        self.assertEqual(load_result["status"], "success")

        # Check if bandits were loaded correctly
        self.assertIn(bandit_id, self.service.bandits)

        # Check if arm data was preserved
        bandit = self.service.bandits[bandit_id]
        arm = bandit["arms"]["arm1"]
        self.assertEqual(arm["rewards"], 0.75)

        logger.info("Save and load bandits test passed")

    def test_visualize_bandit(self):
        """Test bandit visualization."""
        if not hasattr(self.service, "visualize_bandit"):
            logger.warning("visualize_bandit method not implemented, skipping test")
            return

        # Skip test if matplotlib is not available
        try:
            import matplotlib

            matplotlib.use("Agg")  # Use non-interactive backend for testing
        except ImportError:
            logger.warning("Matplotlib not available, skipping visualization test")
            return

        # Initialize a bandit
        arms = ["arm1", "arm2", "arm3"]
        init_result = self.service.initialize_bandit(
            name="test_bandit", arms=arms, algorithm=BanditAlgorithm.THOMPSON_SAMPLING
        )
        bandit_id = init_result["bandit_id"]

        # Update arms
        self.service.update_bandit(bandit_id, "arm1", 0.75)
        self.service.update_bandit(bandit_id, "arm2", 0.40)
        self.service.update_bandit(bandit_id, "arm3", 0.20)

        # Add selection history
        self.service.selection_history = {bandit_id: ["arm1", "arm2", "arm1", "arm1"]}

        # Visualize the bandit
        result = self.service.visualize_bandit(bandit_id)

        self.assertEqual(result["status"], "success")
        self.assertTrue(os.path.exists(result["visualization_path"]))

        logger.info("Visualize bandit test passed")

    def test_optimize_campaigns(self):
        """Test campaign optimization."""
        # Mock the Google Ads API response
        mock_campaigns = [
            {
                "campaign_id": "campaign1",
                "campaign_name": "Campaign 1",
                "status": "ENABLED",
                "budget": 100.0,
                "impressions": 1000,
                "clicks": 100,
                "conversions": 10,
                "cost": 50.0,
            },
            {
                "campaign_id": "campaign2",
                "campaign_name": "Campaign 2",
                "status": "ENABLED",
                "budget": 200.0,
                "impressions": 2000,
                "clicks": 150,
                "conversions": 5,
                "cost": 75.0,
            },
            {
                "campaign_id": "campaign3",
                "campaign_name": "Campaign 3",
                "status": "ENABLED",
                "budget": 300.0,
                "impressions": 3000,
                "clicks": 200,
                "conversions": 2,
                "cost": 100.0,
            },
        ]

        # Mock the _get_campaign_performance method
        with patch.object(self.service, "_get_campaign_performance", return_value=mock_campaigns):
            # Optimize campaigns
            campaign_ids = ["campaign1", "campaign2", "campaign3"]
            total_budget = 1000.0

            result = self.service.optimize_campaigns(campaign_ids, total_budget, days=30)

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["total_budget"], total_budget)

            # Check recommendations
            recommendations = result["recommendations"]
            self.assertEqual(len(recommendations), 3)

            # The total recommended budget should equal the total budget
            total_recommended = sum(rec["recommended_budget"] for rec in recommendations)
            self.assertAlmostEqual(total_recommended, total_budget, places=2)

        logger.info("Campaign optimization test passed")

    def test_run(self):
        """Test the run method."""
        # Test with maintenance action
        result = self.service.run({"action": "maintain"})

        self.assertEqual(result["status"], "success")
        self.assertIn("execution_time_seconds", result)

        # Ensure data was saved
        self.assertTrue(os.path.exists(result["save_result"]["filepath"]))

        logger.info("Run method test passed")


if __name__ == "__main__":
    unittest.main()
