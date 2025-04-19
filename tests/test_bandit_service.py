import unittest
import logging
from unittest.mock import patch, MagicMock
from services.bandit_service.bandit_service import BanditService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestBanditService(unittest.TestCase):
    def setUp(self):
        """Set up the test environment before each test."""
        # Mock GoogleAdsClient to avoid actual API calls
        self.mock_client = MagicMock()
        self.customer_id = "test_customer_id"
        self.service = BanditService(self.mock_client, self.customer_id)
        # Reset the bandits dictionary for a clean state in each test
        self.service.bandits = {}
        logger.info("Set up test environment for BanditService")

    def test_initialization(self):
        """Test that the BanditService initializes correctly."""
        self.assertIsInstance(self.service, BanditService)
        self.assertEqual(self.service.alpha, 1.0)
        self.assertEqual(self.service.beta, 1.0)
        self.assertEqual(self.service.exploration_rate, 0.1)
        self.assertEqual(self.service.bandits, {})
        logger.info("Initialization test passed")

    def test_initialize_bandit(self):
        """Test initializing a bandit with campaign IDs."""
        campaign_ids = ["campaign1", "campaign2", "campaign3"]
        result = self.service.initialize_bandit(campaign_ids)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("bandit_1", self.service.bandits)
        self.assertEqual(self.service.bandits["bandit_1"]["campaigns"], campaign_ids)
        self.assertEqual(len(self.service.bandits["bandit_1"]["arms"]), 3)
        for cid in campaign_ids:
            self.assertIn(cid, self.service.bandits["bandit_1"]["arms"])
            self.assertEqual(self.service.bandits["bandit_1"]["arms"][cid]["alpha"], 1.0)
            self.assertEqual(self.service.bandits["bandit_1"]["arms"][cid]["beta"], 1.0)
        logger.info("Bandit initialization test passed")

    def test_update_bandit(self):
        """Test updating bandit statistics with rewards."""
        campaign_ids = ["campaign1", "campaign2"]
        self.service.initialize_bandit(campaign_ids)
        
        result = self.service.update_bandit("bandit_1", "campaign1", reward=5.0)
        self.assertEqual(result["status"], "success")
        self.assertEqual(self.service.bandits["bandit_1"]["arms"]["campaign1"]["rewards"], 5.0)
        self.assertEqual(self.service.bandits["bandit_1"]["arms"]["campaign1"]["trials"], 1)
        self.assertEqual(self.service.bandits["bandit_1"]["arms"]["campaign1"]["alpha"], 6.0)  # 1 + 5
        self.assertEqual(self.service.bandits["bandit_1"]["arms"]["campaign1"]["beta"], -4.0)  # 1 + (1-5)
        self.assertEqual(self.service.bandits["bandit_1"]["total_rewards"], 5.0)
        self.assertEqual(self.service.bandits["bandit_1"]["total_trials"], 1)
        logger.info("Bandit update test passed")

    def test_update_bandit_invalid_id(self):
        """Test updating a bandit with an invalid ID."""
        result = self.service.update_bandit("invalid_bandit", "campaign1", reward=5.0)
        self.assertEqual(result["status"], "failed")
        self.assertIn("not found", result["message"])
        logger.info("Invalid bandit ID update test passed")

    def test_select_arm(self):
        """Test selecting an arm using Thompson Sampling."""
        campaign_ids = ["campaign1", "campaign2"]
        self.service.initialize_bandit(campaign_ids)
        # Update one campaign to have higher rewards to bias selection
        self.service.update_bandit("bandit_1", "campaign1", reward=10.0)
        
        # Mock np.random.random to force exploitation (not exploration)
        with patch('numpy.random.random', return_value=0.5):
            result = self.service.select_arm("bandit_1")
            self.assertEqual(result["status"], "success")
            self.assertIn(result["selected_campaign"], campaign_ids)
            self.assertIn("Thompson Sampling", result["rationale"])
        logger.info("Arm selection test passed")

    def test_select_arm_exploration(self):
        """Test selecting an arm with exploration mode."""
        campaign_ids = ["campaign1", "campaign2"]
        self.service.initialize_bandit(campaign_ids)
        
        # Mock np.random.random to force exploration
        with patch('numpy.random.random', return_value=0.05):
            result = self.service.select_arm("bandit_1")
            self.assertEqual(result["status"], "success")
            self.assertIn(result["selected_campaign"], campaign_ids)
            self.assertIn("Random selection for exploration", result["rationale"])
        logger.info("Arm selection exploration test passed")

    def test_allocate_budget(self):
        """Test budget allocation across campaigns."""
        campaign_ids = ["campaign1", "campaign2", "campaign3"]
        self.service.initialize_bandit(campaign_ids)
        total_budget = 1000.0
        
        # Mock select_arm to return a specific campaign
        with patch.object(self.service, 'select_arm', return_value={
            "status": "success",
            "selected_campaign": "campaign1",
            "rationale": "Test selection"
        }):
            result = self.service.allocate_budget("bandit_1", total_budget)
            self.assertEqual(result["status"], "success")
            allocations = result["allocations"]
            self.assertEqual(len(allocations), 3)
            # 90% to selected campaign
            self.assertAlmostEqual(allocations["campaign1"], 900 + (100/3), places=2)
            # 10% split among all for exploration
            self.assertAlmostEqual(allocations["campaign2"], 100/3, places=2)
            self.assertAlmostEqual(allocations["campaign3"], 100/3, places=2)
            # Total should sum to total_budget
            total_allocated = sum(allocations.values())
            self.assertAlmostEqual(total_allocated, total_budget, places=2)
        logger.info("Budget allocation test passed")

    def test_get_bandit_stats(self):
        """Test retrieving statistics for a bandit."""
        campaign_ids = ["campaign1", "campaign2"]
        self.service.initialize_bandit(campaign_ids)
        self.service.update_bandit("bandit_1", "campaign1", reward=5.0)
        
        result = self.service.get_bandit_stats("bandit_1")
        self.assertEqual(result["status"], "success")
        stats = result["stats"]
        self.assertEqual(stats["bandit_id"], "bandit_1")
        self.assertEqual(stats["num_campaigns"], 2)
        self.assertEqual(stats["total_rewards"], 5.0)
        self.assertEqual(stats["total_trials"], 1)
        self.assertIn("arms", stats)
        self.assertEqual(stats["arms"]["campaign1"]["rewards"], 5.0)
        self.assertEqual(stats["arms"]["campaign1"]["estimated_success_rate"], 5.0)
        logger.info("Bandit stats retrieval test passed")

if __name__ == '__main__':
    unittest.main() 