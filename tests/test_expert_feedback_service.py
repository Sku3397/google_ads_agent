"""
Unit tests for the Expert Feedback Service.
"""

import unittest
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch
import json
import datetime

from services.expert_feedback_service import ExpertFeedbackService
from services.base_service import BaseService


class TestExpertFeedbackService(unittest.TestCase):
    """Test cases for the Expert Feedback Service."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for data
        self.temp_dir = tempfile.mkdtemp()
        self.feedback_dir = os.path.join(self.temp_dir, "data", "expert_feedback")
        os.makedirs(self.feedback_dir, exist_ok=True)

        # Mock dependencies
        self.mock_ads_api = MagicMock()
        self.mock_optimizer = MagicMock()

        # Configure mocks
        self.mock_ads_api.apply_bid_adjustments.return_value = {
            "status": "success",
            "message": "Bid adjustments applied",
        }
        self.mock_ads_api.add_keywords.return_value = {
            "status": "success",
            "message": "Keywords added",
        }

        # Create test experts
        self.experts = [
            {
                "id": "expert1",
                "name": "Expert One",
                "email": "expert1@example.com",
                "expertise": ["bid_adjustments", "keywords"],
                "role": "reviewer",
            },
            {
                "id": "expert2",
                "name": "Expert Two",
                "email": "expert2@example.com",
                "expertise": ["negative_keywords", "budget"],
                "role": "reviewer",
            },
        ]

        # Create config
        self.config = {
            "expert_feedback": {
                "approval_required": True,
                "auto_apply_approved": True,
                "feedback_retention_days": 90,
                "experts": self.experts,
                "notifications": {"enabled": False},
            }
        }

        # Create the service with temporary data directory
        with patch("os.path.join", side_effect=self._mock_path_join):
            self.service = ExpertFeedbackService(
                ads_api=self.mock_ads_api, optimizer=self.mock_optimizer, config=self.config
            )

    def _mock_path_join(self, *args):
        """Mock path join to use the temporary directory."""
        if args[0] == "data" and args[1] == "expert_feedback":
            return self.feedback_dir
        return os.path.join(*args)

    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test service initialization."""
        self.assertIsInstance(self.service, BaseService)
        self.assertEqual(self.service.feedback_dir, self.feedback_dir)
        self.assertEqual(self.service.experts, self.experts)
        self.assertEqual(self.service.approval_required, True)

    def test_submit_for_review(self):
        """Test submitting recommendations for review."""
        # Create test recommendations
        test_recommendations = [
            {"keyword_id": "123", "keyword_text": "test keyword", "recommended_bid": 1.50}
        ]

        # Submit recommendations
        result = self.service.submit_for_review(
            recommendation_type="bid_adjustments",
            recommendations=test_recommendations,
            priority="high",
        )

        # Verify result
        self.assertEqual(result["status"], "success")
        self.assertIn("submission_id", result)

        # Verify recommendations were stored
        submission_id = result["submission_id"]
        self.assertIn(submission_id, self.service.pending_recommendations)

        submission = self.service.pending_recommendations[submission_id]
        self.assertEqual(submission["recommendation_type"], "bid_adjustments")
        self.assertEqual(submission["recommendations"], test_recommendations)
        self.assertEqual(submission["priority"], "high")
        self.assertEqual(submission["status"], "pending")

    def test_auto_approve(self):
        """Test auto-approval when not requiring approval."""
        # Set auto-approve
        self.service.approval_required = False

        # Create test recommendations
        test_recommendations = [
            {"keyword_id": "123", "keyword_text": "test keyword", "recommended_bid": 1.50}
        ]

        # Submit recommendations
        result = self.service.submit_for_review(
            recommendation_type="bid_adjustments", recommendations=test_recommendations
        )

        # Verify auto-approval
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["auto_approved"])

        # Verify status in pending recommendations
        submission_id = result["submission_id"]
        submission = self.service.pending_recommendations[submission_id]
        self.assertEqual(submission["status"], "approved")
        self.assertEqual(submission["reviewed_by"], "auto")

    def test_get_pending_reviews(self):
        """Test getting pending reviews."""
        # Create some test submissions
        test_recommendations = [{"id": "1"}]

        # Submit multiple recommendations
        self.service.submit_for_review(
            recommendation_type="bid_adjustments",
            recommendations=test_recommendations,
            priority="high",
        )

        self.service.submit_for_review(
            recommendation_type="keywords", recommendations=test_recommendations, priority="normal"
        )

        # Get pending reviews
        result = self.service.get_pending_reviews()

        # Verify result
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["pending_reviews"]), 2)

        # Filter by recommendation type
        result = self.service.get_pending_reviews(recommendation_type="bid_adjustments")
        self.assertEqual(len(result["pending_reviews"]), 1)
        self.assertEqual(result["pending_reviews"][0]["recommendation_type"], "bid_adjustments")

        # Filter by priority
        result = self.service.get_pending_reviews(priority="high")
        self.assertEqual(len(result["pending_reviews"]), 1)
        self.assertEqual(result["pending_reviews"][0]["priority"], "high")

    def test_approve_recommendations(self):
        """Test approving recommendations."""
        # Create a test submission
        test_recommendations = [{"id": "1", "recommended_bid": 1.5}]

        submit_result = self.service.submit_for_review(
            recommendation_type="bid_adjustments", recommendations=test_recommendations
        )

        submission_id = submit_result["submission_id"]

        # Approve the submission
        approve_result = self.service.approve_recommendations(
            submission_id=submission_id, expert_id="expert1", feedback={"comment": "Looks good"}
        )

        # Verify result
        self.assertEqual(approve_result["status"], "success")

        # Verify submission was updated
        submission = self.service.pending_recommendations[submission_id]
        self.assertEqual(submission["status"], "approved")
        self.assertEqual(submission["reviewed_by"], "expert1")
        self.assertEqual(submission["feedback"], {"comment": "Looks good"})

        # Verify the API method was called to apply recommendations
        self.mock_ads_api.apply_bid_adjustments.assert_called_once()

    def test_reject_recommendations(self):
        """Test rejecting recommendations."""
        # Create a test submission
        test_recommendations = [{"id": "1"}]

        submit_result = self.service.submit_for_review(
            recommendation_type="keywords", recommendations=test_recommendations
        )

        submission_id = submit_result["submission_id"]

        # Reject the submission
        reject_result = self.service.reject_recommendations(
            submission_id=submission_id,
            expert_id="expert1",
            feedback={"comment": "Not relevant keywords"},
        )

        # Verify result
        self.assertEqual(reject_result["status"], "success")

        # Verify submission was updated
        submission = self.service.pending_recommendations[submission_id]
        self.assertEqual(submission["status"], "rejected")
        self.assertEqual(submission["reviewed_by"], "expert1")
        self.assertEqual(submission["feedback"]["comment"], "Not relevant keywords")

        # Verify API method was not called (no application for rejected recommendations)
        self.mock_ads_api.add_keywords.assert_not_called()

    def test_modify_recommendations(self):
        """Test modifying recommendations."""
        # Create a test submission
        original_recommendations = [{"id": "1", "keyword_text": "test", "recommended_bid": 1.0}]

        submit_result = self.service.submit_for_review(
            recommendation_type="bid_adjustments", recommendations=original_recommendations
        )

        submission_id = submit_result["submission_id"]

        # Create modified recommendations
        modified_recommendations = [{"id": "1", "keyword_text": "test", "recommended_bid": 1.5}]

        # Modify the submission
        modify_result = self.service.modify_recommendations(
            submission_id=submission_id,
            expert_id="expert1",
            modified_recommendations=modified_recommendations,
            feedback={"comment": "Increased bid for better position"},
        )

        # Verify result
        self.assertEqual(modify_result["status"], "success")

        # Verify submission was updated
        submission = self.service.pending_recommendations[submission_id]
        self.assertEqual(submission["status"], "modified")
        self.assertEqual(submission["reviewed_by"], "expert1")
        self.assertEqual(submission["original_recommendations"], original_recommendations)
        self.assertEqual(submission["recommendations"], modified_recommendations)

        # Verify the API method was called with modified recommendations
        self.mock_ads_api.apply_bid_adjustments.assert_called_once()

    def test_register_expert(self):
        """Test registering a new expert."""
        # Register a new expert
        result = self.service.register_expert(
            expert_id="expert3",
            name="Expert Three",
            email="expert3@example.com",
            expertise=["budget", "creative"],
        )

        # Verify result
        self.assertEqual(result["status"], "success")

        # Verify expert was added
        self.assertEqual(len(self.service.experts), 3)

        # Find the new expert
        new_expert = next((e for e in self.service.experts if e["id"] == "expert3"), None)
        self.assertIsNotNone(new_expert)
        self.assertEqual(new_expert["name"], "Expert Three")
        self.assertEqual(new_expert["expertise"], ["budget", "creative"])

    def test_get_experts(self):
        """Test retrieving experts."""
        # Get all experts
        result = self.service.get_experts()

        # Verify result
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["experts"]), 2)

        # Filter by expertise
        result = self.service.get_experts(expertise="keywords")
        self.assertEqual(len(result["experts"]), 1)
        self.assertEqual(result["experts"][0]["id"], "expert1")

        result = self.service.get_experts(expertise="budget")
        self.assertEqual(len(result["experts"]), 1)
        self.assertEqual(result["experts"][0]["id"], "expert2")


if __name__ == "__main__":
    unittest.main()
