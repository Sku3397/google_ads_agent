"""Tests for the QualityScoreService."""

import unittest
from datetime import datetime
from typing import List, Dict, Any

from services.quality_score_service.quality_score_service import QualityScoreService


class TestQualityScoreService(unittest.TestCase):
    """Test cases for QualityScoreService."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.service = QualityScoreService()
        self.test_keywords: List[Dict[str, Any]] = [
            {"keyword_id": "1", "keyword_text": "test keyword 1", "quality_score": 7},
            {"keyword_id": "2", "keyword_text": "test keyword 2", "quality_score": 4},
        ]

    def test_analyze_quality_scores(self) -> None:
        """Test analyzing quality scores."""
        result = self.service.analyze_quality_scores(
            keywords=self.test_keywords,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
        )

        self.assertIsInstance(result, dict)
        self.assertEqual(result["total_keywords"], len(self.test_keywords))
        self.assertIn("average_quality_score", result)
        self.assertIn("recommendations", result)

    def test_get_improvement_recommendations(self) -> None:
        """Test getting improvement recommendations."""
        keyword_id = "1"
        current_score = 7

        recommendations = self.service.get_improvement_recommendations(
            keyword_id=keyword_id, current_score=current_score
        )

        self.assertIsInstance(recommendations, list)
        self.assertTrue(len(recommendations) > 0)

        first_rec = recommendations[0]
        self.assertEqual(first_rec["keyword_id"], keyword_id)
        self.assertEqual(first_rec["current_score"], current_score)
        self.assertIn("recommendation", first_rec)
        self.assertIn("expected_impact", first_rec)
        self.assertIn("priority", first_rec)


if __name__ == "__main__":
    unittest.main()
