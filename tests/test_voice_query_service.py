"""
Tests for the Voice Query Service.

This module contains tests for the VoiceQueryService, which provides
tools for analyzing and optimizing campaigns for voice search queries.
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import json
import tempfile
import shutil
from datetime import datetime

from services.voice_query_service import VoiceQueryService


class TestVoiceQueryService(unittest.TestCase):
    """Test cases for the Voice Query Service."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create mock dependencies
        self.mock_ads_api = MagicMock()
        self.mock_optimizer = MagicMock()

        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.temp_dir, "data/voice_queries"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "reports/voice_search"), exist_ok=True)

        # Configuration for testing
        self.config = {
            "voice_query.min_query_length": 4,
            "voice_query.confidence_threshold": 0.7,
            "voice_query.custom_patterns_path": os.path.join(
                self.temp_dir, "data/voice_queries/custom_patterns.json"
            ),
        }

        # Create service instance
        self.service = VoiceQueryService(
            ads_api=self.mock_ads_api, optimizer=self.mock_optimizer, config=self.config
        )

        # Replace save and load methods for testing
        self.service.save_data = MagicMock(return_value=True)
        self.service.load_data = MagicMock(return_value=None)

        # Sample voice queries for testing
        self.voice_queries = [
            {
                "query_text": "what are the best running shoes for beginners",
                "impressions": 100,
                "clicks": 10,
                "ctr": 0.1,
                "conversions": 2,
                "cost": 15.0,
                "voice_confidence": 0.85,
                "voice_patterns": ["question_word:what", "voice_pattern:best"],
            },
            {
                "query_text": "where can I find nike shoes on sale near me",
                "impressions": 80,
                "clicks": 12,
                "ctr": 0.15,
                "conversions": 3,
                "cost": 18.0,
                "voice_confidence": 0.9,
                "voice_patterns": [
                    "question_word:where",
                    "voice_pattern:near me",
                    "conversational:I",
                ],
            },
            {
                "query_text": "show me the most comfortable running shoes",
                "impressions": 60,
                "clicks": 8,
                "ctr": 0.13,
                "conversions": 1,
                "cost": 12.0,
                "voice_confidence": 0.75,
                "voice_patterns": ["conversational:show me"],
            },
        ]

        # Sample standard queries for testing
        self.standard_queries = [
            {
                "query_text": "running shoes",
                "impressions": 500,
                "clicks": 50,
                "ctr": 0.1,
                "conversions": 5,
                "cost": 60.0,
                "voice_confidence": 0.2,
                "voice_patterns": [],
            },
            {
                "query_text": "nike sale",
                "impressions": 300,
                "clicks": 30,
                "ctr": 0.1,
                "conversions": 3,
                "cost": 45.0,
                "voice_confidence": 0.1,
                "voice_patterns": [],
            },
        ]

    def tearDown(self):
        """Clean up after each test."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test service initialization."""
        self.assertIsNotNone(self.service)
        self.assertEqual(self.service.ads_api, self.mock_ads_api)
        self.assertEqual(self.service.optimizer, self.mock_optimizer)
        self.assertEqual(self.service.config["voice_query.min_query_length"], 4)
        self.assertEqual(self.service.config["voice_query.confidence_threshold"], 0.7)

    def test_validate_config(self):
        """Test configuration validation."""
        # Test with empty config
        service = VoiceQueryService(config={})
        self.assertIn("voice_query.min_query_length", service.config)
        self.assertIn("voice_query.confidence_threshold", service.config)
        self.assertIn("voice_query.custom_patterns_path", service.config)

        # Test with partial config
        service = VoiceQueryService(config={"voice_query.min_query_length": 5})
        self.assertEqual(service.config["voice_query.min_query_length"], 5)
        self.assertIn("voice_query.confidence_threshold", service.config)

    def test_load_custom_patterns(self):
        """Test loading custom voice patterns."""
        # Create a custom patterns file
        custom_patterns = {
            "industry_specific": ["dental appointment", "legal advice"],
            "local_intents": ["stores in my area", "nearby location"],
            "question_variations": ["tell me about", "I'd like to know"],
        }

        # Write to a temporary file
        patterns_path = os.path.join(self.temp_dir, "custom_patterns.json")
        with open(patterns_path, "w") as f:
            json.dump(custom_patterns, f)

        # Set the config to point to this file
        config = {"voice_query.custom_patterns_path": patterns_path}

        # Create service with this config
        service = VoiceQueryService(config=config)

        # Check that patterns were loaded
        self.assertEqual(
            service.custom_patterns["industry_specific"], custom_patterns["industry_specific"]
        )
        self.assertEqual(service.custom_patterns["local_intents"], custom_patterns["local_intents"])
        self.assertEqual(
            service.custom_patterns["question_variations"], custom_patterns["question_variations"]
        )

    def test_save_custom_patterns(self):
        """Test saving custom voice patterns."""
        # Create custom patterns
        custom_patterns = {
            "industry_specific": ["dental appointment", "legal advice"],
            "local_intents": ["stores in my area", "nearby location"],
            "question_variations": ["tell me about", "I'd like to know"],
        }

        # Call the method
        result = self.service.save_custom_patterns(custom_patterns)

        # Check the result
        self.assertTrue(result)

        # Verify that save_data was called
        self.service.save_data.assert_not_called()  # This should use direct file I/O, not save_data

    def test_calculate_voice_confidence(self):
        """Test calculating voice search confidence score."""
        # Test with a clear voice query
        query = "what are the best running shoes for beginners"
        score = self.service._calculate_voice_confidence(query)
        self.assertGreaterEqual(score, 0.7)  # Should have high confidence

        # Test with a standard query
        query = "running shoes"
        score = self.service._calculate_voice_confidence(query)
        self.assertLessEqual(score, 0.5)  # Should have low confidence

        # Test with conversational query
        query = "I need running shoes for marathon training"
        score = self.service._calculate_voice_confidence(query)
        self.assertGreaterEqual(score, 0.5)  # Should have moderate confidence

        # Test with location-based query
        query = "running shoes near me"
        score = self.service._calculate_voice_confidence(query)
        self.assertGreaterEqual(score, 0.5)  # Should have moderate confidence

    def test_identify_voice_patterns(self):
        """Test identifying voice patterns in a query."""
        # Test with a query containing multiple patterns
        query = "where can I find the best running shoes near me"
        patterns = self.service._identify_voice_patterns(query)

        # Check that expected patterns are found
        self.assertIn("question_word:where", patterns)
        self.assertIn("voice_pattern:best", patterns)
        self.assertIn("voice_pattern:near me", patterns)

        # Test with a standard query
        query = "running shoes"
        patterns = self.service._identify_voice_patterns(query)
        self.assertEqual(len(patterns), 0)  # Should find no patterns

    def test_analyze_search_terms_for_voice_patterns(self):
        """Test analyzing search terms for voice patterns."""
        # Mock _get_simulated_search_queries to return test data
        self.service._get_simulated_search_queries = MagicMock(
            return_value=self.voice_queries + self.standard_queries
        )

        # Call the method
        result = self.service.analyze_search_terms_for_voice_patterns()

        # Check that the result has the expected structure
        self.assertEqual(result["status"], "success")
        self.assertIn("analysis_id", result)
        self.assertIn("total_queries", result)
        self.assertIn("voice_queries", result)
        self.assertIn("standard_queries", result)
        self.assertIn("voice_percentage", result)
        self.assertIn("voice_metrics", result)
        self.assertIn("standard_metrics", result)
        self.assertIn("performance_comparison", result)
        self.assertIn("common_patterns", result)

        # Verify that save_data was called
        self.service.save_data.assert_called_once()

    def test_get_simulated_search_queries(self):
        """Test getting simulated search queries."""
        # Call the method
        queries = self.service._get_simulated_search_queries(days=30)

        # Check that the result is a list of dictionaries
        self.assertIsInstance(queries, list)
        self.assertGreater(len(queries), 0)

        # Check that each query has the expected structure
        for query in queries:
            self.assertIn("query_text", query)
            self.assertIn("impressions", query)
            self.assertIn("clicks", query)
            self.assertIn("ctr", query)
            self.assertIn("conversions", query)
            self.assertIn("cost", query)

    def test_calculate_aggregate_metrics(self):
        """Test calculating aggregate metrics."""
        # Call the method with voice queries
        voice_metrics = self.service._calculate_aggregate_metrics(self.voice_queries)

        # Check that the result has the expected structure
        self.assertIn("impressions", voice_metrics)
        self.assertIn("clicks", voice_metrics)
        self.assertIn("ctr", voice_metrics)
        self.assertIn("conversions", voice_metrics)
        self.assertIn("cost", voice_metrics)
        self.assertIn("cpa", voice_metrics)
        self.assertIn("avg_query_length", voice_metrics)

        # Check calculated values
        self.assertEqual(voice_metrics["impressions"], 240)  # Sum of all impressions
        self.assertEqual(voice_metrics["clicks"], 30)  # Sum of all clicks
        self.assertAlmostEqual(voice_metrics["ctr"], 0.125, places=3)  # 30/240

    def test_compare_metrics(self):
        """Test comparing metrics between voice and standard queries."""
        # Calculate metrics
        voice_metrics = self.service._calculate_aggregate_metrics(self.voice_queries)
        standard_metrics = self.service._calculate_aggregate_metrics(self.standard_queries)

        # Call the method
        comparison = self.service._compare_metrics(voice_metrics, standard_metrics)

        # Check that the result has the expected structure
        self.assertIn("ctr_diff", comparison)
        self.assertIn("cpa_diff", comparison)
        self.assertIn("ctr_interpretation", comparison)
        self.assertIn("cpa_interpretation", comparison)
        self.assertIn("overall", comparison)

    def test_find_common_voice_patterns(self):
        """Test finding common patterns in voice queries."""
        # Call the method
        patterns = self.service._find_common_voice_patterns(self.voice_queries)

        # Check that the result has the expected structure
        self.assertIn("question_word", patterns)
        self.assertIn("voice_pattern", patterns)
        self.assertIn("conversational", patterns)

        # Verify that patterns were found
        self.assertTrue(len(patterns["question_word"]) > 0)

    def test_generate_voice_search_keywords(self):
        """Test generating voice search keywords."""
        # Call the method
        result = self.service.generate_voice_search_keywords(
            seed_keywords=["running shoes", "nike sneakers"],
            question_variations=True,
            conversational_variations=True,
            location_intent=True,
        )

        # Check that the result has the expected structure
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["seed_keywords"], ["running shoes", "nike sneakers"])
        self.assertIn("total_variations", result)
        self.assertIn("grouped_keywords", result)
        self.assertIn("sample_keywords", result)

        # Verify that save_data was called
        self.service.save_data.assert_called_once()

    def test_generate_keyword_variations(self):
        """Test generating keyword variations."""
        # Call the method
        variations = self.service._generate_keyword_variations(
            seed="running shoes",
            question_variations=True,
            conversational_variations=True,
            location_intent=True,
        )

        # Check that variations were generated
        self.assertGreater(len(variations), 0)

        # Check that variations contain the seed keyword
        for variation in variations:
            self.assertIn("running shoes", variation)

        # Check with question variations only
        variations = self.service._generate_keyword_variations(
            seed="running shoes",
            question_variations=True,
            conversational_variations=False,
            location_intent=False,
        )

        # Verify that we have only question variations
        for variation in variations:
            self.assertTrue(
                variation.startswith("what")
                or variation.startswith("where")
                or variation.startswith("how")
                or variation.startswith("which")
                or variation.startswith("who")
                or variation.startswith("when")
                or variation.startswith("why")
            )

    def test_group_keywords_by_pattern(self):
        """Test grouping keywords by pattern."""
        # Create a list of keywords
        keywords = [
            "what are the best running shoes",
            "where can I find running shoes",
            "I need running shoes",
            "show me running shoes",
            "running shoes near me",
            "running shoes in my area",
            "running shoes",
        ]

        # Call the method
        grouped = self.service._group_keywords_by_pattern(keywords)

        # Check that the result has the expected structure
        self.assertIn("question_based", grouped)
        self.assertIn("conversational", grouped)
        self.assertIn("location_based", grouped)
        self.assertIn("other", grouped)

        # Verify grouping
        self.assertEqual(len(grouped["question_based"]), 2)  # "what..." and "where..."
        self.assertEqual(len(grouped["conversational"]), 2)  # "I need..." and "show me..."
        self.assertEqual(len(grouped["location_based"]), 2)  # "...near me" and "...in my area"
        self.assertEqual(len(grouped["other"]), 1)  # "running shoes"

    def test_get_voice_search_recommendations(self):
        """Test getting voice search recommendations."""
        # Call the method with no analysis
        result = self.service.get_voice_search_recommendations()

        # Check that the result has the expected structure
        self.assertEqual(result["status"], "success")
        self.assertIn("total_recommendations", result)
        self.assertIn("recommendations_by_category", result)
        self.assertFalse(result["analysis_based"])

        # Verify that recommendations were generated
        self.assertGreater(result["total_recommendations"], 0)

        # Check recommendations categories
        categories = result["recommendations_by_category"].keys()
        self.assertTrue(
            "keywords" in categories
            or "content" in categories
            or "ads" in categories
            or "structure" in categories
        )

    def test_get_general_voice_recommendations(self):
        """Test getting general voice search recommendations."""
        # Call the method
        recommendations = self.service._get_general_voice_recommendations()

        # Check that recommendations were generated
        self.assertGreater(len(recommendations), 0)

        # Check that each recommendation has the expected structure
        for rec in recommendations:
            self.assertIn("category", rec)
            self.assertIn("title", rec)
            self.assertIn("description", rec)
            self.assertIn("impact", rec)
            self.assertIn("difficulty", rec)

    def test_group_recommendations_by_category(self):
        """Test grouping recommendations by category."""
        # Create recommendations
        recommendations = [
            {"category": "keywords", "title": "Rec 1"},
            {"category": "keywords", "title": "Rec 2"},
            {"category": "content", "title": "Rec 3"},
            {"category": "ads", "title": "Rec 4"},
        ]

        # Call the method
        grouped = self.service._group_recommendations_by_category(recommendations)

        # Check that the result has the expected structure
        self.assertIn("keywords", grouped)
        self.assertIn("content", grouped)
        self.assertIn("ads", grouped)

        # Verify grouping
        self.assertEqual(len(grouped["keywords"]), 2)
        self.assertEqual(len(grouped["content"]), 1)
        self.assertEqual(len(grouped["ads"]), 1)


if __name__ == "__main__":
    unittest.main()
