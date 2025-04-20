"""Tests for the GenerativeContentService."""

import unittest
from unittest.mock import Mock, patch, AsyncMock
import pytest
import pandas as pd
from services.generative_content_service import GenerativeContentService


class TestGenerativeContentService(unittest.TestCase):
    """Test cases for the GenerativeContentService."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_ads_client = Mock()
        self.mock_llm_client = Mock()
        self.service = GenerativeContentService(
            ads_client=self.mock_ads_client,
            llm_client=self.mock_llm_client,
            config={"temperature": 0.7, "max_tokens": 100, "model": "gpt-4"},
        )

        # Test data
        self.test_campaign = {
            "id": "12345678",
            "name": "Test Campaign",
            "status": "ENABLED",
            "target_audience": "Technology enthusiasts",
            "product": "AI Software",
            "industry": "Technology",
        }

        self.test_keywords = [
            "ai software",
            "machine learning tools",
            "artificial intelligence platform",
        ]

    def test_generate_ad_headlines(self):
        """Test generating ad headlines."""
        self.mock_llm_client.generate.return_value = [
            "Revolutionary AI Software",
            "Transform Your Business with AI",
            "Smart AI Solutions for You",
        ]

        headlines = self.service.generate_ad_headlines(campaign=self.test_campaign, num_variants=3)

        self.assertEqual(len(headlines), 3)
        self.mock_llm_client.generate.assert_called_once()
        for headline in headlines:
            self.assertLessEqual(len(headline), 30)  # Google Ads headline limit

    def test_generate_ad_descriptions(self):
        """Test generating ad descriptions."""
        self.mock_llm_client.generate.return_value = [
            "Boost productivity with our advanced AI software. Try it now!",
            "Streamline operations with intelligent automation. Start today!",
            "Leverage cutting-edge AI technology for better results.",
        ]

        descriptions = self.service.generate_ad_descriptions(
            campaign=self.test_campaign, num_variants=3
        )

        self.assertEqual(len(descriptions), 3)
        self.mock_llm_client.generate.assert_called_once()
        for desc in descriptions:
            self.assertLessEqual(len(desc), 90)  # Google Ads description limit

    def test_generate_keyword_variations(self):
        """Test generating keyword variations."""
        self.mock_llm_client.generate.return_value = [
            "best ai software",
            "top machine learning tools",
            "enterprise ai platform",
        ]

        variations = self.service.generate_keyword_variations(
            keywords=self.test_keywords, num_variations=3
        )

        self.assertEqual(len(variations), 3)
        self.mock_llm_client.generate.assert_called_once()
        for var in variations:
            self.assertIsInstance(var, str)

    def test_generate_responsive_search_ad(self):
        """Test generating responsive search ad content."""
        self.mock_llm_client.generate.return_value = {
            "headlines": [
                "Revolutionary AI Software",
                "Transform Your Business with AI",
                "Smart AI Solutions for You",
            ],
            "descriptions": [
                "Boost productivity with our advanced AI software. Try it now!",
                "Streamline operations with intelligent automation. Start today!",
            ],
        }

        ad_content = self.service.generate_responsive_search_ad(campaign=self.test_campaign)

        self.assertIn("headlines", ad_content)
        self.assertIn("descriptions", ad_content)
        self.assertEqual(len(ad_content["headlines"]), 3)
        self.assertEqual(len(ad_content["descriptions"]), 2)

    @patch("services.generative_content_service.openai")
    def test_generate_ad_image(self, mock_openai):
        """Test generating ad image."""
        mock_openai.Image.create.return_value = {"data": [{"url": "https://example.com/image.jpg"}]}

        image_url = self.service.generate_ad_image(
            prompt="Professional AI software interface", size="1200x628"
        )

        self.assertIsInstance(image_url, str)
        self.assertTrue(image_url.startswith("https://"))
        mock_openai.Image.create.assert_called_once()

    def test_optimize_ad_copy(self):
        """Test optimizing ad copy based on performance data."""
        performance_data = pd.DataFrame(
            {
                "headline": ["Old Headline 1", "Old Headline 2"],
                "clicks": [100, 150],
                "impressions": [1000, 1200],
                "conversions": [10, 15],
            }
        )

        self.mock_llm_client.generate.return_value = [
            "New Optimized Headline 1",
            "New Optimized Headline 2",
        ]

        optimized_headlines = self.service.optimize_ad_copy(
            performance_data=performance_data, content_type="headline"
        )

        self.assertEqual(len(optimized_headlines), 2)
        self.mock_llm_client.generate.assert_called_once()

    def test_generate_landing_page_content(self):
        """Test generating landing page content."""
        self.mock_llm_client.generate.return_value = {
            "headline": "Transform Your Business with AI",
            "subheadline": "Powerful Machine Learning Tools",
            "body": "Detailed product description...",
            "cta": "Start Free Trial",
        }

        content = self.service.generate_landing_page_content(campaign=self.test_campaign)

        self.assertIn("headline", content)
        self.assertIn("subheadline", content)
        self.assertIn("body", content)
        self.assertIn("cta", content)

    def test_error_handling(self):
        """Test error handling in content generation."""
        # Test with invalid campaign data
        with self.assertRaises(ValueError):
            self.service.generate_ad_headlines(campaign=None, num_variants=3)

        # Test with API error
        self.mock_llm_client.generate.side_effect = Exception("API Error")
        with self.assertRaises(Exception):
            self.service.generate_ad_headlines(campaign=self.test_campaign, num_variants=3)

    @pytest.mark.asyncio
    async def test_async_content_generation(self):
        """Test asynchronous content generation."""
        self.service.llm_client = AsyncMock()
        self.service.llm_client.generate_async.return_value = [
            "Async Headline 1",
            "Async Headline 2",
        ]

        headlines = await self.service.generate_ad_headlines_async(
            campaign=self.test_campaign, num_variants=2
        )

        self.assertEqual(len(headlines), 2)
        self.service.llm_client.generate_async.assert_called_once()

    def test_content_validation(self):
        """Test content validation methods."""
        # Test headline length validation
        long_headline = "This is a very long headline that exceeds the Google Ads character limit"
        with self.assertRaises(ValueError):
            self.service.validate_headline(long_headline)

        # Test description length validation
        long_description = (
            "This is a very long description that exceeds the Google Ads character limit " * 3
        )
        with self.assertRaises(ValueError):
            self.service.validate_description(long_description)


if __name__ == "__main__":
    unittest.main()
