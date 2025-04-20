from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class CreativeElement:
    """Data class for creative elements."""

    element_type: str  # headline, description, path, etc.
    content: str
    performance_metrics: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class CreativeTest:
    """Data class for creative test configuration."""

    test_id: str
    ad_group_id: str
    elements: Dict[str, List[CreativeElement]]
    start_date: datetime
    end_date: Optional[datetime]
    confidence_level: float
    status: str
    results: Optional[Dict[str, Any]]


class CreativeService:
    """Service for managing ad creative optimization and testing."""

    def __init__(self, client: GoogleAdsClient, customer_id: str):
        """
        Initialize creative optimization service.

        Args:
            client: Google Ads API client
            customer_id: Google Ads customer ID
        """
        self.client = client
        self.customer_id = customer_id
        self.logger = logging.getLogger(__name__)
        self.vectorizer = TfidfVectorizer(stop_words="english")

    def analyze_creative_elements(self, creative_ids: List[str]) -> Dict[str, Any]:
        """
        Analyze performance of individual creative elements.

        Args:
            creative_ids: List of ad creative IDs to analyze

        Returns:
            Dictionary containing analysis results
        """
        try:
            ga_service = self.client.get_service("GoogleAdsService")

            # Query for ad creative data
            query = f"""
                SELECT
                    ad_group_ad.ad.id,
                    ad_group_ad.ad.expanded_text_ad.headline_part1,
                    ad_group_ad.ad.expanded_text_ad.headline_part2,
                    ad_group_ad.ad.expanded_text_ad.description,
                    metrics.impressions,
                    metrics.clicks,
                    metrics.conversions,
                    metrics.cost_micros
                FROM ad_group_ad
                WHERE ad_group_ad.ad.id IN ({','.join(creative_ids)})
                AND ad_group_ad.status = 'ENABLED'
                AND segments.date DURING LAST_30_DAYS
            """

            response = ga_service.search(customer_id=self.customer_id, query=query)

            # Process creative elements
            headlines = []
            descriptions = []
            performance_data = []

            for row in response:
                ad = row.ad_group_ad.ad
                metrics = row.metrics

                # Extract headlines and descriptions
                headlines.extend(
                    [ad.expanded_text_ad.headline_part1, ad.expanded_text_ad.headline_part2]
                )
                descriptions.append(ad.expanded_text_ad.description)

                # Calculate performance metrics
                impressions = metrics.impressions
                clicks = metrics.clicks
                conversions = metrics.conversions
                cost = metrics.cost_micros / 1000000

                ctr = clicks / impressions if impressions > 0 else 0
                conv_rate = conversions / clicks if clicks > 0 else 0
                cpa = cost / conversions if conversions > 0 else 0

                performance_data.append(
                    {"creative_id": ad.id, "ctr": ctr, "conv_rate": conv_rate, "cpa": cpa}
                )

            # Analyze headline characteristics
            headline_analysis = self._analyze_text_elements(headlines, "headline")
            description_analysis = self._analyze_text_elements(descriptions, "description")

            # Combine analyses
            return {
                "headline_analysis": headline_analysis,
                "description_analysis": description_analysis,
                "performance_data": performance_data,
                "recommendations": self._generate_creative_recommendations(
                    headline_analysis, description_analysis, performance_data
                ),
            }

        except GoogleAdsException as ex:
            error_msg = f"Google Ads API error: {ex.failure.errors[0].message}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def _analyze_text_elements(self, texts: List[str], element_type: str) -> Dict[str, Any]:
        """
        Analyze text elements for patterns and characteristics.

        Args:
            texts: List of text elements to analyze
            element_type: Type of element (headline or description)

        Returns:
            Dictionary containing analysis results
        """
        # Convert texts to feature vectors
        text_vectors = self.vectorizer.fit_transform(texts)

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(text_vectors)

        # Analyze text characteristics
        characteristics = {
            "avg_length": np.mean([len(text.split()) for text in texts]),
            "contains_numbers": np.mean([any(c.isdigit() for c in text) for text in texts]),
            "contains_question": np.mean([any(c == "?" for c in text) for text in texts]),
            "contains_exclamation": np.mean([any(c == "!" for c in text) for text in texts]),
            "similarity_score": np.mean(similarity_matrix),
        }

        # Get most common words/phrases
        feature_names = self.vectorizer.get_feature_names_out()
        top_features = np.argsort(text_vectors.sum(axis=0).A1)[-10:]
        common_terms = [feature_names[i] for i in top_features]

        return {"characteristics": characteristics, "common_terms": common_terms}

    def _generate_creative_recommendations(
        self,
        headline_analysis: Dict[str, Any],
        description_analysis: Dict[str, Any],
        performance_data: List[Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations for creative optimization.

        Args:
            headline_analysis: Analysis results for headlines
            description_analysis: Analysis results for descriptions
            performance_data: Performance metrics for creatives

        Returns:
            List of recommendations
        """
        recommendations = []

        # Analyze headline patterns
        headline_chars = headline_analysis["characteristics"]
        if headline_chars["avg_length"] < 4:
            recommendations.append(
                {
                    "type": "headline",
                    "action": "increase_length",
                    "message": "Consider using longer headlines to provide more information",
                }
            )

        if headline_chars["contains_numbers"] < 0.2:
            recommendations.append(
                {
                    "type": "headline",
                    "action": "add_numbers",
                    "message": "Include specific numbers or statistics in headlines",
                }
            )

        # Analyze description patterns
        desc_chars = description_analysis["characteristics"]
        if desc_chars["contains_exclamation"] < 0.1:
            recommendations.append(
                {
                    "type": "description",
                    "action": "add_call_to_action",
                    "message": "Add clear calls-to-action in descriptions",
                }
            )

        # Performance-based recommendations
        avg_ctr = np.mean([p["ctr"] for p in performance_data])
        if avg_ctr < 0.02:  # 2% CTR threshold
            recommendations.append(
                {
                    "type": "general",
                    "action": "improve_relevance",
                    "message": "Consider testing more compelling headlines to improve CTR",
                }
            )

        return recommendations

    def setup_creative_experiment(
        self, ad_group_id: str, elements: Dict[str, List[str]], confidence_level: float = 0.95
    ) -> CreativeTest:
        """
        Setup a new creative testing experiment.

        Args:
            ad_group_id: ID of the ad group for testing
            elements: Dictionary of creative elements to test
            confidence_level: Statistical confidence level required

        Returns:
            CreativeTest object with test configuration
        """
        test_id = f"test_{ad_group_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Convert raw elements to CreativeElement objects
        processed_elements = {}
        for element_type, contents in elements.items():
            processed_elements[element_type] = [
                CreativeElement(
                    element_type=element_type, content=content, performance_metrics={}, metadata={}
                )
                for content in contents
            ]

        return CreativeTest(
            test_id=test_id,
            ad_group_id=ad_group_id,
            elements=processed_elements,
            start_date=datetime.now(),
            end_date=None,
            confidence_level=confidence_level,
            status="INITIALIZED",
            results=None,
        )

    def monitor_creative_test(self, test: CreativeTest) -> Dict[str, Any]:
        """
        Monitor and analyze results of a creative test.

        Args:
            test: CreativeTest object to monitor

        Returns:
            Dictionary containing test results and recommendations
        """
        try:
            ga_service = self.client.get_service("GoogleAdsService")

            # Query for performance data
            query = f"""
                SELECT
                    ad_group_ad.ad.id,
                    metrics.impressions,
                    metrics.clicks,
                    metrics.conversions,
                    metrics.cost_micros
                FROM ad_group_ad
                WHERE ad_group.id = {test.ad_group_id}
                AND segments.date >= '{test.start_date.strftime('%Y-%m-%d')}'
            """

            response = ga_service.search(customer_id=self.customer_id, query=query)

            # Process results
            results = []
            for row in response:
                metrics = row.metrics
                results.append(
                    {
                        "ad_id": row.ad_group_ad.ad.id,
                        "impressions": metrics.impressions,
                        "clicks": metrics.clicks,
                        "conversions": metrics.conversions,
                        "cost": metrics.cost_micros / 1000000,
                    }
                )

            # Calculate statistical significance
            significant_results = self._calculate_significance(results, test.confidence_level)

            return {
                "test_id": test.test_id,
                "status": "RUNNING",
                "results": results,
                "significant_differences": significant_results,
                "recommendations": self._generate_test_recommendations(
                    results, significant_results
                ),
            }

        except GoogleAdsException as ex:
            error_msg = f"Google Ads API error: {ex.failure.errors[0].message}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def _calculate_significance(
        self, results: List[Dict[str, Any]], confidence_level: float
    ) -> Dict[str, Any]:
        """
        Calculate statistical significance of test results.

        Args:
            results: List of performance results
            confidence_level: Required confidence level

        Returns:
            Dictionary containing significance analysis
        """
        from scipy import stats

        # Group results by creative
        grouped_results = {}
        for result in results:
            ad_id = result["ad_id"]
            if ad_id not in grouped_results:
                grouped_results[ad_id] = []
            grouped_results[ad_id].append(result)

        # Calculate CTR confidence intervals
        significance_results = {}
        for ad_id, ad_results in grouped_results.items():
            total_impressions = sum(r["impressions"] for r in ad_results)
            total_clicks = sum(r["clicks"] for r in ad_results)

            if total_impressions > 0:
                ctr = total_clicks / total_impressions
                # Calculate confidence interval using normal approximation
                std_err = np.sqrt((ctr * (1 - ctr)) / total_impressions)
                z_score = stats.norm.ppf((1 + confidence_level) / 2)
                ci_lower = max(0, ctr - z_score * std_err)
                ci_upper = min(1, ctr + z_score * std_err)

                significance_results[ad_id] = {
                    "ctr": ctr,
                    "confidence_interval": (ci_lower, ci_upper),
                    "sample_size": total_impressions,
                }

        return significance_results

    def _generate_test_recommendations(
        self, results: List[Dict[str, Any]], significance_results: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Generate recommendations based on test results.

        Args:
            results: Raw performance results
            significance_results: Statistical significance analysis

        Returns:
            List of recommendations
        """
        recommendations = []

        # Find best performing creative
        best_ctr = 0
        best_ad_id = None

        for ad_id, stats in significance_results.items():
            if stats["ctr"] > best_ctr:
                best_ctr = stats["ctr"]
                best_ad_id = ad_id

        if best_ad_id:
            recommendations.append(
                {
                    "type": "winner",
                    "message": f"Creative {best_ad_id} is the best performer with {best_ctr:.2%} CTR",
                }
            )

        # Check sample sizes
        for ad_id, stats in significance_results.items():
            if stats["sample_size"] < 1000:
                recommendations.append(
                    {
                        "type": "sample_size",
                        "message": f'Creative {ad_id} needs more data (current sample: {stats["sample_size"]})',
                    }
                )

        return recommendations
