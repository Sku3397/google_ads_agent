"""
Keyword Service for Google Ads Management System

This module provides keyword management, analysis, and optimization services.
It integrates with the Keyword Planner API and analyzes search terms reports to
propose new high-intent keywords.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import logging
import os
import json
from google.ads.googleads.errors import GoogleAdsException

# Correct relative import for BaseService
from ..base_service import BaseService

logger = logging.getLogger(__name__)


class KeywordService(BaseService):
    """
    Keyword Service for managing and optimizing keywords in Google Ads.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the Keyword Service"""
        super().__init__(*args, **kwargs)

        # Default thresholds and settings
        self.min_search_volume = 10  # Minimum monthly search volume
        self.max_keyword_suggestions = 50  # Maximum number of keyword suggestions
        self.min_impression_share = 0.1  # Minimum impression share to target (10%)
        self.min_relevance_score = 0.5  # Minimum relevance score (0-1)

        # Override with config values if available
        if self.config.get("keyword", None):
            keyword_config = self.config["keyword"]
            self.min_search_volume = keyword_config.get("min_search_volume", self.min_search_volume)
            self.max_keyword_suggestions = keyword_config.get(
                "max_keyword_suggestions", self.max_keyword_suggestions
            )
            self.min_impression_share = keyword_config.get(
                "min_impression_share", self.min_impression_share
            )
            self.min_relevance_score = keyword_config.get(
                "min_relevance_score", self.min_relevance_score
            )

        self.logger.info(
            f"KeywordService initialized with settings: "
            f"min_search_volume={self.min_search_volume}, "
            f"max_suggestions={self.max_keyword_suggestions}"
        )

    def discover_new_keywords(self, campaign_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Discover new high-intent keywords for a campaign or account.

        Args:
            campaign_id: Optional campaign ID to target specific campaign

        Returns:
            Dictionary with keyword suggestions and metadata
        """
        start_time = datetime.now()

        try:
            self.logger.info(
                f"Starting keyword discovery for campaign_id={campaign_id or 'all campaigns'}"
            )

            # Step 1: Get current keywords for the campaign or account
            current_keywords = self._get_current_keywords(campaign_id)

            if not current_keywords:
                self.logger.warning("No existing keywords found for analysis")
                self._track_execution(start_time, False)
                return {"status": "failed", "message": "No existing keywords found"}

            # Step 2: Get business info from account settings or config
            business_info = self._get_business_info()

            # Step 3: Analyze search terms report
            search_terms = self._analyze_search_terms_report(campaign_id)

            # Step 4: Generate keyword suggestions using Keyword Planner API
            keyword_planner_suggestions = self._get_keyword_planner_suggestions(
                current_keywords, business_info
            )

            # Step 5: Generate keyword suggestions using AI/Optimizer
            ai_suggestions = self._get_ai_keyword_suggestions(
                current_keywords, business_info, search_terms
            )

            # Step 6: Combine, deduplicate, and rank suggestions
            all_suggestions = self._combine_keyword_suggestions(
                keyword_planner_suggestions, ai_suggestions, current_keywords
            )

            # Step 7: Group keywords into suggested ad groups
            ad_group_suggestions = self._group_keywords_into_ad_groups(all_suggestions)

            # Compile results
            result = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "campaign_id": campaign_id,
                "total_suggestions": len(all_suggestions),
                "current_keyword_count": len(current_keywords),
                "keyword_suggestions": all_suggestions[: self.max_keyword_suggestions],
                "ad_group_suggestions": ad_group_suggestions,
                "search_terms_analysis": {
                    "total_search_terms": len(search_terms),
                    "high_performing_terms": [t for t in search_terms if t.get("ctr", 0) > 0.1][
                        :10
                    ],
                    "converting_terms": [t for t in search_terms if t.get("conversions", 0) > 0][
                        :10
                    ],
                },
            }

            # Save results
            self.save_data(
                result,
                f"keyword_suggestions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "reports/keywords",
            )

            self.logger.info(
                f"Keyword discovery completed successfully with {len(all_suggestions)} suggestions"
            )
            self._track_execution(start_time, True)

            return result

        except Exception as e:
            self.logger.error(f"Error during keyword discovery: {str(e)}")
            self._track_execution(start_time, False)
            return {"status": "failed", "message": str(e)}

    def _get_current_keywords(self, campaign_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get current keywords for the specified campaign or all campaigns.

        Args:
            campaign_id: Optional campaign ID to filter keywords

        Returns:
            List of current keyword dictionaries
        """
        try:
            self.logger.info(
                f"Fetching current keywords for campaign_id={campaign_id or 'all campaigns'}"
            )

            # Use the existing API to get keyword performance data
            keywords = self.ads_api.get_keyword_performance(days_ago=30, campaign_id=campaign_id)

            self.logger.info(f"Fetched {len(keywords)} current keywords")
            return keywords

        except Exception as e:
            self.logger.error(f"Error fetching current keywords: {str(e)}")
            return []

    def _get_business_info(self) -> Dict[str, Any]:
        """
        Get business information from account settings or config.

        Returns:
            Dictionary with business information
        """
        # In a real implementation, this would extract business info
        # from the Google Ads account settings or local configuration

        # For now, use default values or config if available
        business_info = {
            "name": "Example Business",
            "industry": "Unknown",
            "products_services": ["Unknown"],
            "target_audience": "General",
            "geographic_focus": "Global",
            "value_props": ["Quality", "Service"],
        }

        # Override with config values if available
        if self.config.get("business_info", None):
            business_info.update(self.config["business_info"])

        return business_info

    def _analyze_search_terms_report(
        self, campaign_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze search terms report to identify valuable terms.

        Args:
            campaign_id: Optional campaign ID to filter search terms

        Returns:
            List of search term dictionaries with performance metrics
        """
        self.logger.info("Analyzing search terms report")

        # This is a placeholder for the actual Google Ads API call
        # In a real implementation, this would call the Google Ads API
        # to fetch and analyze the search terms report

        # Return empty list as a placeholder
        return []

    def _get_keyword_planner_suggestions(
        self, current_keywords: List[Dict[str, Any]], business_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Get keyword suggestions from Keyword Planner API.

        Args:
            current_keywords: List of current keywords
            business_info: Business information dictionary

        Returns:
            List of keyword suggestion dictionaries
        """
        self.logger.info("Fetching keyword suggestions from Keyword Planner")

        # This is a placeholder for the actual Google Ads API call
        # In a real implementation, this would call the Keyword Planner API
        # to get keyword suggestions

        # Return empty list as a placeholder
        return []

    def _get_ai_keyword_suggestions(
        self,
        current_keywords: List[Dict[str, Any]],
        business_info: Dict[str, Any],
        search_terms: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Get keyword suggestions using AI/Optimizer.

        Args:
            current_keywords: List of current keywords
            business_info: Business information dictionary
            search_terms: List of search term dictionaries

        Returns:
            List of keyword suggestion dictionaries
        """
        try:
            self.logger.info("Generating AI-powered keyword suggestions")

            if not self.optimizer:
                self.logger.warning("No optimizer available for AI keyword suggestions")
                return []

            # Extract current keyword texts
            current_keyword_texts = [kw.get("keyword_text", "") for kw in current_keywords]

            # Format existing keywords for the optimizer
            formatted_keywords = []
            for kw in current_keywords:
                formatted_keywords.append(
                    {
                        "keyword_text": kw.get("keyword_text", ""),
                        "match_type": kw.get("match_type", ""),
                        "status": kw.get("status", ""),
                    }
                )

            # Get suggestions from the optimizer
            suggestions = self.optimizer.get_keyword_suggestions(business_info, formatted_keywords)

            # Extract and process keyword suggestions
            result = []
            if "keyword_suggestions" in suggestions:
                for suggestion in suggestions["keyword_suggestions"]:
                    result.append(
                        {
                            "text": suggestion.get("keyword", ""),
                            "match_type": suggestion.get("match_type", "BROAD"),
                            "suggested_bid": suggestion.get("suggested_bid_range", ""),
                            "estimated_volume": suggestion.get("estimated_search_volume", "MEDIUM"),
                            "relevance_score": 0.8,  # Placeholder score
                            "source": "ai_optimizer",
                            "rationale": suggestion.get("rationale", ""),
                        }
                    )

            self.logger.info(f"Generated {len(result)} AI keyword suggestions")
            return result

        except Exception as e:
            self.logger.error(f"Error generating AI keyword suggestions: {str(e)}")
            return []

    def _combine_keyword_suggestions(
        self,
        keyword_planner_suggestions: List[Dict[str, Any]],
        ai_suggestions: List[Dict[str, Any]],
        current_keywords: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Combine, deduplicate, and rank keyword suggestions.

        Args:
            keyword_planner_suggestions: Suggestions from Keyword Planner
            ai_suggestions: Suggestions from AI/Optimizer
            current_keywords: Current keywords to avoid duplicates

        Returns:
            Combined and ranked list of keyword suggestions
        """
        self.logger.info("Combining and ranking keyword suggestions")

        # Create a set of current keyword texts (lowercase for comparison)
        current_keyword_texts = {kw.get("keyword_text", "").lower() for kw in current_keywords}

        # Combine all suggestions
        all_suggestions = []
        seen_keywords = set()

        # Process Keyword Planner suggestions
        for suggestion in keyword_planner_suggestions:
            keyword_text = suggestion.get("text", "").lower()

            # Skip if already in current keywords or already seen
            if keyword_text in current_keyword_texts or keyword_text in seen_keywords:
                continue

            seen_keywords.add(keyword_text)
            all_suggestions.append(suggestion)

        # Process AI suggestions
        for suggestion in ai_suggestions:
            keyword_text = suggestion.get("text", "").lower()

            # Skip if already in current keywords or already seen
            if keyword_text in current_keyword_texts or keyword_text in seen_keywords:
                continue

            seen_keywords.add(keyword_text)
            all_suggestions.append(suggestion)

        # Sort by relevance score (higher is better)
        all_suggestions.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        return all_suggestions

    def _group_keywords_into_ad_groups(
        self, keyword_suggestions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Group keywords into suggested ad groups.

        Args:
            keyword_suggestions: List of keyword suggestions

        Returns:
            List of ad group suggestion dictionaries
        """
        self.logger.info("Grouping keywords into suggested ad groups")

        # This is a placeholder for a more sophisticated grouping algorithm
        # In a real implementation, this would use clustering or semantic grouping

        # Simple grouping by prefix (first word)
        grouped_keywords = {}

        for suggestion in keyword_suggestions:
            keyword_text = suggestion.get("text", "")
            words = keyword_text.split()

            if not words:
                continue

            # Use first word as a simple grouping key
            prefix = words[0].lower()

            if prefix not in grouped_keywords:
                grouped_keywords[prefix] = []

            grouped_keywords[prefix].append(
                {
                    "text": keyword_text,
                    "match_type": suggestion.get("match_type", "BROAD"),
                    "suggested_bid": suggestion.get("suggested_bid", ""),
                }
            )

        # Convert to ad group suggestions
        ad_group_suggestions = []

        for prefix, keywords in grouped_keywords.items():
            # Only create groups with at least 3 keywords
            if len(keywords) >= 3:
                ad_group_suggestions.append(
                    {"name": f"{prefix.title()} Keywords", "theme": prefix, "keywords": keywords}
                )

        return ad_group_suggestions

    def analyze_keyword_performance(
        self, days: int = 30, campaign_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze keyword performance and identify optimization opportunities.

        Args:
            days: Number of days to analyze
            campaign_id: Optional campaign ID to target specific campaign

        Returns:
            Dictionary with keyword performance analysis and recommendations
        """
        start_time = datetime.now()

        try:
            self.logger.info(f"Starting keyword performance analysis for the last {days} days")

            # Fetch keyword data
            keywords = self.ads_api.get_keyword_performance(days_ago=days, campaign_id=campaign_id)

            if not keywords:
                self.logger.warning("No keyword data available for analysis")
                self._track_execution(start_time, False)
                return {"status": "failed", "message": "No keyword data available"}

            # Create pandas DataFrame for easier analysis
            df = pd.DataFrame(keywords)

            # Calculate additional metrics
            df["ctr"] = df.apply(
                lambda row: (
                    row.get("clicks", 0) / row.get("impressions", 1)
                    if row.get("impressions", 0) > 0
                    else 0
                ),
                axis=1,
            )
            df["conversion_rate"] = df.apply(
                lambda row: (
                    row.get("conversions", 0) / row.get("clicks", 1)
                    if row.get("clicks", 0) > 0
                    else 0
                ),
                axis=1,
            )
            df["cost_per_conversion"] = df.apply(
                lambda row: (
                    row.get("cost", 0) / row.get("conversions", 1)
                    if row.get("conversions", 0) > 0
                    else 0
                ),
                axis=1,
            )
            df["impression_share"] = df["search_impression_share"].apply(
                lambda x: float(x) if isinstance(x, (int, float)) else 0
            )

            # Identify different keyword groups
            high_performing = df[(df["ctr"] > 0.1) & (df["impressions"] > 100)].to_dict("records")
            low_ctr = df[(df["ctr"] < 0.01) & (df["impressions"] > 100)].to_dict("records")
            low_quality_score = df[df["quality_score"] < 5].to_dict("records")
            expensive_keywords = df[(df["cost"] > 50) & (df["conversions"] < 1)].to_dict("records")
            impression_opportunity = df[df["impression_share"] < 0.5].to_dict("records")

            # Generate recommendations
            recommendations = self._generate_keyword_recommendations(
                high_performing,
                low_ctr,
                low_quality_score,
                expensive_keywords,
                impression_opportunity,
            )

            # Compile results
            analysis = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "days_analyzed": days,
                "campaign_id": campaign_id,
                "total_keywords": len(keywords),
                "total_clicks": df["clicks"].sum(),
                "total_impressions": df["impressions"].sum(),
                "total_cost": df["cost"].sum(),
                "total_conversions": df["conversions"].sum(),
                "average_ctr": (
                    df["clicks"].sum() / df["impressions"].sum()
                    if df["impressions"].sum() > 0
                    else 0
                ),
                "average_conversion_rate": (
                    df["conversions"].sum() / df["clicks"].sum() if df["clicks"].sum() > 0 else 0
                ),
                "performance_groups": {
                    "high_performing": len(high_performing),
                    "low_ctr": len(low_ctr),
                    "low_quality_score": len(low_quality_score),
                    "expensive_non_converting": len(expensive_keywords),
                    "impression_opportunity": len(impression_opportunity),
                },
                "recommendations": recommendations,
            }

            # Save results
            self.save_data(
                analysis,
                f"keyword_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "reports/keywords",
            )

            self.logger.info("Keyword performance analysis completed successfully")
            self._track_execution(start_time, True)

            return analysis

        except Exception as e:
            self.logger.error(f"Error during keyword performance analysis: {str(e)}")
            self._track_execution(start_time, False)
            return {"status": "failed", "message": str(e)}

    def _generate_keyword_recommendations(
        self,
        high_performing: List[Dict[str, Any]],
        low_ctr: List[Dict[str, Any]],
        low_quality_score: List[Dict[str, Any]],
        expensive_keywords: List[Dict[str, Any]],
        impression_opportunity: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Generate keyword optimization recommendations.

        Args:
            high_performing: High performing keywords
            low_ctr: Low CTR keywords
            low_quality_score: Low quality score keywords
            expensive_keywords: Expensive non-converting keywords
            impression_opportunity: Keywords with impression share opportunity

        Returns:
            List of keyword recommendation dictionaries
        """
        recommendations = []

        # Recommend bid increases for high-performing keywords
        for keyword in high_performing[:10]:  # Limit to top 10
            recommendations.append(
                {
                    "type": "bid_increase",
                    "keyword_id": keyword.get("ad_group_criterion_id", ""),
                    "keyword_text": keyword.get("keyword_text", ""),
                    "current_bid": keyword.get("current_bid", 0),
                    "recommended_bid": keyword.get("current_bid", 0) * 1.2,  # 20% increase
                    "rationale": "High-performing keyword with good CTR and conversion rate",
                    "impact": "Increase impression share and capture more conversions",
                    "priority": "HIGH",
                }
            )

        # Recommend pausing or reducing bids for low CTR keywords
        for keyword in low_ctr[:10]:  # Limit to top 10
            recommendations.append(
                {
                    "type": "bid_decrease" if keyword.get("conversions", 0) > 0 else "pause",
                    "keyword_id": keyword.get("ad_group_criterion_id", ""),
                    "keyword_text": keyword.get("keyword_text", ""),
                    "current_bid": keyword.get("current_bid", 0),
                    "recommended_bid": (
                        keyword.get("current_bid", 0) * 0.7
                        if keyword.get("conversions", 0) > 0
                        else 0
                    ),
                    "rationale": "Low CTR keyword with poor engagement",
                    "impact": "Reduce wasted spend on non-engaging keywords",
                    "priority": "MEDIUM",
                }
            )

        # Recommend improving ad relevance for low quality score keywords
        for keyword in low_quality_score[:10]:  # Limit to top 10
            recommendations.append(
                {
                    "type": "improve_relevance",
                    "keyword_id": keyword.get("ad_group_criterion_id", ""),
                    "keyword_text": keyword.get("keyword_text", ""),
                    "current_quality_score": keyword.get("quality_score", 0),
                    "rationale": "Low quality score affecting ad rank and CPC",
                    "impact": "Improve quality score to lower CPC and increase ad rank",
                    "priority": "HIGH",
                    "actions": [
                        "Update ad text to include keyword",
                        "Improve landing page relevance",
                        "Consider more specific ad group structure",
                    ],
                }
            )

        # Recommend pausing expensive non-converting keywords
        for keyword in expensive_keywords[:10]:  # Limit to top 10
            recommendations.append(
                {
                    "type": "pause",
                    "keyword_id": keyword.get("ad_group_criterion_id", ""),
                    "keyword_text": keyword.get("keyword_text", ""),
                    "current_cost": keyword.get("cost", 0),
                    "conversions": keyword.get("conversions", 0),
                    "rationale": "Expensive keyword with no conversions",
                    "impact": "Eliminate wasted spend on non-converting keywords",
                    "priority": "HIGH",
                }
            )

        # Recommend bid increases for keywords with impression share opportunity
        for keyword in impression_opportunity[:10]:  # Limit to top 10
            if keyword.get("conversions", 0) > 0:  # Only increase bids for converting keywords
                recommendations.append(
                    {
                        "type": "bid_increase",
                        "keyword_id": keyword.get("ad_group_criterion_id", ""),
                        "keyword_text": keyword.get("keyword_text", ""),
                        "current_bid": keyword.get("current_bid", 0),
                        "recommended_bid": keyword.get("current_bid", 0) * 1.15,  # 15% increase
                        "current_impression_share": keyword.get("impression_share", 0),
                        "rationale": "Converting keyword with low impression share",
                        "impact": "Increase impression share to capture more conversions",
                        "priority": "MEDIUM",
                    }
                )

        return recommendations
