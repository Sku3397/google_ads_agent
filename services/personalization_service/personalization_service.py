"""
Personalization Service Implementation

This module provides personalization capabilities for Google Ads campaigns,
optimizing ad delivery based on user characteristics and behavior.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from services.base_service import BaseService


class PersonalizationService(BaseService):
    """
    Service for personalizing ad delivery based on user signals and behavior.

    This service optimizes ad delivery by:
    1. Analyzing user behavior and segmentation
    2. Tailoring ad content, bidding, and targeting to user segments
    3. Tracking and improving personalization effectiveness
    """

    def __init__(
        self,
        ads_api=None,
        optimizer=None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the personalization service.

        Args:
            ads_api: Google Ads API client
            optimizer: AI optimizer instance
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(ads_api, optimizer, config, logger)

        # Configure personalization settings
        self.segment_count = self.config.get("segment_count", 5)
        self.min_observations = self.config.get("min_observations", 100)
        self.update_frequency_days = self.config.get("update_frequency_days", 7)
        self.data_lookback_days = self.config.get("data_lookback_days", 90)

        # Initialize user segments and personalization models
        self.user_segments = {}
        self.personalization_models = {}
        self.segment_performance = {}

        # Load existing segments and models if available
        self._load_state()

        self.logger.info(f"PersonalizationService initialized with {self.segment_count} segments")

    def _load_state(self) -> None:
        """Load previously saved personalization state."""
        try:
            segments = self.load_data("user_segments.json")
            if segments:
                self.user_segments = segments
                self.logger.info(f"Loaded {len(segments)} user segments")

            models = self.load_data("personalization_models.json")
            if models:
                self.personalization_models = models
                self.logger.info(f"Loaded personalization models")

            performance = self.load_data("segment_performance.json")
            if performance:
                self.segment_performance = performance
                self.logger.info(f"Loaded segment performance data")
        except Exception as e:
            self.logger.error(f"Error loading personalization state: {e}")

    def _save_state(self) -> None:
        """Save current personalization state."""
        try:
            self.save_data(self.user_segments, "user_segments.json")
            self.save_data(self.personalization_models, "personalization_models.json")
            self.save_data(self.segment_performance, "segment_performance.json")
            self.logger.info(f"Saved personalization state")
        except Exception as e:
            self.logger.error(f"Error saving personalization state: {e}")

    def create_user_segments(
        self, user_data: pd.DataFrame, features: List[str] = None
    ) -> Dict[str, Any]:
        """
        Create user segments based on behavior and characteristics.

        Args:
            user_data: DataFrame containing user interaction data
            features: List of feature columns to use for segmentation

        Returns:
            Dictionary mapping segment IDs to segment definitions
        """
        start_time = datetime.now()
        self.logger.info(f"Creating user segments from {len(user_data)} records")

        try:
            if len(user_data) < self.min_observations:
                self.logger.warning(
                    f"Insufficient data for segmentation: {len(user_data)} < {self.min_observations}"
                )
                return {}

            # Use default features if none provided
            if not features:
                features = ["device", "location", "time_of_day", "day_of_week", "query_category"]
                self.logger.info(f"Using default features: {features}")

            # Basic data preprocessing
            user_data = self._preprocess_data(user_data, features)

            # Perform clustering or other segmentation method
            # This is a placeholder for actual implementation
            from sklearn.cluster import KMeans

            # Select only the relevant features
            segmentation_data = user_data[features].copy()

            # Handle categorical variables
            for col in segmentation_data.select_dtypes(include=["object"]).columns:
                dummies = pd.get_dummies(segmentation_data[col], prefix=col)
                segmentation_data = pd.concat(
                    [segmentation_data.drop(col, axis=1), dummies], axis=1
                )

            # Fit KMeans
            kmeans = KMeans(n_clusters=self.segment_count, random_state=42)
            segmentation_data = segmentation_data.fillna(0)  # Handle any missing values
            kmeans.fit(segmentation_data)

            # Assign segments
            user_data["segment_id"] = kmeans.labels_

            # Create segment definitions
            segments = {}
            for segment_id in range(self.segment_count):
                segment_data = user_data[user_data["segment_id"] == segment_id]

                # Create a profile for this segment
                profile = {
                    "segment_id": str(segment_id),
                    "size": len(segment_data),
                    "size_percent": round(len(segment_data) / len(user_data) * 100, 2),
                    "features": {},
                    "performance": {
                        "ctr": segment_data["ctr"].mean() if "ctr" in segment_data else None,
                        "conversion_rate": (
                            segment_data["conversion_rate"].mean()
                            if "conversion_rate" in segment_data
                            else None
                        ),
                        "avg_cpc": segment_data["cpc"].mean() if "cpc" in segment_data else None,
                    },
                }

                # Add feature distributions
                for feature in features:
                    if feature in segment_data:
                        if segment_data[feature].dtype == "object":
                            distribution = (
                                segment_data[feature].value_counts(normalize=True).to_dict()
                            )
                            profile["features"][feature] = {
                                "type": "categorical",
                                "distribution": {str(k): v for k, v in distribution.items()},
                            }
                        else:
                            profile["features"][feature] = {
                                "type": "numerical",
                                "mean": segment_data[feature].mean(),
                                "median": segment_data[feature].median(),
                                "min": segment_data[feature].min(),
                                "max": segment_data[feature].max(),
                            }

                segments[str(segment_id)] = profile

            self.user_segments = segments
            self._save_state()

            success = True
            self.logger.info(f"Created {len(segments)} user segments")

        except Exception as e:
            self.logger.error(f"Error creating user segments: {str(e)}")
            segments = {}
            success = False

        self._track_execution(start_time, success)
        return segments

    def _preprocess_data(self, data: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Preprocess user data for segmentation."""
        # Basic preprocessing - handle missing values
        for feature in features:
            if feature in data:
                if data[feature].dtype == "object":
                    data[feature] = data[feature].fillna("unknown")
                else:
                    data[feature] = data[feature].fillna(data[feature].mean())

        return data

    def get_segment_for_user(self, user_data: Dict[str, Any]) -> str:
        """
        Determine the segment for a specific user based on their characteristics.

        Args:
            user_data: Dictionary containing user information

        Returns:
            Segment ID for the user
        """
        # This is a simplified implementation
        # A real implementation would use the clustering model to predict the segment

        # Default to a random segment if we can't determine
        if not self.user_segments:
            return "0"

        # Simple rules-based assignment as fallback
        segment_id = "0"

        try:
            # Try to find the best matching segment based on user characteristics
            best_match_score = -1

            for seg_id, segment in self.user_segments.items():
                match_score = 0

                for feature, feature_info in segment["features"].items():
                    if feature in user_data:
                        if feature_info["type"] == "categorical":
                            # Check if the user's value is common in this segment
                            user_value = str(user_data[feature])
                            if user_value in feature_info["distribution"]:
                                match_score += feature_info["distribution"][user_value]
                        else:
                            # For numerical, check if in range
                            user_value = user_data[feature]
                            if feature_info["min"] <= user_value <= feature_info["max"]:
                                # Higher score if closer to mean
                                distance = abs(user_value - feature_info["mean"])
                                range_size = feature_info["max"] - feature_info["min"]
                                if range_size > 0:
                                    match_score += 1 - (distance / range_size)

                if match_score > best_match_score:
                    best_match_score = match_score
                    segment_id = seg_id

            self.logger.debug(
                f"Assigned user to segment {segment_id} with match score {best_match_score}"
            )

        except Exception as e:
            self.logger.error(f"Error assigning segment to user: {str(e)}")

        return segment_id

    def get_personalized_bid_adjustments(
        self, campaign_id: str, ad_group_id: str, user_segment: str
    ) -> Dict[str, float]:
        """
        Get bid adjustments for a specific user segment.

        Args:
            campaign_id: Google Ads campaign ID
            ad_group_id: Google Ads ad group ID
            user_segment: Segment ID for the user

        Returns:
            Dictionary of bid adjustment factors
        """
        try:
            # Get the segment performance data
            segment_key = f"{campaign_id}_{ad_group_id}_{user_segment}"
            performance = self.segment_performance.get(segment_key, {})

            # Default adjustments
            adjustments = {
                "device": 1.0,
                "location": 1.0,
                "audience": 1.0,
                "time_of_day": 1.0,
                "day_of_week": 1.0,
            }

            # Apply segment-specific adjustments if available
            if performance:
                # Calculate bid adjustment based on segment performance
                # Higher conversion rates or CTR should lead to higher bids
                if "conversion_rate" in performance and performance["conversion_rate"] > 0:
                    base_cr = performance.get(
                        "base_conversion_rate", performance["conversion_rate"]
                    )
                    cr_factor = performance["conversion_rate"] / base_cr
                    adjustments["audience"] = min(
                        max(cr_factor, 0.5), 2.0
                    )  # Limit adjustment range

                # Adjust for device performance
                if "device_performance" in performance:
                    for device, metrics in performance["device_performance"].items():
                        if metrics.get("conversions", 0) > 10:  # Only adjust with sufficient data
                            device_cr = metrics.get("conversion_rate", 0)
                            base_cr = performance.get("conversion_rate", 0.01)
                            if base_cr > 0:
                                device_factor = device_cr / base_cr
                                adjustments["device"] = min(max(device_factor, 0.5), 2.0)

                # Similar adjustments could be made for other dimensions

            self.logger.debug(
                f"Generated bid adjustments for segment {user_segment}: {adjustments}"
            )
            return adjustments

        except Exception as e:
            self.logger.error(f"Error generating bid adjustments: {str(e)}")
            return {
                "device": 1.0,
                "location": 1.0,
                "audience": 1.0,
                "time_of_day": 1.0,
                "day_of_week": 1.0,
            }

    def get_personalized_ads(
        self, ad_group_id: str, user_segment: str, available_ads: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Get personalized ad recommendations for a user segment.

        Args:
            ad_group_id: Google Ads ad group ID
            user_segment: Segment ID for the user
            available_ads: List of available ads

        Returns:
            List of ads ordered by relevance to the segment
        """
        try:
            if not available_ads:
                return []

            # Get the segment definition
            segment = self.user_segments.get(user_segment, {})
            if not segment:
                return available_ads  # Return unmodified if segment not found

            # Get segment performance data for these ads
            ad_performance = {}
            for ad in available_ads:
                ad_id = ad.get("id")
                if not ad_id:
                    continue

                perf_key = f"{ad_group_id}_{ad_id}_{user_segment}"
                performance = self.segment_performance.get(perf_key, {})

                if performance:
                    ad_performance[ad_id] = {
                        "ctr": performance.get("ctr", 0),
                        "conversion_rate": performance.get("conversion_rate", 0),
                        "score": 0,  # Will be calculated
                    }

            # Score and rank the ads based on performance for this segment
            scored_ads = []
            for ad in available_ads:
                ad_id = ad.get("id")
                perf = ad_performance.get(ad_id, {})

                # Calculate a score for this ad
                score = 0
                if perf:
                    ctr_weight = 0.4
                    cr_weight = 0.6
                    score = (perf.get("ctr", 0) * ctr_weight) + (
                        perf.get("conversion_rate", 0) * cr_weight
                    )

                scored_ads.append({"ad": ad, "score": score})

            # Sort by score (highest first)
            scored_ads.sort(key=lambda x: x["score"], reverse=True)

            # Return the sorted ads
            return [item["ad"] for item in scored_ads]

        except Exception as e:
            self.logger.error(f"Error getting personalized ads: {str(e)}")
            return available_ads  # Return unmodified on error

    def update_segment_performance(self, performance_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Update segment performance metrics based on new data.

        Args:
            performance_data: DataFrame with performance metrics by segment

        Returns:
            Updated segment performance dictionary
        """
        start_time = datetime.now()
        self.logger.info(f"Updating segment performance from {len(performance_data)} records")

        try:
            # Group data by segment
            grouped = performance_data.groupby("segment_id")

            updated_segments = {}

            for segment_id, group in grouped:
                # Calculate performance metrics
                metrics = {
                    "impressions": group["impressions"].sum(),
                    "clicks": group["clicks"].sum(),
                    "conversions": group["conversions"].sum(),
                    "cost": group["cost"].sum(),
                    "ctr": (
                        group["clicks"].sum() / group["impressions"].sum()
                        if group["impressions"].sum() > 0
                        else 0
                    ),
                    "conversion_rate": (
                        group["conversions"].sum() / group["clicks"].sum()
                        if group["clicks"].sum() > 0
                        else 0
                    ),
                    "cpa": (
                        group["cost"].sum() / group["conversions"].sum()
                        if group["conversions"].sum() > 0
                        else 0
                    ),
                    "last_updated": datetime.now().isoformat(),
                }

                # If we have device-level data
                if "device" in group.columns:
                    device_grouped = group.groupby("device")
                    device_performance = {}

                    for device, device_group in device_grouped:
                        device_performance[device] = {
                            "impressions": device_group["impressions"].sum(),
                            "clicks": device_group["clicks"].sum(),
                            "conversions": device_group["conversions"].sum(),
                            "ctr": (
                                device_group["clicks"].sum() / device_group["impressions"].sum()
                                if device_group["impressions"].sum() > 0
                                else 0
                            ),
                            "conversion_rate": (
                                device_group["conversions"].sum() / device_group["clicks"].sum()
                                if device_group["clicks"].sum() > 0
                                else 0
                            ),
                        }

                    metrics["device_performance"] = device_performance

                # Update the segments
                segment_key = str(segment_id)

                # If we have campaign and ad group columns, create more specific keys
                if "campaign_id" in group.columns and "ad_group_id" in group.columns:
                    # Group by campaign and ad group
                    for (campaign_id, ad_group_id), campaign_group in group.groupby(
                        ["campaign_id", "ad_group_id"]
                    ):
                        campaign_metrics = {
                            "impressions": campaign_group["impressions"].sum(),
                            "clicks": campaign_group["clicks"].sum(),
                            "conversions": campaign_group["conversions"].sum(),
                            "cost": campaign_group["cost"].sum(),
                            "ctr": (
                                campaign_group["clicks"].sum() / campaign_group["impressions"].sum()
                                if campaign_group["impressions"].sum() > 0
                                else 0
                            ),
                            "conversion_rate": (
                                campaign_group["conversions"].sum() / campaign_group["clicks"].sum()
                                if campaign_group["clicks"].sum() > 0
                                else 0
                            ),
                            "last_updated": datetime.now().isoformat(),
                        }

                        campaign_segment_key = f"{campaign_id}_{ad_group_id}_{segment_id}"
                        updated_segments[campaign_segment_key] = campaign_metrics

                # Also update the overall segment performance
                updated_segments[segment_key] = metrics

            # Update the segment performance dict
            self.segment_performance.update(updated_segments)

            # Save state
            self._save_state()

            self.logger.info(f"Updated performance for {len(updated_segments)} segment entries")
            success = True

        except Exception as e:
            self.logger.error(f"Error updating segment performance: {str(e)}")
            success = False
            updated_segments = {}

        self._track_execution(start_time, success)
        return updated_segments

    def recommend_ad_customizers(self, ad_group_id: str, user_segment: str) -> Dict[str, List[str]]:
        """
        Recommend ad customizers for personalization based on user segment.

        Args:
            ad_group_id: Google Ads ad group ID
            user_segment: Segment ID for the user

        Returns:
            Dictionary of customizer variables and recommended values
        """
        try:
            # Get the segment definition
            segment = self.user_segments.get(user_segment, {})
            if not segment:
                return {}

            # Default customizer recommendations
            customizers = {
                "headline_customizers": [],
                "description_customizers": [],
                "path_customizers": [],
            }

            # Use segment data to generate customizer recommendations
            segment_features = segment.get("features", {})

            # Generate headline customizers based on segment characteristics
            if "device" in segment_features:
                device_dist = segment_features["device"].get("distribution", {})
                primary_device = (
                    max(device_dist.items(), key=lambda x: x[1])[0] if device_dist else None
                )

                if primary_device == "mobile":
                    customizers["headline_customizers"].append("Mobile-Friendly Solution")
                    customizers["description_customizers"].append("Optimized for your smartphone")
                elif primary_device == "tablet":
                    customizers["headline_customizers"].append("Perfect for Tablets")
                    customizers["description_customizers"].append(
                        "Designed for your tablet experience"
                    )
                elif primary_device == "desktop":
                    customizers["headline_customizers"].append("Professional Desktop Tools")
                    customizers["description_customizers"].append(
                        "Full-featured desktop experience"
                    )

            # Time-based customizers
            current_hour = datetime.now().hour
            if 0 <= current_hour < 6:
                customizers["headline_customizers"].append("Night Owl Special")
            elif 6 <= current_hour < 12:
                customizers["headline_customizers"].append("Morning Boost")
            elif 12 <= current_hour < 18:
                customizers["headline_customizers"].append("Afternoon Deal")
            else:
                customizers["headline_customizers"].append("Evening Offer")

            # Location-based customizers (if available)
            if "location" in segment_features:
                location_dist = segment_features["location"].get("distribution", {})
                primary_location = (
                    max(location_dist.items(), key=lambda x: x[1])[0] if location_dist else None
                )

                if primary_location:
                    customizers["headline_customizers"].append(f"Popular in {primary_location}")
                    customizers["description_customizers"].append(
                        f"Trusted by {primary_location} customers"
                    )

            self.logger.debug(f"Generated ad customizers for segment {user_segment}")
            return customizers

        except Exception as e:
            self.logger.error(f"Error generating ad customizers: {str(e)}")
            return {
                "headline_customizers": [],
                "description_customizers": [],
                "path_customizers": [],
            }

    def run_personalization_update(self) -> bool:
        """
        Run a full update of personalization models and segments.

        Returns:
            Boolean indicating success
        """
        start_time = datetime.now()
        self.logger.info("Starting personalization update")

        try:
            # Check if we have the required APIs
            if not self.ads_api:
                self.logger.error("Cannot run personalization update: ads_api not available")
                return False

            # 1. Collect user data from recent campaigns
            lookback_date = datetime.now() - timedelta(days=self.data_lookback_days)

            # This would be implemented to fetch real data from the API
            # For now, we'll create a placeholder implementation

            # 2. Create or update user segments
            try:
                # Placeholder: In reality, we would fetch this data from Google Ads API
                # For demonstration, we'll create a simple mock dataset
                user_data = pd.DataFrame(
                    {
                        "user_id": range(1000),
                        "device": np.random.choice(
                            ["mobile", "desktop", "tablet"], 1000, p=[0.6, 0.3, 0.1]
                        ),
                        "location": np.random.choice(
                            ["urban", "suburban", "rural"], 1000, p=[0.5, 0.3, 0.2]
                        ),
                        "time_of_day": np.random.choice(
                            ["morning", "afternoon", "evening", "night"], 1000
                        ),
                        "day_of_week": np.random.choice(["weekday", "weekend"], 1000, p=[0.7, 0.3]),
                        "query_category": np.random.choice(
                            ["commercial", "informational", "navigational"], 1000
                        ),
                        "ctr": np.random.beta(2, 10, 1000),  # Beta distribution for CTR
                        "conversion_rate": np.random.beta(
                            1, 20, 1000
                        ),  # Beta distribution for conversion rate
                        "cpc": np.random.gamma(2, 0.5, 1000),  # Gamma distribution for CPC
                    }
                )

                segments = self.create_user_segments(user_data)
                self.logger.info(f"Created/updated {len(segments)} user segments")

                # 3. Update performance metrics for segments
                performance_data = pd.DataFrame(
                    {
                        "segment_id": [
                            str(x) for x in np.random.randint(0, self.segment_count, 500)
                        ],
                        "campaign_id": [f"campaign_{x}" for x in np.random.randint(1, 5, 500)],
                        "ad_group_id": [f"adgroup_{x}" for x in np.random.randint(1, 10, 500)],
                        "device": np.random.choice(["mobile", "desktop", "tablet"], 500),
                        "impressions": np.random.randint(10, 1000, 500),
                        "clicks": np.random.randint(0, 100, 500),
                        "conversions": np.random.randint(0, 10, 500),
                        "cost": np.random.uniform(10, 1000, 500),
                    }
                )

                self.update_segment_performance(performance_data)

                self.logger.info(f"Personalization update completed successfully")
                success = True

            except Exception as e:
                self.logger.error(f"Error during personalization update: {str(e)}")
                success = False

        except Exception as e:
            self.logger.error(f"Error running personalization update: {str(e)}")
            success = False

        self._track_execution(start_time, success)
        return success
