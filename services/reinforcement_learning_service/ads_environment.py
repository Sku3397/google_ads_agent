"""
Google Ads Environment for Reinforcement Learning

This module implements a Gym environment for training RL agents on Google Ads optimization.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, List
import logging
from google.ads.googleads.client import GoogleAdsClient
from datetime import datetime, timedelta


class GoogleAdsEnv(gym.Env):
    """Custom Environment for Google Ads optimization that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        client: GoogleAdsClient,
        campaign_ids: List[str],
        observation_window: int = 7,
        action_type: str = "bidding",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the environment.

        Args:
            client: Google Ads API client
            campaign_ids: List of campaign IDs to optimize
            observation_window: Number of days of historical data to include in state
            action_type: Type of actions to optimize ("bidding" or "keywords")
            config: Additional configuration parameters
        """
        super().__init__()

        self.client = client
        self.campaign_ids = campaign_ids
        self.observation_window = observation_window
        self.action_type = action_type
        self.config = config or {}

        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # Set up action space
        if action_type == "bidding":
            # Continuous action space for bid multipliers
            self.action_space = spaces.Box(
                low=0.5,  # Minimum bid multiplier
                high=2.0,  # Maximum bid multiplier
                shape=(1,),
                dtype=np.float32,
            )
        else:  # keyword optimization
            # Discrete action space for keyword actions
            # 0: Add, 1: Remove, 2: Pause, 3: Enable
            self.action_space = spaces.Discrete(4)

        # Define observation space
        # Features: performance metrics, time features, campaign stats, etc.
        num_features = 20  # Adjust based on actual features used
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32
        )

        # Initialize tracking variables
        self.current_step = 0
        self.max_steps = self.config.get("max_steps", 30)
        self.current_state = None
        self.performance_history = []

        # Feature normalization parameters
        self.feature_means = None
        self.feature_stds = None

        self.logger.info(
            f"Environment initialized with {len(campaign_ids)} campaigns, "
            f"state dimension {num_features}, "
            f"action space type {action_type}"
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset

        Returns:
            Initial state and info dictionary
        """
        super().reset(seed=seed)

        # Reset episode variables
        self.current_step = 0
        self.performance_history = []

        # Get initial state
        self.current_state = self._get_current_state()

        # Get current metrics
        initial_metrics = self._get_performance_metrics()

        info = {"initial_metrics": initial_metrics}

        return self.current_state, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take an action in the environment.

        Args:
            action: Action to take (bid multiplier or keyword action)

        Returns:
            next_state, reward, terminated, truncated, info
        """
        # Apply action to Google Ads
        if self.action_type == "bidding":
            self._apply_bid_multiplier(action)
        else:
            self._apply_keyword_action(action)

        # Wait for sufficient time to observe results
        # In practice, this would be handled by the training loop

        # Get new state and calculate reward
        next_state = self._get_current_state()
        reward = self._calculate_reward(self.current_state, next_state, action)

        # Update current state
        self.current_state = next_state
        self.current_step += 1

        # Check if episode is done
        terminated = False  # No natural termination
        truncated = self.current_step >= self.max_steps

        # Store performance metrics
        self.performance_history.append({"state": next_state, "action": action, "reward": reward})

        # Get current metrics
        current_metrics = self._get_performance_metrics()

        info = {"step": self.current_step, "metrics": current_metrics}

        return next_state, reward, terminated, truncated, info

    def _get_current_state(self) -> np.ndarray:
        """
        Get the current state representation.

        Returns:
            State vector as numpy array
        """
        try:
            # Calculate time window
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.observation_window)

            # Collect metrics from Google Ads API
            metrics = []
            for campaign_id in self.campaign_ids:
                campaign_metrics = self._get_campaign_metrics(campaign_id, start_date, end_date)
                metrics.extend(self._process_campaign_metrics(campaign_metrics))

            # Convert to numpy array
            state = np.array(metrics, dtype=np.float32)

            # Normalize state
            if self.feature_means is None:
                # Initialize normalization parameters
                self.feature_means = state.mean(axis=0)
                self.feature_stds = state.std(axis=0) + 1e-8

            normalized_state = (state - self.feature_means) / self.feature_stds
            return normalized_state

        except Exception as e:
            self.logger.error(f"Error getting state: {str(e)}")
            return np.zeros(len(self.campaign_ids) * 20, dtype=np.float32)

    def _get_campaign_metrics(
        self, campaign_id: str, start_date: datetime, end_date: datetime
    ) -> Dict[str, float]:
        """
        Get campaign performance metrics from Google Ads API.

        Args:
            campaign_id: Campaign ID
            start_date: Start date for metrics
            end_date: End date for metrics

        Returns:
            Dictionary of metrics
        """
        try:
            ga_service = self.client.get_service("GoogleAdsService")

            # Format dates for query
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            # Enhanced query with more metrics
            query = f"""
                SELECT
                    campaign.id,
                    campaign.name,
                    campaign.status,
                    campaign.advertising_channel_type,
                    campaign.bidding_strategy_type,
                    campaign_budget.amount_micros,
                    campaign_budget.period,
                    metrics.impressions,
                    metrics.clicks,
                    metrics.conversions,
                    metrics.conversions_value,
                    metrics.cost_micros,
                    metrics.average_cpc,
                    metrics.ctr,
                    metrics.average_cost_per_conversion,
                    metrics.conversion_rate,
                    metrics.cross_device_conversions,
                    metrics.conversions_from_interactions_rate,
                    metrics.interaction_rate,
                    metrics.top_impression_percentage,
                    metrics.absolute_top_impression_percentage,
                    metrics.search_impression_share,
                    metrics.search_budget_lost_impression_share,
                    metrics.search_rank_lost_impression_share
                FROM campaign
                WHERE 
                    campaign.id = {campaign_id}
                    AND segments.date BETWEEN '{start_str}' AND '{end_str}'
                    AND campaign.status = 'ENABLED'
            """

            response = ga_service.search(customer_id=self.client.customer_id, query=query)

            # Process metrics with error handling
            metrics = {
                "impressions": 0.0,
                "clicks": 0.0,
                "conversions": 0.0,
                "cost_micros": 0.0,
                "conversion_value": 0.0,
                "average_cpc": 0.0,
                "ctr": 0.0,
                "conversion_rate": 0.0,
                "roas": 0.0,
                "budget_amount": 0.0,
                "budget_utilization": 0.0,
                "top_impression_pct": 0.0,
                "abs_top_impression_pct": 0.0,
                "impression_share": 0.0,
                "lost_is_budget": 0.0,
                "lost_is_rank": 0.0,
            }

            for row in response:
                metrics.update(
                    {
                        "impressions": getattr(row.metrics, "impressions", 0),
                        "clicks": getattr(row.metrics, "clicks", 0),
                        "conversions": getattr(row.metrics, "conversions", 0),
                        "cost_micros": getattr(row.metrics, "cost_micros", 0),
                        "conversion_value": getattr(row.metrics, "conversions_value", 0),
                        "average_cpc": getattr(row.metrics, "average_cpc", 0),
                        "ctr": getattr(row.metrics, "ctr", 0),
                        "conversion_rate": getattr(row.metrics, "conversion_rate", 0),
                        "budget_amount": getattr(row.campaign_budget, "amount_micros", 0),
                        "top_impression_pct": getattr(row.metrics, "top_impression_percentage", 0),
                        "abs_top_impression_pct": getattr(
                            row.metrics, "absolute_top_impression_percentage", 0
                        ),
                        "impression_share": getattr(row.metrics, "search_impression_share", 0),
                        "lost_is_budget": getattr(
                            row.metrics, "search_budget_lost_impression_share", 0
                        ),
                        "lost_is_rank": getattr(
                            row.metrics, "search_rank_lost_impression_share", 0
                        ),
                    }
                )

                # Calculate derived metrics
                if metrics["cost_micros"] > 0 and metrics["conversion_value"] > 0:
                    metrics["roas"] = (metrics["conversion_value"] * 1_000_000) / metrics[
                        "cost_micros"
                    ]

                if metrics["budget_amount"] > 0:
                    metrics["budget_utilization"] = (
                        metrics["cost_micros"] / metrics["budget_amount"]
                    )

            return metrics

        except Exception as e:
            logging.error(f"Error fetching campaign metrics: {str(e)}")
            return metrics

    def _process_campaign_metrics(self, metrics: Dict[str, float]) -> List[float]:
        """
        Process raw campaign metrics into state features with advanced feature engineering.

        Args:
            metrics: Raw metrics from API

        Returns:
            List of processed features
        """
        # Basic performance metrics
        basic_features = [
            metrics["impressions"] / 1000,  # Scale down large numbers
            metrics["clicks"] / 100,
            metrics["conversions"],
            metrics["cost_micros"] / 1_000_000,  # Convert to actual currency
            metrics["conversion_value"],
            metrics["average_cpc"] / 1_000_000,
            metrics["ctr"],
            metrics["conversion_rate"],
            metrics["roas"],
        ]

        # Competition and quality metrics
        competition_features = [
            metrics["impression_share"],
            metrics["lost_is_budget"],
            metrics["lost_is_rank"],
            metrics["top_impression_pct"],
            metrics["abs_top_impression_pct"],
        ]

        # Efficiency metrics
        efficiency_features = [
            metrics["conversion_rate"] * metrics["ctr"],  # Conv per impression
            metrics["roas"] * metrics["conversion_rate"],  # Value per conversion
            metrics["budget_utilization"],
            metrics["cost_micros"] / (metrics["impressions"] + 1),  # CPM
            metrics["conversion_value"] / (metrics["clicks"] + 1),  # Value per click
        ]

        # Time-based features
        current_time = datetime.now()
        time_features = [
            current_time.hour / 24,  # Hour of day
            current_time.weekday() / 7,  # Day of week
            np.sin(2 * np.pi * current_time.hour / 24),  # Cyclical encoding
            np.cos(2 * np.pi * current_time.hour / 24),
            np.sin(2 * np.pi * current_time.weekday() / 7),
            np.cos(2 * np.pi * current_time.weekday() / 7),
        ]

        # Combine all features
        return basic_features + competition_features + efficiency_features + time_features

    def _calculate_reward(
        self, state: np.ndarray, next_state: np.ndarray, action: np.ndarray
    ) -> float:
        """
        Calculate reward based on state transition and action.

        Args:
            state: Current state
            next_state: Next state
            action: Action taken

        Returns:
            Reward value
        """
        # Get raw metrics from normalized states
        current_metrics = self._denormalize_state(state)
        next_metrics = self._denormalize_state(next_state)

        # Calculate reward components
        conversion_reward = (
            next_metrics[2] - current_metrics[2]  # Change in conversions
        ) * self.config.get("conversion_weight", 20.0)

        cost_penalty = (next_metrics[3] - current_metrics[3]) * self.config.get(  # Change in cost
            "cost_weight", -0.1
        )

        roas_reward = (next_metrics[8] - current_metrics[8]) * self.config.get(  # Change in ROAS
            "roas_weight", 15.0
        )

        # Combine rewards
        total_reward = conversion_reward + cost_penalty + roas_reward

        # Add action penalty for extreme actions
        if self.action_type == "bidding":
            action_penalty = -abs(action[0] - 1.0) * self.config.get("action_penalty", 0.1)
            total_reward += action_penalty

        return float(total_reward)

    def _denormalize_state(self, normalized_state: np.ndarray) -> np.ndarray:
        """
        Convert normalized state back to original scale.

        Args:
            normalized_state: Normalized state vector

        Returns:
            Denormalized state vector
        """
        return normalized_state * self.feature_stds + self.feature_means

    def _apply_bid_multiplier(self, action: np.ndarray):
        """
        Apply bid multiplier action to Google Ads campaigns.

        Args:
            action: Bid multiplier value
        """
        try:
            # Get current keyword performance to make informed decisions
            ga_service = self.client.get_service("GoogleAdsService")

            for campaign_id in self.campaign_ids:
                # Get keywords for the campaign
                keywords = self._get_campaign_keywords(campaign_id)

                for keyword in keywords:
                    # Apply bid multiplier with safety checks
                    current_bid = keyword["current_bid"]
                    new_bid = current_bid * float(action[0])

                    # Apply bid limits
                    min_bid = current_bid * 0.5
                    max_bid = current_bid * 2.0
                    new_bid = np.clip(new_bid, min_bid, max_bid)

                    # Update the bid
                    success, message = self.client.apply_optimization(
                        optimization_type="bid_adjustment",
                        entity_type="keyword",
                        entity_id=keyword["criterion_id"],
                        changes={"bid_micros": int(new_bid * 1_000_000)},
                    )

                    if not success:
                        logging.warning(
                            f"Failed to update bid for keyword {keyword['keyword_text']}: {message}"
                        )

        except Exception as e:
            logging.error(f"Error applying bid multiplier: {str(e)}")

    def _apply_keyword_action(self, action: int):
        """
        Apply keyword action to Google Ads campaigns.

        Args:
            action: Keyword action (0: Add, 1: Remove, 2: Pause, 3: Enable)
        """
        try:
            for campaign_id in self.campaign_ids:
                keywords = self._get_campaign_keywords(campaign_id)

                if action == 0:  # Add new keyword
                    # Get suggestions based on performance
                    suggestions = self._get_keyword_suggestions(campaign_id)
                    if suggestions:
                        best_suggestion = suggestions[0]  # Take the top suggestion
                        success, message = self.client.apply_optimization(
                            optimization_type="add",
                            entity_type="keyword",
                            entity_id=None,
                            changes={
                                "campaign_id": campaign_id,
                                "keyword_text": best_suggestion["text"],
                                "match_type": best_suggestion["match_type"],
                                "bid_micros": best_suggestion["suggested_bid_micros"],
                            },
                        )

                elif action in [1, 2, 3]:  # Remove, Pause, Enable
                    # Sort keywords by performance
                    keywords.sort(key=lambda k: k["performance_score"], reverse=True)

                    if keywords:
                        target_keyword = keywords[-1] if action == 1 else keywords[0]
                        status = (
                            "REMOVED" if action == 1 else "PAUSED" if action == 2 else "ENABLED"
                        )

                        success, message = self.client.apply_optimization(
                            optimization_type="status_change",
                            entity_type="keyword",
                            entity_id=target_keyword["criterion_id"],
                            changes={"status": status},
                        )

                        if not success:
                            logging.warning(f"Failed to {status.lower()} keyword: {message}")

        except Exception as e:
            logging.error(f"Error applying keyword action: {str(e)}")

    def _get_campaign_keywords(self, campaign_id: str) -> List[Dict[str, Any]]:
        """Helper method to get campaign keywords with performance metrics."""
        try:
            keywords = self.client.get_keyword_performance(
                days_ago=self.observation_window, campaign_id=campaign_id
            )

            # Add performance score for decision making
            for keyword in keywords:
                keyword["performance_score"] = self._calculate_keyword_score(keyword)

            return keywords
        except Exception as e:
            logging.error(f"Error getting campaign keywords: {str(e)}")
            return []

    def _calculate_keyword_score(self, keyword: Dict[str, Any]) -> float:
        """Calculate a performance score for a keyword."""
        try:
            conv_value = keyword.get("conversions", 0) * keyword.get("conversion_value", 0)
            cost = keyword.get("cost", 0)
            clicks = keyword.get("clicks", 0)
            impressions = keyword.get("impressions", 0)

            # Combine multiple metrics into a single score
            score_components = [
                conv_value / (cost + 1),  # ROAS
                keyword.get("conversion_rate", 0),
                keyword.get("ctr", 0),
                clicks / (impressions + 1),
            ]

            return np.mean([x for x in score_components if not np.isnan(x)])
        except Exception as e:
            logging.error(f"Error calculating keyword score: {str(e)}")
            return 0.0

    def _get_keyword_suggestions(self, campaign_id: str) -> List[Dict[str, Any]]:
        """Get keyword suggestions based on performance data."""
        # This would integrate with Google Ads API's keyword planner
        # For now, return an empty list as placeholder
        return []

    def _get_performance_metrics(self) -> Dict[str, float]:
        """
        Get current performance metrics for monitoring.

        Returns:
            Dictionary of current performance metrics
        """
        if not self.current_state is None:
            metrics = self._denormalize_state(self.current_state)
            return {
                "impressions": metrics[0],
                "clicks": metrics[1],
                "conversions": metrics[2],
                "cost": metrics[3],
                "conversion_value": metrics[4],
                "average_cpc": metrics[5],
                "ctr": metrics[6],
                "conversion_rate": metrics[7],
                "roas": metrics[8],
            }
        return {}

    def render(self, mode="human"):
        """
        Render the environment.

        For Google Ads, this could display current performance metrics.
        """
        if mode == "human":
            metrics = self._get_performance_metrics()
            print("\nCurrent Performance Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value}")
