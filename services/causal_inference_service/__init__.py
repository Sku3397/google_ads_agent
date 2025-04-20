"""
CausalInferenceService for measuring true causal impact of Google Ads optimizations.

This service uses causal inference techniques to:
1. Measure true uplift from bid changes
2. Isolate campaign performance from external factors
3. Perform counterfactual analysis
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from causalimpact import CausalImpact
from statsmodels.tsa.statespace.structural import UnobservedComponents
from sklearn.preprocessing import StandardScaler

from ..base_service import BaseService

logger = logging.getLogger(__name__)


class CausalInferenceService(BaseService):
    """Service for causal inference and uplift measurement in Google Ads campaigns."""

    def __init__(self, config: Dict[str, any]):
        """
        Initialize the causal inference service.

        Args:
            config: Configuration dictionary containing:
                - lookback_days: Days of historical data to use
                - confidence_level: Confidence level for impact analysis (0-1)
                - control_campaign_ids: List of control campaign IDs
                - seasonality_period: Days for seasonality adjustment (7 for weekly)
        """
        super().__init__(config)
        self.lookback_days = config.get("lookback_days", 90)
        self.confidence_level = config.get("confidence_level", 0.95)
        self.control_campaign_ids = config.get("control_campaign_ids", [])
        self.seasonality_period = config.get("seasonality_period", 7)

        # Initialize storage for synthetic controls
        self.synthetic_controls = {}

        # Get Google Ads API client
        self.ads_client = config.get("ads_client")
        if not self.ads_client:
            raise ValueError("ads_client is required in config")

    def measure_bid_impact(
        self, campaign_id: str, metric: str, intervention_date: datetime, post_period_days: int = 14
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Measure the causal impact of bid changes on a campaign metric.

        Args:
            campaign_id: ID of the campaign that received bid changes
            metric: Metric to analyze ('clicks', 'conversions', 'cost', etc.)
            intervention_date: Date when bid changes were applied
            post_period_days: Days after intervention to analyze

        Returns:
            Dict containing:
                - absolute_effect: Absolute lift in metric
                - relative_effect: Relative lift as percentage
                - confidence_intervals: Upper/lower bounds at configured confidence
                - p_value: Statistical significance
        """
        try:
            # Get pre/post intervention data
            pre_data = self._get_campaign_data(
                campaign_id,
                start_date=intervention_date - timedelta(days=self.lookback_days),
                end_date=intervention_date,
            )

            post_data = self._get_campaign_data(
                campaign_id,
                start_date=intervention_date,
                end_date=intervention_date + timedelta(days=post_period_days),
            )

            # Get control campaign data for same period
            control_data = self._get_control_data(
                self.control_campaign_ids,
                start_date=intervention_date - timedelta(days=self.lookback_days),
                end_date=intervention_date + timedelta(days=post_period_days),
            )

            # Combine into time series
            data = pd.concat([pd.Series(pre_data[metric]), pd.Series(post_data[metric])])

            # Create CausalImpact model
            ci = CausalImpact(
                data,
                pre_period=(0, len(pre_data) - 1),
                post_period=(len(pre_data), len(data) - 1),
                model_args={"nseasons": self.seasonality_period},
            )

            # Extract results
            results = {
                "absolute_effect": float(ci.summary_data["abs_effect"][0]),
                "relative_effect": float(ci.summary_data["rel_effect"][0]),
                "confidence_intervals": {
                    "lower": float(ci.summary_data["lower"][0]),
                    "upper": float(ci.summary_data["upper"][0]),
                },
                "p_value": float(ci.summary_data["p"][0]),
            }

            logger.info(
                f"Measured causal impact for campaign {campaign_id}: "
                f"{results['relative_effect']:.1%} lift in {metric}"
            )

            return results

        except Exception as e:
            logger.error(f"Error measuring bid impact: {str(e)}")
            raise

    def build_synthetic_control(
        self,
        target_campaign_id: str,
        candidate_campaign_ids: List[str],
        metric: str,
        training_days: int = 30,
    ) -> Dict[str, float]:
        """
        Build a synthetic control for counterfactual analysis.

        Args:
            target_campaign_id: Campaign to build control for
            candidate_campaign_ids: Potential control campaign IDs
            metric: Metric to optimize for
            training_days: Days of data to use for fitting

        Returns:
            Dict of campaign_id: weight pairs for synthetic control
        """
        try:
            # Get historical data
            end_date = datetime.now() - timedelta(days=1)
            start_date = end_date - timedelta(days=training_days)

            target_data = self._get_campaign_data(
                target_campaign_id, start_date=start_date, end_date=end_date
            )

            candidate_data = self._get_control_data(
                candidate_campaign_ids, start_date=start_date, end_date=end_date
            )

            # Normalize data
            scaler = StandardScaler()
            target_normalized = scaler.fit_transform(target_data[metric].values.reshape(-1, 1))
            candidate_normalized = scaler.transform(candidate_data[metric].values.reshape(-1, 1))

            # Fit optimal weights using regression
            weights = np.linalg.lstsq(candidate_normalized, target_normalized, rcond=None)[0]

            # Create campaign_id: weight mapping
            control_weights = {
                campaign_id: float(weight)
                for campaign_id, weight in zip(candidate_campaign_ids, weights)
            }

            # Store for future use
            self.synthetic_controls[target_campaign_id] = control_weights

            logger.info(
                f"Built synthetic control for campaign {target_campaign_id} "
                f"using {len(control_weights)} campaigns"
            )

            return control_weights

        except Exception as e:
            logger.error(f"Error building synthetic control: {str(e)}")
            raise

    def _get_campaign_data(
        self, campaign_id: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Get historical campaign performance data.

        Args:
            campaign_id: Campaign ID to get data for
            start_date: Start date for data range
            end_date: End date for data range

        Returns:
            DataFrame with daily campaign metrics
        """
        try:
            # Calculate days ago for API call
            days_ago = (datetime.now().date() - start_date.date()).days

            # Get campaign performance data
            campaign_data = self.ads_client.get_campaign_performance(days_ago=days_ago)

            # Filter for specific campaign and date range
            df = pd.DataFrame(campaign_data)
            df["date"] = pd.date_range(start=start_date, end=end_date, freq="D")
            df = df[df["id"] == campaign_id]

            # Ensure data completeness
            if len(df) == 0:
                raise ValueError(f"No data found for campaign {campaign_id}")

            if len(df) != (end_date - start_date).days + 1:
                raise ValueError(f"Incomplete data for campaign {campaign_id}")

            return df

        except Exception as e:
            logger.error(f"Error getting campaign data: {str(e)}")
            raise

    def _get_control_data(
        self, campaign_ids: List[str], start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Get historical data for control campaigns.

        Args:
            campaign_ids: List of control campaign IDs
            start_date: Start date for data range
            end_date: End date for data range

        Returns:
            DataFrame with daily metrics for control campaigns
        """
        try:
            if not campaign_ids:
                raise ValueError("No control campaign IDs provided")

            # Calculate days ago for API call
            days_ago = (datetime.now().date() - start_date.date()).days

            # Get performance data for all campaigns
            campaign_data = self.ads_client.get_campaign_performance(days_ago=days_ago)

            # Filter for control campaigns and date range
            df = pd.DataFrame(campaign_data)
            df["date"] = pd.date_range(start=start_date, end=end_date, freq="D")
            df = df[df["id"].isin(campaign_ids)]

            # Ensure data completeness
            if len(df) == 0:
                raise ValueError(f"No data found for control campaigns")

            expected_rows = len(campaign_ids) * ((end_date - start_date).days + 1)
            if len(df) != expected_rows:
                raise ValueError(f"Incomplete data for control campaigns")

            # Aggregate across control campaigns
            df_agg = (
                df.groupby("date")
                .agg({"impressions": "sum", "clicks": "sum", "conversions": "sum", "cost": "sum"})
                .reset_index()
            )

            return df_agg

        except Exception as e:
            logger.error(f"Error getting control campaign data: {str(e)}")
            raise
