"""
Trend Forecasting Service for Google Ads Agent

This service provides advanced trend forecasting capabilities that go beyond
basic forecasting to identify emerging trends and seasonal patterns.
"""

import logging
import os
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from fbprophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm

from services.base_service import BaseService


@dataclass
class TrendForecast:
    """Data class representing a trend forecast"""

    keyword: str
    forecast_date: datetime
    forecasted_value: float
    lower_bound: float
    upper_bound: float
    confidence: float
    model_type: str
    features_used: List[str]


class TrendForecastingService(BaseService):
    """
    Service for forecasting trends in Google Ads campaigns.

    This service provides advanced trend forecasting capabilities beyond basic
    forecasting, including:
    - Long-term trend predictions
    - Seasonal pattern identification
    - Emerging trend detection
    - Multi-variate forecasting
    - External trend signal integration
    """

    def __init__(
        self,
        ads_api=None,
        optimizer=None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the Trend Forecasting Service.

        Args:
            ads_api: Google Ads API client instance
            optimizer: AI optimizer instance for generating suggestions
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(ads_api, optimizer, config, logger)

        # Set default forecast horizons
        self.forecast_horizons = {
            "short_term": 7,  # 7 days
            "medium_term": 30,  # 30 days
            "long_term": 90,  # 90 days
        }

        # Set model configurations
        self.model_configs = {
            "prophet": {
                "daily_seasonality": False,
                "weekly_seasonality": True,
                "yearly_seasonality": True,
                "seasonality_mode": "multiplicative",
                "changepoint_prior_scale": 0.05,
            },
            "sarima": {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 7)},  # Weekly seasonality
            "auto_arima": {
                "seasonal": True,
                "m": 7,  # Weekly seasonality
                "suppress_warnings": True,
            },
        }

        # Store forecast history
        self.forecast_history = {}

        # Set paths for storing models and data
        self.model_dir = os.path.join("data", "trend_forecasting", "models")
        self.forecast_dir = os.path.join("data", "trend_forecasting", "forecasts")

        # Create directories if they don't exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.forecast_dir, exist_ok=True)

        self.logger.info("Trend Forecasting Service initialized")

    def forecast_keyword_performance(
        self,
        keyword: str,
        campaign_id: Optional[str] = None,
        horizon: str = "medium_term",
        metric: str = "clicks",
        model_type: str = "prophet",
        include_external_signals: bool = False,
    ) -> TrendForecast:
        """
        Forecast performance for a specific keyword.

        Args:
            keyword: The keyword to forecast
            campaign_id: Optional campaign ID to limit data to
            horizon: Forecast horizon ('short_term', 'medium_term', 'long_term')
            metric: Metric to forecast ('clicks', 'impressions', 'conversions', 'cost')
            model_type: Model to use ('prophet', 'sarima', 'auto_arima', 'ensemble')
            include_external_signals: Whether to include external signals in the forecast

        Returns:
            A TrendForecast object with the forecast results
        """
        start_time = datetime.now()

        try:
            # Get historical data for this keyword
            if campaign_id:
                # Get data for this keyword in this campaign
                days_to_fetch = max(90, self.forecast_horizons.get(horizon, 30) * 3)
                historical_data = self._get_keyword_historical_data(
                    keyword, campaign_id, days_to_fetch
                )
            else:
                # Get data for this keyword across all campaigns
                days_to_fetch = max(90, self.forecast_horizons.get(horizon, 30) * 3)
                historical_data = self._get_keyword_historical_data(keyword, None, days_to_fetch)

            if historical_data.empty:
                raise ValueError(f"No historical data found for keyword '{keyword}'")

            # Prepare data for forecasting
            forecast_df = self._prepare_data_for_forecasting(historical_data, metric)

            # Include external signals if requested
            if include_external_signals:
                forecast_df = self._add_external_signals(forecast_df, keyword)

            # Determine forecast period
            forecast_days = self.forecast_horizons.get(horizon, 30)

            # Choose appropriate model
            if model_type == "ensemble":
                forecast_result = self._ensemble_forecast(forecast_df, forecast_days, metric)
            elif model_type == "prophet":
                forecast_result = self._prophet_forecast(forecast_df, forecast_days)
            elif model_type == "sarima":
                forecast_result = self._sarima_forecast(forecast_df, forecast_days, metric)
            elif model_type == "auto_arima":
                forecast_result = self._auto_arima_forecast(forecast_df, forecast_days, metric)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Process forecast results
            forecast_date = datetime.now() + timedelta(days=forecast_days)
            forecasted_value = forecast_result["forecasted_value"]
            lower_bound = forecast_result["lower_bound"]
            upper_bound = forecast_result["upper_bound"]
            confidence = forecast_result["confidence"]
            features_used = forecast_result.get("features_used", [metric])

            # Create and return forecast object
            forecast = TrendForecast(
                keyword=keyword,
                forecast_date=forecast_date,
                forecasted_value=forecasted_value,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                confidence=confidence,
                model_type=model_type,
                features_used=features_used,
            )

            # Save the forecast for later reference
            self._save_forecast(keyword, forecast, metric, horizon)

            # Track execution
            self._track_execution(start_time, True)

            return forecast

        except Exception as e:
            self.logger.error(f"Error forecasting keyword performance: {str(e)}")
            self._track_execution(start_time, False)
            raise

    def detect_emerging_trends(
        self,
        campaign_id: Optional[str] = None,
        lookback_days: int = 90,
        min_growth_rate: float = 0.2,
        min_data_points: int = 14,
    ) -> List[Dict[str, Any]]:
        """
        Detect emerging trends in keyword performance.

        Args:
            campaign_id: Optional campaign ID to limit data to
            lookback_days: Number of days to look back
            min_growth_rate: Minimum growth rate to consider a trend emerging
            min_data_points: Minimum number of data points required

        Returns:
            List of emerging trends with growth rates and confidence
        """
        start_time = datetime.now()

        try:
            # Get historical keyword performance data
            keyword_data = self._get_campaign_keyword_data(campaign_id, lookback_days)

            if keyword_data.empty:
                self.logger.warning(f"No keyword data found for trend detection")
                return []

            # Group by keyword and date, sum metrics
            keyword_data = (
                keyword_data.groupby(["keyword", "date"])
                .agg({"clicks": "sum", "impressions": "sum", "conversions": "sum", "cost": "sum"})
                .reset_index()
            )

            # Get unique keywords
            keywords = keyword_data["keyword"].unique()

            # Analyze each keyword for emerging trends
            emerging_trends = []

            for keyword in keywords:
                # Get data for this keyword
                kw_data = keyword_data[keyword_data["keyword"] == keyword].sort_values("date")

                # Skip if we don't have enough data points
                if len(kw_data) < min_data_points:
                    continue

                # Calculate growth metrics
                trend_metrics = self._calculate_trend_metrics(kw_data)

                # Check if this is an emerging trend
                if (
                    trend_metrics["recent_growth_rate"] > min_growth_rate
                    and trend_metrics["confidence"] >= 0.7
                ):

                    emerging_trends.append(
                        {
                            "keyword": keyword,
                            "growth_rate": trend_metrics["recent_growth_rate"],
                            "confidence": trend_metrics["confidence"],
                            "current_volume": trend_metrics["current_volume"],
                            "predicted_volume": trend_metrics["predicted_volume"],
                            "trend_strength": trend_metrics["trend_strength"],
                            "seasonality_impact": trend_metrics["seasonality_impact"],
                        }
                    )

            # Sort by growth rate (descending)
            emerging_trends = sorted(emerging_trends, key=lambda x: x["growth_rate"], reverse=True)

            # Track execution
            self._track_execution(start_time, True)

            return emerging_trends

        except Exception as e:
            self.logger.error(f"Error detecting emerging trends: {str(e)}")
            self._track_execution(start_time, False)
            return []

    def identify_seasonal_patterns(
        self, campaign_id: Optional[str] = None, lookback_days: int = 365, metric: str = "clicks"
    ) -> List[Dict[str, Any]]:
        """
        Identify seasonal patterns in campaign or keyword performance.

        Args:
            campaign_id: Optional campaign ID to limit data to
            lookback_days: Number of days to look back
            metric: Metric to analyze ('clicks', 'impressions', 'conversions', 'cost')

        Returns:
            List of seasonal patterns with period, strength, and phase
        """
        start_time = datetime.now()

        try:
            # Get historical data
            if campaign_id:
                # Get data for this campaign
                historical_data = self._get_campaign_historical_data(campaign_id, lookback_days)
            else:
                # Get data across all campaigns
                historical_data = self._get_account_historical_data(lookback_days)

            if historical_data.empty:
                self.logger.warning("No historical data found for seasonal pattern analysis")
                return []

            # Resample to daily frequency if needed
            if "date" in historical_data.columns:
                historical_data = historical_data.set_index("date")

            # Ensure we have the requested metric
            if metric not in historical_data.columns:
                raise ValueError(f"Metric '{metric}' not found in historical data")

            # Need enough data for seasonal decomposition
            if len(historical_data) < 2 * 7:  # At least 2 weeks
                self.logger.warning("Not enough data for seasonal decomposition")
                return []

            # Identify potential seasonal periods to check
            seasonal_periods = [7, 14, 30, 90, 365]  # day, 2 weeks, month, quarter, year
            applicable_periods = [p for p in seasonal_periods if len(historical_data) >= 2 * p]

            seasonal_patterns = []

            for period in applicable_periods:
                # Only consider periods where we have enough data
                if len(historical_data) < 2 * period:
                    continue

                try:
                    # Decompose the time series
                    decomposition = seasonal_decompose(
                        historical_data[metric].fillna(method="ffill").fillna(0),
                        model="multiplicative",
                        period=period,
                    )

                    # Calculate seasonal strength
                    seasonal_strength = np.var(decomposition.seasonal) / np.var(
                        decomposition.seasonal + decomposition.resid
                    )

                    # If the seasonal strength is significant
                    if seasonal_strength > 0.1:  # Arbitrary threshold
                        # Find the phase (when the seasonal peak occurs)
                        seasonal_series = decomposition.seasonal.dropna()
                        if len(seasonal_series) >= period:
                            peak_day = seasonal_series[:period].idxmax()

                            # Create a seasonal pattern entry
                            pattern = {
                                "period": period,
                                "period_name": self._get_period_name(period),
                                "strength": seasonal_strength,
                                "peak_day": (
                                    peak_day.strftime("%Y-%m-%d")
                                    if hasattr(peak_day, "strftime")
                                    else str(peak_day)
                                ),
                                "metric": metric,
                            }

                            seasonal_patterns.append(pattern)
                except Exception as decomp_error:
                    self.logger.warning(
                        f"Error in seasonal decomposition for period {period}: {str(decomp_error)}"
                    )
                    continue

            # Sort by strength
            seasonal_patterns = sorted(seasonal_patterns, key=lambda x: x["strength"], reverse=True)

            # Track execution
            self._track_execution(start_time, True)

            return seasonal_patterns

        except Exception as e:
            self.logger.error(f"Error identifying seasonal patterns: {str(e)}")
            self._track_execution(start_time, False)
            return []

    def generate_trend_report(
        self,
        campaign_id: Optional[str] = None,
        lookback_days: int = 90,
        forecast_horizon: str = "medium_term",
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive trend report with visualizations.

        Args:
            campaign_id: Optional campaign ID to limit data to
            lookback_days: Number of days to look back
            forecast_horizon: Forecast horizon ('short_term', 'medium_term', 'long_term')

        Returns:
            Dictionary with trend report data
        """
        start_time = datetime.now()

        try:
            report = {
                "generated_at": datetime.now().isoformat(),
                "campaign_id": campaign_id,
                "lookback_days": lookback_days,
                "forecast_horizon": forecast_horizon,
                "emerging_trends": [],
                "seasonal_patterns": [],
                "top_keyword_forecasts": [],
                "overall_trend": {},
                "visualizations": {},
            }

            # Get emerging trends
            emerging_trends = self.detect_emerging_trends(
                campaign_id=campaign_id, lookback_days=lookback_days
            )
            report["emerging_trends"] = emerging_trends

            # Get seasonal patterns
            seasonal_patterns = self.identify_seasonal_patterns(
                campaign_id=campaign_id,
                lookback_days=max(
                    lookback_days, 365
                ),  # Need at least a year for good seasonal patterns
            )
            report["seasonal_patterns"] = seasonal_patterns

            # Get historical data for visualizations
            if campaign_id:
                historical_data = self._get_campaign_historical_data(campaign_id, lookback_days)
                keywords = self._get_top_keywords_for_campaign(campaign_id, 10)
            else:
                historical_data = self._get_account_historical_data(lookback_days)
                keywords = self._get_top_keywords_for_account(10)

            # Generate forecasts for top keywords
            top_keyword_forecasts = []
            for keyword in keywords:
                try:
                    forecast = self.forecast_keyword_performance(
                        keyword=keyword,
                        campaign_id=campaign_id,
                        horizon=forecast_horizon,
                        model_type="prophet",
                    )
                    top_keyword_forecasts.append(
                        {
                            "keyword": keyword,
                            "forecast_date": forecast.forecast_date.isoformat(),
                            "forecasted_value": forecast.forecasted_value,
                            "lower_bound": forecast.lower_bound,
                            "upper_bound": forecast.upper_bound,
                            "confidence": forecast.confidence,
                        }
                    )
                except Exception as kw_error:
                    self.logger.warning(f"Error forecasting keyword '{keyword}': {str(kw_error)}")

            report["top_keyword_forecasts"] = top_keyword_forecasts

            # Generate overall trend analysis
            if not historical_data.empty:
                overall_trend = self._analyze_overall_trend(historical_data)
                report["overall_trend"] = overall_trend

            # Generate visualizations (in production, would save actual image files)
            report["visualizations"] = {
                "trends_chart": f"trends_chart_{campaign_id if campaign_id else 'account'}.png",
                "seasonal_patterns_chart": f"seasonal_{campaign_id if campaign_id else 'account'}.png",
                "forecast_chart": f"forecast_{campaign_id if campaign_id else 'account'}.png",
            }

            # Save visualizations
            if not historical_data.empty:
                self._save_trend_visualizations(
                    historical_data, report["visualizations"], campaign_id
                )

            # Save the report
            report_filename = f"trend_report_{campaign_id if campaign_id else 'account'}_{datetime.now().strftime('%Y%m%d')}.json"
            self.save_data(report, report_filename, directory=self.forecast_dir)

            # Track execution
            self._track_execution(start_time, True)

            return report

        except Exception as e:
            self.logger.error(f"Error generating trend report: {str(e)}")
            self._track_execution(start_time, False)
            return {"error": str(e), "generated_at": datetime.now().isoformat()}

    def discover_trending_keywords(
        self, industry: str, location: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Discover trending keywords in the specified industry and location.

        Args:
            industry: Industry to find trending keywords for
            location: Optional location to limit results to
            limit: Maximum number of keywords to return

        Returns:
            List of trending keywords with scores and metadata
        """
        start_time = datetime.now()

        try:
            # In a production environment, this would connect to:
            # 1. Google Trends API
            # 2. Social media trend APIs
            # 3. News APIs
            # 4. Industry-specific trend sources

            # For this implementation, we'll simulate discovering trending keywords
            trending_keywords = self._simulate_trending_keywords(industry, location, limit)

            # Track execution
            self._track_execution(start_time, True)

            return trending_keywords

        except Exception as e:
            self.logger.error(f"Error discovering trending keywords: {str(e)}")
            self._track_execution(start_time, False)
            return []

    def _get_keyword_historical_data(
        self, keyword: str, campaign_id: Optional[str], days: int
    ) -> pd.DataFrame:
        """
        Get historical data for a specific keyword.

        Args:
            keyword: Keyword to get data for
            campaign_id: Optional campaign ID to limit data to
            days: Number of days of data to retrieve

        Returns:
            DataFrame with historical keyword data
        """
        if not self.ads_api:
            raise ValueError("Google Ads API client not initialized")

        try:
            # Get keyword performance data
            keyword_data = self.ads_api.get_keyword_performance(
                days_ago=days, campaign_id=campaign_id
            )

            # Filter to the specific keyword
            keyword_data = [k for k in keyword_data if k["keyword_text"].lower() == keyword.lower()]

            if not keyword_data:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(keyword_data)

            # Ensure we have a date column (might need to be added based on the API structure)
            if "date" not in df.columns and "segments.date" in df.columns:
                df["date"] = pd.to_datetime(df["segments.date"])
            elif "date" not in df.columns:
                # If no date information, we can't do trend analysis
                raise ValueError("Keyword data does not include date information")

            return df

        except Exception as e:
            self.logger.error(f"Error getting keyword historical data: {str(e)}")
            return pd.DataFrame()

    def _get_campaign_keyword_data(self, campaign_id: Optional[str], days: int) -> pd.DataFrame:
        """
        Get historical keyword data for a campaign or all campaigns.

        Args:
            campaign_id: Optional campaign ID to limit data to
            days: Number of days of data to retrieve

        Returns:
            DataFrame with historical keyword data
        """
        if not self.ads_api:
            raise ValueError("Google Ads API client not initialized")

        try:
            # Get keyword performance data
            keyword_data = self.ads_api.get_keyword_performance(
                days_ago=days, campaign_id=campaign_id
            )

            if not keyword_data:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(keyword_data)

            # Ensure we have necessary columns
            if "keyword_text" not in df.columns:
                raise ValueError("Keyword data does not include keyword text")

            # Rename columns for consistency
            df = df.rename(columns={"keyword_text": "keyword"})

            # Ensure we have a date column
            if "date" not in df.columns and "segments.date" in df.columns:
                df["date"] = pd.to_datetime(df["segments.date"])
            elif "date" not in df.columns:
                # If no date information, we can't do trend analysis
                raise ValueError("Keyword data does not include date information")

            return df

        except Exception as e:
            self.logger.error(f"Error getting campaign keyword data: {str(e)}")
            return pd.DataFrame()

    def _get_campaign_historical_data(self, campaign_id: str, days: int) -> pd.DataFrame:
        """
        Get historical performance data for a campaign.

        Args:
            campaign_id: Campaign ID to get data for
            days: Number of days of data to retrieve

        Returns:
            DataFrame with historical campaign data
        """
        if not self.ads_api:
            raise ValueError("Google Ads API client not initialized")

        try:
            # Get campaign performance data
            campaign_data = self.ads_api.get_campaign_performance(days_ago=days)

            # Filter to the specific campaign
            campaign_data = [c for c in campaign_data if c["id"] == campaign_id]

            if not campaign_data:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(campaign_data)

            # Ensure we have a date column
            if "date" not in df.columns and "segments.date" in df.columns:
                df["date"] = pd.to_datetime(df["segments.date"])
            elif "date" not in df.columns:
                # If no date information, we can't do trend analysis
                self.logger.warning(
                    "Campaign data does not include date information, using dummy dates"
                )
                # Create dummy dates for the last 'days' days
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=days - 1)
                df["date"] = pd.date_range(start=start_date, end=end_date, periods=len(df))

            return df

        except Exception as e:
            self.logger.error(f"Error getting campaign historical data: {str(e)}")
            return pd.DataFrame()

    def _get_account_historical_data(self, days: int) -> pd.DataFrame:
        """
        Get historical performance data across all campaigns.

        Args:
            days: Number of days of data to retrieve

        Returns:
            DataFrame with historical account data
        """
        if not self.ads_api:
            raise ValueError("Google Ads API client not initialized")

        try:
            # Get campaign performance data for all campaigns
            campaign_data = self.ads_api.get_campaign_performance(days_ago=days)

            if not campaign_data:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(campaign_data)

            # Ensure we have a date column
            if "date" not in df.columns and "segments.date" in df.columns:
                df["date"] = pd.to_datetime(df["segments.date"])
            elif "date" not in df.columns:
                # If no date information, we can't do trend analysis
                self.logger.warning(
                    "Campaign data does not include date information, using dummy dates"
                )
                # Create dummy dates for the last 'days' days
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=days - 1)
                df["date"] = pd.date_range(start=start_date, end=end_date, periods=len(df))

            # Aggregate data across all campaigns by date
            if len(df) > 0:
                df = (
                    df.groupby("date")
                    .agg(
                        {"impressions": "sum", "clicks": "sum", "conversions": "sum", "cost": "sum"}
                    )
                    .reset_index()
                )

            return df

        except Exception as e:
            self.logger.error(f"Error getting account historical data: {str(e)}")
            return pd.DataFrame()

    def _prepare_data_for_forecasting(self, df: pd.DataFrame, metric: str) -> pd.DataFrame:
        """
        Prepare data for forecasting models.

        Args:
            df: DataFrame with historical data
            metric: Metric to forecast

        Returns:
            DataFrame ready for forecasting
        """
        # Copy to avoid modifying the original
        forecast_df = df.copy()

        # Ensure we have the required metric
        if metric not in forecast_df.columns:
            raise ValueError(f"Metric '{metric}' not found in historical data")

        # Ensure we have a date column as a datetime type
        if "date" not in forecast_df.columns:
            raise ValueError("Data must include a 'date' column")

        forecast_df["date"] = pd.to_datetime(forecast_df["date"])

        # Set date as index
        forecast_df = forecast_df.set_index("date").sort_index()

        # Resample to daily frequency and forward fill
        forecast_df = forecast_df.resample("D").mean().fillna(method="ffill")

        # For remaining NAs (at the beginning), use backfill
        forecast_df = forecast_df.fillna(method="bfill")

        # If still NAs, fill with zeros
        forecast_df = forecast_df.fillna(0)

        # Reset index to use date as a column
        forecast_df = forecast_df.reset_index()

        # For Prophet specifically, rename columns
        forecast_df_prophet = forecast_df.copy()
        forecast_df_prophet = forecast_df_prophet.rename(columns={"date": "ds", metric: "y"})

        return forecast_df_prophet

    def _prophet_forecast(self, df: pd.DataFrame, forecast_days: int) -> Dict[str, Any]:
        """
        Generate a forecast using Facebook Prophet.

        Args:
            df: DataFrame prepared for Prophet (with 'ds' and 'y' columns)
            forecast_days: Number of days to forecast

        Returns:
            Dictionary with forecast results
        """
        try:
            # Initialize Prophet with configuration
            model = Prophet(**self.model_configs["prophet"])

            # Add any additional regressors
            features_used = ["y"]
            for col in df.columns:
                if col not in ["ds", "y"]:
                    model.add_regressor(col)
                    features_used.append(col)

            # Fit the model
            model.fit(df)

            # Create future dataframe
            future = model.make_future_dataframe(periods=forecast_days)

            # Add regressor values to future dataframe
            for col in df.columns:
                if col not in ["ds", "y"]:
                    # For simplicity, use the last value
                    # In a real implementation, you would predict or provide future values
                    future[col] = df[col].iloc[-1]

            # Generate forecast
            forecast = model.predict(future)

            # Get values for the forecast date
            future_forecast = forecast.iloc[-1]

            # Compute the mean confidence interval width as a proxy for model confidence
            confidence_interval_width = (forecast["yhat_upper"] - forecast["yhat_lower"]).mean()
            max_value = forecast["yhat"].max()
            confidence = 1 - min(1, confidence_interval_width / (max_value + 1e-10))

            # Return forecast results
            return {
                "forecasted_value": future_forecast["yhat"],
                "lower_bound": future_forecast["yhat_lower"],
                "upper_bound": future_forecast["yhat_upper"],
                "confidence": confidence,
                "features_used": features_used,
            }

        except Exception as e:
            self.logger.error(f"Error in Prophet forecast: {str(e)}")
            # Fallback to a simpler forecast
            return self._simple_forecast(df, forecast_days)

    def _sarima_forecast(self, df: pd.DataFrame, forecast_days: int, metric: str) -> Dict[str, Any]:
        """
        Generate a forecast using SARIMA model.

        Args:
            df: DataFrame with historical data
            forecast_days: Number of days to forecast
            metric: Metric to forecast

        Returns:
            Dictionary with forecast results
        """
        try:
            # For SARIMA, we need the target as a series
            if "y" in df.columns:
                # If already prepared for Prophet
                y = df["y"]
            else:
                # Otherwise get the specified metric
                y = df[metric]

            # Configure SARIMA model
            model = SARIMAX(
                y,
                order=self.model_configs["sarima"]["order"],
                seasonal_order=self.model_configs["sarima"]["seasonal_order"],
            )

            # Fit the model
            fit_model = model.fit(disp=False)

            # Generate forecast
            forecast = fit_model.forecast(steps=forecast_days)

            # Get forecast value for the target date
            forecasted_value = forecast[-1]

            # Get prediction intervals
            pred_interval = fit_model.get_forecast(steps=forecast_days).conf_int()
            lower_bound = pred_interval.iloc[-1, 0]
            upper_bound = pred_interval.iloc[-1, 1]

            # Compute confidence based on interval width
            confidence_interval_width = upper_bound - lower_bound
            max_value = y.max()
            confidence = 1 - min(1, confidence_interval_width / (max_value + 1e-10))

            # Return forecast results
            return {
                "forecasted_value": forecasted_value,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "confidence": confidence,
                "features_used": [metric],
            }

        except Exception as e:
            self.logger.error(f"Error in SARIMA forecast: {str(e)}")
            # Fallback to a simpler forecast
            return self._simple_forecast(df, forecast_days)

    def _auto_arima_forecast(
        self, df: pd.DataFrame, forecast_days: int, metric: str
    ) -> Dict[str, Any]:
        """
        Generate a forecast using Auto ARIMA.

        Args:
            df: DataFrame with historical data
            forecast_days: Number of days to forecast
            metric: Metric to forecast

        Returns:
            Dictionary with forecast results
        """
        try:
            # For Auto ARIMA, we need the target as a series
            if "y" in df.columns:
                # If already prepared for Prophet
                y = df["y"]
            else:
                # Otherwise get the specified metric
                y = df[metric]

            # Use pmdarima to automatically find the best model
            model = pm.auto_arima(
                y,
                seasonal=self.model_configs["auto_arima"]["seasonal"],
                m=self.model_configs["auto_arima"]["m"],
                suppress_warnings=self.model_configs["auto_arima"]["suppress_warnings"],
            )

            # Generate forecast
            forecast, conf_int = model.predict(n_periods=forecast_days, return_conf_int=True)

            # Get forecast value for the target date
            forecasted_value = forecast[-1]
            lower_bound = conf_int[-1, 0]
            upper_bound = conf_int[-1, 1]

            # Compute confidence based on interval width
            confidence_interval_width = upper_bound - lower_bound
            max_value = y.max()
            confidence = 1 - min(1, confidence_interval_width / (max_value + 1e-10))

            # Return forecast results
            return {
                "forecasted_value": forecasted_value,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "confidence": confidence,
                "features_used": [metric],
            }

        except Exception as e:
            self.logger.error(f"Error in Auto ARIMA forecast: {str(e)}")
            # Fallback to a simpler forecast
            return self._simple_forecast(df, forecast_days)

    def _ensemble_forecast(
        self, df: pd.DataFrame, forecast_days: int, metric: str
    ) -> Dict[str, Any]:
        """
        Generate a forecast using an ensemble of models.

        Args:
            df: DataFrame with historical data
            forecast_days: Number of days to forecast
            metric: Metric to forecast

        Returns:
            Dictionary with forecast results
        """
        try:
            # Get forecasts from individual models
            prophet_forecast = self._prophet_forecast(df, forecast_days)

            try:
                sarima_forecast = self._sarima_forecast(df, forecast_days, metric)
            except Exception as sarima_error:
                self.logger.warning(
                    f"SARIMA forecast failed, using default values: {str(sarima_error)}"
                )
                sarima_forecast = {
                    "forecasted_value": prophet_forecast["forecasted_value"],
                    "lower_bound": prophet_forecast["lower_bound"],
                    "upper_bound": prophet_forecast["upper_bound"],
                    "confidence": 0.5,
                }

            try:
                auto_arima_forecast = self._auto_arima_forecast(df, forecast_days, metric)
            except Exception as arima_error:
                self.logger.warning(
                    f"Auto ARIMA forecast failed, using default values: {str(arima_error)}"
                )
                auto_arima_forecast = {
                    "forecasted_value": prophet_forecast["forecasted_value"],
                    "lower_bound": prophet_forecast["lower_bound"],
                    "upper_bound": prophet_forecast["upper_bound"],
                    "confidence": 0.5,
                }

            # Weight the forecasts by their confidence
            total_confidence = (
                prophet_forecast["confidence"]
                + sarima_forecast["confidence"]
                + auto_arima_forecast["confidence"]
            )

            if total_confidence == 0:
                # If all confidences are 0, use simple average
                forecasted_value = (
                    prophet_forecast["forecasted_value"]
                    + sarima_forecast["forecasted_value"]
                    + auto_arima_forecast["forecasted_value"]
                ) / 3

                lower_bound = (
                    prophet_forecast["lower_bound"]
                    + sarima_forecast["lower_bound"]
                    + auto_arima_forecast["lower_bound"]
                ) / 3

                upper_bound = (
                    prophet_forecast["upper_bound"]
                    + sarima_forecast["upper_bound"]
                    + auto_arima_forecast["upper_bound"]
                ) / 3

                confidence = 0.6  # Default confidence for ensemble
            else:
                # Weighted average based on confidence
                forecasted_value = (
                    prophet_forecast["forecasted_value"] * prophet_forecast["confidence"]
                    + sarima_forecast["forecasted_value"] * sarima_forecast["confidence"]
                    + auto_arima_forecast["forecasted_value"] * auto_arima_forecast["confidence"]
                ) / total_confidence

                lower_bound = (
                    prophet_forecast["lower_bound"] * prophet_forecast["confidence"]
                    + sarima_forecast["lower_bound"] * sarima_forecast["confidence"]
                    + auto_arima_forecast["lower_bound"] * auto_arima_forecast["confidence"]
                ) / total_confidence

                upper_bound = (
                    prophet_forecast["upper_bound"] * prophet_forecast["confidence"]
                    + sarima_forecast["upper_bound"] * sarima_forecast["confidence"]
                    + auto_arima_forecast["upper_bound"] * auto_arima_forecast["confidence"]
                ) / total_confidence

                # Ensemble confidence is the weighted average of individual confidences
                confidence = total_confidence / 3

            # Return ensemble forecast results
            return {
                "forecasted_value": forecasted_value,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "confidence": confidence,
                "features_used": prophet_forecast.get("features_used", [metric]),
            }

        except Exception as e:
            self.logger.error(f"Error in ensemble forecast: {str(e)}")
            # Fallback to Prophet forecast or simple forecast
            try:
                return self._prophet_forecast(df, forecast_days)
            except:
                return self._simple_forecast(df, forecast_days)

    def _simple_forecast(self, df: pd.DataFrame, forecast_days: int) -> Dict[str, Any]:
        """
        Generate a simple forecast using moving average and trend.

        Args:
            df: DataFrame with historical data
            forecast_days: Number of days to forecast

        Returns:
            Dictionary with forecast results
        """
        try:
            # Get the target series
            if "y" in df.columns:
                y = df["y"]
            elif "clicks" in df.columns:
                y = df["clicks"]
            elif "impressions" in df.columns:
                y = df["impressions"]
            else:
                y = df.iloc[:, -1]  # Last column as a fallback

            # Calculate average and trend
            n = len(y)
            if n < 2:
                return {
                    "forecasted_value": y.iloc[0] if n > 0 else 0,
                    "lower_bound": 0,
                    "upper_bound": y.iloc[0] * 2 if n > 0 else 0,
                    "confidence": 0.3,
                    "features_used": ["simple_trend"],
                }

            # Use last 30 days or all available data if less
            window = min(30, n)
            recent_y = y[-window:]

            # Calculate moving average
            avg = recent_y.mean()

            # Calculate simple linear trend
            x = np.arange(window)
            if np.std(recent_y) > 0:  # Only calculate trend if there's variation
                slope, intercept = np.polyfit(x, recent_y, 1)
                trend = slope
            else:
                trend = 0

            # Project forward
            forecast_value = avg + trend * forecast_days

            # Ensure non-negative forecast for metrics like clicks
            forecast_value = max(0, forecast_value)

            # Simple confidence interval based on historical volatility
            std = np.std(recent_y)
            lower_bound = max(0, forecast_value - 2 * std)
            upper_bound = forecast_value + 2 * std

            # Confidence based on data stability
            cv = std / (avg + 1e-10)  # Coefficient of variation
            confidence = max(0.3, 1 - min(1, cv))  # Higher CV = lower confidence

            return {
                "forecasted_value": forecast_value,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "confidence": confidence,
                "features_used": ["simple_trend"],
            }

        except Exception as e:
            self.logger.error(f"Error in simple forecast: {str(e)}")
            # Ultimate fallback
            return {
                "forecasted_value": 0,
                "lower_bound": 0,
                "upper_bound": 0,
                "confidence": 0.1,
                "features_used": [],
            }

    def _add_external_signals(self, df: pd.DataFrame, keyword: str) -> pd.DataFrame:
        """
        Add external signals to the forecast dataframe.

        Args:
            df: DataFrame with historical data
            keyword: Keyword to get signals for

        Returns:
            DataFrame with added external signals
        """
        # Copy to avoid modifying the original
        forecast_df = df.copy()

        try:
            # Use the ContextualSignalService if available
            if hasattr(self, "contextual_signal_service"):
                # In a real implementation, you would query for external signals
                # and incorporate them into the forecast data
                pass

            # Alternatively, add engineered features

            # Add day of week
            if "ds" in forecast_df.columns:
                forecast_df["day_of_week"] = forecast_df["ds"].dt.dayofweek

            # Add month
            if "ds" in forecast_df.columns:
                forecast_df["month"] = forecast_df["ds"].dt.month

            # Add holiday flag (simplified)
            # In a real implementation, you would use a proper holiday calendar
            if "ds" in forecast_df.columns:
                is_holiday = np.zeros(len(forecast_df))
                # Mark major US holidays (very simplified)
                for i, date in enumerate(forecast_df["ds"]):
                    if (
                        (date.month == 1 and date.day == 1)
                        or (date.month == 12 and date.day == 25)
                        or (date.month == 7 and date.day == 4)
                        or (date.month == 11 and date.day in [22, 23, 24, 25, 26])
                        or (
                            date.month == 5
                            and date.day >= 25
                            and date.day <= 31
                            and date.dayofweek == 0
                        )
                    ):
                        is_holiday[i] = 1
                forecast_df["is_holiday"] = is_holiday

            return forecast_df

        except Exception as e:
            self.logger.error(f"Error adding external signals: {str(e)}")
            return df  # Return original dataframe if there's an error

    def _calculate_trend_metrics(self, kw_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate trend metrics for a keyword.

        Args:
            kw_data: DataFrame with keyword performance data

        Returns:
            Dictionary with trend metrics
        """
        # Calculate trend metrics
        metrics = {}

        try:
            # Sort by date
            kw_data = kw_data.sort_values("date")

            # Calculate growth rate metrics
            if "clicks" in kw_data.columns and len(kw_data) >= 7:
                # Split into recent and older data
                recent_data = kw_data.iloc[-7:]  # Last 7 days
                older_data = kw_data.iloc[:-7]  # Days before that

                if len(older_data) > 0:
                    # Calculate average clicks
                    recent_avg_clicks = recent_data["clicks"].mean()
                    older_avg_clicks = older_data["clicks"].mean()

                    # Calculate growth rate
                    if older_avg_clicks > 0:
                        growth_rate = (recent_avg_clicks - older_avg_clicks) / older_avg_clicks
                    else:
                        growth_rate = 1.0 if recent_avg_clicks > 0 else 0.0

                    metrics["recent_growth_rate"] = growth_rate
                else:
                    metrics["recent_growth_rate"] = 0.0

                metrics["current_volume"] = recent_data["clicks"].sum()
            else:
                metrics["recent_growth_rate"] = 0.0
                metrics["current_volume"] = (
                    kw_data["clicks"].sum() if "clicks" in kw_data.columns else 0
                )

            # Calculate trend strength using linear regression
            if len(kw_data) >= 14 and "clicks" in kw_data.columns:
                x = np.arange(len(kw_data))
                y = kw_data["clicks"].values

                if np.std(y) > 0:  # Only if there's variation
                    # Fit linear trend
                    slope, intercept = np.polyfit(x, y, 1)

                    # R-squared as trend strength
                    y_pred = slope * x + intercept
                    ss_total = np.sum((y - np.mean(y)) ** 2)
                    ss_residual = np.sum((y - y_pred) ** 2)

                    if ss_total > 0:
                        r_squared = 1 - (ss_residual / ss_total)
                        # Adjust for direction - positive trend should have positive strength
                        trend_strength = r_squared * (1 if slope > 0 else -1)
                    else:
                        trend_strength = 0
                else:
                    trend_strength = 0

                metrics["trend_strength"] = trend_strength

                # Calculate predicted future volume
                days_ahead = 30  # Predict a month ahead
                predicted_value = slope * (len(kw_data) + days_ahead) + intercept
                metrics["predicted_volume"] = max(
                    0, predicted_value * days_ahead
                )  # Prevent negative prediction
            else:
                metrics["trend_strength"] = 0
                metrics["predicted_volume"] = metrics[
                    "current_volume"
                ]  # Just use current as prediction

            # Calculate seasonality impact
            if len(kw_data) >= 30 and "clicks" in kw_data.columns:
                try:
                    # Set date as index
                    kw_data_ts = kw_data.set_index("date")["clicks"]

                    # Check for sufficient variation
                    if np.std(kw_data_ts) > 0:
                        # Decompose with a weekly seasonality
                        decomposition = seasonal_decompose(
                            kw_data_ts.fillna(method="ffill").fillna(0),
                            model="multiplicative",
                            period=7,  # Weekly
                        )

                        # Extract seasonal component
                        seasonal = decomposition.seasonal.dropna()

                        # Calculate seasonality impact as the ratio of max to min seasonal factor
                        seasonality_impact = (
                            seasonal.max() / seasonal.min() if seasonal.min() > 0 else 1.0
                        )
                    else:
                        seasonality_impact = 1.0  # No seasonal impact if no variation
                except Exception as decomp_error:
                    self.logger.warning(f"Error in seasonality decomposition: {str(decomp_error)}")
                    seasonality_impact = 1.0
            else:
                seasonality_impact = 1.0  # Not enough data for seasonality

            metrics["seasonality_impact"] = seasonality_impact

            # Calculate confidence based on data quality and consistency
            metrics_present = sum(
                1
                for m in ["recent_growth_rate", "trend_strength", "seasonality_impact"]
                if m in metrics
            )
            data_length_factor = min(1.0, len(kw_data) / 30)  # More data = higher confidence

            confidence = 0.5 * data_length_factor + 0.5 * (metrics_present / 3)
            metrics["confidence"] = confidence

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating trend metrics: {str(e)}")
            return {
                "recent_growth_rate": 0,
                "current_volume": 0,
                "predicted_volume": 0,
                "trend_strength": 0,
                "seasonality_impact": 1.0,
                "confidence": 0.3,
            }

    def _get_top_keywords_for_campaign(self, campaign_id: str, limit: int = 10) -> List[str]:
        """
        Get top keywords for a campaign by clicks.

        Args:
            campaign_id: Campaign ID
            limit: Maximum number of keywords to return

        Returns:
            List of top keywords
        """
        if not self.ads_api:
            return []

        try:
            # Get keyword performance for this campaign
            keyword_data = self.ads_api.get_keyword_performance(
                days_ago=30, campaign_id=campaign_id
            )

            if not keyword_data:
                return []

            # Convert to DataFrame
            df = pd.DataFrame(keyword_data)

            # Group by keyword and sum clicks
            if "keyword_text" in df.columns and "clicks" in df.columns:
                top_keywords = (
                    df.groupby("keyword_text")["clicks"].sum().sort_values(ascending=False)
                )
                return top_keywords.head(limit).index.tolist()
            else:
                return []

        except Exception as e:
            self.logger.error(f"Error getting top keywords for campaign: {str(e)}")
            return []

    def _get_top_keywords_for_account(self, limit: int = 10) -> List[str]:
        """
        Get top keywords across all campaigns by clicks.

        Args:
            limit: Maximum number of keywords to return

        Returns:
            List of top keywords
        """
        if not self.ads_api:
            return []

        try:
            # Get keyword performance across all campaigns
            keyword_data = self.ads_api.get_keyword_performance(days_ago=30)

            if not keyword_data:
                return []

            # Convert to DataFrame
            df = pd.DataFrame(keyword_data)

            # Group by keyword and sum clicks
            if "keyword_text" in df.columns and "clicks" in df.columns:
                top_keywords = (
                    df.groupby("keyword_text")["clicks"].sum().sort_values(ascending=False)
                )
                return top_keywords.head(limit).index.tolist()
            else:
                return []

        except Exception as e:
            self.logger.error(f"Error getting top keywords for account: {str(e)}")
            return []

    def _analyze_overall_trend(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze overall performance trend.

        Args:
            historical_data: DataFrame with historical performance data

        Returns:
            Dictionary with trend analysis
        """
        trend_analysis = {}

        try:
            # Copy and ensure date is index
            df = historical_data.copy()
            if "date" in df.columns:
                df = df.set_index("date")

            # Calculate trends for key metrics
            for metric in ["clicks", "impressions", "conversions", "cost"]:
                if metric in df.columns:
                    # Calculate simple linear trend
                    x = np.arange(len(df))
                    y = df[metric].values

                    if len(y) >= 2 and np.std(y) > 0:
                        # Fit linear trend
                        slope, intercept = np.polyfit(x, y, 1)

                        # Get average value
                        avg = np.mean(y)

                        # Calculate relative trend (% change per day)
                        relative_trend = slope / avg if avg > 0 else 0

                        # Calculate R-squared
                        y_pred = slope * x + intercept
                        ss_total = np.sum((y - np.mean(y)) ** 2)
                        ss_residual = np.sum((y - y_pred) ** 2)

                        if ss_total > 0:
                            r_squared = 1 - (ss_residual / ss_total)
                        else:
                            r_squared = 0

                        trend_analysis[metric] = {
                            "direction": "up" if slope > 0 else "down",
                            "strength": r_squared,
                            "daily_change": slope,
                            "daily_change_percent": relative_trend * 100,
                            "current_value": y[-1],
                            "average_value": avg,
                        }
                    else:
                        trend_analysis[metric] = {
                            "direction": "stable",
                            "strength": 0,
                            "daily_change": 0,
                            "daily_change_percent": 0,
                            "current_value": y[-1] if len(y) > 0 else 0,
                            "average_value": avg if len(y) > 0 else 0,
                        }

            # Analyze performance stability
            stability = {}
            for metric in ["clicks", "impressions", "conversions", "cost"]:
                if metric in df.columns:
                    values = df[metric].values
                    if len(values) > 0 and np.mean(values) > 0:
                        cv = np.std(values) / np.mean(values)  # Coefficient of variation
                        stability[metric] = 1 - min(1, cv)  # Higher CV = lower stability
                    else:
                        stability[metric] = 1.0  # Default to stable if no data

            trend_analysis["stability"] = stability

            # Identify weekly patterns
            weekly_patterns = {}
            if len(df) >= 14:  # Need at least two weeks of data
                for metric in ["clicks", "impressions", "conversions", "cost"]:
                    if metric in df.columns:
                        # Add day of week
                        if not df.index.empty and hasattr(df.index, "dayofweek"):
                            df["day_of_week"] = df.index.dayofweek

                            # Group by day of week
                            day_of_week_avg = df.groupby("day_of_week")[metric].mean()

                            if not day_of_week_avg.empty:
                                # Find best and worst days
                                best_day = day_of_week_avg.idxmax()
                                worst_day = day_of_week_avg.idxmin()

                                # Calculate day-of-week effect
                                overall_avg = df[metric].mean()
                                day_effect = {
                                    day: (value / overall_avg - 1) * 100 if overall_avg > 0 else 0
                                    for day, value in day_of_week_avg.items()
                                }

                                weekly_patterns[metric] = {
                                    "best_day": self._day_number_to_name(best_day),
                                    "worst_day": self._day_number_to_name(worst_day),
                                    "day_effect": {
                                        self._day_number_to_name(day): effect
                                        for day, effect in day_effect.items()
                                    },
                                }

            trend_analysis["weekly_patterns"] = weekly_patterns

            return trend_analysis

        except Exception as e:
            self.logger.error(f"Error analyzing overall trend: {str(e)}")
            return {"error": str(e)}

    def _save_trend_visualizations(
        self,
        historical_data: pd.DataFrame,
        visualization_paths: Dict[str, str],
        campaign_id: Optional[str] = None,
    ) -> None:
        """
        Generate and save trend visualizations.

        Args:
            historical_data: DataFrame with historical data
            visualization_paths: Dictionary of visualization file paths
            campaign_id: Optional campaign ID for file naming
        """
        try:
            # Copy data
            df = historical_data.copy()

            # Ensure date is datetime
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])

            # Ensure plots directory exists
            plots_dir = os.path.join(self.forecast_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            # 1. Trends chart - plot key metrics over time
            try:
                plt.figure(figsize=(10, 6))

                # Plot each metric on its own scaled axis
                for metric in ["clicks", "impressions", "conversions", "cost"]:
                    if metric in df.columns:
                        plt.plot("date", metric, data=df, label=metric)

                plt.title(
                    f"Performance Trends {f'for Campaign {campaign_id}' if campaign_id else 'Across All Campaigns'}"
                )
                plt.xlabel("Date")
                plt.ylabel("Value")
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Save the plot
                trends_path = os.path.join(plots_dir, visualization_paths["trends_chart"])
                plt.savefig(trends_path)
                plt.close()

                self.logger.info(f"Saved trends chart to {trends_path}")

            except Exception as trends_error:
                self.logger.error(f"Error generating trends chart: {str(trends_error)}")

            # 2. Seasonal patterns chart
            try:
                if "date" in df.columns and "clicks" in df.columns and len(df) >= 14:
                    plt.figure(figsize=(12, 8))

                    # Add day of week
                    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek

                    # Create subplots
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

                    # Day of week patterns
                    sns.boxplot(x="day_of_week", y="clicks", data=df, ax=ax1)
                    ax1.set_title("Clicks by Day of Week")
                    ax1.set_xlabel("Day of Week")
                    ax1.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

                    # Time series decomposition if enough data
                    if len(df) >= 30:
                        # Set date as index
                        df_ts = df.set_index("date")["clicks"]

                        # Decompose with a weekly seasonality
                        decomposition = seasonal_decompose(
                            df_ts.fillna(method="ffill").fillna(0),
                            model="multiplicative",
                            period=7,  # Weekly
                        )

                        # Plot seasonal component
                        decomposition.seasonal.plot(ax=ax2)
                        ax2.set_title("Seasonal Component (Weekly Pattern)")
                        ax2.set_xlabel("Date")
                        ax2.set_ylabel("Seasonal Factor")

                    plt.tight_layout()

                    # Save the plot
                    seasonal_path = os.path.join(
                        plots_dir, visualization_paths["seasonal_patterns_chart"]
                    )
                    plt.savefig(seasonal_path)
                    plt.close()

                    self.logger.info(f"Saved seasonal patterns chart to {seasonal_path}")

            except Exception as seasonal_error:
                self.logger.error(
                    f"Error generating seasonal patterns chart: {str(seasonal_error)}"
                )

            # 3. Forecast chart
            try:
                if "date" in df.columns and "clicks" in df.columns and len(df) >= 14:
                    plt.figure(figsize=(10, 6))

                    # Prepare data for Prophet
                    prophet_df = df[["date", "clicks"]].rename(
                        columns={"date": "ds", "clicks": "y"}
                    )

                    # Initialize Prophet with configuration
                    model = Prophet(**self.model_configs["prophet"])

                    # Fit the model
                    model.fit(prophet_df)

                    # Create future dataframe
                    future = model.make_future_dataframe(periods=30)  # 30 days forecast

                    # Generate forecast
                    forecast = model.predict(future)

                    # Plot the forecast
                    fig = model.plot(forecast)
                    ax = fig.gca()
                    ax.set_title(
                        f"Clicks Forecast {f'for Campaign {campaign_id}' if campaign_id else 'Across All Campaigns'}"
                    )
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Clicks")

                    # Save the plot
                    forecast_path = os.path.join(plots_dir, visualization_paths["forecast_chart"])
                    plt.savefig(forecast_path)
                    plt.close()

                    self.logger.info(f"Saved forecast chart to {forecast_path}")

            except Exception as forecast_error:
                self.logger.error(f"Error generating forecast chart: {str(forecast_error)}")

        except Exception as e:
            self.logger.error(f"Error saving trend visualizations: {str(e)}")

    def _save_forecast(
        self, keyword: str, forecast: TrendForecast, metric: str, horizon: str
    ) -> None:
        """
        Save a forecast for later reference.

        Args:
            keyword: Keyword the forecast is for
            forecast: The forecast object
            metric: Metric that was forecast
            horizon: Forecast horizon
        """
        try:
            # Create a dictionary representation of the forecast
            forecast_dict = {
                "keyword": keyword,
                "forecast_date": forecast.forecast_date.isoformat(),
                "forecasted_value": forecast.forecasted_value,
                "lower_bound": forecast.lower_bound,
                "upper_bound": forecast.upper_bound,
                "confidence": forecast.confidence,
                "model_type": forecast.model_type,
                "features_used": forecast.features_used,
                "metric": metric,
                "horizon": horizon,
                "created_at": datetime.now().isoformat(),
            }

            # Store in forecast history
            if keyword not in self.forecast_history:
                self.forecast_history[keyword] = []

            self.forecast_history[keyword].append(forecast_dict)

            # Limit history to last 10 forecasts per keyword
            if len(self.forecast_history[keyword]) > 10:
                self.forecast_history[keyword] = self.forecast_history[keyword][-10:]

            # Save to disk
            filename = f"forecast_{keyword.replace(' ', '_')}_{horizon}_{metric}.json"
            self.save_data(forecast_dict, filename, directory=self.forecast_dir)

        except Exception as e:
            self.logger.error(f"Error saving forecast: {str(e)}")

    def _simulate_trending_keywords(
        self, industry: str, location: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Simulate discovering trending keywords (for demonstration purposes).

        Args:
            industry: Industry to find trending keywords for
            location: Optional location to limit results to
            limit: Maximum number of keywords to return

        Returns:
            List of trending keywords with scores and metadata
        """
        # In a real implementation, this would connect to external APIs

        # Industry-specific keyword templates
        industry_keywords = {
            "retail": [
                "{season} {product}",
                "best {product} for {occasion}",
                "{brand} {product} {attribute}",
                "{product} on sale",
                "discount {product}",
                "{product} near me",
                "{attribute} {product}",
                "{occasion} {gift}",
                "{brand} vs {brand}",
            ],
            "travel": [
                "{destination} vacation",
                "cheap flights to {destination}",
                "{destination} {accommodation}",
                "all inclusive {destination}",
                "{destination} with {feature}",
                "{season} vacation in {destination}",
                "{destination} {activity}",
                "best time to visit {destination}",
                "{transportation} to {destination}",
            ],
            "technology": [
                "new {device} {year}",
                "best {device} for {use_case}",
                "{brand} {device} review",
                "how to {action} {device}",
                "{device} vs {device}",
                "cheap {device}",
                "{software} tutorial",
                "{device} {issue} fix",
                "{software} alternatives",
            ],
            "health": [
                "{condition} symptoms",
                "how to prevent {condition}",
                "best {supplement} for {benefit}",
                "{diet} meal plan",
                "{exercise} for {body_part}",
                "{condition} treatment",
                "{health_product} reviews",
                "natural remedies for {condition}",
                "{exercise} workout plan",
            ],
        }

        # Get the appropriate templates for the industry
        templates = industry_keywords.get(industry.lower(), industry_keywords["retail"])

        # Fill-in values for the templates
        fill_values = {
            "season": ["summer", "winter", "fall", "spring", "holiday"],
            "product": ["shoes", "jackets", "laptops", "phones", "furniture", "gifts", "watches"],
            "occasion": [
                "wedding",
                "graduation",
                "birthday",
                "anniversary",
                "Christmas",
                "Halloween",
            ],
            "brand": ["Nike", "Apple", "Samsung", "Amazon", "Sony", "Google"],
            "attribute": [
                "waterproof",
                "lightweight",
                "durable",
                "premium",
                "budget",
                "professional",
            ],
            "gift": ["gift ideas", "presents", "gift boxes", "gift cards"],
            "destination": ["Hawaii", "Bali", "Paris", "New York", "Japan", "Caribbean", "Mexico"],
            "accommodation": ["hotels", "resorts", "Airbnb", "vacation rentals", "all-inclusive"],
            "feature": ["kids", "pets", "elderly", "families", "couples"],
            "activity": ["hiking", "beaches", "restaurants", "nightlife", "museums"],
            "season": ["summer", "winter", "spring", "fall", "holiday"],
            "transportation": ["flights", "train", "cruise", "bus", "rental car"],
            "device": [
                "laptop",
                "smartphone",
                "tablet",
                "smartwatch",
                "headphones",
                "camera",
                "TV",
            ],
            "year": [str(datetime.now().year), str(datetime.now().year + 1)],
            "use_case": ["gaming", "work", "students", "travel", "photography", "video editing"],
            "action": ["fix", "upgrade", "set up", "optimize", "clean", "troubleshoot"],
            "software": ["Windows", "iOS", "Android", "Photoshop", "Office", "Zoom"],
            "issue": ["battery", "slow", "won't start", "freezing", "overheating", "blue screen"],
            "condition": [
                "diabetes",
                "anxiety",
                "depression",
                "allergies",
                "insomnia",
                "arthritis",
            ],
            "supplement": ["vitamin D", "probiotics", "omega-3", "protein", "collagen", "zinc"],
            "benefit": ["immunity", "sleep", "energy", "skin", "hair", "joints", "weight loss"],
            "diet": ["keto", "vegan", "paleo", "mediterranean", "intermittent fasting"],
            "exercise": ["yoga", "pilates", "HIIT", "cardio", "strength training", "running"],
            "body_part": ["abs", "arms", "legs", "back", "core", "full body"],
            "health_product": ["fitness tracker", "massage gun", "supplements", "air purifier"],
        }

        # Adjust based on location if provided
        if location:
            fill_values["product"].append(f"{location} specialties")
            fill_values["destination"] = [location] + fill_values["destination"]
            # Add local brands based on location
            # This would be more sophisticated in a real implementation

        # Generate trending keywords
        trending_keywords = []
        for _ in range(min(30, limit * 2)):  # Generate extras so we can filter
            template = np.random.choice(templates)

            # Fill in the template with random values
            keyword = template
            for placeholder, values in fill_values.items():
                if "{" + placeholder + "}" in keyword:
                    keyword = keyword.replace("{" + placeholder + "}", np.random.choice(values))

            # Generate a random trending score and volume
            trend_score = np.random.uniform(0.6, 1.0)
            volume = int(np.random.lognormal(10, 1))

            # Generate random growth rate biased towards positive
            growth_rate = np.random.normal(0.2, 0.1)

            # Add some metadata
            trending_keywords.append(
                {
                    "keyword": keyword,
                    "trend_score": trend_score,
                    "volume": volume,
                    "growth_rate": growth_rate,
                    "industry": industry,
                    "location": location,
                    "source": np.random.choice(
                        ["google_trends", "social_media", "news_analysis", "search_volume_analysis"]
                    ),
                }
            )

        # Sort by trend score and limit
        trending_keywords = sorted(trending_keywords, key=lambda x: x["trend_score"], reverse=True)
        return trending_keywords[:limit]

    def _get_period_name(self, period: int) -> str:
        """
        Convert a period number to a human-readable name.

        Args:
            period: Period in days

        Returns:
            Human-readable period name
        """
        if period == 7:
            return "Weekly"
        elif period == 14:
            return "Bi-weekly"
        elif period == 30:
            return "Monthly"
        elif period == 90:
            return "Quarterly"
        elif period == 365:
            return "Yearly"
        else:
            return f"{period}-day"

    def _day_number_to_name(self, day_number: int) -> str:
        """
        Convert a day number (0-6) to a day name.

        Args:
            day_number: Day number (0=Monday, 6=Sunday)

        Returns:
            Day name
        """
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return days[day_number] if 0 <= day_number < 7 else f"Day {day_number}"
