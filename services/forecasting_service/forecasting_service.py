"""
Forecasting Service for Google Ads Management System

This module provides forecasting capabilities for predicting future performance metrics,
search trends, budget requirements, and other key indicators for Google Ads campaigns.
"""

import logging
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import pickle
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy import stats
import joblib

# Import forecasting libraries
try:
    from prophet import Prophet  # type: ignore

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not available. Install with: pip install prophet")

try:
    import statsmodels.api as sm  # type: ignore

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("Statsmodels not available. Install with: pip install statsmodels")

# Correct relative import for BaseService
from ..base_service import BaseService

logger = logging.getLogger(__name__)


class ForecastingService(BaseService):
    """
    Service for forecasting future performance metrics and trends within Google Ads.

    The ForecastingService implements multiple time series forecasting techniques
    to predict future values for key performance indicators, upcoming search trends,
    required budgets, and other metrics to enable proactive campaign management.

    Features:
    - Performance metric forecasting (clicks, impressions, conversions, etc.)
    - Budget forecasting and planning
    - Trend identification and prediction
    - Seasonal pattern detection
    - Integration with Google Ads demand forecasts
    """

    def __init__(
        self,
        ads_api=None,
        optimizer=None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the ForecastingService.

        Args:
            ads_api: Google Ads API client instance
            optimizer: AI optimizer instance for generating suggestions
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(ads_api, optimizer, config, logger)

        # Initialize forecasting components
        self.models = {}
        self.forecasts = {}
        self.metrics_history = {}

        # Default forecasting parameters
        self.forecast_horizon = 30  # days by default
        self.confidence_level = 0.95  # 95% confidence intervals
        self.training_window = 90  # days of historical data used for training

        # Override defaults with config values if provided
        if config and "forecasting" in config:
            self.forecast_horizon = config["forecasting"].get("horizon_days", self.forecast_horizon)
            self.confidence_level = config["forecasting"].get(
                "confidence_level", self.confidence_level
            )
            self.training_window = config["forecasting"].get(
                "training_window_days", self.training_window
            )

        # Create directory for model storage
        self.models_dir = os.path.join("data", "forecasting", "models")
        os.makedirs(self.models_dir, exist_ok=True)

        # Load existing models and forecasts if available
        self._load_models()

        self.logger.info("ForecastingService initialized")

    def _load_models(self):
        """Load saved forecasting models if they exist."""
        try:
            model_index_path = os.path.join(self.models_dir, "model_index.json")
            if os.path.exists(model_index_path):
                with open(model_index_path, "r") as f:
                    model_index = json.load(f)

                for model_info in model_index.get("models", []):
                    model_path = os.path.join(self.models_dir, model_info["filename"])
                    if os.path.exists(model_path):
                        # For statsmodels-based models that are saved with joblib
                        self.models[model_info["id"]] = {
                            "model": joblib.load(model_path),
                            "type": model_info["type"],
                            "metric": model_info["metric"],
                            "campaign_id": model_info.get("campaign_id"),
                            "last_trained": model_info["last_trained"],
                            "performance": model_info.get("performance", {}),
                        }

                self.logger.info(f"Loaded {len(self.models)} forecasting models")

                # Load latest forecasts if available
                forecasts_path = os.path.join("data", "forecasting", "latest_forecasts.json")
                if os.path.exists(forecasts_path):
                    with open(forecasts_path, "r") as f:
                        self.forecasts = json.load(f)
                    self.logger.info(f"Loaded existing forecasts from {forecasts_path}")

        except Exception as e:
            self.logger.error(f"Error loading forecasting models: {str(e)}")

    def _save_models(self):
        """Save forecasting models and model index."""
        try:
            # Create model index
            model_index = {"models": []}

            for model_id, model_data in self.models.items():
                # Skip if the model is not saveable (e.g., Prophet models)
                if not hasattr(model_data["model"], "save"):
                    continue

                filename = f"model_{model_id}.pkl"
                model_path = os.path.join(self.models_dir, filename)

                # Save the model using joblib
                joblib.dump(model_data["model"], model_path)

                # Add to index
                model_index["models"].append(
                    {
                        "id": model_id,
                        "filename": filename,
                        "type": model_data["type"],
                        "metric": model_data["metric"],
                        "campaign_id": model_data.get("campaign_id"),
                        "last_trained": model_data["last_trained"],
                        "performance": model_data.get("performance", {}),
                    }
                )

            # Save model index
            with open(os.path.join(self.models_dir, "model_index.json"), "w") as f:
                json.dump(model_index, f, indent=2, default=str)

            # Save latest forecasts
            with open(os.path.join("data", "forecasting", "latest_forecasts.json"), "w") as f:
                json.dump(self.forecasts, f, indent=2, default=str)

            self.logger.info(f"Saved {len(model_index['models'])} forecasting models")

        except Exception as e:
            self.logger.error(f"Error saving forecasting models: {str(e)}")

    def fetch_historical_data(
        self,
        metrics: List[str],
        days: int = 90,
        campaign_id: Optional[str] = None,
        aggregate_by: str = "day",
    ) -> pd.DataFrame:
        """
        Fetch historical performance data for forecasting.

        Args:
            metrics: List of metrics to fetch (e.g., 'clicks', 'impressions', 'conversions')
            days: Number of days of historical data to fetch
            campaign_id: Optional campaign ID to filter data for a specific campaign
            aggregate_by: Time aggregation level ('day', 'week', 'month')

        Returns:
            Pandas DataFrame with historical data
        """
        self.logger.info(
            f"Fetching {days} days of historical data for metrics: {', '.join(metrics)}"
        )

        try:
            # Use the Ads API to fetch historical data
            if not self.ads_api:
                raise ValueError("Google Ads API client not initialized")

            # Fetch data for different metrics
            # In this case we'll focus on keyword performance data since it's most
            # commonly used for forecasting
            data = self.ads_api.get_keyword_performance(days, campaign_id)

            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(data)

            # Check if we have the required metrics
            for metric in metrics:
                if metric not in df.columns:
                    self.logger.warning(f"Metric '{metric}' not found in the retrieved data")

            # Ensure we have a date column
            if "day" not in df.columns and "date" not in df.columns:
                # If we don't have an explicit date column, we'll use the date from
                # the request
                today = datetime.now().date()
                df["date"] = [today - timedelta(days=i) for i in range(days)]

            date_col = "day" if "day" in df.columns else "date"

            # Convert date column to datetime if it's not already
            if df[date_col].dtype != "datetime64[ns]":
                df[date_col] = pd.to_datetime(df[date_col])

            # Set date as index
            df = df.set_index(date_col)

            # Aggregate by the specified time period
            if aggregate_by == "week":
                df = df.resample("W").sum()
            elif aggregate_by == "month":
                df = df.resample("M").sum()
            elif aggregate_by == "day":
                df = df.resample("D").sum()

            # Sort by date ascending (oldest to newest)
            df = df.sort_index()

            # Store in memory for later use
            history_key = f"{'all' if campaign_id is None else campaign_id}_{aggregate_by}"
            self.metrics_history[history_key] = df

            return df

        except Exception as e:
            self.logger.error(f"Error fetching historical data: {str(e)}")
            raise

    def forecast_metrics(
        self,
        metrics: List[str],
        days_to_forecast: int = 30,
        campaign_id: Optional[str] = None,
        model_type: str = "auto",
    ) -> Dict[str, Any]:
        """
        Forecast future values for specified metrics.

        Args:
            metrics: List of metrics to forecast
            days_to_forecast: Number of days to forecast into the future
            campaign_id: Optional campaign ID to forecast for a specific campaign
            model_type: Type of forecasting model to use:
                - 'auto': Automatically select the best model
                - 'arima': Use ARIMA model
                - 'ets': Use Exponential Smoothing
                - 'ensemble': Use an ensemble of multiple models

        Returns:
            Dictionary with forecast results
        """
        self.logger.info(f"Forecasting {', '.join(metrics)} for the next {days_to_forecast} days")

        # Set the forecast horizon
        self.forecast_horizon = days_to_forecast

        try:
            # Fetch historical data if needed
            history_key = f"{'all' if campaign_id is None else campaign_id}_day"
            if history_key not in self.metrics_history:
                # Fetch at least 3x the forecast horizon for good model training
                training_days = max(self.training_window, days_to_forecast * 3)
                self.fetch_historical_data(metrics, training_days, campaign_id, "day")

            # Get the historical data
            df = self.metrics_history[history_key]

            # Check if we have enough data
            min_training_rows = max(
                30, days_to_forecast * 2
            )  # At least 30 days or 2x forecast horizon
            if len(df) < min_training_rows:
                self.logger.warning(
                    f"Not enough historical data for reliable forecasting. "
                    f"Have {len(df)} rows, need at least {min_training_rows}."
                )

            # Initialize results
            results = {
                "metrics": {},
                "campaign_id": campaign_id,
                "forecast_horizon": days_to_forecast,
                "confidence_level": self.confidence_level,
                "forecast_date": datetime.now().isoformat(),
                "models_used": {},
            }

            # Forecast each metric
            for metric in metrics:
                if metric not in df.columns:
                    self.logger.warning(f"Metric '{metric}' not in historical data, skipping")
                    continue

                # Get the time series for this metric
                series = df[metric]

                # Create a model ID for this forecast
                model_id = f"{metric}_{'all' if campaign_id is None else campaign_id}"

                # Get the best model for this metric
                if model_type == "auto":
                    # Try different models and select the best
                    model_results = self._train_and_evaluate_models(
                        series, model_id, metric, campaign_id
                    )
                    best_model = model_results["best_model"]
                    best_model_type = model_results["best_model_type"]
                    results["models_used"][metric] = best_model_type
                elif model_type == "ensemble":
                    # Use an ensemble of multiple models
                    forecast, lower, upper = self._ensemble_forecast(series, days_to_forecast)
                    results["models_used"][metric] = "ensemble"
                else:
                    # Use the specified model type
                    best_model = self._create_model(model_type, series)
                    best_model_type = model_type
                    results["models_used"][metric] = model_type

                # Generate forecast
                if model_type != "ensemble":
                    forecast, lower, upper = self._generate_forecast(
                        best_model, best_model_type, series, days_to_forecast
                    )

                # Convert forecast to a list for JSON serialization
                forecast_dates = [
                    (datetime.now().date() + timedelta(days=i + 1)).isoformat()
                    for i in range(days_to_forecast)
                ]

                results["metrics"][metric] = {
                    "forecast": forecast.tolist() if isinstance(forecast, np.ndarray) else forecast,
                    "lower_bound": lower.tolist() if isinstance(lower, np.ndarray) else lower,
                    "upper_bound": upper.tolist() if isinstance(upper, np.ndarray) else upper,
                    "dates": forecast_dates,
                }

            # Store the forecast results
            self.forecasts[
                f"{'all' if campaign_id is None else campaign_id}_{datetime.now().strftime('%Y%m%d')}"
            ] = results

            # Save models and forecasts
            self._save_models()

            return results

        except Exception as e:
            self.logger.error(f"Error generating forecasts: {str(e)}")
            raise

    def _train_and_evaluate_models(self, series, model_id, metric, campaign_id=None):
        """
        Train and evaluate multiple forecasting models to find the best one.

        Args:
            series: Time series data
            model_id: Unique ID for the model
            metric: Name of the metric being forecast
            campaign_id: Optional campaign ID

        Returns:
            Dictionary with best model information
        """
        # Split data into training and validation sets
        train_size = int(len(series) * 0.8)
        train, validation = series[:train_size], series[train_size:]
        validation_horizon = len(validation)

        models_to_try = {
            "arima": self._create_model("arima", train),
            "ets": self._create_model("ets", train),
        }

        # Evaluate each model
        performance = {}
        for model_name, model in models_to_try.items():
            try:
                # Generate forecast on validation data
                forecast, _, _ = self._generate_forecast(
                    model, model_name, train, validation_horizon
                )

                # Calculate error metrics
                mse = np.mean((validation - forecast[:validation_horizon]) ** 2)
                mae = np.mean(np.abs(validation - forecast[:validation_horizon]))
                mape = (
                    np.mean(np.abs((validation - forecast[:validation_horizon]) / validation)) * 100
                    if np.all(validation != 0)
                    else np.inf
                )

                performance[model_name] = {
                    "mse": mse,
                    "mae": mae,
                    "mape": mape if not np.isinf(mape) else None,
                }

            except Exception as e:
                self.logger.warning(f"Error evaluating {model_name} model: {str(e)}")
                performance[model_name] = {"mse": np.inf, "mae": np.inf, "mape": np.inf}

        # Select the best model based on MSE
        best_model_type = min(performance.items(), key=lambda x: x[1]["mse"])[0]

        # Retrain the best model on the full dataset
        best_model = self._create_model(best_model_type, series)

        # Store the model
        self.models[model_id] = {
            "model": best_model,
            "type": best_model_type,
            "metric": metric,
            "campaign_id": campaign_id,
            "last_trained": datetime.now().isoformat(),
            "performance": performance[best_model_type],
        }

        return {
            "best_model": best_model,
            "best_model_type": best_model_type,
            "performance": performance,
        }

    def _create_model(self, model_type, series):
        """
        Create a time series forecasting model.

        Args:
            model_type: Type of model to create
            series: Time series data to train on

        Returns:
            Trained forecasting model
        """
        if model_type == "arima":
            # Auto ARIMA-like behavior: use a default ARIMA(1,1,1) model which works well
            # for many time series
            model = ARIMA(series, order=(1, 1, 1))
            return model.fit()

        elif model_type == "ets":
            # Exponential Smoothing model
            model = ExponentialSmoothing(
                series,
                trend="add",
                seasonal="add",
                seasonal_periods=7,  # Weekly seasonality is common in Ads data
            )
            return model.fit()

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _generate_forecast(self, model, model_type, series, horizon):
        """
        Generate a forecast using the specified model.

        Args:
            model: Trained forecasting model
            model_type: Type of model
            series: Historical time series data
            horizon: Number of periods to forecast

        Returns:
            Tuple of (forecast, lower_bound, upper_bound)
        """
        if model_type == "arima":
            # Generate forecast
            forecast_result = model.forecast(steps=horizon)
            forecast = forecast_result

            # Compute confidence intervals
            pred_ci = model.get_forecast(steps=horizon).conf_int(alpha=1 - self.confidence_level)
            lower = pred_ci.iloc[:, 0].values
            upper = pred_ci.iloc[:, 1].values

        elif model_type == "ets":
            # Generate forecast
            forecast_result = model.forecast(steps=horizon)
            forecast = forecast_result.values

            # Compute approximate confidence intervals
            # This is a simplified approach - ETS in statsmodels doesn't directly provide confidence intervals
            residuals = model.resid
            residual_std = np.std(residuals)
            z_value = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)

            margin = z_value * residual_std * np.sqrt(np.arange(1, horizon + 1))
            lower = forecast - margin
            upper = forecast + margin

        # Ensure no negative values for metrics that can't be negative
        forecast = np.maximum(forecast, 0)
        lower = np.maximum(lower, 0)

        return forecast, lower, upper

    def _ensemble_forecast(self, series, horizon):
        """
        Generate an ensemble forecast by combining multiple models.

        Args:
            series: Historical time series data
            horizon: Number of periods to forecast

        Returns:
            Tuple of (forecast, lower_bound, upper_bound)
        """
        # Train individual models
        arima_model = self._create_model("arima", series)
        ets_model = self._create_model("ets", series)

        # Generate forecasts from each model
        arima_forecast, arima_lower, arima_upper = self._generate_forecast(
            arima_model, "arima", series, horizon
        )
        ets_forecast, ets_lower, ets_upper = self._generate_forecast(
            ets_model, "ets", series, horizon
        )

        # Simple ensemble: average the forecasts
        forecast = (arima_forecast + ets_forecast) / 2

        # Use the wider confidence interval for conservatism
        lower = np.minimum(arima_lower, ets_lower)
        upper = np.maximum(arima_upper, ets_upper)

        return forecast, lower, upper

    def forecast_budget(
        self,
        campaign_id: str,
        days_to_forecast: int = 30,
        target_metric: str = "conversions",
        target_value: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Forecast the budget required to achieve target metrics.

        Args:
            campaign_id: Campaign ID to forecast budget for
            days_to_forecast: Number of days to forecast into the future
            target_metric: Metric to target (e.g., 'conversions', 'clicks')
            target_value: Optional target value to achieve

        Returns:
            Dictionary with budget forecast results
        """
        self.logger.info(
            f"Forecasting budget for campaign {campaign_id} for the next {days_to_forecast} days"
        )

        try:
            # Fetch historical data for this campaign
            metrics = ["cost", "clicks", "conversions", "impressions"]
            history_key = f"{campaign_id}_day"

            if history_key not in self.metrics_history:
                self.fetch_historical_data(metrics, self.training_window, campaign_id, "day")

            # Get historical data
            df = self.metrics_history[history_key]

            # Check for required metrics
            required_metrics = ["cost", target_metric]
            for metric in required_metrics:
                if metric not in df.columns:
                    raise ValueError(f"Required metric '{metric}' not found in historical data")

            # Calculate historical performance ratios
            if target_metric == "conversions":
                # Cost per conversion
                if "conversions" in df.columns and df["conversions"].sum() > 0:
                    cpa = df["cost"].sum() / df["conversions"].sum()
                else:
                    cpa = None

                # Conversion rate
                if "clicks" in df.columns and df["clicks"].sum() > 0:
                    conv_rate = df["conversions"].sum() / df["clicks"].sum()
                else:
                    conv_rate = None

            # Calculate click-through rate
            if (
                "clicks" in df.columns
                and "impressions" in df.columns
                and df["impressions"].sum() > 0
            ):
                ctr = df["clicks"].sum() / df["impressions"].sum()
            else:
                ctr = None

            # Calculate cost per click
            if "clicks" in df.columns and df["clicks"].sum() > 0:
                cpc = df["cost"].sum() / df["clicks"].sum()
            else:
                cpc = None

            # Forecast the target metric
            metric_forecast = self.forecast_metrics([target_metric], days_to_forecast, campaign_id)

            # Calculate required budget based on forecasted metrics and historical ratios
            if target_value is not None:
                # Calculate budget needed to achieve the target value
                if target_metric == "conversions" and cpa is not None:
                    required_budget = target_value * cpa
                elif target_metric == "clicks" and cpc is not None:
                    required_budget = target_value * cpc
                else:
                    required_budget = None
            else:
                # Calculate budget based on forecasted values
                forecasted_values = metric_forecast["metrics"][target_metric]["forecast"]
                total_forecasted = sum(forecasted_values)

                if target_metric == "conversions" and cpa is not None:
                    required_budget = total_forecasted * cpa
                elif target_metric == "clicks" and cpc is not None:
                    required_budget = total_forecasted * cpc
                else:
                    required_budget = None

            # Forecast the cost directly
            cost_forecast = self.forecast_metrics(["cost"], days_to_forecast, campaign_id)

            # Prepare the result
            result = {
                "campaign_id": campaign_id,
                "forecast_horizon": days_to_forecast,
                "forecast_date": datetime.now().isoformat(),
                "historical_performance": {
                    "cpc": cpc,
                    "ctr": ctr,
                    "cpa": cpa,
                    "conversion_rate": conv_rate,
                },
                "metric_forecast": metric_forecast["metrics"][target_metric],
                "cost_forecast": cost_forecast["metrics"]["cost"],
                "total_forecasted_cost": sum(cost_forecast["metrics"]["cost"]["forecast"]),
            }

            if required_budget is not None:
                result["required_budget_for_target"] = required_budget

                # Calculate daily budget
                result["suggested_daily_budget"] = required_budget / days_to_forecast

            return result

        except Exception as e:
            self.logger.error(f"Error forecasting budget: {str(e)}")
            raise

    def detect_search_trends(
        self, days_lookback: int = 90, min_growth_rate: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detect emerging search trends based on historical performance data.

        Args:
            days_lookback: Number of days of historical data to analyze
            min_growth_rate: Minimum growth rate to consider a trend

        Returns:
            Dictionary with detected trends
        """
        self.logger.info(f"Detecting search trends from the last {days_lookback} days")

        try:
            # Get performance data for all campaigns
            metrics = ["impressions", "clicks"]
            self.fetch_historical_data(metrics, days_lookback, None, "day")

            # We need keyword-level data for trend detection
            if not self.ads_api:
                raise ValueError("Google Ads API client not initialized")

            # Get keyword performance data
            keyword_data = self.ads_api.get_keyword_performance(days_lookback)

            # Convert to DataFrame
            df = pd.DataFrame(keyword_data)

            # Check if we have the necessary data
            if "keyword_text" not in df.columns:
                raise ValueError("Keyword text not found in the retrieved data")

            # Ensure we have a date column
            if "day" not in df.columns and "date" not in df.columns:
                raise ValueError("Date column not found in the retrieved data")

            date_col = "day" if "day" in df.columns else "date"

            # Convert date column to datetime if it's not already
            if df[date_col].dtype != "datetime64[ns]":
                df[date_col] = pd.to_datetime(df[date_col])

            # Calculate the number of days to split for trend analysis
            # We'll compare recent performance to earlier performance
            split_days = days_lookback // 3
            recent_cutoff = datetime.now().date() - timedelta(days=split_days)

            # Split data into recent and earlier periods
            df_recent = df[df[date_col].dt.date >= recent_cutoff]
            df_earlier = df[df[date_col].dt.date < recent_cutoff]

            # Group by keyword to get aggregate performance
            recent_keywords = (
                df_recent.groupby("keyword_text")
                .agg({"impressions": "sum", "clicks": "sum", "cost": "sum"})
                .reset_index()
            )

            earlier_keywords = (
                df_earlier.groupby("keyword_text")
                .agg({"impressions": "sum", "clicks": "sum", "cost": "sum"})
                .reset_index()
            )

            # Merge to compare performance
            keywords = recent_keywords.merge(
                earlier_keywords, on="keyword_text", how="outer", suffixes=("_recent", "_earlier")
            ).fillna(0)

            # Calculate growth rates
            keywords["impressions_growth"] = (
                keywords["impressions_recent"] - keywords["impressions_earlier"]
            ) / (
                keywords["impressions_earlier"] + 1
            )  # Add 1 to avoid division by zero

            keywords["clicks_growth"] = (keywords["clicks_recent"] - keywords["clicks_earlier"]) / (
                keywords["clicks_earlier"] + 1
            )

            # Identify emerging trends (keywords with significant growth)
            trending_keywords = keywords[
                (keywords["impressions_growth"] >= min_growth_rate)
                & (keywords["impressions_recent"] >= 100)  # Min threshold to filter out noise
            ].sort_values("impressions_growth", ascending=False)

            # Identify declining trends
            declining_keywords = keywords[
                (keywords["impressions_growth"] <= -min_growth_rate)
                & (keywords["impressions_earlier"] >= 100)
            ].sort_values("impressions_growth", ascending=True)

            # Prepare results
            result = {
                "analysis_date": datetime.now().isoformat(),
                "days_analyzed": days_lookback,
                "trending_keywords": [],
                "declining_keywords": [],
                "stable_volume_keywords": [],
            }

            # Add trending keywords
            for _, row in trending_keywords.head(20).iterrows():
                result["trending_keywords"].append(
                    {
                        "keyword": row["keyword_text"],
                        "recent_impressions": int(row["impressions_recent"]),
                        "recent_clicks": int(row["clicks_recent"]),
                        "growth_rate": float(row["impressions_growth"]),
                        "click_growth_rate": float(row["clicks_growth"]),
                    }
                )

            # Add declining keywords
            for _, row in declining_keywords.head(20).iterrows():
                result["declining_keywords"].append(
                    {
                        "keyword": row["keyword_text"],
                        "recent_impressions": int(row["impressions_recent"]),
                        "earlier_impressions": int(row["impressions_earlier"]),
                        "decline_rate": float(row["impressions_growth"]),
                    }
                )

            # Identify stable high-volume keywords
            stable_keywords = keywords[
                (abs(keywords["impressions_growth"]) < min_growth_rate)
                & (keywords["impressions_recent"] >= 500)
            ].sort_values("impressions_recent", ascending=False)

            for _, row in stable_keywords.head(20).iterrows():
                result["stable_volume_keywords"].append(
                    {
                        "keyword": row["keyword_text"],
                        "impressions": int(row["impressions_recent"]),
                        "clicks": int(row["clicks_recent"]),
                    }
                )

            return result

        except Exception as e:
            self.logger.error(f"Error detecting search trends: {str(e)}")
            raise

    def get_demand_forecasts(self) -> Dict[str, Any]:
        """
        Retrieve demand forecasts from Google Ads Insights API.

        Returns:
            Dictionary with demand forecast information
        """
        self.logger.info("Retrieving demand forecasts from Google Ads Insights API")

        try:
            # This requires the Google Ads API client to support the Insights API
            if not self.ads_api:
                raise ValueError("Google Ads API client not initialized")

            # Check if the API client has the method to get demand forecasts
            if not hasattr(self.ads_api, "get_demand_forecasts"):
                # If not available, we'll return a warning
                return {
                    "status": "warning",
                    "message": "Demand forecasts feature not supported by the current Google Ads API client",
                    "documentation_url": "https://support.google.com/google-ads/answer/10787044",
                }

            # If supported, get the demand forecasts
            # Note: This is a placeholder - the actual implementation would depend
            # on the specific API method provided by the Google Ads API client
            demand_forecasts = self.ads_api.get_demand_forecasts()

            return {
                "status": "success",
                "forecast_date": datetime.now().isoformat(),
                "demand_forecasts": demand_forecasts,
            }

        except Exception as e:
            self.logger.error(f"Error retrieving demand forecasts: {str(e)}")
            return {"status": "error", "message": str(e)}

    def run(self, **kwargs):
        """
        Run the forecasting service's primary functionality.

        Args:
            **kwargs: Additional parameters for specific operations
                - operation: The specific operation to run
                    - "forecast_metrics": Forecast performance metrics
                    - "forecast_budget": Forecast budget requirements
                    - "detect_trends": Detect search trends
                    - "demand_forecasts": Get demand forecasts from Google Ads

        Returns:
            Results dictionary
        """
        operation = kwargs.get("operation", "forecast_metrics")

        if operation == "forecast_metrics":
            metrics = kwargs.get("metrics", ["clicks", "impressions", "conversions"])
            days = kwargs.get("days", self.forecast_horizon)
            campaign_id = kwargs.get("campaign_id")
            model_type = kwargs.get("model_type", "auto")

            return self.forecast_metrics(metrics, days, campaign_id, model_type)

        elif operation == "forecast_budget":
            campaign_id = kwargs.get("campaign_id")
            days = kwargs.get("days", 30)
            target_metric = kwargs.get("target_metric", "conversions")
            target_value = kwargs.get("target_value")

            if not campaign_id:
                return {"status": "error", "message": "Campaign ID is required"}

            return self.forecast_budget(campaign_id, days, target_metric, target_value)

        elif operation == "detect_trends":
            days_lookback = kwargs.get("days_lookback", 90)
            min_growth_rate = kwargs.get("min_growth_rate", 0.1)

            return self.detect_search_trends(days_lookback, min_growth_rate)

        elif operation == "demand_forecasts":
            return self.get_demand_forecasts()

        else:
            return {"status": "error", "message": f"Unknown operation: {operation}"}
