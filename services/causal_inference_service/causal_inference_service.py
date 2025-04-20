"""
Causal Inference Service for Google Ads campaigns.

This service uses causal inference techniques to measure the true impact of campaign changes
and experiments on key performance metrics.
"""

import logging
import os
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt  # type: ignore

# Import causal inference libraries
try:
    from causalimpact import CausalImpact  # type: ignore

    CAUSAL_IMPACT_AVAILABLE = True
except ImportError:
    CAUSAL_IMPACT_AVAILABLE = False
    logging.warning("CausalImpact not available. Install with: pip install pycausalimpact")

try:
    import statsmodels.api as sm  # type: ignore
    import statsmodels.formula.api as smf  # type: ignore

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("statsmodels not available. Install with: pip install statsmodels")

try:
    import econml  # type: ignore
    from econml.dml import CausalForestDML  # type: ignore

    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    logging.warning("EconML not available. Install with: pip install econml")

from services.base_service import BaseService

logger = logging.getLogger(__name__)


class CausalInferenceService(BaseService):
    """Service for performing causal inference analysis on Google Ads campaigns."""

    def __init__(self, ads_client: Any, config: Dict[str, Any]) -> None:
        """
        Initialize the CausalInferenceService.

        Args:
            ads_client: The Google Ads API client
            config: Configuration dictionary
        """
        super().__init__(ads_client, config)
        self.min_pre_period_days = config.get("min_pre_period_days", 30)
        self.min_post_period_days = config.get("min_post_period_days", 14)
        self.significance_level = config.get("significance_level", 0.05)

    def analyze_campaign_change_impact(
        self,
        campaign_id: str,
        change_date: datetime,
        metric: str,
        control_campaigns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze the causal impact of a campaign change.

        Args:
            campaign_id: ID of the campaign that was changed
            change_date: Date when the change was made
            metric: Metric to analyze (e.g., 'clicks', 'conversions', 'cost')
            control_campaigns: Optional list of control campaign IDs

        Returns:
            Dict containing analysis results including:
            - estimated_effect: The estimated effect size
            - confidence_interval: (lower, upper) bounds
            - p_value: Statistical significance
            - is_significant: Boolean indicating if effect is significant
        """
        try:
            # Get pre-period data
            pre_period_start = change_date - timedelta(days=self.min_pre_period_days)
            pre_period_data = self._get_campaign_data(
                campaign_id, pre_period_start, change_date, metric
            )

            # Get post-period data
            post_period_end = change_date + timedelta(days=self.min_post_period_days)
            post_period_data = self._get_campaign_data(
                campaign_id, change_date, post_period_end, metric
            )

            # Get control campaign data if provided
            control_data = None
            if control_campaigns:
                control_data = self._get_control_campaign_data(
                    control_campaigns, pre_period_start, post_period_end, metric
                )

            # Perform causal impact analysis
            impact = self._run_causal_impact_analysis(
                pre_period_data, post_period_data, control_data
            )

            return {
                "estimated_effect": impact.get("estimated_effect", 0),
                "confidence_interval": impact.get("confidence_interval", (0, 0)),
                "p_value": impact.get("p_value", 1.0),
                "is_significant": impact.get("is_significant", False),
                "relative_effect": impact.get("relative_effect", 0),
                "report": impact.get("report", ""),
            }

        except Exception as e:
            logger.error(f"Error analyzing campaign change impact: {str(e)}")
            return {
                "error": str(e),
                "estimated_effect": 0,
                "confidence_interval": (0, 0),
                "p_value": 1.0,
                "is_significant": False,
            }

    def _get_campaign_data(
        self, campaign_id: str, start_date: datetime, end_date: datetime, metric: str
    ) -> pd.DataFrame:
        """
        Get campaign performance data for the specified period.

        Args:
            campaign_id: Campaign ID
            start_date: Start date for data collection
            end_date: End date for data collection
            metric: Metric to collect

        Returns:
            DataFrame with daily metric values
        """
        try:
            # Query the Google Ads API for campaign performance data
            query = f"""
                SELECT
                    segments.date,
                    metrics.{metric}
                FROM campaign
                WHERE 
                    campaign.id = {campaign_id}
                    AND segments.date BETWEEN '{start_date.strftime('%Y-%m-%d')}'
                    AND '{end_date.strftime('%Y-%m-%d')}'
                ORDER BY segments.date
            """

            response = self.ads_client.get_service("GoogleAdsService").search(
                customer_id=self.customer_id, query=query
            )

            # Convert response to DataFrame
            data = []
            for row in response:
                data.append({"date": row.segments.date, metric: getattr(row.metrics, metric)})

            return pd.DataFrame(data)

        except Exception as e:
            logger.error(f"Error getting campaign data: {str(e)}")
            raise

    def _get_control_campaign_data(
        self, campaign_ids: List[str], start_date: datetime, end_date: datetime, metric: str
    ) -> pd.DataFrame:
        """
        Get performance data for control campaigns.

        Args:
            campaign_ids: List of control campaign IDs
            start_date: Start date for data collection
            end_date: End date for data collection
            metric: Metric to collect

        Returns:
            DataFrame with daily metric values for control campaigns
        """
        try:
            # Query the Google Ads API for control campaign data
            campaign_id_str = ",".join(campaign_ids)
            query = f"""
                SELECT
                    segments.date,
                    campaign.id,
                    metrics.{metric}
                FROM campaign
                WHERE 
                    campaign.id IN ({campaign_id_str})
                    AND segments.date BETWEEN '{start_date.strftime('%Y-%m-%d')}'
                    AND '{end_date.strftime('%Y-%m-%d')}'
                ORDER BY segments.date
            """

            response = self.ads_client.get_service("GoogleAdsService").search(
                customer_id=self.customer_id, query=query
            )

            # Convert response to DataFrame
            data = []
            for row in response:
                data.append(
                    {
                        "date": row.segments.date,
                        "campaign_id": row.campaign.id,
                        metric: getattr(row.metrics, metric),
                    }
                )

            return pd.DataFrame(data)

        except Exception as e:
            logger.error(f"Error getting control campaign data: {str(e)}")
            raise

    def _run_causal_impact_analysis(
        self,
        pre_data: pd.DataFrame,
        post_data: pd.DataFrame,
        control_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Run causal impact analysis using the CausalImpact package.

        Args:
            pre_data: Pre-intervention period data
            post_data: Post-intervention period data
            control_data: Optional control campaign data

        Returns:
            Dictionary containing analysis results
        """
        try:
            # Combine pre and post data
            data = pd.concat([pre_data, post_data])

            # Set up the pre/post periods
            pre_period = [0, len(pre_data) - 1]
            post_period = [len(pre_data), len(data) - 1]

            # Create the CausalImpact model
            ci = CausalImpact(
                data, pre_period, post_period, model_args={"niter": 1000, "standardize_data": True}
            )

            # Extract results
            summary = ci.summary()
            report = ci.summary(output="report")

            return {
                "estimated_effect": summary["AbsEffect"][0],
                "confidence_interval": (
                    summary["AbsEffect.lower"][0],
                    summary["AbsEffect.upper"][0],
                ),
                "p_value": summary["p"][0],
                "is_significant": summary["p"][0] < self.significance_level,
                "relative_effect": summary["RelEffect"][0],
                "report": report,
            }

        except Exception as e:
            logger.error(f"Error running causal impact analysis: {str(e)}")
            raise

    def measure_treatment_effect(
        self,
        pre_period: List[str],
        post_period: List[str],
        control_data: pd.DataFrame,
        treatment_data: pd.DataFrame,
        metric: str,
        model_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Measure the causal impact of an intervention (e.g., a campaign change)
        using Google's CausalImpact package.

        Args:
            pre_period: [start_date, end_date] for the pre-intervention period.
            post_period: [start_date, end_date] for the post-intervention period.
            control_data: DataFrame with time series data for control group(s).
                          Must have a datetime index and columns for relevant metrics.
            treatment_data: DataFrame with time series data for the treated group.
                            Must have a datetime index and a column for the target metric.
            metric: The name of the metric column in treatment_data to analyze.
            model_args: Optional arguments to pass to the CausalImpact model.

        Returns:
            Dictionary containing the causal impact analysis summary.
        """
        start_time = datetime.now()
        self.logger.info(f"Measuring causal impact for metric '{metric}'...")

        if not CAUSAL_IMPACT_AVAILABLE:
            error_msg = (
                "CausalImpact package not available. Install with: pip install pycausalimpact"
            )
            self.logger.error(error_msg)
            return {"status": "error", "message": error_msg}

        # Validate inputs
        if not isinstance(pre_period, list) or len(pre_period) != 2:
            return {
                "status": "error",
                "message": "pre_period must be a list of [start_date, end_date]",
            }

        if not isinstance(post_period, list) or len(post_period) != 2:
            return {
                "status": "error",
                "message": "post_period must be a list of [start_date, end_date]",
            }

        try:
            # Prepare the data for CausalImpact
            # Ensure datetime index
            if not isinstance(treatment_data.index, pd.DatetimeIndex):
                try:
                    treatment_data.index = pd.to_datetime(treatment_data.index)
                except Exception as e:
                    return {
                        "status": "error",
                        "message": f"Cannot convert treatment_data index to datetime: {str(e)}",
                    }

            if not isinstance(control_data.index, pd.DatetimeIndex):
                try:
                    control_data.index = pd.to_datetime(control_data.index)
                except Exception as e:
                    return {
                        "status": "error",
                        "message": f"Cannot convert control_data index to datetime: {str(e)}",
                    }

            # Combine control and treatment data
            # The first column should be the treated time series (metric of interest)
            if metric not in treatment_data.columns:
                return {
                    "status": "error",
                    "message": f"Metric '{metric}' not found in treatment_data columns",
                }

            all_data = pd.concat([treatment_data[[metric]], control_data], axis=1)

            # Convert period dates to datetime if they're strings
            if isinstance(pre_period[0], str):
                pre_period_dt = [pd.to_datetime(date) for date in pre_period]
            else:
                pre_period_dt = pre_period

            if isinstance(post_period[0], str):
                post_period_dt = [pd.to_datetime(date) for date in post_period]
            else:
                post_period_dt = post_period

            # Set default model arguments
            default_args = self.config["default_model_args"].copy()
            if model_args:
                default_args.update(model_args)

            # Run CausalImpact
            ci = CausalImpact(all_data, pre_period_dt, post_period_dt, **default_args)

            # Get summary and results
            summary_data = ci.summary_data
            report = ci.summary(output="report")

            # Extract key metrics from the summary
            absolute_effect = summary_data.iloc[0]["abs_effect"]
            relative_effect = summary_data.iloc[0]["rel_effect"]
            p_value = summary_data.iloc[0]["p"]

            # Get confidence intervals
            ci_lower = summary_data.iloc[0]["abs_effect_lower"]
            ci_upper = summary_data.iloc[0]["abs_effect_upper"]

            # Save the plot if output_dir is configured
            plot_path = None
            if self.config["output_dir"]:
                plot = ci.plot()
                plot_filename = (
                    f"{metric}_causal_impact_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                plot_path = os.path.join(self.config["output_dir"], plot_filename)
                plt.savefig(plot_path)
                plt.close()

            # Create analysis ID and store results
            analysis_id = f"ci_{metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            analysis_results = {
                "id": analysis_id,
                "status": "success",
                "method": "causal_impact",
                "pre_period": pre_period,
                "post_period": post_period,
                "metric": metric,
                "estimated_absolute_effect": float(absolute_effect),
                "estimated_relative_effect": float(relative_effect),
                "p_value": float(p_value),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "report": report,
                "plot_path": plot_path,
                "timestamp": datetime.now().isoformat(),
            }

            # Store the analysis
            self.analyses[analysis_id] = analysis_results

            self.logger.info(
                f"Causal impact analysis completed. Absolute effect: {absolute_effect:.2f}, p-value: {p_value:.4f}"
            )

        except Exception as e:
            self.logger.error(f"Error during causal impact analysis: {str(e)}")
            analysis_results = {
                "status": "error",
                "method": "causal_impact",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            self._track_execution(start_time, success=False)
            return analysis_results

        self._track_execution(start_time, success=True)
        return analysis_results

    def difference_in_differences(
        self,
        data: pd.DataFrame,
        time_col: str,
        treatment_col: str,
        outcome_col: str,
        pre_post_col: str = None,
        intervention_time: Union[str, datetime] = None,
        covariates: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform difference-in-differences (DiD) analysis to measure causal effects.

        Args:
            data: DataFrame containing panel data for treatment and control groups
            time_col: Name of the column containing time information
            treatment_col: Name of the column indicating treatment assignment (0=control, 1=treatment)
            outcome_col: Name of the column containing the outcome metric
            pre_post_col: Name of the column indicating pre/post periods (optional, will be created from intervention_time if not provided)
            intervention_time: The time when intervention occurred (optional if pre_post_col is provided)
            covariates: List of covariate columns to include in the model (optional)

        Returns:
            Dictionary with DiD analysis results
        """
        start_time = datetime.now()
        self.logger.info(
            f"Running difference-in-differences analysis for outcome '{outcome_col}'..."
        )

        if not STATSMODELS_AVAILABLE:
            error_msg = "statsmodels package not available. Install with: pip install statsmodels"
            self.logger.error(error_msg)
            return {"status": "error", "message": error_msg}

        try:
            # Make a copy to avoid modifying the original data
            df = data.copy()

            # Ensure time column is datetime
            if pd.api.types.is_string_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col])

            # Create pre/post indicator if not provided
            if pre_post_col is None:
                if intervention_time is None:
                    return {
                        "status": "error",
                        "message": "Either pre_post_col or intervention_time must be provided",
                    }

                if isinstance(intervention_time, str):
                    intervention_time = pd.to_datetime(intervention_time)

                pre_post_col = "post_treatment"
                df[pre_post_col] = (df[time_col] >= intervention_time).astype(int)

            # Create interaction term
            df["did_interaction"] = df[treatment_col] * df[pre_post_col]

            # Prepare formula for regression
            covariates_formula = "" if not covariates else " + " + " + ".join(covariates)
            formula = f"{outcome_col} ~ {treatment_col} + {pre_post_col} + did_interaction{covariates_formula}"

            # Run the DiD regression
            model = smf.ols(formula=formula, data=df)
            results = model.fit(cov_type="HC3")  # Using robust standard errors

            # Extract key results
            did_effect = results.params["did_interaction"]
            p_value = results.pvalues["did_interaction"]
            ci = results.conf_int().loc["did_interaction"].tolist()

            # Create visualization
            plot_path = None
            if VISUALIZATION_AVAILABLE:
                # Prepare data for plotting
                agg_data = df.groupby([time_col, treatment_col])[outcome_col].mean().reset_index()

                # Create time series plot
                plt.figure(figsize=(12, 6))

                # Plot treatment group
                treatment_data = agg_data[agg_data[treatment_col] == 1]
                plt.plot(
                    treatment_data[time_col],
                    treatment_data[outcome_col],
                    "b-",
                    label="Treatment Group",
                )

                # Plot control group
                control_data = agg_data[agg_data[treatment_col] == 0]
                plt.plot(
                    control_data[time_col], control_data[outcome_col], "r-", label="Control Group"
                )

                # Add vertical line at intervention time
                if intervention_time:
                    plt.axvline(
                        x=intervention_time, color="k", linestyle="--", label="Intervention Time"
                    )

                plt.title(f"Difference-in-Differences: {outcome_col}")
                plt.xlabel("Time")
                plt.ylabel(outcome_col)
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Save the plot
                plot_filename = f"did_{outcome_col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plot_path = os.path.join(self.config["output_dir"], plot_filename)
                plt.savefig(plot_path)
                plt.close()

            # Create analysis ID and store results
            analysis_id = f"did_{outcome_col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            analysis_results = {
                "id": analysis_id,
                "status": "success",
                "method": "difference_in_differences",
                "outcome_metric": outcome_col,
                "estimated_effect": float(did_effect),
                "p_value": float(p_value),
                "ci_lower": float(ci[0]),
                "ci_upper": float(ci[1]),
                "formula": formula,
                "model_summary": results.summary().as_text(),
                "params": {k: float(v) for k, v in results.params.items()},
                "plot_path": plot_path,
                "timestamp": datetime.now().isoformat(),
            }

            # Check statistical significance
            if p_value < 0.05:
                if did_effect > 0:
                    analysis_results["interpretation"] = (
                        "Positive statistically significant effect detected"
                    )
                else:
                    analysis_results["interpretation"] = (
                        "Negative statistically significant effect detected"
                    )
            else:
                analysis_results["interpretation"] = "No statistically significant effect detected"

            # Store the analysis
            self.analyses[analysis_id] = analysis_results

            self.logger.info(
                f"DiD analysis completed. Effect: {did_effect:.4f}, p-value: {p_value:.4f}"
            )

        except Exception as e:
            self.logger.error(f"Error during difference-in-differences analysis: {str(e)}")
            analysis_results = {
                "status": "error",
                "method": "difference_in_differences",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            self._track_execution(start_time, success=False)
            return analysis_results

        self._track_execution(start_time, success=True)
        return analysis_results

    def create_synthetic_control(
        self,
        data: pd.DataFrame,
        treatment_unit: str,
        control_units: List[str],
        unit_column: str,
        time_column: str,
        outcome_column: str,
        pre_treatment_end: Union[str, datetime],
        post_treatment_start: Union[str, datetime],
        covariates: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a synthetic control to evaluate causal effects when randomized control is not possible.

        Args:
            data: DataFrame containing panel data
            treatment_unit: Identifier for the treated unit
            control_units: List of identifiers for potential control units
            unit_column: Column name identifying units
            time_column: Column name identifying time periods
            outcome_column: Column name for the outcome variable
            pre_treatment_end: End of pre-treatment period
            post_treatment_start: Start of post-treatment period
            covariates: Optional list of covariates to match on

        Returns:
            Dictionary with synthetic control analysis results
        """
        start_time = datetime.now()
        self.logger.info(f"Creating synthetic control for {treatment_unit}...")

        if not STATSMODELS_AVAILABLE:
            error_msg = "statsmodels package not available. Install with: pip install statsmodels"
            self.logger.error(error_msg)
            return {"status": "error", "message": error_msg}

        try:
            # Convert date strings to datetime objects if needed
            if isinstance(pre_treatment_end, str):
                pre_treatment_end = pd.to_datetime(pre_treatment_end)
            if isinstance(post_treatment_start, str):
                post_treatment_start = pd.to_datetime(post_treatment_start)

            # Ensure time column is datetime
            data_copy = data.copy()
            if pd.api.types.is_string_dtype(data_copy[time_column]):
                data_copy[time_column] = pd.to_datetime(data_copy[time_column])

            # Split data into treatment and donor pool
            treatment_data = data_copy[data_copy[unit_column] == treatment_unit].copy()
            control_data = data_copy[data_copy[unit_column].isin(control_units)].copy()

            # Split into pre and post periods
            pre_treatment = data_copy[data_copy[time_column] <= pre_treatment_end].copy()
            post_treatment = data_copy[data_copy[time_column] >= post_treatment_start].copy()

            # Calculate pre-treatment averages for outcome and covariates
            pre_treat_means = (
                pre_treatment.groupby(unit_column)
                .agg(
                    {
                        outcome_column: "mean",
                        **(({col: "mean" for col in covariates}) if covariates else {}),
                    }
                )
                .reset_index()
            )

            # Calculate distance from treatment unit to each control unit
            treatment_means = pre_treat_means[pre_treat_means[unit_column] == treatment_unit].iloc[
                0
            ]

            distances = []
            for _, row in pre_treat_means[
                pre_treat_means[unit_column] != treatment_unit
            ].iterrows():
                # Calculate Euclidean distance based on outcome and covariates
                features = [outcome_column] + (covariates if covariates else [])
                distance = np.sqrt(np.sum([(treatment_means[f] - row[f]) ** 2 for f in features]))
                distances.append({"unit": row[unit_column], "distance": distance})

            # Sort by distance and select top N units for synthetic control
            distances_df = pd.DataFrame(distances)
            distances_df = distances_df.sort_values("distance")
            top_units = distances_df.head(min(5, len(distances_df)))["unit"].tolist()

            # Create synthetic control by weighting closest units
            # Simple inverse distance weighting approach
            total_inverse_distance = sum(
                1 / row["distance"]
                for _, row in distances_df.iterrows()
                if row["unit"] in top_units
            )

            weights = {
                row["unit"]: (1 / row["distance"]) / total_inverse_distance
                for _, row in distances_df.iterrows()
                if row["unit"] in top_units
            }

            # Apply weights to create synthetic control
            synthetic_control = []

            for time_point in sorted(data_copy[time_column].unique()):
                # Get outcome values for each control unit at this time
                time_data = data_copy[data_copy[time_column] == time_point]

                # Calculate weighted average
                weighted_sum = 0
                for unit, weight in weights.items():
                    unit_data = time_data[time_data[unit_column] == unit]
                    if not unit_data.empty:
                        weighted_sum += unit_data[outcome_column].iloc[0] * weight

                # Get treatment unit value at this time
                treatment_value = None
                treatment_time_data = time_data[time_data[unit_column] == treatment_unit]
                if not treatment_time_data.empty:
                    treatment_value = treatment_time_data[outcome_column].iloc[0]

                synthetic_control.append(
                    {
                        "time": time_point,
                        "treatment_value": treatment_value,
                        "synthetic_control_value": weighted_sum,
                        "difference": (
                            treatment_value - weighted_sum if treatment_value is not None else None
                        ),
                    }
                )

            # Convert to DataFrame
            results_df = pd.DataFrame(synthetic_control)

            # Calculate effect
            pre_period_diff = results_df[results_df["time"] <= pre_treatment_end][
                "difference"
            ].mean()
            post_period_diff = results_df[results_df["time"] >= post_treatment_start][
                "difference"
            ].mean()

            causal_effect = post_period_diff - pre_period_diff

            # Prepare visualization
            plot_path = None
            if VISUALIZATION_AVAILABLE:
                plt.figure(figsize=(12, 6))

                # Plot actual treatment series
                plt.plot(
                    results_df["time"],
                    results_df["treatment_value"],
                    "b-",
                    label=f"Actual {treatment_unit}",
                )

                # Plot synthetic control
                plt.plot(
                    results_df["time"],
                    results_df["synthetic_control_value"],
                    "r--",
                    label="Synthetic Control",
                )

                # Add vertical line at intervention time
                plt.axvline(x=pre_treatment_end, color="k", linestyle="--", label="Intervention")

                plt.title(f"Synthetic Control Analysis: {outcome_column} for {treatment_unit}")
                plt.xlabel("Time")
                plt.ylabel(outcome_column)
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Save plot
                plot_filename = f"synthetic_control_{treatment_unit}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plot_path = os.path.join(self.config["output_dir"], plot_filename)
                plt.savefig(plot_path)
                plt.close()

            # Create analysis ID and store results
            analysis_id = f"sc_{treatment_unit}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            analysis_results = {
                "id": analysis_id,
                "status": "success",
                "method": "synthetic_control",
                "treatment_unit": treatment_unit,
                "outcome_metric": outcome_column,
                "estimated_effect": float(causal_effect),
                "pre_treatment_difference": float(pre_period_diff),
                "post_treatment_difference": float(post_period_diff),
                "control_units_used": top_units,
                "weights": weights,
                "plot_path": plot_path,
                "timestamp": datetime.now().isoformat(),
            }

            # Determine direction and magnitude
            if abs(causal_effect) < 0.05 * abs(results_df["treatment_value"].mean()):
                analysis_results["interpretation"] = "No substantial effect detected"
            elif causal_effect > 0:
                analysis_results["interpretation"] = (
                    f"Positive effect of {causal_effect:.2f} detected"
                )
            else:
                analysis_results["interpretation"] = (
                    f"Negative effect of {causal_effect:.2f} detected"
                )

            # Store the analysis
            self.analyses[analysis_id] = analysis_results

            self.logger.info(f"Synthetic control analysis completed. Effect: {causal_effect:.4f}")

        except Exception as e:
            self.logger.error(f"Error creating synthetic control: {str(e)}")
            analysis_results = {
                "status": "error",
                "method": "synthetic_control",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            self._track_execution(start_time, success=False)
            return analysis_results

        self._track_execution(start_time, success=True)
        return analysis_results

    def run_uplift_analysis(
        self, data: pd.DataFrame, treatment_col: str, outcome_col: str, features: List[str]
    ) -> Dict[str, Any]:
        """
        Perform uplift modeling to identify segments most responsive to treatment.

        Args:
            data: DataFrame containing features, treatment assignment, and outcome.
            treatment_col: Name of the column indicating treatment assignment (0 or 1).
            outcome_col: Name of the column indicating the outcome (e.g., conversion).
            features: List of feature column names to use for modeling.

        Returns:
            Dictionary with uplift analysis results (e.g., segment-specific uplift scores).
        """
        start_time = datetime.now()
        self.logger.info("Running uplift analysis...")

        if not ECONML_AVAILABLE:
            error_msg = "econml package not available. Install with: pip install econml"
            self.logger.error(error_msg)
            return {"status": "error", "message": error_msg}

        try:
            # Prepare features and target variables
            X = data[features].copy()
            T = data[treatment_col].copy()
            Y = data[outcome_col].copy()

            # Create and fit causal forest model for heterogeneous treatment effects
            model = CausalForestDML(
                n_estimators=100, min_samples_leaf=10, max_depth=5, verbose=0, random_state=42
            )
            model.fit(Y, T, X=X)

            # Estimate conditional average treatment effects (CATE)
            cate_estimates = model.effect(X)

            # Add CATE estimates to the data
            data_with_cate = data.copy()
            data_with_cate["cate"] = cate_estimates

            # Segment analysis
            num_segments = min(
                5, len(data) // 50
            )  # Create at most 5 segments, but ensure enough data in each

            # Create segments based on estimated uplift
            data_with_cate["uplift_percentile"] = pd.qcut(
                data_with_cate["cate"], q=num_segments, labels=False
            )

            # Calculate average uplift and metrics by segment
            segment_analysis = (
                data_with_cate.groupby("uplift_percentile")
                .agg(
                    {
                        "cate": "mean",
                        outcome_col: "mean",
                        treatment_col: "mean",
                        **{
                            feature: "mean"
                            for feature in features
                            if np.issubdtype(data[feature].dtype, np.number)
                        },
                    }
                )
                .reset_index()
            )

            # Rename segments for better interpretability (1 = highest uplift)
            segment_analysis["segment"] = num_segments - segment_analysis["uplift_percentile"]
            segment_analysis = segment_analysis.sort_values("segment")

            # Create visualizations
            plot_path = None
            if VISUALIZATION_AVAILABLE:
                plt.figure(figsize=(10, 6))

                # Plot average CATE by segment
                bars = plt.bar(segment_analysis["segment"], segment_analysis["cate"], alpha=0.7)

                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.002,
                        f"{height:.3f}",
                        ha="center",
                        va="bottom",
                    )

                plt.title(f"Average Treatment Effect by Segment for {outcome_col}")
                plt.xlabel("Segment (1 = Highest Uplift)")
                plt.ylabel(f"Estimated Effect on {outcome_col}")
                plt.xticks(segment_analysis["segment"])
                plt.grid(axis="y", alpha=0.3)

                # Save plot
                plot_filename = (
                    f"uplift_segments_{outcome_col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                plot_path = os.path.join(self.config["output_dir"], plot_filename)
                plt.savefig(plot_path)
                plt.close()

            # Format segment analysis as a list of dicts
            segments = []
            for _, row in segment_analysis.iterrows():
                segment_data = {
                    "segment": int(row["segment"]),
                    "size": int(
                        sum(data_with_cate["uplift_percentile"] == row["uplift_percentile"])
                    ),
                    "average_uplift": float(row["cate"]),
                    "outcome_rate": float(row[outcome_col]),
                    "features": {
                        feature: float(row[feature])
                        for feature in features
                        if feature in row.index and np.issubdtype(data[feature].dtype, np.number)
                    },
                }
                segments.append(segment_data)

            # Create analysis ID and store results
            analysis_id = f"uplift_{outcome_col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            analysis_results = {
                "id": analysis_id,
                "status": "success",
                "method": "uplift_modeling",
                "outcome_metric": outcome_col,
                "average_treatment_effect": float(np.mean(cate_estimates)),
                "segments": segments,
                "plot_path": plot_path,
                "timestamp": datetime.now().isoformat(),
            }

            # Generate insights
            if segments and len(segments) > 0:
                best_segment = max(segments, key=lambda x: x["average_uplift"])
                worst_segment = min(segments, key=lambda x: x["average_uplift"])

                insights = [
                    f"Best segment (#{best_segment['segment']}) shows an uplift of {best_segment['average_uplift']:.4f}",
                    f"Worst segment (#{worst_segment['segment']}) shows an uplift of {worst_segment['average_uplift']:.4f}",
                ]

                # Identify key distinguishing features
                if best_segment["features"] and worst_segment["features"]:
                    diff_features = []
                    for feature, value in best_segment["features"].items():
                        worst_value = worst_segment["features"].get(feature)
                        if worst_value is not None:
                            pct_diff = (value - worst_value) / (
                                abs(worst_value) if worst_value != 0 else 1
                            )
                            if abs(pct_diff) > 0.2:  # 20% difference threshold
                                diff_features.append(
                                    {
                                        "feature": feature,
                                        "best_value": value,
                                        "worst_value": worst_value,
                                        "pct_diff": pct_diff,
                                    }
                                )

                    if diff_features:
                        sorted_diffs = sorted(
                            diff_features, key=lambda x: abs(x["pct_diff"]), reverse=True
                        )
                        for i, diff in enumerate(sorted_diffs[:3]):  # Top 3 differences
                            insights.append(
                                f"Feature '{diff['feature']}' is {abs(diff['pct_diff']) * 100:.1f}% "
                                + ("higher" if diff["pct_diff"] > 0 else "lower")
                                + f" in the best segment ({diff['best_value']:.2f} vs {diff['worst_value']:.2f})"
                            )

                analysis_results["insights"] = insights

            # Store the analysis
            self.analyses[analysis_id] = analysis_results

            self.logger.info(f"Uplift analysis completed with {len(segments)} segments")

        except Exception as e:
            self.logger.error(f"Error during uplift analysis: {str(e)}")
            analysis_results = {
                "status": "error",
                "method": "uplift_modeling",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            self._track_execution(start_time, success=False)
            return analysis_results

        self._track_execution(start_time, success=True)
        return analysis_results

    def design_ab_experiment(
        self,
        hypothesis: str,
        control_group: Dict,
        treatment_group: Dict,
        metrics: List[str],
        duration_days: int,
    ) -> Dict[str, Any]:
        """
        Design an A/B experiment structure for Google Ads.

        Args:
            hypothesis: The hypothesis being tested.
            control_group: Definition of the control group (e.g., specific campaigns/ad groups).
            treatment_group: Definition of the treatment group and the change being applied.
            metrics: List of primary and secondary metrics to track.
            duration_days: Planned duration of the experiment.

        Returns:
            Dictionary describing the experiment design.
        """
        start_time = datetime.now()
        self.logger.info(f"Designing A/B experiment for hypothesis: {hypothesis}")

        experiment_design = {
            "hypothesis": hypothesis,
            "control_group": control_group,
            "treatment_group": treatment_group,
            "metrics": metrics,
            "duration_days": duration_days,
            "status": "designed",
            "start_date": None,  # To be set when experiment starts
            "end_date": None,
            "required_sample_size": None,  # TODO: Implement power analysis
        }

        # TODO: Implement power analysis to estimate required sample size/duration
        self.logger.info(f"A/B experiment designed: {experiment_design}")
        self._track_execution(start_time, success=True)
        return experiment_design

    def analyze_experiment_results(
        self, experiment_data: pd.DataFrame, metric: str, group_col: str = "group"
    ) -> Dict[str, Any]:  # 'control' or 'treatment'
        """
        Analyze the results of a completed A/B experiment using statistical tests.

        Args:
            experiment_data: DataFrame with experiment results (one row per user/unit),
                             including the metric and group assignment.
            metric: The primary metric to analyze.
            group_col: The column indicating group assignment ('control' or 'treatment').

        Returns:
            Dictionary with statistical analysis results (p-value, confidence interval, etc.).
        """
        start_time = datetime.now()
        self.logger.info(f"Analyzing experiment results for metric: {metric}")

        # Placeholder: Implement statistical significance testing (e.g., t-test, Z-test)
        try:
            # control_results = experiment_data[experiment_data[group_col] == 'control'][metric]
            # treatment_results = experiment_data[experiment_data[group_col] == 'treatment'][metric]

            # from scipy.stats import ttest_ind
            # t_stat, p_value = ttest_ind(treatment_results, control_results, equal_var=False) # Welch's t-test

            # Calculate confidence interval for the difference
            # ... (implementation needed)

            # Placeholder result
            results = {
                "status": "placeholder",
                "message": "Statistical analysis not yet implemented.",
                "metric": metric,
                "control_mean": None,  # control_results.mean(),
                "treatment_mean": None,  # treatment_results.mean(),
                "difference": None,  # treatment_results.mean() - control_results.mean(),
                "p_value": None,  # p_value,
                "confidence_interval": None,  # [lower, upper]
            }
            self.logger.warning("Experiment result analysis is a placeholder.")

        except Exception as e:
            self.logger.error(f"Error analyzing experiment results: {str(e)}")
            results = {"status": "error", "message": str(e)}
            self._track_execution(start_time, success=False)
            return results

        self._track_execution(start_time, success=True)
        return results

    def run(self, **kwargs):
        """
        Placeholder run method. Causal analyses are typically run on demand or after experiments.
        """
        self.logger.info("CausalInferenceService run method called (currently a placeholder).")
        pass


# Import visualization libraries if available
try:
    import matplotlib.pyplot as plt

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
